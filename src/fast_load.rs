//! Fast loader for the existing `Tachiom<M>` on-disk format.
//!
//! HNSW is decoded with bincode/serde. The flat sections that follow are read
//! directly.
//!
//! Wire layout (little-endian; fixed-int bincode encodes `usize` and lengths
//! as `u64`):
//!
//! ```text
//! Tachiom<M>
//!   centroids: HNSW<...>                          bincode/serde
//!   ---- manual decoding begins here ----
//!   inverted_lists: Vec<u32>                      u64 len + u32[]
//!   offsets:        Vec<usize>                    u64 len + u64[]
//!   residuals: MultiVectorDataset<MultiVecTwoLevelProductQuantizer<M, f16>>
//!     data:    Box<[u8]>                          u64 len + u8[]
//!     offsets: Box<[usize]>                       u64 len + u64[]
//!     encoder:
//!       token_dim:        usize                   u64
//!       dsub:             usize                   u64
//!       coarse_centroids: PlainDenseDataset<f32, SquaredEuclideanDistance>
//!                                                 n_vecs u64, data (u64 len + f32[]), d u64
//!       pq_centroids:     Box<[PlainDenseDataset<f32, ..>]>
//!                                                 u64 len (= M), then M x the same triple
//!       with_norms:       bool                    u8, 0 or 1
//!   max_doc_tokens: usize                         u64
//!   dataset_mean:   Option<Box<[f32]>>            u8 tag; 1 => u64 len + f32[]
//! ```

use bincode::error::DecodeError;
use half::f16;
use std::fs::File;
use std::io::{BufReader, Read};

use vectorium::{
    DenseDataset, IndexIoError, MultiVecTwoLevelProductQuantizer, MultiVectorDataset,
    PlainDenseDataset, PlainDenseQuantizer, SquaredEuclideanDistance,
};

use crate::tachiom::{HNSWCentroids, Tachiom};

const READ_BUFFER_BYTES: usize = 1 << 20;

const CHUNK_BYTES: usize = 4 * 1024 * 1024;

// Must match vectorium::IndexSerializer::save_index.
fn config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian()
}

type Result<T> = std::result::Result<T, IndexIoError>;

fn corrupt_error(message: String) -> IndexIoError {
    IndexIoError::Decode(DecodeError::OtherString(message))
}

fn corrupt<T>(message: String) -> Result<T> {
    Err(corrupt_error(message))
}

fn allocate<T>(len: usize, field: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| corrupt_error(format!("{field}: cannot allocate {len} elements: {e}")))?;
    Ok(values)
}

/// Buffered reader that tracks bytes consumed independently of read-ahead.
struct Reader {
    inner: BufReader<File>,
    pos: u64,
    len: u64,
}

impl Read for Reader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Reader {
    fn open(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            inner: BufReader::with_capacity(READ_BUFFER_BYTES, file),
            pos: 0,
            len,
        })
    }

    fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    fn fill(&mut self, buf: &mut [u8], field: &str) -> Result<()> {
        if buf.len() as u64 > self.remaining() {
            return corrupt(format!(
                "{field}: needs {} bytes at offset {}, file has {} left",
                buf.len(),
                self.pos,
                self.remaining()
            ));
        }
        self.read_exact(buf)?;
        Ok(())
    }

    fn u64(&mut self, field: &str) -> Result<u64> {
        let mut bytes = [0u8; 8];
        self.fill(&mut bytes, field)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn usize(&mut self, field: &str) -> Result<usize> {
        let value = self.u64(field)?;
        usize::try_from(value)
            .map_err(|_| corrupt_error(format!("{field}: {value} does not fit usize")))
    }

    // Bincode uses one-byte 0/1 tags for bool and Option.
    fn flag(&mut self, field: &str) -> Result<bool> {
        let mut byte = [0];
        self.fill(&mut byte, field)?;
        match byte[0] {
            0 => Ok(false),
            1 => Ok(true),
            other => corrupt(format!("{field}: expected 0 or 1, got {other}")),
        }
    }

    fn byte_array(&mut self, field: &str) -> Result<Vec<u8>> {
        let bytes = self.usize(field)?;
        if bytes as u64 > self.remaining() {
            return corrupt(format!(
                "{field}: length {bytes} overruns the file ({} bytes left at offset {})",
                self.remaining(),
                self.pos
            ));
        }
        let mut buf = allocate(bytes, field)?;
        buf.resize(bytes, 0);
        self.fill(&mut buf, field)?;
        Ok(buf)
    }

    fn typed_array<T>(
        &mut self,
        width: usize,
        field: &str,
        convert: impl Fn(&[u8]) -> Result<T>,
    ) -> Result<Vec<T>> {
        let n = self.usize(field)?;
        self.typed_values(n, width, field, convert)
    }

    fn typed_values<T>(
        &mut self,
        n: usize,
        width: usize,
        field: &str,
        convert: impl Fn(&[u8]) -> Result<T>,
    ) -> Result<Vec<T>> {
        assert!(width != 0 && CHUNK_BYTES.is_multiple_of(width));
        let total_bytes = n.checked_mul(width).ok_or_else(|| {
            corrupt_error(format!(
                "{field}: {n} elements of {width} bytes overflow usize"
            ))
        })?;
        let remaining = self.remaining();
        if total_bytes as u64 > remaining {
            return corrupt(format!(
                "{field}: length {n} overruns the file \
                 ({remaining} bytes left at offset {})",
                self.pos
            ));
        }
        let mut out = allocate(n, field)?;
        if total_bytes == 0 {
            return Ok(out);
        }
        let mut scratch = vec![0u8; CHUNK_BYTES.min(total_bytes)];
        let mut left = total_bytes;
        while left > 0 {
            let take = left.min(scratch.len());
            self.fill(&mut scratch[..take], field)?;
            for bytes in scratch[..take].chunks_exact(width) {
                out.push(convert(bytes)?);
            }
            left -= take;
        }
        Ok(out)
    }

    fn u32_array(&mut self, field: &str) -> Result<Vec<u32>> {
        self.typed_array(4, field, |c| {
            Ok(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        })
    }

    fn f32_array(&mut self, field: &str) -> Result<Vec<f32>> {
        self.typed_array(4, field, |c| {
            Ok(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        })
    }

    fn usize_array(&mut self, field: &str) -> Result<Vec<usize>> {
        self.typed_array(8, field, |c| {
            let value = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            usize::try_from(value)
                .map_err(|_| corrupt_error(format!("{field}: {value} does not fit usize")))
        })
    }
}

fn read_f32_dataset(
    reader: &mut Reader,
    field: &str,
) -> Result<PlainDenseDataset<f32, SquaredEuclideanDistance>> {
    let n_vecs = reader.usize(&format!("{field}.n_vecs"))?;
    let data = reader.f32_array(&format!("{field}.data"))?;
    let d = reader.usize(&format!("{field}.encoder.d"))?;
    if n_vecs.checked_mul(d) != Some(data.len()) {
        return corrupt(format!(
            "{field}: {} values but n_vecs {n_vecs} x d {d}",
            data.len()
        ));
    }
    Ok(DenseDataset::from_raw(
        data.into_boxed_slice(),
        n_vecs,
        PlainDenseQuantizer::new(d),
    ))
}

pub(crate) fn load_index<const M: usize>(path: &str) -> Result<Tachiom<M>> {
    let mut reader = Reader::open(path)?;

    let centroids: HNSWCentroids = bincode::serde::decode_from_std_read(&mut reader, config())?;

    let inverted_lists = reader.u32_array("Tachiom.inverted_lists")?;
    let offsets = reader.usize_array("Tachiom.offsets")?;

    let residual_data = reader.byte_array("residuals.data")?;
    let doc_offsets = reader.usize_array("residuals.offsets")?;

    let token_dim = reader.usize("residuals.encoder.token_dim")?;
    reader.usize("residuals.encoder.dsub")?;
    let coarse_centroids = read_f32_dataset(&mut reader, "residuals.encoder.coarse_centroids")?;

    let n_codebooks = reader.usize("residuals.encoder.pq_centroids.len")?;
    if n_codebooks != M {
        return corrupt(format!(
            "residuals.encoder: {n_codebooks} codebooks, M is {M}"
        ));
    }
    let mut codebooks = allocate(M, "residuals.encoder.pq_centroids")?;
    for i in 0..M {
        let field = format!("residuals.encoder.pq_centroids[{i}]");
        codebooks.push(read_f32_dataset(&mut reader, &field)?);
    }
    let with_norms = reader.flag("residuals.encoder.with_norms")?;

    let max_doc_tokens = reader.usize("Tachiom.max_doc_tokens")?;
    let dataset_mean = if reader.flag("Tachiom.dataset_mean")? {
        Some(reader.f32_array("Tachiom.dataset_mean")?.into_boxed_slice())
    } else {
        None
    };

    let encoder = MultiVecTwoLevelProductQuantizer::<M, f16>::from_pretrained(
        token_dim,
        coarse_centroids,
        codebooks,
        with_norms,
    );
    let residuals = MultiVectorDataset::from_raw(
        residual_data.into_boxed_slice(),
        doc_offsets.into_boxed_slice(),
        encoder,
    );

    Ok(Tachiom {
        centroids,
        inverted_lists,
        offsets,
        residuals,
        max_doc_tokens,
        dataset_mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::io::Seek;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vectorium::{
        DenseMultiVectorView, IndexSerializer, MultiVectorDataset, PlainMultiVecQuantizer,
    };

    use crate::hnsw::HNSWBuildConfiguration;
    use crate::tachiom::{TachiomBuildParams, TachiomInputDataset};

    const M: usize = 32;
    const DIM: usize = 64;
    const N_DOCS: usize = 400;
    const TOKENS_PER_DOC: usize = 8; // Enough training tokens for 256-entry PQ codebooks.

    fn build_fixture(center: bool, normalize: bool) -> Tachiom<M> {
        let mut rng = StdRng::seed_from_u64(7);
        let n_tokens = N_DOCS * TOKENS_PER_DOC;
        let flat: Vec<f16> = (0..n_tokens * DIM)
            .map(|_| f16::from_f32(rng.gen_range(-1.0f32..1.0f32)))
            .collect();
        let offsets: Vec<usize> = (0..=N_DOCS).map(|i| i * TOKENS_PER_DOC * DIM).collect();
        let raw: TachiomInputDataset = MultiVectorDataset::from_raw(
            flat.into_boxed_slice(),
            offsets.into_boxed_slice(),
            PlainMultiVecQuantizer::<f16>::new(DIM),
        );
        let params = TachiomBuildParams {
            token_ids: (0..n_tokens).map(|i| i % 97).collect(),
            total_centroids: 128,
            tac_n_iter: 3,
            tac_micro_threshold: None,
            tac_small_threshold: None,
            pq_sample_size: n_tokens,
            pq_n_iter: 3,
            normalize,
            pq_seed: Some(42),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(8)
                .with_ef_construction(40),
            center_dataset: center,
        };
        Tachiom::<M>::build_index(raw, &params)
    }

    fn temp_path(name: &str) -> String {
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path: PathBuf = std::env::temp_dir();
        path.push(format!(
            "tachiom_fast_load_{}_{}_{name}.bin",
            std::process::id(),
            n,
        ));
        path.to_str().unwrap().to_owned()
    }

    fn legacy_load_index<const M: usize>(filename: &str) -> Tachiom<M> {
        let file = File::open(filename).unwrap();
        bincode::serde::decode_from_std_read(&mut BufReader::new(file), config()).unwrap()
    }

    fn queries(rng: &mut StdRng, n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| (0..4 * DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
            .collect()
    }

    fn search_all(index: &Tachiom<M>, qs: &[Vec<f32>]) -> Vec<Vec<(f32, u32)>> {
        qs.iter()
            .map(|q| {
                index.search(
                    DenseMultiVectorView::new(q, DIM),
                    10,
                    8,
                    64,
                    40,
                    None,
                    None,
                    None,
                )
            })
            .collect()
    }

    #[test]
    fn fast_load_matches_legacy() {
        for center in [false, true] {
            for normalize in [false, true] {
                let path = temp_path("equivalence");
                build_fixture(center, normalize).save_index(&path).unwrap();
                let original = std::fs::read(&path).unwrap();

                let legacy = legacy_load_index::<M>(&path);
                let fast = Tachiom::<M>::load_index(&path).unwrap();

                let bytes = |ix: &Tachiom<M>| bincode::serde::encode_to_vec(ix, config()).unwrap();
                assert_eq!(bytes(&fast), original, "fast-load re-encoding differs");
                assert_eq!(bytes(&legacy), original, "legacy re-encoding differs");
                assert_eq!(fast.dataset_mean.is_some(), center);

                let mut rng = StdRng::seed_from_u64(11);
                let qs = queries(&mut rng, 5);
                let fast_hits = search_all(&fast, &qs);
                assert!(
                    fast_hits.iter().any(|hits| !hits.is_empty()),
                    "fixture searches returned nothing, the comparison would be vacuous"
                );
                assert_eq!(fast_hits, search_all(&legacy, &qs), "search results differ");

                std::fs::remove_file(&path).unwrap();
            }
        }
    }

    #[test]
    fn buffered_prefix_uses_logical_position() {
        let path = temp_path("split");
        let index = build_fixture(true, true);
        let expected_len = index.inverted_lists.len() as u64;
        index.save_index(&path).unwrap();
        assert!(
            index.centroids.max_level() > 0,
            "fixture must have multiple HNSW levels"
        );

        let mut reader = Reader::open(&path).unwrap();
        let _: HNSWCentroids = bincode::serde::decode_from_std_read(&mut reader, config()).unwrap();
        let fd_pos = reader.inner.get_mut().stream_position().unwrap();
        assert!(
            fd_pos > reader.pos,
            "expected read-ahead: fd at {fd_pos}, logical cursor at {}",
            reader.pos
        );
        assert_eq!(reader.u64("inverted_lists length").unwrap(), expected_len);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn trailing_bytes_match_legacy_behavior() {
        let path = temp_path("trailing_bytes");
        build_fixture(true, true).save_index(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0xAB, 0xCD, 0xEF, 0x01]);
        std::fs::write(&path, &bytes).unwrap();

        let legacy = legacy_load_index::<M>(&path);
        let fast = Tachiom::<M>::load_index(&path).unwrap();
        let encode = |ix: &Tachiom<M>| bincode::serde::encode_to_vec(ix, config()).unwrap();
        assert_eq!(encode(&fast), encode(&legacy));

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn typed_array_crosses_chunk_boundary() {
        let path = temp_path("chunked_array");
        let n = CHUNK_BYTES / 4 + 1;
        let mut bytes = Vec::with_capacity(8 + n * 4);
        bytes.extend_from_slice(&(n as u64).to_le_bytes());
        for value in 0..n as u32 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();

        let mut reader = Reader::open(&path).unwrap();
        let values = reader.u32_array("chunked").unwrap();
        assert_eq!(values.len(), n);
        assert_eq!(values[0], 0);
        assert_eq!(values[n - 1], n as u32 - 1);
        assert_eq!(reader.remaining(), 0);

        std::fs::remove_file(&path).unwrap();
    }
}

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

use vectorium::core::dataset::Dataset;
use vectorium::core::index::IndexStats;
use vectorium::{
    DenseDataset, IndexIoError, MultiVecTwoLevelProductQuantizer, MultiVectorDataset,
    PlainDenseDataset, PlainDenseQuantizer, SquaredEuclideanDistance, VectorEncoder,
};

use crate::tachiom::{HNSWCentroids, Tachiom};

const KSUB: usize = 256;

const COARSE_ID_BYTES: usize = std::mem::size_of::<u32>();

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
            .map_err(|_| IndexIoError::Decode(DecodeError::OutsideUsizeRange(value)))
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

    fn f32_array_exact(&mut self, field: &str, expected: usize) -> Result<Vec<f32>> {
        let n = self.usize(field)?;
        if n != expected {
            return corrupt(format!("{field}: length {n}, expected {expected}"));
        }
        self.typed_values(n, 4, field, |c| {
            Ok(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        })
    }

    fn usize_array(&mut self, field: &str) -> Result<Vec<usize>> {
        self.typed_array(8, field, |c| {
            let value = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            usize::try_from(value)
                .map_err(|_| IndexIoError::Decode(DecodeError::OutsideUsizeRange(value)))
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

fn validate_offsets(field: &str, offsets: &[usize], data_len: usize) -> Result<()> {
    if offsets.first() != Some(&0) {
        return corrupt(format!(
            "{field}: first offset must be 0, got {:?}",
            offsets.first()
        ));
    }
    if offsets.last() != Some(&data_len) {
        return corrupt(format!(
            "{field}: last offset {:?} does not equal data length {data_len}",
            offsets.last()
        ));
    }
    if let Some(i) = offsets.windows(2).position(|w| w[1] < w[0]) {
        return corrupt(format!(
            "{field}: offsets decrease at index {}: {} -> {}",
            i + 1,
            offsets[i],
            offsets[i + 1]
        ));
    }
    Ok(())
}

fn validate_encoded_documents(
    data: &[u8],
    offsets: &[usize],
    n_coarse: usize,
    bytes_per_token: usize,
) -> Result<()> {
    for (doc_id, bounds) in offsets.windows(2).enumerate() {
        let document = &data[bounds[0]..bounds[1]];
        let span = document.len();
        if !span.is_multiple_of(bytes_per_token) {
            return corrupt(format!(
                "residuals document {doc_id}: {span} bytes is not divisible by encoded width {bytes_per_token}"
            ));
        }
        let n_tokens = span / bytes_per_token;
        let coarse_bytes = n_tokens.checked_mul(COARSE_ID_BYTES).ok_or_else(|| {
            corrupt_error(format!(
                "residuals document {doc_id}: coarse ID byte count overflows"
            ))
        })?;
        // Vectorium's scorer assumes coarse IDs are in range.
        for (token, bytes) in document[..coarse_bytes]
            .as_chunks::<COARSE_ID_BYTES>()
            .0
            .iter()
            .enumerate()
        {
            let id = u32::from_le_bytes(*bytes) as usize;
            if id >= n_coarse {
                return corrupt(format!(
                    "residuals document {doc_id}, token {token}: coarse centroid {id} is out of range for {n_coarse} centroids"
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn load_index<const M: usize>(path: &str) -> Result<Tachiom<M>> {
    if M == 0 || !M.is_multiple_of(4) {
        return corrupt(format!(
            "Tachiom M must be nonzero and divisible by 4, got {M}"
        ));
    }
    let mut reader = Reader::open(path)?;

    let centroids: HNSWCentroids = bincode::serde::decode_from_std_read(&mut reader, config())?;
    let n_centroids = centroids.n_elements();
    let levels = centroids.nodes_per_level();
    if n_centroids == 0 || levels.last() != Some(&n_centroids) {
        return corrupt(format!(
            "invalid HNSW ground level size {:?} for {n_centroids} centroids",
            levels.last()
        ));
    }
    if n_centroids > u32::MAX as usize {
        return corrupt(format!("{n_centroids} centroids exceed the u32 ID range"));
    }

    let inverted_lists = reader.u32_array("Tachiom.inverted_lists")?;
    let offsets = reader.usize_array("Tachiom.offsets")?;
    validate_offsets("Tachiom.offsets", &offsets, inverted_lists.len())?;
    // Search reads offsets[cidx + 1], so the sentinel is required.
    if offsets.len() - 1 != n_centroids {
        return corrupt(format!(
            "Tachiom.offsets has {} entries for {n_centroids} centroids",
            offsets.len()
        ));
    }

    let residual_data = reader.byte_array("residuals.data")?;
    let doc_offsets = reader.usize_array("residuals.offsets")?;
    validate_offsets("residuals.offsets", &doc_offsets, residual_data.len())?;
    let n_docs = doc_offsets.len() - 1;
    if n_docs > u32::MAX as usize {
        return corrupt(format!("{n_docs} documents exceed the u32 ID range"));
    }
    if let Some((position, &doc_id)) = inverted_lists
        .iter()
        .enumerate()
        .find(|&(_, &doc_id)| doc_id as usize >= n_docs)
    {
        return corrupt(format!(
            "Tachiom.inverted_lists[{position}] contains document {doc_id}, but the index has {n_docs} documents"
        ));
    }

    let token_dim = reader.usize("residuals.encoder.token_dim")?;
    let dsub = reader.usize("residuals.encoder.dsub")?;
    let coarse_centroids = read_f32_dataset(&mut reader, "residuals.encoder.coarse_centroids")?;
    let n_coarse = coarse_centroids.len();
    // dsub is serialized separately but derived from token_dim and M.
    if token_dim == 0
        || !token_dim.is_multiple_of(M)
        || dsub != token_dim / M
        || coarse_centroids.output_dim() != token_dim
    {
        return corrupt(format!(
            "residuals.encoder: token_dim {token_dim}, dsub {dsub}, coarse d {} with M {M}",
            coarse_centroids.output_dim()
        ));
    }
    if n_centroids != n_coarse || centroids.dim() != token_dim {
        return corrupt(format!(
            "centroid mismatch: HNSW is {n_centroids} x {}, residual encoder is {n_coarse} x {token_dim}",
            centroids.dim()
        ));
    }

    let n_codebooks = reader.usize("residuals.encoder.pq_centroids.len")?;
    if n_codebooks != M {
        return corrupt(format!(
            "residuals.encoder: {n_codebooks} codebooks, M is {M}"
        ));
    }
    let mut codebooks = allocate(M, "residuals.encoder.pq_centroids")?;
    for i in 0..M {
        let field = format!("residuals.encoder.pq_centroids[{i}]");
        let codebook = read_f32_dataset(&mut reader, &field)?;
        if codebook.len() != KSUB || codebook.output_dim() != dsub {
            return corrupt(format!(
                "{field}: {} x {}, expected {KSUB} x dsub {dsub}",
                codebook.len(),
                codebook.output_dim()
            ));
        }
        codebooks.push(codebook);
    }
    let with_norms = reader.flag("residuals.encoder.with_norms")?;

    let max_doc_tokens = reader.usize("Tachiom.max_doc_tokens")?;
    let dataset_mean = if reader.flag("Tachiom.dataset_mean")? {
        Some(
            reader
                .f32_array_exact("Tachiom.dataset_mean", token_dim)?
                .into_boxed_slice(),
        )
    } else {
        None
    };

    let encoder = MultiVecTwoLevelProductQuantizer::<M, f16>::from_pretrained(
        token_dim,
        coarse_centroids,
        codebooks,
        with_norms,
    );
    validate_encoded_documents(&residual_data, &doc_offsets, n_coarse, encoder.output_dim())?;
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
    use std::sync::OnceLock;
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

    // Locate fields by walking the serialized layout rather than using fixed offsets.

    struct Layout {
        mid_prefix_pos: u64,
        mid_bulk_array_pos: u64,
        inverted_lists_len_pos: u64,
        inverted_lists_first_pos: u64,
        residuals_data_len_pos: u64,
        residuals_data_first_pos: u64,
        offsets_first_pos: u64,
        offsets_second_pos: u64,
        offsets_last_pos: u64,
        residuals_offsets_first_pos: u64,
        residuals_offsets_second_pos: u64,
        residuals_offsets_last_pos: u64,
        dsub_pos: u64,
        coarse_n_vecs_pos: u64,
        coarse_n_vecs_value: u64,
        pq_centroids_len_pos: u64,
        with_norms_pos: u64,
        dataset_mean_flag_pos: u64,
        dataset_mean_len_pos: u64,
        n_docs: usize,
    }

    struct MalformedFixture {
        bytes: Vec<u8>,
        layout: Layout,
    }

    fn malformed_fixture() -> &'static MalformedFixture {
        static FIXTURE: OnceLock<MalformedFixture> = OnceLock::new();

        FIXTURE.get_or_init(|| {
            let path = temp_path("malformed_base");
            build_fixture(true, true).save_index(&path).unwrap();

            let layout = locate(&path);
            let bytes = std::fs::read(&path).unwrap();
            std::fs::remove_file(&path).unwrap();

            MalformedFixture { bytes, layout }
        })
    }

    fn locate(path: &str) -> Layout {
        let mut reader = Reader::open(path).unwrap();
        let _: HNSWCentroids = bincode::serde::decode_from_std_read(&mut reader, config()).unwrap();
        let mid_prefix_pos = reader.pos / 2;

        let inverted_lists_len_pos = reader.pos;
        let inverted_lists_first_pos = inverted_lists_len_pos + 8;
        reader.u32_array("inverted_lists").unwrap();

        let offsets_len = reader.pos;
        let offsets = reader.usize_array("offsets").unwrap();
        let offsets_first_pos = offsets_len + 8;
        let offsets_second_pos = offsets_first_pos + 8;
        let offsets_last_pos = offsets_first_pos + (offsets.len() as u64 - 1) * 8;

        let residuals_data_len_pos = reader.pos;
        let residuals_data_first_pos = residuals_data_len_pos + 8;
        reader.byte_array("residuals.data").unwrap();
        let mid_bulk_array_pos = (residuals_data_len_pos + reader.pos) / 2;

        let residuals_offsets_len = reader.pos;
        let doc_offsets = reader.usize_array("residuals.offsets").unwrap();
        let n_docs = doc_offsets.len() - 1;
        let residuals_offsets_first_pos = residuals_offsets_len + 8;
        let residuals_offsets_second_pos = residuals_offsets_first_pos + 8;
        let residuals_offsets_last_pos =
            residuals_offsets_first_pos + (doc_offsets.len() as u64 - 1) * 8;

        reader.usize("token_dim").unwrap();
        let dsub_pos = reader.pos;
        reader.usize("dsub").unwrap();
        let coarse_n_vecs_pos = reader.pos;
        let coarse_n_vecs_value = reader.u64("coarse_centroids.n_vecs").unwrap();
        reader.f32_array("coarse_centroids.data").unwrap();
        reader.usize("coarse_centroids.d").unwrap();

        let pq_centroids_len_pos = reader.pos;
        let n_codebooks = reader.usize("pq_centroids.len").unwrap();
        for _ in 0..n_codebooks {
            reader.usize("cb.n_vecs").unwrap();
            reader.f32_array("cb.data").unwrap();
            reader.usize("cb.d").unwrap();
        }
        let with_norms_pos = reader.pos;
        reader.flag("with_norms").unwrap();

        reader.usize("max_doc_tokens").unwrap();
        let dataset_mean_flag_pos = reader.pos;
        reader.flag("dataset_mean").unwrap();
        let dataset_mean_len_pos = reader.pos;

        Layout {
            mid_prefix_pos,
            mid_bulk_array_pos,
            inverted_lists_len_pos,
            inverted_lists_first_pos,
            residuals_data_len_pos,
            residuals_data_first_pos,
            offsets_first_pos,
            offsets_second_pos,
            offsets_last_pos,
            residuals_offsets_first_pos,
            residuals_offsets_second_pos,
            residuals_offsets_last_pos,
            dsub_pos,
            coarse_n_vecs_pos,
            coarse_n_vecs_value,
            pq_centroids_len_pos,
            with_norms_pos,
            dataset_mean_flag_pos,
            dataset_mean_len_pos,
            n_docs,
        }
    }

    fn read_u64_at(bytes: &[u8], pos: u64) -> u64 {
        let pos = pos as usize;
        u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap())
    }

    fn write_u64_at(bytes: &mut [u8], pos: u64, value: u64) {
        let pos = pos as usize;
        bytes[pos..pos + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn write_u32_at(bytes: &mut [u8], pos: u64, value: u32) {
        let pos = pos as usize;
        bytes[pos..pos + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn assert_corrupt_contains(name: &str, bytes: &[u8], expected: &str) {
        let path = temp_path(name);
        std::fs::write(&path, bytes).unwrap();

        let result = Tachiom::<M>::load_index(&path);
        std::fs::remove_file(&path).unwrap();

        match result {
            Err(IndexIoError::Decode(DecodeError::OtherString(message))) => assert!(
                message.contains(expected),
                "{name}: expected {expected:?}, got {message:?}"
            ),
            Err(error) => panic!("{name}: unexpected error {error:?}"),
            Ok(_) => panic!("{name}: expected load_index to reject this fixture"),
        }
    }

    fn assert_decode_error(name: &str, bytes: &[u8]) {
        let path = temp_path(name);
        std::fs::write(&path, bytes).unwrap();

        let result = Tachiom::<M>::load_index(&path);
        std::fs::remove_file(&path).unwrap();

        match result {
            Err(IndexIoError::Decode(_)) => {}
            Err(error) => panic!("{name}: unexpected error {error:?}"),
            Ok(_) => panic!("{name}: expected load_index to reject this fixture"),
        }
    }

    #[test]
    fn rejects_truncated_files() {
        let fixture = malformed_fixture();
        let base = &fixture.bytes;

        for (name, end) in [
            (
                "truncated_in_hnsw_prefix",
                fixture.layout.mid_prefix_pos as usize,
            ),
            (
                "truncated_in_bulk_array",
                fixture.layout.mid_bulk_array_pos as usize,
            ),
            ("truncated_one_byte_short", base.len() - 1),
        ] {
            assert_decode_error(name, &base[..end]);
        }
    }

    #[test]
    fn rejects_nonzero_first_offsets() {
        let fixture = malformed_fixture();

        for (name, pos) in [
            (
                "tachiom_offsets_first_nonzero",
                fixture.layout.offsets_first_pos,
            ),
            (
                "residuals_offsets_first_nonzero",
                fixture.layout.residuals_offsets_first_pos,
            ),
        ] {
            let mut bytes = fixture.bytes.clone();
            write_u64_at(&mut bytes, pos, 5);
            assert_corrupt_contains(name, &bytes, "first offset must be 0");
        }
    }

    #[test]
    fn rejects_non_monotonic_offsets() {
        let fixture = malformed_fixture();

        for (name, pos) in [
            (
                "tachiom_offsets_decreasing",
                fixture.layout.offsets_second_pos,
            ),
            (
                "residuals_offsets_decreasing",
                fixture.layout.residuals_offsets_second_pos,
            ),
        ] {
            let mut bytes = fixture.bytes.clone();
            let next = read_u64_at(&bytes, pos + 8);
            write_u64_at(&mut bytes, pos, next + 1);
            assert_corrupt_contains(name, &bytes, "offsets decrease at index");
        }
    }

    #[test]
    fn rejects_mismatched_last_offsets() {
        let fixture = malformed_fixture();

        for (name, pos) in [
            (
                "tachiom_offsets_last_mismatch",
                fixture.layout.offsets_last_pos,
            ),
            (
                "residuals_offsets_last_mismatch",
                fixture.layout.residuals_offsets_last_pos,
            ),
        ] {
            let mut bytes = fixture.bytes.clone();
            let original = read_u64_at(&bytes, pos);
            write_u64_at(&mut bytes, pos, original + 1);
            assert_corrupt_contains(name, &bytes, "does not equal data length");
        }
    }

    #[test]
    fn rejects_declared_length_overruns() {
        let fixture = malformed_fixture();
        let base = &fixture.bytes;
        let overflowing_u32_len = (usize::MAX / 4 + 1) as u64;

        for (name, len_pos, width, value, expected) in [
            (
                "inverted_lists_length_overflow",
                fixture.layout.inverted_lists_len_pos,
                4u64,
                overflowing_u32_len,
                "bytes overflow usize",
            ),
            (
                "inverted_lists_length_past_remaining",
                fixture.layout.inverted_lists_len_pos,
                4,
                0,
                "overruns the file",
            ),
            (
                "residuals_data_length_past_remaining",
                fixture.layout.residuals_data_len_pos,
                1,
                0,
                "overruns the file",
            ),
        ] {
            let mut bytes = base.clone();
            let declared = if value == 0 {
                (base.len() as u64 - len_pos - 8) / width + 1
            } else {
                value
            };
            write_u64_at(&mut bytes, len_pos, declared);
            assert_corrupt_contains(name, &bytes, expected);
        }
    }

    #[test]
    fn rejects_invalid_flags_and_shapes() {
        let fixture = malformed_fixture();

        for (name, pos) in [
            ("with_norms_flag_invalid", fixture.layout.with_norms_pos),
            (
                "dataset_mean_tag_invalid",
                fixture.layout.dataset_mean_flag_pos,
            ),
        ] {
            let mut bytes = fixture.bytes.clone();
            bytes[pos as usize] = 2;
            assert_corrupt_contains(name, &bytes, "expected 0 or 1");
        }

        let mut bytes = fixture.bytes.clone();
        write_u64_at(
            &mut bytes,
            fixture.layout.pq_centroids_len_pos,
            M as u64 + 1,
        );
        assert_corrupt_contains("pq_centroids_count_mismatch", &bytes, "codebooks, M is");

        let mut bytes = fixture.bytes.clone();
        let original = read_u64_at(&bytes, fixture.layout.coarse_n_vecs_pos);
        write_u64_at(&mut bytes, fixture.layout.coarse_n_vecs_pos, original + 1);
        assert_corrupt_contains(
            "coarse_dataset_n_vecs_mismatch",
            &bytes,
            "values but n_vecs",
        );
    }

    #[test]
    fn rejects_encoder_shape_that_would_panic() {
        let fixture = malformed_fixture();

        let mut bytes = fixture.bytes.clone();
        let original = read_u64_at(&bytes, fixture.layout.dsub_pos);
        write_u64_at(&mut bytes, fixture.layout.dsub_pos, original + 1);
        assert_corrupt_contains("dsub_mismatch", &bytes, "residuals.encoder: token_dim");
    }

    #[test]
    fn rejects_cross_field_mismatches() {
        let fixture = malformed_fixture();

        let mut bytes = fixture.bytes.clone();
        write_u32_at(
            &mut bytes,
            fixture.layout.residuals_data_first_pos,
            fixture.layout.coarse_n_vecs_value as u32,
        );
        assert_corrupt_contains(
            "residual_coarse_id_out_of_range",
            &bytes,
            "is out of range for",
        );

        let mut bytes = fixture.bytes.clone();
        write_u32_at(
            &mut bytes,
            fixture.layout.inverted_lists_first_pos,
            fixture.layout.n_docs as u32,
        );
        assert_corrupt_contains("posting_doc_id_out_of_range", &bytes, "but the index has");

        let mut bytes = fixture.bytes.clone();
        let second = read_u64_at(&bytes, fixture.layout.residuals_offsets_second_pos);
        write_u64_at(
            &mut bytes,
            fixture.layout.residuals_offsets_second_pos,
            second + 1,
        );
        assert_corrupt_contains(
            "residual_document_span_ragged",
            &bytes,
            "is not divisible by encoded width",
        );

        let mut bytes = fixture.bytes.clone();
        let mean_len = read_u64_at(&bytes, fixture.layout.dataset_mean_len_pos);
        write_u64_at(
            &mut bytes,
            fixture.layout.dataset_mean_len_pos,
            mean_len - 1,
        );
        assert_corrupt_contains(
            "dataset_mean_length_mismatch",
            &bytes,
            "Tachiom.dataset_mean: length",
        );
    }

    #[test]
    fn missing_file_is_io_error() {
        let path = temp_path("absent");
        match Tachiom::<M>::load_index(&path) {
            Err(IndexIoError::Io(_)) => {}
            Err(error) => panic!("expected an Io error, got {error:?}"),
            Ok(_) => panic!("expected load_index to fail on a missing file"),
        }
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

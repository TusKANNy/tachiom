use clap::Parser;
use half::f16;
use ndarray::Array1;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

use tachiom::graph::Graph;
use tachiom::hnsw::HNSWBuildConfiguration;
use tachiom::tachiom::Tachiom;
use vectorium::core::index::Index;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::vector_encoder::VectorEncoder;
use vectorium::{
    DenseDataset, IndexSerializer, MultiVecTwoLevelProductQuantizer, MultiVectorDataset,
    PlainDenseDataset, PlainDenseQuantizer,
};

// Blocked layout constants: M subspaces, KSUB centroids per subspace.
const M: usize = 32;
const KSUB: usize = 256;

type CentroidDataset = DenseDataset<PlainDenseQuantizer<f16, DotProduct>>;
type HNSWCentroids = tachiom::hnsw::HNSW<CentroidDataset, Graph>;

#[derive(Parser, Debug)]
#[clap(
    about = "Build a Tachiom index from pre-built components (old pipeline output format).\n\
                Lets you test the search path independently of index construction."
)]
struct Args {
    /// [K, dim] f32 coarse centroids (.npy)
    #[clap(long)]
    centroids_file: String,

    /// [M*KSUB*dsub] f32 PQ centroids in AoS layout [M][KSUB][dsub] (.npy)
    #[clap(long)]
    pq_centroids_file: String,

    /// [N, M] u8 PQ codes (.npy)
    #[clap(long)]
    residuals_file: String,

    /// [N] f32 residual norms (.npy) — provide to enable norm scaling
    #[clap(long)]
    norms_file: Option<String>,

    /// [N] u64 (or u32) coarse centroid assignment per token (.npy)
    #[clap(long)]
    assignments_file: String,

    /// document lengths file (.npy, i32 or i64)
    #[clap(long)]
    doclens_file: String,

    /// Output path for the serialized Tachiom index
    #[clap(short = 'o', long)]
    output_file: String,

    /// HNSW M: neighbours per node in the centroid graph
    #[clap(long, default_value_t = 32)]
    hnsw_m: usize,

    /// HNSW ef_construction
    #[clap(long, default_value_t = 1500)]
    ef_construction: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("=== Tachiom Build From Parts ===");
    println!("Centroids:      {}", args.centroids_file);
    println!("PQ centroids:   {}", args.pq_centroids_file);
    println!("PQ codes:       {}", args.residuals_file);
    println!("Norms:          {:?}", args.norms_file);
    println!("Assignments:    {}", args.assignments_file);
    println!("Doclens:        {}", args.doclens_file);
    println!("Output:         {}", args.output_file);

    let total_start = Instant::now();

    // ── Load coarse centroids (f32) ───────────────────────────────────────────
    println!("\nLoading coarse centroids...");
    let (centroids_f32, n_centroids, token_dim) = read_f32_2d_npy(&args.centroids_file)?;
    println!("  {} centroids × dim={}", n_centroids, token_dim);
    let dsub = token_dim / M;
    anyhow::ensure!(
        token_dim % M == 0,
        "token_dim {} not divisible by M {}",
        token_dim,
        M
    );

    // Keep f16 copy for HNSW; f32 copy is used for the encoder's coarse_centroids.
    let centroids_f16: Vec<f16> = centroids_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // ── Load PQ centroids (f32, AoS: [M][KSUB][dsub]) ────────────────────────
    println!("Loading PQ centroids...");
    let pq_flat: Vec<f32> = {
        let arr: Array1<f32> =
            Array1::read_npy(BufReader::new(File::open(&args.pq_centroids_file)?))?;
        anyhow::ensure!(
            arr.len() == M * KSUB * dsub,
            "PQ centroids length {} != M*KSUB*dsub={}*{}*{}={}",
            arr.len(),
            M,
            KSUB,
            dsub,
            M * KSUB * dsub
        );
        arr.into_raw_vec_and_offset().0
    };

    // ── Load PQ codes (u8, [N, M]) ────────────────────────────────────────────
    println!("Loading PQ codes...");
    let (codes, n_tokens) = read_u8_2d_npy(&args.residuals_file, M)?;
    println!("  {} tokens × M={}", n_tokens, M);

    // ── Load residual norms (f32, [N]) ────────────────────────────────────────
    let norms: Option<Vec<f32>> = if let Some(ref path) = args.norms_file {
        println!("Loading residual norms...");
        let arr: Array1<f32> = Array1::read_npy(BufReader::new(File::open(path)?))?;
        anyhow::ensure!(
            arr.len() == n_tokens,
            "norms length {} != n_tokens {}",
            arr.len(),
            n_tokens
        );
        Some(arr.into_raw_vec_and_offset().0)
    } else {
        None
    };
    let with_norms = norms.is_some();

    // ── Load assignments ([N] u64 or u32) ─────────────────────────────────────
    println!("Loading assignments...");
    let assignments: Vec<usize> = read_assignments_npy(&args.assignments_file, n_tokens)?;

    // ── Load doclens ──────────────────────────────────────────────────────────
    println!("Loading doclens...");
    let doclens: Vec<usize> = read_doclens_npy(&args.doclens_file)?;
    let n_docs = doclens.len();
    let total_tok: usize = doclens.iter().sum();
    anyhow::ensure!(
        total_tok == n_tokens,
        "sum(doclens)={} != n_tokens={}",
        total_tok,
        n_tokens
    );
    println!("  {} docs, {} tokens", n_docs, n_tokens);

    // ── Build MultiVecTwoLevelProductQuantizer via from_pretrained ────────────
    println!("\nBuilding encoder from pretrained components...");

    let coarse_centroids_ds = PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
        centroids_f32.into_boxed_slice(),
        n_centroids,
        PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(token_dim),
    );

    // Split AoS PQ flat buffer into per-subspace PlainDenseDatasets.
    // AoS layout: pq_flat[m * KSUB * dsub + k * dsub + d] = subspace m, centroid k, dim d.
    // PlainDenseDataset AoS: centroid k starts at k * dsub. Same layout. ✓
    let pq_centroid_datasets: Vec<PlainDenseDataset<f32, SquaredEuclideanDistance>> = (0..M)
        .map(|m| {
            PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
                pq_flat[m * KSUB * dsub..(m + 1) * KSUB * dsub]
                    .to_vec()
                    .into_boxed_slice(),
                KSUB,
                PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dsub),
            )
        })
        .collect();

    let encoder = MultiVecTwoLevelProductQuantizer::<M, f16>::from_pretrained(
        token_dim,
        coarse_centroids_ds,
        pq_centroid_datasets,
        with_norms,
    );
    let output_dim = encoder.output_dim(); // = COARSE_ID_BYTES + M + (with_norms ? NORM_BYTES : 0)

    // ── Assemble per-document encoded data (blocked layout) ───────────────────
    // Per document: [coarse_ids: 4·n_tok] [pq_codes: M·n_tok] [norms: 2·n_tok if with_norms]
    println!(
        "Assembling encoded documents ({} bytes/token)...",
        output_dim
    );

    let mut enc_data: Vec<u8> = Vec::with_capacity(n_tokens * output_dim);
    let mut enc_offsets: Vec<usize> = Vec::with_capacity(n_docs + 1);
    enc_offsets.push(0);

    let mut token_start = 0usize;
    for &n_tok in &doclens {
        // Block 1: coarse centroid IDs (u32 LE, 4 bytes each)
        for t in 0..n_tok {
            let id = assignments[token_start + t] as u32;
            enc_data.extend_from_slice(&id.to_le_bytes());
        }
        // Block 2: PQ codes (M bytes each)
        for t in 0..n_tok {
            let base = (token_start + t) * M;
            enc_data.extend_from_slice(&codes[base..base + M]);
        }
        // Block 3: residual norms (f16 LE, 2 bytes each) — only when with_norms
        if let Some(ref norms_vec) = norms {
            for t in 0..n_tok {
                let norm = f16::from_f32(norms_vec[token_start + t]);
                enc_data.extend_from_slice(&norm.to_le_bytes());
            }
        }

        enc_offsets.push(enc_data.len());
        token_start += n_tok;
    }

    let residuals = MultiVectorDataset::from_raw(
        enc_data.into_boxed_slice(),
        enc_offsets.into_boxed_slice(),
        encoder,
    );

    // ── Build HNSW on f16 coarse centroids ────────────────────────────────────
    println!(
        "Building HNSW on {} centroids (M={}, ef_construction={})...",
        n_centroids, args.hnsw_m, args.ef_construction
    );
    let t_hnsw = Instant::now();

    let centroid_dataset: CentroidDataset = DenseDataset::from_raw(
        centroids_f16.into_boxed_slice(),
        n_centroids,
        PlainDenseQuantizer::<f16, DotProduct>::new(token_dim),
    );
    let hnsw_cfg = HNSWBuildConfiguration::default()
        .with_num_neighbors(args.hnsw_m)
        .with_ef_construction(args.ef_construction);
    let centroids_hnsw: HNSWCentroids =
        <HNSWCentroids as Index<CentroidDataset>>::build_index(centroid_dataset, &hnsw_cfg);

    println!("  HNSW built in {:.2?}", t_hnsw.elapsed());

    // ── Assemble Tachiom index from parts ─────────────────────────────────────
    println!("Building inverted lists...");
    let index = Tachiom::<M>::from_parts(centroids_hnsw, &assignments, residuals);
    index.print_space_usage_bytes();

    // ── Save ──────────────────────────────────────────────────────────────────
    println!("\nSaving index to {}...", args.output_file);
    let t_save = Instant::now();
    index
        .save_index(&args.output_file)
        .map_err(|e| anyhow::anyhow!("Failed to save index: {:?}", e))?;
    println!("Saved in {:.2?}", t_save.elapsed());
    println!("\nTotal time: {:.2?}", total_start.elapsed());

    Ok(())
}

// ============================================================================
// I/O helpers
// ============================================================================

fn read_f32_2d_npy(path: &str) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    use ndarray::Array2;
    let arr: Array2<f32> = Array2::read_npy(BufReader::new(File::open(path)?))?;
    let (rows, cols) = arr.dim();
    Ok((arr.into_raw_vec_and_offset().0, rows, cols))
}

/// Read a [N, M] u8 array and verify the second dimension matches `expected_m`.
fn read_u8_2d_npy(path: &str, expected_m: usize) -> anyhow::Result<(Vec<u8>, usize)> {
    use ndarray::Array2;
    let arr: Array2<u8> = Array2::read_npy(BufReader::new(File::open(path)?))?;
    let (n, m) = arr.dim();
    anyhow::ensure!(
        m == expected_m,
        "PQ codes second dim {} != expected M={}",
        m,
        expected_m
    );
    Ok((arr.into_raw_vec_and_offset().0, n))
}

/// Read assignments as u64 or u32, converting to usize.
fn read_assignments_npy(path: &str, expected_len: usize) -> anyhow::Result<Vec<usize>> {
    let mut reader = BufReader::new(File::open(path)?);
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    anyhow::ensure!(shape.len() == 1, "assignments must be 1D");
    anyhow::ensure!(
        shape[0] == expected_len,
        "assignments length {} != n_tokens {}",
        shape[0],
        expected_len
    );
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader.read_exact(&mut raw)?;
    Ok(match elem_size {
        8 => raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        4 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => anyhow::bail!("unsupported assignments dtype (elem_size={})", elem_size),
    })
}

/// Read doclens; accepts i32, i64, u32, u64.
fn read_doclens_npy(path: &str) -> anyhow::Result<Vec<usize>> {
    let mut reader = BufReader::new(File::open(path)?);
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    anyhow::ensure!(shape.len() == 1, "doclens must be 1D");
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader.read_exact(&mut raw)?;
    Ok(match elem_size {
        4 => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        8 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => anyhow::bail!("unsupported doclens dtype (elem_size={})", elem_size),
    })
}

fn parse_npy_header(reader: &mut impl Read) -> anyhow::Result<(Vec<usize>, usize, usize)> {
    let mut n = 0usize;
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    n += 6;
    anyhow::ensure!(&magic == b"\x93NUMPY", "not a NumPy file");
    let mut ver = [0u8; 2];
    reader.read_exact(&mut ver)?;
    n += 2;
    let header_len: usize = if ver[0] == 1 {
        let mut hl = [0u8; 2];
        reader.read_exact(&mut hl)?;
        n += 2;
        u16::from_le_bytes(hl) as usize
    } else {
        let mut hl = [0u8; 4];
        reader.read_exact(&mut hl)?;
        n += 4;
        u32::from_le_bytes(hl) as usize
    };
    let mut hdr = vec![0u8; header_len];
    reader.read_exact(&mut hdr)?;
    n += header_len;
    let hdr = String::from_utf8_lossy(&hdr);
    anyhow::ensure!(
        !hdr.contains("'fortran_order': True"),
        "Fortran-order not supported"
    );
    let descr =
        extract_quoted(&hdr, "'descr': '").ok_or_else(|| anyhow::anyhow!("'descr' not found"))?;
    anyhow::ensure!(!descr.starts_with('>'), "big-endian not supported");
    let elem_size = dtype_elem_size(descr.trim_start_matches(['<', '=', '|']))
        .ok_or_else(|| anyhow::anyhow!("unrecognised dtype '{}'", descr))?;
    let tok = "'shape': (";
    let si = hdr
        .find(tok)
        .ok_or_else(|| anyhow::anyhow!("'shape' not found"))?;
    let rest = &hdr[si + tok.len()..];
    let ei = rest
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("shape ')' not found"))?;
    let shape: Vec<usize> = rest[..ei]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("{}", e))
        })
        .collect::<anyhow::Result<_>>()?;
    Ok((shape, elem_size, n))
}

fn extract_quoted<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let start = s.find(prefix)? + prefix.len();
    let rest = &s[start..];
    let end = rest.find('\'')?;
    Some(&rest[..end])
}

fn dtype_elem_size(code: &str) -> Option<usize> {
    match code {
        "i1" | "u1" | "b1" => Some(1),
        "i2" | "u2" | "f2" => Some(2),
        "i4" | "u4" | "f4" => Some(4),
        "i8" | "u8" | "f8" => Some(8),
        _ => None,
    }
}

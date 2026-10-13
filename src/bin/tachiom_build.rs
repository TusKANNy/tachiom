use clap::{Parser, ValueEnum};
use half::f16;
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

use tachiom::hnsw::{HNSWBuildConfiguration, HNSWSearchConfiguration};
use tachiom::kmeans::{KMeansAssign, KMeansParams};
use tachiom::tachiom::{Clustering, TacParams, Tachiom, TachiomBuildParams, TachiomInputDataset};
use vectorium::{IndexSerializer, MultiVectorDataset, PlainMultiVecQuantizer};

#[derive(ValueEnum, Clone, Copy, Debug)]
enum ClusteringArg {
    Tac,
    Kmeans,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum KMeansAssignArg {
    Flat,
    Hnsw,
}

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Build a Tachiom IVF-PQ index from a multivector (late-interaction) dataset"
)]
struct Args {
    /// Path to the f16 token-vector file (.npy, shape [N, dim])
    #[clap(short = 'i', long)]
    vectors_file: String,

    /// Path to the token-type ID file (.npy, dtype i64 or u32, shape [N]).
    /// Required for TAC; optional for k-means (stratifies the PQ sample when given).
    #[clap(long)]
    token_ids_file: Option<String>,

    /// Path to the document-length file (.npy, shape [n_docs])
    #[clap(long)]
    doclens_file: String,

    /// Output path for the serialized Tachiom index
    #[clap(short = 'o', long)]
    output_file: String,

    /// Total coarse-centroid budget
    #[clap(long, default_value_t = 4_194_304)]
    total_centroids: usize,

    /// Coarse-clustering method
    #[clap(long, value_enum, default_value_t = ClusteringArg::Tac)]
    clustering: ClusteringArg,

    /// K-means iterations per token type in TAC
    #[clap(long, default_value_t = 10)]
    tac_n_iter: usize,

    /// TAC micro threshold: token groups smaller than this get 1 centroid each
    #[clap(long, default_value_t = 128)]
    tac_micro_threshold: usize,

    /// TAC small threshold: token groups in [micro, small) get 2 centroids each
    #[clap(long, default_value_t = 256)]
    tac_small_threshold: usize,

    /// Lloyd iterations for k-means
    #[clap(long, default_value_t = 10)]
    kmeans_n_iter: usize,

    /// k-means trains on min(N, points_per_centroid × total_centroids) tokens
    #[clap(long, default_value_t = 64)]
    kmeans_points_per_centroid: usize,

    /// RNG seed for k-means sampling and initialisation
    #[clap(long, default_value_t = 42)]
    kmeans_seed: u64,

    /// Nearest-centroid search inside k-means
    #[clap(long, value_enum, default_value_t = KMeansAssignArg::Hnsw)]
    kmeans_assign: KMeansAssignArg,

    /// HNSW M for k-means assignment
    #[clap(long, default_value_t = 32)]
    kmeans_hnsw_m: usize,

    /// HNSW ef_construction for k-means assignment
    #[clap(long, default_value_t = 200)]
    kmeans_hnsw_ef_construction: usize,

    /// HNSW ef_search for k-means assignment
    #[clap(long, default_value_t = 16)]
    kmeans_hnsw_ef_search: usize,

    /// Tokens sampled for PQ training
    #[clap(long, default_value_t = 10_000_000)]
    pq_sample_size: usize,

    /// K-means iterations for PQ subspace training
    #[clap(long, default_value_t = 10)]
    pq_n_iter: usize,

    /// Normalise residuals before PQ (stores norms in payload)
    #[clap(long)]
    normalize: bool,

    /// RNG seed for reproducible PQ training
    #[clap(long, default_value_t = 42)]
    pq_seed: u64,

    /// HNSW M: neighbours per node in the centroid graph
    #[clap(long, default_value_t = 32)]
    hnsw_m: usize,

    /// HNSW ef_construction
    #[clap(long, default_value_t = 1500)]
    ef_construction: usize,

    /// Number of PQ subspaces (only M=32 is currently supported)
    #[clap(long, default_value_t = 32)]
    pq_subspaces: usize,

    /// Subtract the dataset mean from all token vectors before building.
    /// Pass --center-dataset false to disable.
    #[clap(long, action = clap::ArgAction::Set, default_value_t = true)]
    center_dataset: bool,

    /// Print a `[timing] <stage>: <elapsed>` line for each build stage.
    /// Can also be enabled with the TACHIOM_TIMINGS environment variable.
    #[clap(long)]
    timings: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.timings {
        tachiom::timing::set_enabled(true);
    }

    if args.pq_subspaces != 32 {
        anyhow::bail!("Only --pq-subspaces 32 is currently supported");
    }

    println!("=== Tachiom Build ===");
    println!("Vectors:         {}", args.vectors_file);
    println!("Token IDs:       {:?}", args.token_ids_file);
    println!("Doc lengths:     {}", args.doclens_file);
    println!("Output:          {}", args.output_file);
    println!("Total centroids: {}", args.total_centroids);
    println!("Clustering:      {:?}", args.clustering);
    println!("PQ sample:       {}", args.pq_sample_size);
    println!("PQ iters:        {}", args.pq_n_iter);
    println!("Normalize:       {}", args.normalize);
    println!("HNSW M:          {}", args.hnsw_m);
    println!("HNSW ef_constr:  {}", args.ef_construction);
    println!("CPU cores:       {}", rayon::current_num_threads());

    let total_start = Instant::now();

    // ── Load token vectors (f16) ──────────────────────────────────────────────
    println!("\nLoading token vectors...");
    let load_start = Instant::now();
    let (flat_f16, dim) = read_f16_npy(&args.vectors_file)?;
    let n_tokens = flat_f16.len() / dim;
    println!(
        "  {} tokens × dim={} in {:.2?}",
        n_tokens,
        dim,
        load_start.elapsed()
    );

    // ── Load token type IDs ───────────────────────────────────────────────────
    println!("Loading token IDs...");
    let token_ids = match &args.token_ids_file {
        Some(path) => read_token_ids_npy(path)?,
        None => Vec::new(),
    };
    anyhow::ensure!(
        token_ids.is_empty() || token_ids.len() == n_tokens,
        "token_ids length ({}) != n_tokens ({})",
        token_ids.len(),
        n_tokens
    );

    // ── Load document lengths ─────────────────────────────────────────────────
    println!("Loading document lengths...");
    let doclens = read_doclens_npy(&args.doclens_file)?;
    let n_docs = doclens.len();
    let total_tokens_check: usize = doclens.iter().sum();
    anyhow::ensure!(
        total_tokens_check == n_tokens,
        "sum(doclens)={} != n_tokens={}",
        total_tokens_check,
        n_tokens
    );
    println!("  {} docs, {} tokens total", n_docs, n_tokens);
    tachiom::timing::report("data_loading", load_start);

    // ── Build TachiomInputDataset ─────────────────────────────────────────────
    // PlainMultiVecQuantizer is a pass-through encoder; no transformation needed.
    println!("\nBuilding raw multivector dataset...");
    let prep_start = Instant::now();
    let encoder = PlainMultiVecQuantizer::<f16>::new(dim);
    let mut offsets: Vec<usize> = Vec::with_capacity(doclens.len() + 1);
    offsets.push(0usize);
    for &n_tok in &doclens {
        offsets.push(offsets.last().unwrap() + n_tok * dim);
    }
    anyhow::ensure!(
        *offsets.last().unwrap() == flat_f16.len(),
        "sum(doclens)*dim={} != flat_f16.len()={}",
        offsets.last().unwrap(),
        flat_f16.len()
    );
    let raw_dataset: TachiomInputDataset = MultiVectorDataset::from_raw(
        flat_f16.into_boxed_slice(),
        offsets.into_boxed_slice(),
        encoder,
    );
    tachiom::timing::report("dataset_assembly", prep_start);

    // ── Build index ───────────────────────────────────────────────────────────
    let clustering = match args.clustering {
        ClusteringArg::Tac => {
            anyhow::ensure!(
                !token_ids.is_empty(),
                "--clustering tac requires --token-ids-file"
            );
            Clustering::Tac(TacParams {
                n_iter: args.tac_n_iter,
                micro_threshold: Some(args.tac_micro_threshold),
                small_threshold: Some(args.tac_small_threshold),
            })
        }
        ClusteringArg::Kmeans => Clustering::KMeans(KMeansParams {
            n_iter: args.kmeans_n_iter,
            points_per_centroid: args.kmeans_points_per_centroid,
            seed: args.kmeans_seed,
            assign: match args.kmeans_assign {
                KMeansAssignArg::Flat => KMeansAssign::Flat,
                KMeansAssignArg::Hnsw => KMeansAssign::Hnsw {
                    build: HNSWBuildConfiguration::default()
                        .with_num_neighbors(args.kmeans_hnsw_m)
                        .with_ef_construction(args.kmeans_hnsw_ef_construction),
                    search: HNSWSearchConfiguration::default()
                        .with_ef_search(args.kmeans_hnsw_ef_search),
                },
            },
        }),
    };

    println!("\n=== Building Tachiom index ===");
    let build_params = TachiomBuildParams {
        token_ids,
        total_centroids: args.total_centroids,
        clustering,
        pq_sample_size: args.pq_sample_size,
        pq_n_iter: args.pq_n_iter,
        normalize: args.normalize,
        pq_seed: Some(args.pq_seed),
        hnsw_params: HNSWBuildConfiguration::default()
            .with_num_neighbors(args.hnsw_m)
            .with_ef_construction(args.ef_construction),
        center_dataset: args.center_dataset,
    };

    let build_start = Instant::now();
    let index = Tachiom::<32>::build_index(raw_dataset, &build_params);
    let build_time = build_start.elapsed();
    println!("Build time: {:.2?}", build_time);

    println!("\n=== Space Usage ===");
    index.print_space_usage_bytes();

    // ── Save index ────────────────────────────────────────────────────────────
    println!("\nSaving index to {}...", args.output_file);
    let save_start = Instant::now();
    index
        .save_index(&args.output_file)
        .map_err(|e| anyhow::anyhow!("Failed to save index: {:?}", e))?;
    println!("Saved in {:.2?}", save_start.elapsed());

    println!("\n=== Done — total time: {:.2?} ===", total_start.elapsed());

    Ok(())
}

// ============================================================================
// I/O helpers
// ============================================================================

fn read_f16_npy(path: &str) -> anyhow::Result<(Vec<f16>, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    anyhow::ensure!(shape.len() == 2, "Expected 2D array for token vectors");
    anyhow::ensure!(elem_size == 2, "Expected f16 (2-byte) dtype");

    let (n_vecs, dim) = (shape[0], shape[1]);
    let mut raw = vec![0u8; n_vecs * dim * 2];
    reader.read_exact(&mut raw)?;
    let data = raw
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok((data, dim))
}

fn read_token_ids_npy(path: &str) -> anyhow::Result<Vec<usize>> {
    let mut reader = BufReader::new(File::open(path)?);
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    anyhow::ensure!(shape.len() == 1, "Expected 1D token-ID array");
    anyhow::ensure!(
        elem_size == 4 || elem_size == 8,
        "Unsupported token-ID elem size: {}",
        elem_size
    );
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader.read_exact(&mut raw)?;
    Ok(match elem_size {
        8 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
    })
}

fn read_doclens_npy(path: &str) -> anyhow::Result<Vec<usize>> {
    let mut reader = BufReader::new(File::open(path)?);
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    anyhow::ensure!(shape.len() == 1, "Expected 1D doclens array");
    anyhow::ensure!(
        elem_size == 4 || elem_size == 8,
        "Unsupported doclens elem size: {}",
        elem_size
    );
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader.read_exact(&mut raw)?;
    Ok(match elem_size {
        8 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
    })
}

fn parse_npy_header(reader: &mut impl Read) -> anyhow::Result<(Vec<usize>, usize, usize)> {
    let mut n = 0usize;
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    n += 6;
    anyhow::ensure!(&magic == b"\x93NUMPY", "Not a NumPy file");
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
    let mut hdr_bytes = vec![0u8; header_len];
    reader.read_exact(&mut hdr_bytes)?;
    n += header_len;
    let hdr = String::from_utf8_lossy(&hdr_bytes);
    anyhow::ensure!(
        !hdr.contains("'fortran_order': True"),
        "Fortran-order arrays not supported"
    );
    let descr = {
        let prefix = "'descr': '";
        let start = hdr
            .find(prefix)
            .ok_or_else(|| anyhow::anyhow!("'descr' not found"))?
            + prefix.len();
        let rest = &hdr[start..];
        let end = rest
            .find('\'')
            .ok_or_else(|| anyhow::anyhow!("descr end quote not found"))?;
        rest[..end].to_string()
    };
    anyhow::ensure!(!descr.starts_with('>'), "Big-endian dtype not supported");
    let elem_size = dtype_elem_size(descr.trim_start_matches(['<', '=', '|']))
        .ok_or_else(|| anyhow::anyhow!("Unrecognised dtype '{}'", descr))?;
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

fn dtype_elem_size(code: &str) -> Option<usize> {
    match code {
        "i1" | "u1" | "b1" => Some(1),
        "i2" | "u2" | "f2" => Some(2),
        "i4" | "u4" | "f4" => Some(4),
        "i8" | "u8" | "f8" => Some(8),
        _ => None,
    }
}

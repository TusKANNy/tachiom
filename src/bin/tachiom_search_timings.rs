use clap::Parser;
use ndarray::{Array3, s};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;

use tachiom_private::tachiom::Tachiom;
use vectorium::core::index::Index;
use vectorium::{DenseMultiVectorView, IndexSerializer};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Search a Tachiom IVF-PQ index and report per-stage timing breakdown"
)]
struct Args {
    /// Path to the serialized Tachiom index
    #[clap(short = 'i', long)]
    index_file: String,

    /// Path to the query file (.npy, shape [n_queries, n_tokens_per_query, dim], f32)
    #[clap(short = 'q', long)]
    query_file: String,

    /// Output file for results (tab-separated: query_id doc_id rank score)
    #[clap(short = 'o', long)]
    output_path: Option<String>,

    /// Number of results to return per query
    #[clap(long, default_value_t = 10)]
    k: usize,

    /// Coarse centroids probed per query token
    #[clap(long, default_value_t = 4)]
    k_centroids: usize,

    /// Maximum candidate documents to score in stage 2
    #[clap(long, default_value_t = 1000)]
    k_docs_to_score: usize,

    /// HNSW ef_search for centroid lookup
    #[clap(long, default_value_t = 64)]
    ef_search: usize,

    /// Alpha pruning threshold (fraction relative to k-th candidate score)
    #[clap(long)]
    alpha: Option<f32>,

    /// Beta early-exit staleness counter
    #[clap(long)]
    beta: Option<usize>,

    /// Lambda for distance-adaptive early termination in HNSW
    #[clap(long)]
    lambda: Option<f32>,

    /// Number of timing runs (results from run 1 are saved)
    #[clap(long, default_value_t = 1)]
    num_runs: usize,

    /// Number of PQ subspaces the index was built with (only 32 supported)
    #[clap(long, default_value_t = 32)]
    pq_subspaces: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.pq_subspaces != 32 {
        anyhow::bail!("Only --pq-subspaces 32 is currently supported");
    }

    println!("=== Tachiom Search (per-stage timings) ===");
    println!("Index:           {}", args.index_file);
    println!("Queries:         {}", args.query_file);
    println!("k:               {}", args.k);
    println!("k_centroids:     {}", args.k_centroids);
    println!("k_docs_to_score: {}", args.k_docs_to_score);
    println!("ef_search:       {}", args.ef_search);
    println!("alpha:           {:?}", args.alpha);
    println!("beta:            {:?}", args.beta);
    println!("lambda:          {:?}", args.lambda);
    println!("num_runs:        {}", args.num_runs);

    // ── Load index ────────────────────────────────────────────────────────────
    println!("\nLoading index...");
    let load_start = Instant::now();
    let index = Tachiom::<32>::load_index(&args.index_file)
        .map_err(|e| anyhow::anyhow!("Failed to load index: {:?}", e))?;
    println!(
        "Loaded {} docs, dim={} in {:.2?}",
        index.n_elements(),
        index.dim(),
        load_start.elapsed()
    );
    index.print_space_usage_bytes();

    // ── Load queries ──────────────────────────────────────────────────────────
    println!("\nLoading queries from {}...", args.query_file);
    let queries_arr: Array3<f32> = Array3::read_npy(BufReader::new(File::open(&args.query_file)?))?;
    let (n_queries, n_tokens_per_query, query_dim) = queries_arr.dim();
    anyhow::ensure!(
        query_dim == index.dim(),
        "Query dim {} != index dim {}",
        query_dim,
        index.dim()
    );
    println!(
        "  {} queries × {} tokens × dim={}",
        n_queries, n_tokens_per_query, query_dim
    );

    println!(
        "\nSearching {} queries ({} runs)...",
        n_queries, args.num_runs
    );

    // Pre-flatten queries once.
    let query_flat: Vec<Vec<f32>> = (0..n_queries)
        .map(|q| queries_arr.slice(s![q, .., ..]).iter().copied().collect())
        .collect();

    let mut total_time_us = 0u128;
    let mut total_stage1_ns = 0u128;
    let mut total_stage2_ns = 0u128;
    let mut total_stage3_ns = 0u128;

    let mut results = Vec::<(f32, u32)>::with_capacity(n_queries * args.k);

    let t0 = Instant::now();
    for run in 0..args.num_runs {
        if run > 0 {
            results.clear();
        }
        for q in 0..n_queries {
            let query = DenseMultiVectorView::new(&query_flat[q], query_dim);

            let (scored, timings) = index.search_with_timings(
                query,
                args.k,
                args.k_centroids,
                args.k_docs_to_score,
                args.ef_search,
                args.alpha,
                args.beta,
                args.lambda,
            );

            total_stage1_ns += timings.stage1_ns;
            total_stage2_ns += timings.stage2_ns;
            total_stage3_ns += timings.stage3_ns;

            results.extend(scored);
        }
    }
    total_time_us += t0.elapsed().as_micros();

    let n = (n_queries * args.num_runs) as u128;
    let avg_us = total_time_us / n;
    let avg_stage1_us = total_stage1_ns / 1000 / n;
    let avg_stage2_us = total_stage2_ns / 1000 / n;
    let avg_stage3_us = total_stage3_ns / 1000 / n;
    let stages_total_us = avg_stage1_us + avg_stage2_us + avg_stage3_us;

    println!("[######] Average Query Time:        {avg_us} μs");
    println!("[Stage1] Coarse-score accumulation: {avg_stage1_us} μs");
    println!("[Stage2] Candidate selection+prune: {avg_stage2_us} μs");
    println!("[Stage3] Full rerank + top-k:       {avg_stage3_us} μs");
    println!("[Sum  ] Stages 1+2+3:               {stages_total_us} μs");

    if stages_total_us > 0 {
        let pct = |x: u128| (x as f64 * 100.0) / stages_total_us as f64;
        println!(
            "[Pct  ] Stage1 {:.1}% | Stage2 {:.1}% | Stage3 {:.1}%",
            pct(avg_stage1_us),
            pct(avg_stage2_us),
            pct(avg_stage3_us)
        );
    }

    // ── Save results ──────────────────────────────────────────────────────────
    if let Some(ref output_path) = args.output_path {
        println!("\nWriting results to {}...", output_path);
        let mut file = File::create(output_path)?;
        for (i, (score, doc_id)) in results.iter().enumerate() {
            let query_id = i / args.k;
            let rank = (i % args.k) + 1;
            writeln!(file, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score)?;
        }
    }

    Ok(())
}

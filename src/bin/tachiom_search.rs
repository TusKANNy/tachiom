use clap::Parser;
use ndarray::{Array3, s};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;

use tachiom_private::tachiom::{Tachiom, TachiomInputDataset, TachiomSearchParams};
use vectorium::core::index::Index;
use vectorium::distances::Distance;
use vectorium::{DenseMultiVectorView, IndexSerializer};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Search a Tachiom IVF-PQ index with multivector (late-interaction) queries"
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

    println!("=== Tachiom Search ===");
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

    let search_params = TachiomSearchParams {
        k_centroids: args.k_centroids,
        k_docs_to_score: args.k_docs_to_score,
        ef_search: args.ef_search,
        alpha: args.alpha,
        beta: args.beta,
        lambda: args.lambda,
    };

    println!(
        "\nSearching {} queries ({} runs)...",
        n_queries, args.num_runs
    );

    // Pre-flatten all queries once to avoid repeated allocation in the hot path.
    let query_flat: Vec<Vec<f32>> = (0..n_queries)
        .map(|q| {
            let mut v = Vec::with_capacity(n_tokens_per_query * query_dim);
            let slice = queries_arr.slice(s![q, .., ..]);
            for t in 0..n_tokens_per_query {
                for d in 0..query_dim {
                    v.push(slice[[t, d]]);
                }
            }
            v
        })
        .collect();

    let mut total_time_us = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(n_queries * args.k);

    let t0 = Instant::now();
    for run in 0..args.num_runs {
        if run > 0 {
            results.clear();
        }
        for q in 0..n_queries {
            let query = DenseMultiVectorView::new(&query_flat[q], query_dim);

            let scored = <Tachiom<32> as Index<TachiomInputDataset>>::search(
                &index,
                query,
                args.k,
                &search_params,
            );

            results.extend(
                scored
                    .into_iter()
                    .map(|s| (s.distance.distance(), s.vector as usize)),
            );
        }
    }
    total_time_us += t0.elapsed().as_micros();

    let avg_us = total_time_us / (n_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_us} μs");

    // ── Save results ──────────────────────────────────────────────────────────
    if let Some(ref output_path) = args.output_path {
        println!("Writing results to {}...", output_path);
        let mut file = File::create(output_path)?;
        for (i, (score, doc_id)) in results.iter().enumerate() {
            let query_id = i / args.k;
            let rank = (i % args.k) + 1;
            writeln!(file, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score)?;
        }
    }

    Ok(())
}

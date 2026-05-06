use clap::Parser;
use half::f16;
use ndarray::Array2;
use ndarray_npy::WriteNpyExt;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;

use tachiom::tac::{TacBuilder, TacResult};

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about = "Train Token-Aware Clustering (TAC) on multivector data"
)]
struct Args {
    /// Path to the f16 token vectors file (.npy, dtype f16/u16, shape [N, dim])
    #[clap(short = 'i', long)]
    vectors_file: String,

    /// Path to the token-type ID file (.npy, dtype i64 or u32, shape [N])
    #[clap(long)]
    token_ids_file: String,

    /// Total centroid budget distributed across all token types
    #[clap(long)]
    total_centroids: usize,

    /// Number of k-means iterations per token group
    #[clap(long, default_value_t = 10)]
    tac_n_iter: usize,

    /// Output directory for centroids.npy, assignments.npy and timing.txt
    #[clap(short = 'o', long)]
    output_dir: String,

    /// Verbose output (spread scores, top allocations, etc.)
    #[clap(long)]
    verbose: bool,

    /// Test mode: load only the first 100 000 vectors (reads stop early, no full-file load)
    #[clap(long)]
    test_mode: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("=== Token-Aware Clustering (TAC) ===");
    println!("Input file:      {}", args.vectors_file);
    println!("Token IDs file:  {}", args.token_ids_file);
    println!("Total centroids: {}", args.total_centroids);
    println!("K-means iters:   {}", args.tac_n_iter);
    println!("Output dir:      {}", args.output_dir);
    println!("CPU cores:       {}", rayon::current_num_threads());

    std::fs::create_dir_all(&args.output_dir)?;

    let limit = args.test_mode.then_some(100_000_usize);
    if args.test_mode {
        println!("⚠ TEST MODE: loading at most 100 000 vectors");
    }

    // ── Load token vectors ────────────────────────────────────────────────────
    println!("\nLoading token vectors from {}...", args.vectors_file);
    let load_start = Instant::now();

    let (data_f16, dim) = read_f16_npy(&args.vectors_file, limit)?;
    let n_vectors = data_f16.len() / dim;

    // ── Load token IDs ────────────────────────────────────────────────────────
    println!("Loading token IDs from {}...", args.token_ids_file);
    let token_ids = read_token_ids_npy(&args.token_ids_file, limit)?;

    anyhow::ensure!(
        token_ids.len() == n_vectors,
        "token_ids length ({}) does not match number of vectors ({})",
        token_ids.len(),
        n_vectors
    );

    println!(
        "✓ Loaded {} vectors × dim={} ({:.2} GB) in {:.2?}",
        n_vectors,
        dim,
        data_f16.len() as f64 * 2.0 / 1e9,
        load_start.elapsed()
    );

    // ── Run TAC ───────────────────────────────────────────────────────────────
    println!("\n=== Running Token-Aware Clustering ===");
    let train_start = Instant::now();

    let tac = TacBuilder::new()
        .n_iter(args.tac_n_iter)
        .verbose(args.verbose)
        .build();

    let result: TacResult = tac.train(&data_f16, dim, &token_ids, args.total_centroids);

    let train_time = train_start.elapsed();
    println!("✓ TAC completed in {:.2?}", train_time);
    println!("✓ Final number of centroids: {}", result.n_centroids);

    // ── Statistics ────────────────────────────────────────────────────────────
    print_clustering_stats(&result.assignments, result.n_centroids, n_vectors);

    // ── Save results ──────────────────────────────────────────────────────────
    println!("\n=== Saving Results ===");

    let stem = format!(
        "it{}_k{}_n{}",
        args.tac_n_iter, result.n_centroids, n_vectors
    );

    save_centroids(
        &result.centroids,
        result.n_centroids,
        dim,
        &args.output_dir,
        &stem,
    )?;
    save_assignments(&result.assignments, &args.output_dir, &stem)?;

    let timing_path = Path::new(&args.output_dir).join("timing.txt");
    let mut timing_file = File::create(&timing_path)?;
    writeln!(
        timing_file,
        "TAC training time (s): {:.6}",
        train_time.as_secs_f64()
    )?;
    println!("✓ Timing written to {}", timing_path.display());

    println!("\n=== Summary ===");
    println!("Total time:       {:.2?}", load_start.elapsed());
    println!(
        "  - Data loading: {:.2?}",
        train_start.duration_since(load_start)
    );
    println!("  - TAC training: {:.2?}", train_time);
    println!("Results saved to: {}", args.output_dir);

    Ok(())
}

// ============================================================================
// I/O helpers
// ============================================================================

/// Read a 2-D f16 numpy file, loading at most `limit` rows.
///
/// NumPy stores float16 as dtype `<f2`; the raw bytes are identical to u16
/// bit patterns in little-endian order.  We parse the header to get the full
/// shape, then `read_exact` only `min(limit, n_vecs) * dim * 2` bytes so that
/// large files are never fully loaded in test mode.
fn read_f16_npy(path: &str, limit: Option<usize>) -> anyhow::Result<(Vec<f16>, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let (shape, elem_size) = parse_npy_header(&mut reader)?;
    anyhow::ensure!(
        shape.len() == 2,
        "Expected 2-D array for token vectors, got {}D",
        shape.len()
    );
    anyhow::ensure!(
        elem_size == 2,
        "Expected 2-byte dtype (f16/u16), got {} B/elem",
        elem_size
    );

    let (total_vecs, dim) = (shape[0], shape[1]);
    let n_read = limit.map_or(total_vecs, |l| l.min(total_vecs));

    let mut raw = vec![0u8; n_read * dim * 2];
    reader.read_exact(&mut raw)?;

    let data: Vec<f16> = raw
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    Ok((data, dim))
}

/// Read a 1-D token-type ID numpy file, loading at most `limit` entries.
///
/// Accepts i64 (ColBERT / PLAID standard) or u32 — the dtype is detected from
/// the numpy header, so no trial-and-error re-open is needed.
fn read_token_ids_npy(path: &str, limit: Option<usize>) -> anyhow::Result<Vec<usize>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let (shape, elem_size) = parse_npy_header(&mut reader)?;
    anyhow::ensure!(
        shape.len() == 1,
        "Expected 1-D array for token IDs, got {}D",
        shape.len()
    );
    anyhow::ensure!(
        elem_size == 4 || elem_size == 8,
        "Unsupported token-ID element size: {} bytes (expected 4=u32 or 8=i64)",
        elem_size
    );

    let total = shape[0];
    let n_read = limit.map_or(total, |l| l.min(total));

    let mut raw = vec![0u8; n_read * elem_size];
    reader.read_exact(&mut raw)?;

    let ids: Vec<usize> = match elem_size {
        8 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        4 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => unreachable!(),
    };

    Ok(ids)
}

/// Parse a NumPy v1/v2 file header, returning `(shape, elem_size_bytes)`.
///
/// Leaves `reader` positioned at the first byte of the data section.
/// Errors on Fortran-order or big-endian arrays (unusual in practice).
fn parse_npy_header(reader: &mut impl Read) -> anyhow::Result<(Vec<usize>, usize)> {
    // Magic: "\x93NUMPY"
    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    anyhow::ensure!(&magic == b"\x93NUMPY", "Not a NumPy file (magic mismatch)");

    // Version (major, minor)
    let mut ver = [0u8; 2];
    reader.read_exact(&mut ver)?;

    // HEADER_LEN: 2 bytes for v1, 4 bytes for v2+
    let header_len: usize = if ver[0] == 1 {
        let mut hl = [0u8; 2];
        reader.read_exact(&mut hl)?;
        u16::from_le_bytes(hl) as usize
    } else {
        let mut hl = [0u8; 4];
        reader.read_exact(&mut hl)?;
        u32::from_le_bytes(hl) as usize
    };

    // Header dict
    let mut hdr_bytes = vec![0u8; header_len];
    reader.read_exact(&mut hdr_bytes)?;
    let hdr = String::from_utf8_lossy(&hdr_bytes);

    anyhow::ensure!(
        !hdr.contains("'fortran_order': True"),
        "Fortran-order (column-major) NumPy arrays are not supported"
    );

    // Extract dtype descriptor, e.g. '<u2', '<i8', '<f2', '<u4'
    let descr = extract_quoted(&hdr, "'descr': '")
        .ok_or_else(|| anyhow::anyhow!("'descr' not found in numpy header: {}", hdr))?;

    anyhow::ensure!(
        !descr.starts_with('>'),
        "Big-endian numpy dtype '{}' is not supported",
        descr
    );

    let elem_size = dtype_elem_size(descr.trim_start_matches(['<', '=', '|']))
        .ok_or_else(|| anyhow::anyhow!("Unrecognised numpy dtype '{}'", descr))?;

    // Extract shape, e.g. (184030080, 128) or (184030080,)
    let shape_inner = {
        let tok = "'shape': (";
        let si = hdr
            .find(tok)
            .ok_or_else(|| anyhow::anyhow!("'shape' not found in numpy header"))?;
        let rest = &hdr[si + tok.len()..];
        let ei = rest
            .find(')')
            .ok_or_else(|| anyhow::anyhow!("shape closing ')' not found"))?;
        rest[..ei].to_string()
    };

    let shape: Vec<usize> = shape_inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("shape token '{}': {}", s.trim(), e))
        })
        .collect::<anyhow::Result<_>>()?;

    Ok((shape, elem_size))
}

/// Extract the string value after `prefix` up to the next single quote.
fn extract_quoted<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let start = s.find(prefix)? + prefix.len();
    let rest = &s[start..];
    let end = rest.find('\'')?;
    Some(&rest[..end])
}

/// Map a NumPy type-code string (endian prefix already stripped) to byte width.
fn dtype_elem_size(code: &str) -> Option<usize> {
    match code {
        "i1" | "u1" | "b1" => Some(1),
        "i2" | "u2" | "f2" => Some(2),
        "i4" | "u4" | "f4" => Some(4),
        "i8" | "u8" | "f8" => Some(8),
        _ => None,
    }
}

// ============================================================================
// Output helpers
// ============================================================================

fn save_centroids(
    centroids_f16: &[f16],
    n_centroids: usize,
    dim: usize,
    output_dir: &str,
    stem: &str,
) -> anyhow::Result<()> {
    let path = Path::new(output_dir).join(format!("centroids_{}.npy", stem));
    println!("Saving centroids to {}...", path.display());

    let centroids_f32: Vec<f32> = centroids_f16.iter().map(|x| x.to_f32()).collect();
    let arr = Array2::from_shape_vec((n_centroids, dim), centroids_f32)
        .map_err(|e| anyhow::anyhow!("Centroid reshape: {}", e))?;

    arr.write_npy(File::create(&path)?)?;
    println!("✓ Centroids saved ({} × {})", n_centroids, dim);
    Ok(())
}

fn save_assignments(assignments: &[u32], output_dir: &str, stem: &str) -> anyhow::Result<()> {
    let path = Path::new(output_dir).join(format!("assignments_{}.npy", stem));
    println!("Saving assignments to {}...", path.display());

    ndarray::Array1::from_vec(assignments.to_vec()).write_npy(File::create(&path)?)?;
    println!("✓ Assignments saved ({} entries)", assignments.len());
    Ok(())
}

fn print_clustering_stats(assignments: &[u32], n_centroids: usize, n_vectors: usize) {
    let mut counts = vec![0u32; n_centroids];
    for &c in assignments {
        counts[c as usize] += 1;
    }
    let min = *counts.iter().min().unwrap_or(&0);
    let max = *counts.iter().max().unwrap_or(&0);
    let avg = n_vectors as f64 / n_centroids as f64;

    println!("\n=== Clustering Statistics ===");
    println!("Total centroids:  {}", n_centroids);
    println!("Min cluster size: {}", min);
    println!("Max cluster size: {}", max);
    println!("Avg cluster size: {:.2}", avg);
    println!("Imbalance ratio:  {:.2}", max as f64 / avg);
}

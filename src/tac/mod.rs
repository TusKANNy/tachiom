//! Token-Aware Clustering (TAC).
//!
//! TAC is a clustering algorithm for multivector data where each token vector carries
//! a *token type id* (e.g. a vocabulary id from a late-interaction model such as
//! ColBERT).  Rather than running one global k-means over all token
//! vectors, TAC runs a separate k-means per token type and distributes the total
//! centroid budget proportionally to each group's size and internal spread.
//!
//! # Quick start
//!
//! ```ignore
//! use tachiom::tac::TacBuilder;
//!
//! let tac = TacBuilder::new().n_iter(20).verbose(true).build();
//! let result = tac.train(&data_f16, dim, &token_ids, total_centroids);
//! // result.centroids  – flat Vec<f16> of all centroids, ordered by token id
//! // result.assignments – global assignment: token_vec[i] -> centroid idx
//! ```

mod strategies;

use half::f16;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use vectorium::{
    Dataset, KMeans, KMeansBuilder, PlainDenseDataset, PlainDenseQuantizer,
    SquaredEuclideanDistance,
};

pub use strategies::allocate_centroids_damped_spread;
pub use strategies::compute_spread_measure;

// ============================================================================
// Default thresholds for token group classification
// ============================================================================

const DEFAULT_MICRO_THRESHOLD: usize = 128;
const DEFAULT_SMALL_THRESHOLD: usize = 256;

// ============================================================================
// Output
// ============================================================================

/// Result returned by [`TokenAwareClustering::train`].
pub struct TacResult {
    /// Flat buffer of all centroids in f16, ordered by token id (ascending).
    ///
    /// Layout: `[centroids_of_token0 || centroids_of_token1 || ...]`,
    /// where each block is `n_centroids_for_token_i * dim` values.
    pub centroids: Vec<f16>,

    /// Global assignment vector: `assignments[i]` is the centroid index
    /// assigned to the i-th input token vector.
    pub assignments: Vec<u32>,

    /// Vector dimension (token embedding size).
    pub dim: usize,

    /// Total number of centroids produced across all token types.
    pub n_centroids: usize,
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for [`TokenAwareClustering`].
pub struct TacBuilder {
    n_iter: usize,
    verbose: bool,
    /// Cap on training vectors per token group.
    /// `None` (default) = automatic formula: sample when group > 1M.
    /// `Some(usize::MAX)` = never sample, always use the full group.
    /// `Some(n)` = use at most `n` vectors for training.
    max_sample_size: Option<usize>,
    micro_threshold: Option<usize>,
    small_threshold: Option<usize>,
}

impl Default for TacBuilder {
    fn default() -> Self {
        TacBuilder {
            n_iter: 10,
            verbose: false,
            max_sample_size: None,
            micro_threshold: None,
            small_threshold: None,
        }
    }
}

impl TacBuilder {
    pub fn new() -> Self {
        TacBuilder::default()
    }

    /// Number of k-means iterations per token group (default: 10).
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Enable verbose progress output (default: false).
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Maximum training vectors per token group.
    ///
    /// - `None` (default): auto formula — when a group exceeds 1M vectors, sample
    ///   `min(10M, max(1M, 2·39·k, n/(2·n_iter)))` vectors for training, then
    ///   run the final assignment pass on the full group.
    /// - `Some(usize::MAX)`: disable sampling, always train on every vector.
    /// - `Some(n)`: hard cap at `n` training vectors regardless of group size.
    pub fn max_sample_size(mut self, max_sample_size: Option<usize>) -> Self {
        self.max_sample_size = max_sample_size;
        self
    }

    /// Override the micro threshold (default: 128).
    /// Token groups with fewer than this many vectors receive 1 centroid each.
    pub fn micro_threshold(mut self, micro_threshold: usize) -> Self {
        self.micro_threshold = Some(micro_threshold);
        self
    }

    /// Override the small threshold (default: 256).
    /// Token groups in `[micro_threshold, small_threshold)` receive 2 centroids each.
    pub fn small_threshold(mut self, small_threshold: usize) -> Self {
        self.small_threshold = Some(small_threshold);
        self
    }

    pub fn build(self) -> TokenAwareClustering {
        TokenAwareClustering {
            n_iter: self.n_iter,
            verbose: self.verbose,
            max_sample_size: self.max_sample_size,
            micro_threshold: self.micro_threshold,
            small_threshold: self.small_threshold,
        }
    }
}

// ============================================================================
// TokenAwareClustering
// ============================================================================

/// Token-Aware Clustering algorithm.
///
/// Instantiate via [`TacBuilder`] and call [`train`](TokenAwareClustering::train).
pub struct TokenAwareClustering {
    n_iter: usize,
    verbose: bool,
    max_sample_size: Option<usize>,
    micro_threshold: Option<usize>,
    small_threshold: Option<usize>,
}

impl Default for TokenAwareClustering {
    fn default() -> Self {
        TacBuilder::default().build()
    }
}

impl TokenAwareClustering {
    pub fn builder() -> TacBuilder {
        TacBuilder::new()
    }

    /// Run token-aware k-means on a flat buffer of f16 token vectors.
    ///
    /// # Arguments
    /// * `data`             – Flat f16 buffer; vector `i` occupies `data[i*dim .. (i+1)*dim]`.
    /// * `dim`              – Per-token embedding dimension.
    /// * `token_ids`        – One token-type id per vector (must have length `data.len() / dim`).
    /// * `total_centroids`  – Total centroid budget distributed across all token types.
    ///
    /// # Panics
    /// Panics if `token_ids.len() != data.len() / dim`.
    pub fn train(
        &self,
        data: &[f16],
        dim: usize,
        token_ids: &[usize],
        total_centroids: usize,
    ) -> TacResult {
        let n_vectors = data.len() / dim;
        assert_eq!(
            token_ids.len(),
            n_vectors,
            "token_ids.len() ({}) must equal data.len() / dim ({})",
            token_ids.len(),
            n_vectors
        );

        let total_centroids = if total_centroids > n_vectors {
            eprintln!(
                "[TAC] Warning: total_centroids ({}) > n_vectors ({}); \
                 capping to n_vectors to avoid k-means crash.",
                total_centroids, n_vectors
            );
            n_vectors
        } else {
            total_centroids
        };

        let train_start = Instant::now();

        // ── Group vector indices by token id ──────────────────────────────────
        let mut token_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &tid) in token_ids.iter().enumerate() {
            token_groups.entry(tid).or_default().push(idx);
        }

        if self.verbose {
            println!(
                "=== TAC: {} vectors × dim={}, {} unique token types, budget={} centroids ===",
                n_vectors,
                dim,
                token_groups.len(),
                total_centroids
            );
        }

        // ── Minimum budget check ──────────────────────────────────────────────
        // TAC needs at minimum 1 centroid per micro type (< micro_threshold vecs),
        // 2 per small type (micro_threshold..small_threshold vecs), and 4 per active type (≥ small_threshold vecs).
        // If the budget cannot cover even these floors, TAC produces meaningless
        // per-type clusters and panics during assignment lookup.
        // Fall back to a single global k-means on all vectors instead.
        let micro_threshold = self.micro_threshold.unwrap_or(DEFAULT_MICRO_THRESHOLD);
        let small_threshold = self.small_threshold.unwrap_or(DEFAULT_SMALL_THRESHOLD);
        let n_micro = token_groups
            .values()
            .filter(|v| v.len() < micro_threshold)
            .count();
        let n_small = token_groups
            .values()
            .filter(|v| v.len() >= micro_threshold && v.len() < small_threshold)
            .count();
        let n_active = token_groups
            .values()
            .filter(|v| v.len() >= small_threshold)
            .count();
        let min_tac_budget = n_micro + n_small * 2 + n_active * 4;

        if total_centroids < min_tac_budget {
            eprintln!(
                "[TAC] Warning: total_centroids ({}) < minimum TAC budget \
                 ({} = {}×1 + {}×2 + {}×4). \
                 TAC is designed for large-scale datasets; falling back to \
                 global k-means.",
                total_centroids, min_tac_budget, n_micro, n_small, n_active
            );
            let all_indices: Vec<usize> = (0..n_vectors).collect();
            let (centroids, assignments) = train_kmeans_for_token(
                data,
                &all_indices,
                dim,
                total_centroids,
                self.n_iter,
                self.max_sample_size,
            );
            return TacResult {
                centroids: centroids.values().to_vec(),
                assignments,
                dim,
                n_centroids: total_centroids,
            };
        }

        // ── Centroid budget allocation ─────────────────────────────────────────
        let allocation = allocate_centroids_damped_spread(
            &token_groups,
            data,
            dim,
            total_centroids,
            micro_threshold,
            small_threshold,
            self.verbose,
        );

        // ── Per-token k-means (parallel) ──────────────────────────────────────
        if self.verbose {
            println!("\n=== Training per-token k-means ===");
        }

        let results: Vec<(
            usize,
            PlainDenseDataset<f16, SquaredEuclideanDistance>,
            Vec<u32>,
        )> = token_groups
            .par_iter()
            .map(|(&token_id, indices)| {
                let k = allocation[&token_id];
                let (centroids, local_assignments) = train_kmeans_for_token(
                    data,
                    indices,
                    dim,
                    k,
                    self.n_iter,
                    self.max_sample_size,
                );
                (token_id, centroids, local_assignments)
            })
            .collect();

        // ── Concatenate centroids, remap assignments to global ids ────────────
        if self.verbose {
            println!("\n=== Concatenating centroids and remapping assignments ===");
        }

        // Sort by token_id for a deterministic, reproducible layout.
        let mut sorted = results;
        sorted.sort_by_key(|(tid, _, _)| *tid);

        let mut all_centroids: Vec<f16> = Vec::new();
        let mut global_assignments = vec![0u32; n_vectors];
        let mut centroid_offset = 0u32;

        for (token_id, centroids, local_assignments) in sorted {
            let n_token_centroids = centroids.len();
            all_centroids.extend_from_slice(centroids.values());

            for (local_idx, &global_idx) in token_groups[&token_id].iter().enumerate() {
                global_assignments[global_idx] = local_assignments[local_idx] + centroid_offset;
            }

            centroid_offset += n_token_centroids as u32;
        }

        let n_centroids = all_centroids.len() / dim;

        if self.verbose {
            println!(
                "✓ TAC complete in {:.2?} — {} centroids produced",
                train_start.elapsed(),
                n_centroids
            );
        }

        TacResult {
            centroids: all_centroids,
            assignments: global_assignments,
            dim,
            n_centroids,
        }
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Train k-means for a single token group; returns centroids + per-vector assignments.
///
/// Training may run on a sampled subset (see `max_sample_size`); assignments are
/// always computed on the full group.
///
/// `indices` indexes into the global `data` flat buffer (one entry per vector).
/// `token_id` is used as the RNG seed for both subsampling and KMeans initialisation,
/// making results fully reproducible regardless of which rayon thread handles this group.
fn train_kmeans_for_token(
    data: &[f16],
    indices: &[usize],
    dim: usize,
    k: usize,
    n_iter: usize,
    max_sample_size: Option<usize>,
) -> (PlainDenseDataset<f16, SquaredEuclideanDistance>, Vec<u32>) {
    let encoder = PlainDenseQuantizer::<f16, SquaredEuclideanDistance>::new(dim);

    if k == 0 || indices.is_empty() {
        return (
            PlainDenseDataset::from_raw(Box::new([]), 0, encoder),
            Vec::new(),
        );
    }

    let n = indices.len();

    // Determine training sample size.
    // When n > 1M, cap training at min(10M, n, max(1M, 2·39·k, n/(2·n_iter)))
    // so that head tokens (millions of vectors, small k) don't dominate training time.
    const AUTO_THRESHOLD: usize = 1_000_000;
    const AUTO_MAX: usize = 10_000_000;
    const MIN_PTS_PER_CENTROID: usize = 39;

    let training_n = match max_sample_size {
        Some(cap) => cap.min(n),
        None => {
            if n > AUTO_THRESHOLD {
                let by_cluster = 2 * MIN_PTS_PER_CENTROID * k;
                let by_iter = n / (2 * n_iter).max(1);
                let candidate = AUTO_THRESHOLD.max(by_cluster).max(by_iter);
                AUTO_MAX.min(n).min(candidate)
            } else {
                n
            }
        }
    };

    // Always extract the full token group into a contiguous buffer (used for assignment).
    let mut full_data: Vec<f16> = Vec::with_capacity(n * dim);
    for &idx in indices {
        full_data.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
    }
    let full_dataset = PlainDenseDataset::<f16, SquaredEuclideanDistance>::from_raw(
        full_data.into_boxed_slice(),
        n,
        encoder.clone(),
    );

    let centroids = if training_n < n {
        // Fixed seed for reproducible subsampling.
        let mut rng = StdRng::seed_from_u64(525);
        let local_indices: Vec<usize> = (0..n).collect();
        let sampled: Vec<usize> = local_indices
            .choose_multiple(&mut rng, training_n)
            .copied()
            .collect();

        let full_slice = full_dataset.values();
        let mut train_data: Vec<f16> = Vec::with_capacity(training_n * dim);
        for &li in &sampled {
            train_data.extend_from_slice(&full_slice[li * dim..(li + 1) * dim]);
        }
        let training_dataset = PlainDenseDataset::<f16, SquaredEuclideanDistance>::from_raw(
            train_data.into_boxed_slice(),
            training_n,
            encoder,
        );
        // Fixed seed for reproducible centroid initialization.
        KMeansBuilder::new()
            .n_iter(n_iter)
            .n_redo(1)
            .seed(Some(524))
            .verbose(false)
            .build()
            .train(&training_dataset, k, None)
    } else {
        // Fixed seed for reproducible centroid initialization.
        KMeansBuilder::new()
            .n_iter(n_iter)
            .n_redo(1)
            .seed(Some(524))
            .verbose(false)
            .build()
            .train(&full_dataset, k, None)
    };

    // Assign every vector in the full group to its nearest centroid.
    let assignments: Vec<u32> = KMeans::compute_assignments(&full_dataset, &centroids, 50_000)
        .iter()
        .map(|&(_, idx)| idx as u32)
        .collect();

    (centroids, assignments)
}

//! Global k-means for coarse centroids.
//!
//! Unlike [TAC](crate::tac), this ignores token ids and clusters all token vectors
//! together, so it applies to any multivector corpus (multilingual vocabularies,
//! image patches, ...).  Training runs vectorium's Lloyd's k-means on a sample of
//! the corpus; every vector is then assigned to its nearest centroid.
//!
//! [`KMeansAssign`] selects how the nearest centroid is found, both during training
//! and in the final assignment pass.

use half::f16;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index;
use rayon::prelude::*;
use std::time::Instant;

use vectorium::core::index::Index;
use vectorium::core::vector::DenseVectorView;
use vectorium::{
    DenseDatasetGeneric, KMeans, KMeansBuilder, PlainDenseDataset, PlainDenseQuantizer,
    SquaredEuclideanDistance,
};

use crate::graph::Graph;
use crate::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};

type Centroids = PlainDenseDataset<f16, SquaredEuclideanDistance>;
type CentroidHnsw = HNSW<Centroids, Graph>;

/// Nearest-centroid search used by k-means.
pub enum KMeansAssign {
    /// Exhaustive scan over all centroids: exact, `O(n · k · d)` per iteration.
    Flat,
    /// HNSW over the centroids, rebuilt every iteration and once more for the
    /// final assignment.  Approximate, but scales to millions of centroids.
    Hnsw {
        build: HNSWBuildConfiguration,
        search: HNSWSearchConfiguration,
    },
}

impl KMeansAssign {
    /// HNSW assignment with the parameters kANNolo uses for IVF training
    /// (`M = 32`, `ef_construction = 200`, `ef_search = 16`).
    pub fn hnsw_default() -> Self {
        KMeansAssign::Hnsw {
            build: HNSWBuildConfiguration::default()
                .with_num_neighbors(32)
                .with_ef_construction(200),
            search: HNSWSearchConfiguration::default().with_ef_search(16),
        }
    }
}

/// Parameters for [`train_kmeans`].
pub struct KMeansParams {
    /// Lloyd iterations (default: 10).
    pub n_iter: usize,
    /// Training sample size as a multiple of the centroid count: k-means trains on
    /// `min(n_vectors, points_per_centroid · k)` vectors (default: 64).
    pub points_per_centroid: usize,
    /// Seed for sampling and initialisation (default: 42).
    pub seed: u64,
    /// Nearest-centroid search (default: HNSW).
    pub assign: KMeansAssign,
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            n_iter: 10,
            points_per_centroid: 64,
            seed: 42,
            assign: KMeansAssign::hnsw_default(),
        }
    }
}

/// Result returned by [`train_kmeans`].
pub struct KMeansResult {
    /// Flat f16 buffer of all centroids, `n_centroids × dim`.
    pub centroids: Vec<f16>,
    /// `assignments[i]` is the centroid index of the i-th input vector.
    pub assignments: Vec<u32>,
    /// Vector dimension.
    pub dim: usize,
    /// Number of centroids produced.
    pub n_centroids: usize,
}

/// Run k-means on a flat buffer of f16 vectors; vector `i` is `data[i*dim..(i+1)*dim]`.
///
/// `k` is capped to the number of vectors.
pub fn train_kmeans(data: &[f16], dim: usize, k: usize, params: &KMeansParams) -> KMeansResult {
    let n = data.len() / dim;
    assert!(
        n > 0 && k > 0,
        "k-means needs at least one vector and one centroid"
    );
    let k = if k > n {
        eprintln!("[k-means] Warning: k ({k}) > n_vectors ({n}); capping to n_vectors.");
        n
    } else {
        k
    };

    // ── Training sample ──────────────────────────────────────────────────────
    let sample_start = Instant::now();
    let n_train = n.min(params.points_per_centroid.saturating_mul(k).max(k));
    let train_values: Vec<f16> = if n_train == n {
        data.to_vec()
    } else {
        let mut rng = StdRng::seed_from_u64(params.seed);
        let mut ids = index::sample(&mut rng, n, n_train).into_vec();
        ids.sort_unstable();
        let mut values = Vec::with_capacity(n_train * dim);
        for i in ids {
            values.extend_from_slice(&data[i * dim..(i + 1) * dim]);
        }
        values
    };
    let train_set = Centroids::from_raw(
        train_values.into_boxed_slice(),
        n_train,
        PlainDenseQuantizer::new(dim),
    );
    crate::timing::report("kmeans.sample", sample_start);

    println!("[k-means] {n} vectors × dim={dim}, k={k}, training on {n_train}");

    let kmeans = KMeansBuilder::new()
        .n_iter(params.n_iter)
        .n_redo(1)
        .seed(Some(params.seed))
        .build();

    // ── Train, then assign every vector ──────────────────────────────────────
    let (centroids, assignments) = match &params.assign {
        KMeansAssign::Flat => {
            let train_start = Instant::now();
            let centroids = kmeans.train(&train_set, k, None);
            crate::timing::report("kmeans.train", train_start);
            drop(train_set);

            let assign_start = Instant::now();
            let full = DenseDatasetGeneric::from_raw(data, n, PlainDenseQuantizer::new(dim));
            let assignments = KMeans::compute_assignments(&full, &centroids, 0)
                .into_iter()
                .map(|(_, c)| c as u32)
                .collect();
            crate::timing::report("kmeans.assign", assign_start);
            (centroids.values().to_vec(), assignments)
        }
        KMeansAssign::Hnsw { build, search } => {
            let train_start = Instant::now();
            let centroids = kmeans.train_with_index::<CentroidHnsw, f16, f16>(
                &train_set,
                k,
                None,
                |c| CentroidHnsw::build_index(c, build),
                search,
            );
            crate::timing::report("kmeans.train", train_start);
            drop(train_set);

            let assign_start = Instant::now();
            let values = centroids.values().to_vec();
            let index = CentroidHnsw::build_index(centroids, build);
            let assignments = data
                .par_chunks_exact(dim)
                .map_init(
                    || Vec::<f32>::with_capacity(dim),
                    |query, v| {
                        query.clear();
                        query.extend(v.iter().map(|x| x.to_f32()));
                        index.search(DenseVectorView::new(query.as_slice()), 1, search)[0].vector
                            as u32
                    },
                )
                .collect();
            crate::timing::report("kmeans.assign", assign_start);
            (values, assignments)
        }
    };

    KMeansResult {
        n_centroids: centroids.len() / dim,
        centroids,
        assignments,
        dim,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_data(n: usize, dim: usize) -> Vec<f16> {
        let mut rng = StdRng::seed_from_u64(7);
        (0..n * dim)
            .map(|_| f16::from_f32(rng.gen_range(-1.0..1.0)))
            .collect()
    }

    fn sq_dist(a: &[f16], b: &[f16]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x.to_f32() - y.to_f32()).powi(2))
            .sum()
    }

    /// Fraction of vectors assigned to their exact nearest returned centroid.
    fn nearest_fraction(data: &[f16], dim: usize, result: &KMeansResult) -> f64 {
        let hits = data
            .chunks_exact(dim)
            .zip(&result.assignments)
            .filter(|(v, a)| {
                let assigned = sq_dist(v, &result.centroids[**a as usize * dim..][..dim]);
                result
                    .centroids
                    .chunks_exact(dim)
                    .all(|c| assigned <= sq_dist(v, c) + 1e-4)
            })
            .count();
        hits as f64 / result.assignments.len() as f64
    }

    #[test]
    fn assignments_are_nearest_centroids() {
        let (n, dim, k) = (5_000, 16, 50);
        let data = random_data(n, dim);
        for (assign, min_exact) in [
            (KMeansAssign::Flat, 1.0),
            (KMeansAssign::hnsw_default(), 0.99),
        ] {
            // points_per_centroid = 20 trains on 1 000 of the 5 000 vectors.
            let params = KMeansParams {
                points_per_centroid: 20,
                assign,
                ..Default::default()
            };
            let result = train_kmeans(&data, dim, k, &params);
            assert_eq!(result.n_centroids, k);
            assert_eq!(result.centroids.len(), k * dim);
            assert_eq!(result.assignments.len(), n);
            assert!(nearest_fraction(&data, dim, &result) >= min_exact);
        }
    }

    #[test]
    fn k_is_capped_to_n_vectors() {
        let data = random_data(6, 4);
        let params = KMeansParams {
            assign: KMeansAssign::Flat,
            ..Default::default()
        };
        let result = train_kmeans(&data, 4, 100, &params);
        assert_eq!(result.n_centroids, 6);
        assert!(result.assignments.iter().all(|&c| (c as usize) < 6));
    }
}

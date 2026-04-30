use half::f16;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use serde::{Deserialize, Serialize};

use crate::graph::Graph;
use crate::hnsw::{
    EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration,
};
use crate::tac::TacBuilder;

use vectorium::core::dataset::ScoredVector;
use vectorium::core::index::Index;
use vectorium::core::vector::DenseVectorView;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::vector_encoder::{QueryEvaluator, VectorEncoder};
use vectorium::{
    Dataset, DenseDataset, IndexSerializer, MultiVecTwoLevelProductQuantizer, MultiVectorDataset,
    PlainDenseDataset, PlainDenseQuantizer, PlainMultiVecQuantizer, SpaceUsage,
};

// Type aliases for better readability
type CentroidDataset = DenseDataset<PlainDenseQuantizer<f16, DotProduct>>;
type HNSWCentroids = HNSW<CentroidDataset, Graph>;
type ResidualDataset<const M: usize> = MultiVectorDataset<MultiVecTwoLevelProductQuantizer<M, f16>>;

/// Min-heap wrapper for maintaining top-k scores (smaller = evicted first).
#[derive(Debug, Clone)]
struct MinHeapScore {
    score: f32,
    doc_id: u32,
}

impl Eq for MinHeapScore {}
impl PartialEq for MinHeapScore {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.doc_id == other.doc_id
    }
}
impl Ord for MinHeapScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap becomes a min-heap
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}
impl PartialOrd for MinHeapScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Serialize, Deserialize)]
/// A IVF (Inverted File) index structure for late-interaction multivector retrieval.
///
/// This index organizes token vectors using a coarse quantization approach:
/// - Centroids are stored in an HNSW index for fast nearest neighbor search
/// - Each centroid has an associated inverted list of document IDs
/// - Token vectors are stored in a multivector dataset with a two-level product quantizer
///
/// # Type Parameters
/// * `M`: The number of subspaces in the two-level product quantizer
pub struct Tachiom<const M: usize> {
    /// HNSW index over centroids (f16 vectors with inner product metric)
    pub centroids: HNSWCentroids,

    /// Flattened list of document IDs assigned to centroids (deduplicated per centroid)
    /// All document IDs assigned to centroid i are stored contiguously
    pub inverted_lists: Vec<u32>,

    /// Offsets array to locate inverted lists for each centroid
    /// inverted_lists[offsets[i]..offsets[i+1]] contains the IDs for centroid i
    pub offsets: Vec<usize>,

    /// Dataset containing all token vectors with parametric quantization.
    /// When the encoder has `with_norms = true`, per-token norms are embedded in the
    /// encoded byte payload — no external norm storage is needed.
    pub residuals: ResidualDataset<M>,

    /// Maximum number of tokens in any single document (cached for scratchpad pre-allocation).
    pub max_doc_tokens: usize,
}

impl<const M: usize> Tachiom<M> {
    /// Build an IVF index from centroids HNSW and token->centroid `assignments`.
    ///
    /// `assignments` must be a slice with one centroid id per token (same order as dataset).
    /// This function computes `inverted_lists` and `offsets`.
    pub fn from_parts(
        centroids: HNSWCentroids,
        assignments: &[usize],
        dataset: ResidualDataset<M>,
    ) -> Self {
        let n_documents = dataset.len() as usize;

        let n_centroids = centroids.n_elements() as usize;

        // Group documents per centroid (deduplicated via FxHashSet for faster hashing)
        let mut groups: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); n_centroids];

        let output_dim = dataset.encoder().output_dim();
        let ds_offsets = dataset.offsets();

        // Calculate token count from dataset offsets
        let n_tokens = ds_offsets.last().unwrap() / output_dim;

        assert_eq!(
            assignments.len(),
            n_tokens,
            "assignments.len={} must equal number of tokens in dataset={}",
            assignments.len(),
            n_tokens
        );

        let mut token_idx = 0;
        for doc_id in 0..n_documents {
            let doc_tokens = (ds_offsets[doc_id + 1] - ds_offsets[doc_id]) / output_dim;
            for _ in 0..doc_tokens {
                let c_id = assignments[token_idx];
                if c_id >= n_centroids {
                    panic!(
                        "assignment centroid id {} >= n_centroids {}",
                        c_id, n_centroids
                    );
                }
                groups[c_id].insert(doc_id as u32);
                token_idx += 1;
            }
        }

        // Flatten groups into inverted_lists (document IDs) and build offsets
        let mut inverted_lists: Vec<u32> = Vec::new();
        let mut offsets: Vec<usize> = Vec::with_capacity(n_centroids + 1);
        offsets.push(0);

        for grp in groups.iter() {
            for &doc_id in grp.iter() {
                inverted_lists.push(doc_id);
            }
            offsets.push(inverted_lists.len());
        }

        // Compute maximum tokens per document for scratchpad pre-allocation
        let max_doc_tokens = ds_offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) / output_dim)
            .max()
            .unwrap_or(0);

        Tachiom {
            centroids,
            inverted_lists,
            offsets,
            residuals: dataset,
            max_doc_tokens,
        }
    }

    /// Search the index for the k nearest neighbor documents to the query.
    ///
    /// # Arguments
    /// * `query` - Query multivector in f32 format
    /// * `k` - Number of final results to return
    /// * `k_centroids` - Number of centroids to search
    /// * `k_docs_to_score` - Maximum number of documents to rerank
    /// * `ef_search` - The ef construction parameter for HNSW search
    /// * `alpha` - Optional pruning threshold (fraction relative to k-th score)
    /// * `beta` - Optional early termination staleness counter
    pub fn search<'a>(
        &'a self,
        query: vectorium::DenseMultiVectorView<'a, f32>,
        k: usize,
        k_centroids: usize,
        k_docs_to_score: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        lambda: Option<f32>,
    ) -> Vec<(f32, u32)> {
        let early_termination = if let Some(lambda_val) = lambda {
            EarlyTerminationStrategy::DistanceAdaptive { lambda: lambda_val }
        } else {
            EarlyTerminationStrategy::None
        };

        let search_params = HNSWSearchConfiguration::default()
            .with_ef_search(ef_search)
            .with_early_termination(early_termination);

        let mut doc_scores: FxHashMap<u32, f32> = FxHashMap::default();
        doc_scores.reserve(4096); // Accumulates docs from all tokens; expect thousands to tens of thousands

        let n_query_tokens = query.num_vecs();
        let query_dim = query.dim();
        let mut best_per_doc: FxHashMap<u32, f32> = FxHashMap::default();
        best_per_doc.reserve(128); // Per-token map, reused each iteration; pre-allocate reasonable size

        for qi in 0..n_query_tokens {
            let q_token_f32 = &query.values()[qi * query_dim..(qi + 1) * query_dim];
            let q_token = DenseVectorView::new(q_token_f32);

            let centroids_res = self.centroids.search(q_token, k_centroids, &search_params);

            best_per_doc.clear();
            for scored_vec in &centroids_res {
                let cidx = scored_vec.vector as usize;
                let dist = scored_vec.distance.0;
                let off_start = self.offsets[cidx];
                let off_end = if cidx + 1 < self.offsets.len() {
                    self.offsets[cidx + 1]
                } else {
                    self.inverted_lists.len()
                };

                for &doc_id in self.inverted_lists[off_start..off_end].iter() {
                    best_per_doc
                        .entry(doc_id)
                        .and_modify(|prev| *prev = dist.max(*prev))
                        .or_insert(dist);
                }
            }

            for (&doc_id, &best_sim) in best_per_doc.iter() {
                *doc_scores.entry(doc_id).or_insert(0.0) += best_sim;
            }
        }

        // Stage 2: Candidate selection and pruning
        let mut docs_with_scores: Vec<(u32, f32)> = doc_scores.into_iter().collect();

        let take_n = std::cmp::min(k_docs_to_score, docs_with_scores.len());

        // Use partition instead of full sort: find the k-th best in O(n) instead of O(n log n)
        if docs_with_scores.len() > take_n {
            // Partition at top-k position (score comparison is reversed for descending)
            docs_with_scores.select_nth_unstable_by(take_n - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            docs_with_scores.truncate(take_n);
        }

        // Sort only the top-k candidates (much smaller set now)
        docs_with_scores
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));

        let mut candidates = docs_with_scores;

        // Apply alpha-based pruning if specified
        if let Some(alpha_val) = alpha {
            if !candidates.is_empty() && k > 0 {
                let kth_idx = std::cmp::min(k, candidates.len()) - 1;
                // Handle negative scores gracefully (wider acceptance means a lower threshold)
                let threshold = candidates[kth_idx].1 - candidates[kth_idx].1.abs() * alpha_val;
                // Keep candidates with score >= threshold (more positive is better)
                candidates.retain(|&(_, s)| s >= threshold);
            }
        }

        // Stage 3: Full distance computation and top-k selection

        // Use a simple vector to track top-k results (sorting at the end)
        let mut result_scores: Vec<(f32, u32)> = Vec::new();

        // Create the base evaluator once, giving it the full global norms vector
        // Norms, if present, are embedded in the encoded payload and extracted internally
        // by the evaluator. No external scratchpad or norm slice is needed.
        let query_evaluator = self.residuals.encoder().query_evaluator(query);

        let score_doc = |doc_id: u32| -> f32 {
            let doc_view = self.residuals.get(doc_id as u64);
            query_evaluator.compute_distance(doc_view).0
        };

        if beta.is_some() && candidates.len() >= k && k > 0 {
            // Beta-based early termination: use a min-heap to avoid O(n log n) sorting per insertion
            let mut heap: BinaryHeap<MinHeapScore> = BinaryHeap::with_capacity(k);
            let beta_val = beta.unwrap();

            // Score first k documents
            for (doc_id, _) in candidates.iter().take(k) {
                let score = score_doc(*doc_id);
                heap.push(MinHeapScore {
                    score,
                    doc_id: *doc_id,
                });
            }

            // Score remaining candidates with early termination
            let mut n_stalls = 0usize;
            for (doc_id, _) in candidates.iter().skip(k) {
                let score = score_doc(*doc_id);

                // Only insert if better than worst in heap
                if let Some(worst) = heap.peek() {
                    if score > worst.score {
                        heap.push(MinHeapScore {
                            score,
                            doc_id: *doc_id,
                        });
                        if heap.len() > k {
                            heap.pop();
                        }
                        n_stalls = 0;
                    } else {
                        n_stalls += 1;
                        if n_stalls >= beta_val {
                            break;
                        }
                    }
                }
            }

            // Extract results from heap
            result_scores.reserve(heap.len());
            while let Some(item) = heap.pop() {
                result_scores.push((item.score, item.doc_id));
            }
            // Results come out in ascending order, so reverse for descending
            result_scores.reverse();
        } else {
            // Score all candidates
            for (doc_id, _) in candidates.iter() {
                let score = score_doc(*doc_id);
                result_scores.push((score, *doc_id));
            }
            result_scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            result_scores.truncate(k);
        }

        result_scores
    }
}

// ============================================================================
// Index trait support types
// ============================================================================

/// Parameters controlling how a [`Tachiom`] index is built.
pub struct TachiomBuildParams {
    /// Token-type IDs for TAC, one per token in dataset order.
    ///
    /// `token_ids[i]` is the vocabulary / token-type id of the i-th token in
    /// the flattened dataset (documents concatenated in order).  TAC uses these
    /// to allocate the centroid budget proportionally across token types.
    pub token_ids: Vec<usize>,

    /// Total coarse-centroid budget (number of IVF clusters).
    pub total_centroids: usize,

    /// K-means iterations per token type inside TAC (default: 10).
    pub tac_n_iter: usize,

    /// Number of tokens sampled for PQ training.
    pub pq_sample_size: usize,

    /// K-means iterations for PQ subspace training (default: 10).
    pub pq_n_iter: usize,

    /// Normalise residuals before PQ encoding; embeds original norm in payload.
    pub normalize: bool,

    /// Optional RNG seed for reproducible PQ training.
    pub pq_seed: Option<u64>,

    /// HNSW build configuration for the coarse-centroid index.
    pub hnsw_params: HNSWBuildConfiguration,
}

/// Parameters controlling a single [`Tachiom`] search.
pub struct TachiomSearchParams {
    /// Number of coarse centroids to probe per query token.
    pub k_centroids: usize,

    /// Maximum number of candidate documents to score in stage 2.
    pub k_docs_to_score: usize,

    /// HNSW `ef_search` for centroid lookup.
    pub ef_search: usize,

    /// Alpha-based candidate pruning threshold (fraction of the k-th score).
    pub alpha: Option<f32>,

    /// Beta-based early-termination staleness counter.
    pub beta: Option<usize>,

    /// Lambda for distance-adaptive early termination in HNSW centroid search.
    pub lambda: Option<f32>,
}

impl Default for TachiomSearchParams {
    fn default() -> Self {
        Self {
            k_centroids: 20,
            k_docs_to_score: 500,
            ef_search: 30,
            alpha: Some(0.45),
            beta: None,
            lambda: None,
        }
    }
}

// ============================================================================
// Index trait implementation
// ============================================================================

/// The raw multivector input type for building a Tachiom index.
///
/// Each document is stored as a variable-length sequence of f16 token vectors.
pub type TachiomInputDataset = MultiVectorDataset<PlainMultiVecQuantizer<f16>>;

impl<const M: usize> Index<TachiomInputDataset> for Tachiom<M> {
    type BuildParams = TachiomBuildParams;
    type SearchParams = TachiomSearchParams;

    /// Number of indexed documents.
    fn n_elements(&self) -> usize {
        self.residuals.len()
    }

    /// Dimensionality of each token vector (before quantization).
    fn dim(&self) -> usize {
        self.residuals.encoder().input_dim()
    }

    fn print_space_usage_bytes(&self) {
        let centroid_hnsw_size = self.centroids.space_usage_bytes();
        let inv_lists_size = self.inverted_lists.space_usage_bytes();
        let offsets_size = self.offsets.space_usage_bytes();
        let residuals_size = self.residuals.space_usage_bytes();
        let total = centroid_hnsw_size + inv_lists_size + offsets_size + residuals_size;
        println!(
            "[Tachiom] Space: centroids_hnsw={centroid_hnsw_size}B, \
             inverted_lists={inv_lists_size}B, offsets={offsets_size}B, \
             residuals={residuals_size}B, total={total}B"
        );
    }

    /// Build a Tachiom index from a raw multivector dataset.
    ///
    /// Pipeline:
    /// 1. Run TAC (or plain k-means) to obtain coarse centroids + per-token assignments.
    /// 2. Train the residual PQ via [`MultiVecTwoLevelProductQuantizer::train_from_coarse`].
    /// 3. Encode all documents in parallel using the trained encoder.
    /// 4. Build an HNSW index over the coarse centroids.
    /// 5. Build inverted lists mapping centroids → documents.
    fn build_index(dataset: TachiomInputDataset, params: &TachiomBuildParams) -> Self {
        let token_dim = dataset.encoder().input_dim();
        let n_docs = dataset.len();

        // ── Extract flat f16 token data and per-doc token counts ──────────────
        let flat_f16: &[f16] = dataset.values();
        let n_tokens = flat_f16.len() / token_dim;
        let doc_token_counts: Vec<usize> = dataset
            .offsets()
            .windows(2)
            .map(|w| (w[1] - w[0]) / token_dim)
            .collect();

        assert_eq!(doc_token_counts.len(), n_docs);

        println!(
            "[Tachiom::build_index] {} docs, {} tokens, dim={}",
            n_docs, n_tokens, token_dim
        );

        // ── Step 1: Token-Aware Clustering ───────────────────────────────────
        assert_eq!(
            params.token_ids.len(),
            n_tokens,
            "token_ids length ({}) must equal n_tokens ({})",
            params.token_ids.len(),
            n_tokens
        );
        println!("[Tachiom::build_index] Step 1: Token-Aware Clustering...");
        let tac = TacBuilder::new().n_iter(params.tac_n_iter).build();
        let tac_result = tac.train(
            flat_f16,
            token_dim,
            &params.token_ids,
            params.total_centroids,
        );
        let n_centroids = tac_result.n_centroids;
        let assignments_usize: Vec<usize> =
            tac_result.assignments.iter().map(|&x| x as usize).collect();
        let centroids_f16 = tac_result.centroids;

        // ── Step 2: Build centroid dataset (f16) and coarse centroids (f32) ───
        println!("[Tachiom::build_index] Step 2: Building encoder...");

        let centroids_f32: Vec<f32> = centroids_f16.iter().map(|x| x.to_f32()).collect();

        let coarse_centroids_ds = PlainDenseDataset::<f32, SquaredEuclideanDistance>::from_raw(
            centroids_f32.into_boxed_slice(),
            n_centroids,
            PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(token_dim),
        );

        // Pass flat_f16 directly — train_from_coarse converts to f32 on the fly
        // for the sampled subset only, so no full N×D×4 copy is needed.
        let encoder = MultiVecTwoLevelProductQuantizer::<M, f16>::train_from_coarse(
            coarse_centroids_ds,
            flat_f16,
            &assignments_usize,
            params.pq_sample_size,
            params.pq_n_iter,
            params.normalize,
            params.pq_seed,
        );

        // ── Step 3: Encode all documents ──────────────────────────────────────
        // Use push_encoded_with_ids to bypass search_nearest over ncoarse centroids —
        // with millions of centroids a brute-force search per token is infeasible.
        // Instead we use the TAC assignments computed in Step 1 as direct lookups.
        println!(
            "[Tachiom::build_index] Step 3: Encoding {} documents...",
            n_docs
        );
        let residuals = {
            use rayon::prelude::*;
            let output_dim = encoder.output_dim();

            // Build cumulative token-start per document (needed to slice flat_f16 and assignments).
            let mut token_starts = Vec::with_capacity(n_docs + 1);
            token_starts.push(0usize);
            for &n in &doc_token_counts {
                token_starts.push(token_starts.last().unwrap() + n);
            }

            let enc_ref = &encoder;
            let f16_ref = flat_f16;
            let asgn_ref = &assignments_usize;

            // Encode each document in parallel; the pre-computed coarse IDs from TAC
            // are used directly, so no centroid search is performed.
            let encoded_docs: Vec<Vec<u8>> = (0..n_docs)
                .into_par_iter()
                .map(|doc_id| {
                    let tok_start = token_starts[doc_id];
                    let n_tok = doc_token_counts[doc_id];
                    let tok_slice =
                        &f16_ref[tok_start * token_dim..(tok_start + n_tok) * token_dim];
                    let coarse_ids: Vec<u32> = asgn_ref[tok_start..tok_start + n_tok]
                        .iter()
                        .map(|&x| x as u32)
                        .collect();
                    let view = vectorium::DenseMultiVectorView::new(tok_slice, token_dim);
                    let mut buf = Vec::with_capacity(n_tok * output_dim);
                    enc_ref.push_encoded_with_ids(view, &coarse_ids, &mut buf);
                    buf
                })
                .collect();

            // Assemble flat encoded buffer and offsets.
            let total_len: usize = encoded_docs.iter().map(|d| d.len()).sum();
            let mut enc_data = Vec::with_capacity(total_len);
            let mut enc_offsets: Vec<usize> = Vec::with_capacity(n_docs + 1);
            enc_offsets.push(0);
            for doc in &encoded_docs {
                enc_data.extend_from_slice(doc);
                enc_offsets.push(enc_data.len());
            }

            MultiVectorDataset::from_raw(
                enc_data.into_boxed_slice(),
                enc_offsets.into_boxed_slice(),
                encoder,
            )
        };

        // ── Step 4: Build HNSW on coarse centroids ────────────────────────────
        println!(
            "[Tachiom::build_index] Step 4: Building HNSW on {} centroids...",
            n_centroids
        );
        let centroid_dataset: CentroidDataset = DenseDataset::from_raw(
            centroids_f16.into_boxed_slice(),
            n_centroids,
            PlainDenseQuantizer::<f16, DotProduct>::new(token_dim),
        );
        let centroids_hnsw = HNSWCentroids::build_index(centroid_dataset, &params.hnsw_params);

        // ── Step 5: Build inverted lists ──────────────────────────────────────
        println!("[Tachiom::build_index] Step 5: Building inverted lists...");
        Tachiom::from_parts(centroids_hnsw, &assignments_usize, residuals)
    }

    fn search<'q>(
        &'q self,
        query: vectorium::DenseMultiVectorView<'q, f32>,
        k: usize,
        search_params: &TachiomSearchParams,
    ) -> Vec<ScoredVector<DotProduct>> {
        self.search(
            query,
            k,
            search_params.k_centroids,
            search_params.k_docs_to_score,
            search_params.ef_search,
            search_params.alpha,
            search_params.beta,
            search_params.lambda,
        )
        .into_iter()
        .map(|(score, doc_id)| ScoredVector {
            distance: DotProduct::from(score),
            vector: doc_id as u64,
        })
        .collect()
    }
}

impl<const M: usize> IndexSerializer for Tachiom<M> {}

//! Python bindings for Tachiom (PyO3).
//!
//! Compiled when the `python` feature is enabled.  Builds a `cdylib` exposing a
//! single `tachiom` module with one `Tachiom` class.

use crate::hnsw::HNSWBuildConfiguration;
use crate::tac::{TacBuilder, TacResult};
use crate::tachiom::{Tachiom, TachiomBuildParams, TachiomInputDataset};
use vectorium::core::index::Index;
use vectorium::vector_encoder::{MultiVecEncoder, VectorEncoder};
use vectorium::{
    Dataset, DenseMultiVectorView, IndexSerializer, MultiVectorDataset, PlainMultiVecQuantizer,
};

use half::f16;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

use std::fs::File;
use std::io::{BufReader, Read};

/// PQ subspace count.  Hard-coded to match the rest of the codebase; the public
/// `pq_subspaces` kwarg is validated against this value (warns if different).
const M_FIXED: usize = 32;

// ============================================================================
// Module
// ============================================================================

#[pymodule]
fn tachiom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTachiom>()?;
    m.add_class::<PyTac>()?;
    Ok(())
}

// ============================================================================
// PyTachiom class
// ============================================================================

#[pyclass(name = "Tachiom", module = "tachiom", unsendable)]
pub struct PyTachiom {
    inner: Tachiom<M_FIXED>,
}

#[pymethods]
impl PyTachiom {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Build a Tachiom index from raw .npy inputs (full pipeline: TAC → PQ → HNSW).
    #[classmethod]
    #[pyo3(signature = (
        vectors_path,
        token_ids_path,
        doclens_path,
        *,
        total_centroids = 4_194_304,
        tac_n_iter = 10,
        pq_sample_size = 10_000_000,
        pq_n_iter = 10,
        normalize = false,
        pq_seed = 42,
        hnsw_m = 32,
        ef_construction = 1500,
        pq_subspaces = 32,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        vectors_path: &str,
        token_ids_path: &str,
        doclens_path: &str,
        total_centroids: usize,
        tac_n_iter: usize,
        pq_sample_size: usize,
        pq_n_iter: usize,
        normalize: bool,
        pq_seed: u64,
        hnsw_m: usize,
        ef_construction: usize,
        pq_subspaces: usize,
    ) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        let (dataset, token_ids) = load_input_dataset(vectors_path, token_ids_path, doclens_path)?;

        let params = TachiomBuildParams {
            token_ids,
            total_centroids,
            tac_n_iter,
            pq_sample_size,
            pq_n_iter,
            normalize,
            pq_seed: Some(pq_seed),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(hnsw_m)
                .with_ef_construction(ef_construction),
        };

        let inner = py.allow_threads(|| Tachiom::<M_FIXED>::build_index(dataset, &params));
        Ok(PyTachiom { inner })
    }

    /// Build a Tachiom index from in-memory numpy arrays (full pipeline: TAC → PQ → HNSW).
    ///
    /// Equivalent to `build()` but accepts numpy arrays instead of file paths.
    /// Supports memory-mapped arrays (`np.load(..., mmap_mode='r')`) to minimise RAM
    /// usage during construction — data is read from the buffer with a single copy.
    ///
    /// `vectors` must be f16 and C-contiguous.  Cast with `.astype(np.float16)` if needed.
    #[classmethod]
    #[pyo3(signature = (
        vectors,
        token_ids,
        doclens,
        *,
        total_centroids = 4_194_304,
        tac_n_iter = 10,
        pq_sample_size = 10_000_000,
        pq_n_iter = 10,
        normalize = false,
        pq_seed = 42,
        hnsw_m = 32,
        ef_construction = 1500,
        pq_subspaces = 32,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build_from_arrays(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        vectors: PyReadonlyArray2<'_, u16>,
        token_ids: PyReadonlyArray1<'_, u32>,
        doclens: PyReadonlyArray1<'_, i32>,
        total_centroids: usize,
        tac_n_iter: usize,
        pq_sample_size: usize,
        pq_n_iter: usize,
        normalize: bool,
        pq_seed: u64,
        hnsw_m: usize,
        ef_construction: usize,
        pq_subspaces: usize,
    ) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        let (dataset, token_ids_vec) = dataset_from_arrays(&vectors, &token_ids, &doclens)?;

        let params = TachiomBuildParams {
            token_ids: token_ids_vec,
            total_centroids,
            tac_n_iter,
            pq_sample_size,
            pq_n_iter,
            normalize,
            pq_seed: Some(pq_seed),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(hnsw_m)
                .with_ef_construction(ef_construction),
        };

        let inner = py.allow_threads(|| Tachiom::<M_FIXED>::build_index(dataset, &params));
        Ok(PyTachiom { inner })
    }

    /// Build a Tachiom index using pre-computed coarse centroids and assignments.
    /// Skips the TAC step.  Useful for isolating retrieval differences between
    /// clustering and residual encoding.
    #[classmethod]
    #[pyo3(signature = (
        vectors_path,
        token_ids_path,
        doclens_path,
        centroids_path,
        assignments_path,
        *,
        pq_sample_size = 10_000_000,
        pq_n_iter = 10,
        normalize = false,
        pq_seed = 42,
        hnsw_m = 32,
        ef_construction = 1500,
        pq_subspaces = 32,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build_from_tac(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        vectors_path: &str,
        token_ids_path: &str,
        doclens_path: &str,
        centroids_path: &str,
        assignments_path: &str,
        pq_sample_size: usize,
        pq_n_iter: usize,
        normalize: bool,
        pq_seed: u64,
        hnsw_m: usize,
        ef_construction: usize,
        pq_subspaces: usize,
    ) -> PyResult<Self> {
        warn_pq_subspaces(py, pq_subspaces)?;
        let (dataset, token_ids) = load_input_dataset(vectors_path, token_ids_path, doclens_path)?;
        let n_tokens = token_ids.len();

        let (centroids_f32, n_centroids, _dim) = read_f32_2d_npy(centroids_path)?;
        let centroids_f16: Vec<f16> = centroids_f32.iter().map(|&x| f16::from_f32(x)).collect();

        let assignments = read_assignments_npy(assignments_path, n_tokens)?;

        let params = TachiomBuildParams {
            token_ids,
            total_centroids: n_centroids, // unused by build_index_from_tac, required by struct
            tac_n_iter: 0,                // unused
            pq_sample_size,
            pq_n_iter,
            normalize,
            pq_seed: Some(pq_seed),
            hnsw_params: HNSWBuildConfiguration::default()
                .with_num_neighbors(hnsw_m)
                .with_ef_construction(ef_construction),
        };

        let inner = py.allow_threads(|| {
            Tachiom::<M_FIXED>::build_index_from_tac(
                centroids_f16,
                n_centroids,
                assignments,
                dataset,
                &params,
            )
        });
        Ok(PyTachiom { inner })
    }

    /// Load a previously-saved Tachiom index from disk.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_owned = path.to_owned();
        let inner = py
            .allow_threads(|| Tachiom::<M_FIXED>::load_index(&path_owned))
            .map_err(|e| PyIOError::new_err(format!("Failed to load index: {e:?}")))?;
        Ok(PyTachiom { inner })
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    /// Save the index to disk (bincode-flavoured serialization).
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let path_owned = path.to_owned();
        py.allow_threads(|| self.inner.save_index(&path_owned))
            .map_err(|e| PyIOError::new_err(format!("Failed to save index: {e:?}")))?;
        Ok(())
    }

    // ── Search ───────────────────────────────────────────────────────────────

    /// Search a single multivector query.
    ///
    /// `query` must be a 2D C-contiguous f32 array of shape `(n_tokens, dim)`.
    /// Returns `(scores, doc_ids)` as 1D ndarrays of length `k` (sentinel-padded
    /// when fewer than `k` results are produced).
    #[pyo3(signature = (
        query, k = 10, *,
        k_centroids = 20,
        k_docs_to_score = 500,
        ef_search = 30,
        alpha = Some(0.45),
        beta = None,
        lambda_ = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'py, f32>,
        k: usize,
        k_centroids: usize,
        k_docs_to_score: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        lambda_: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<u32>>)> {
        let dim = self.inner.residuals.encoder().input_dim();
        let q_slice = require_contiguous_2d(&query, dim, "query")?;
        let q_view = DenseMultiVectorView::new(q_slice, dim);

        let result: Vec<(f32, u32)> = py.allow_threads(|| {
            self.inner.search(
                q_view,
                k,
                k_centroids,
                k_docs_to_score,
                ef_search,
                alpha,
                beta,
                lambda_,
            )
        });

        let (scores, doc_ids) = pad_result(result, k);
        Ok((
            scores.into_pyarray(py).unbind(),
            doc_ids.into_pyarray(py).unbind(),
        ))
    }

    /// Search a batch of multivector queries.
    ///
    /// `tokens` is a flat 2D C-contiguous f32 array of shape `(total_tokens, dim)` with
    /// all query token vectors concatenated in query order.  `n_queries` states the number
    /// of queries and is always required.
    ///
    /// **Uniform mode** (`offsets = None`): all queries are assumed to have the same token
    /// count.  `total_tokens` must divide evenly by `n_queries`; the per-query stride is
    /// `total_tokens / n_queries`.
    ///
    /// **Ragged mode** (`offsets` provided): a 1D u64 array of length `n_queries + 1`.
    /// `offsets[i]..offsets[i+1]` is the row range in `tokens` for query `i`.
    /// `n_queries` is validated against `len(offsets) - 1`.
    ///
    /// Returns `(scores, doc_ids)` — both 2D ndarrays of shape `(n_queries, k)`,
    /// sentinel-padded when fewer than `k` results are produced for a given query.
    ///
    /// `num_threads`:
    /// - `0` — rayon's default thread pool (typically all cores).
    /// - `1` — serial loop (reproducible single-thread benchmarks).
    /// - `n` — temporary rayon pool of size `n` for this call.
    #[pyo3(signature = (
        tokens, n_queries, k = 10, *,
        offsets = None,
        num_threads = 0,
        k_centroids = 20,
        k_docs_to_score = 500,
        ef_search = 30,
        alpha = Some(0.45),
        beta = None,
        lambda_ = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn batch_search<'py>(
        &self,
        py: Python<'py>,
        tokens: PyReadonlyArray2<'py, f32>,
        n_queries: usize,
        k: usize,
        offsets: Option<PyReadonlyArray1<'py, u64>>,
        num_threads: usize,
        k_centroids: usize,
        k_docs_to_score: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        lambda_: Option<f32>,
    ) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<u32>>)> {
        let dim = self.inner.residuals.encoder().input_dim();

        let tokens_shape = tokens.shape();
        if tokens_shape.len() != 2 || tokens_shape[1] != dim {
            return Err(PyValueError::new_err(format!(
                "tokens must have shape (total_tokens, {dim}); got {tokens_shape:?}"
            )));
        }
        if !tokens.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "tokens must be C-contiguous; call np.ascontiguousarray(tokens) first",
            ));
        }
        if n_queries == 0 {
            return Err(PyValueError::new_err("n_queries must be > 0"));
        }
        let total_tokens = tokens_shape[0];
        let tokens_slice = tokens
            .as_slice()
            .map_err(|_| PyValueError::new_err("tokens could not be exposed as a slice"))?;

        let views: Vec<DenseMultiVectorView<f32>> = match offsets {
            None => {
                if total_tokens % n_queries != 0 {
                    return Err(PyValueError::new_err(format!(
                        "total_tokens={total_tokens} is not divisible by n_queries={n_queries}; \
                         provide an offsets array for variable-length queries"
                    )));
                }
                let stride = total_tokens / n_queries * dim;
                (0..n_queries)
                    .map(|i| {
                        DenseMultiVectorView::new(&tokens_slice[i * stride..(i + 1) * stride], dim)
                    })
                    .collect()
            }
            Some(off) => {
                if !off.is_c_contiguous() {
                    return Err(PyValueError::new_err("offsets must be C-contiguous"));
                }
                let off_slice = off.as_slice().map_err(|_| {
                    PyValueError::new_err("offsets could not be exposed as a slice")
                })?;
                if off_slice.len() != n_queries + 1 {
                    return Err(PyValueError::new_err(format!(
                        "offsets length ({}) must equal n_queries + 1 ({})",
                        off_slice.len(),
                        n_queries + 1
                    )));
                }
                if *off_slice.last().unwrap() as usize != total_tokens {
                    return Err(PyValueError::new_err(format!(
                        "offsets[-1]={} != total_tokens={total_tokens}",
                        off_slice.last().unwrap()
                    )));
                }
                let mut views = Vec::with_capacity(n_queries);
                for i in 0..n_queries {
                    let start = off_slice[i] as usize;
                    let end = off_slice[i + 1] as usize;
                    if start > end {
                        return Err(PyValueError::new_err(format!(
                            "offsets[{i}]={start} > offsets[{}]={end}: must be non-decreasing",
                            i + 1
                        )));
                    }
                    views.push(DenseMultiVectorView::new(
                        &tokens_slice[start * dim..end * dim],
                        dim,
                    ));
                }
                views
            }
        };

        let results: Vec<Vec<(f32, u32)>> = py.allow_threads(|| {
            self.inner.batch_search(
                &views,
                k,
                k_centroids,
                k_docs_to_score,
                ef_search,
                alpha,
                beta,
                lambda_,
                num_threads,
            )
        });

        let (scores_arr, doc_ids_arr) = pad_results_batch(results, n_queries, k);
        Ok((
            scores_arr.into_pyarray(py).unbind(),
            doc_ids_arr.into_pyarray(py).unbind(),
        ))
    }

    // ── Inspection ───────────────────────────────────────────────────────────

    /// Number of indexed documents.
    #[getter]
    fn len(&self) -> usize {
        self.inner.n_elements()
    }

    /// Token vector dimensionality (before quantization).
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Total number of tokens across all documents.
    #[getter]
    fn n_tokens(&self) -> usize {
        let dim = self.inner.residuals.encoder().output_dim();
        if dim == 0 {
            return 0;
        }
        self.inner
            .residuals
            .offsets()
            .last()
            .map(|&end| end / dim)
            .unwrap_or(0)
    }

    /// Number of coarse centroids in the IVF.
    #[getter]
    fn n_centroids(&self) -> usize {
        self.inner.centroids.n_elements()
    }

    /// Print a per-component size breakdown of the index.
    fn print_space_usage(&self) {
        let (ch, il, off, res) = self.inner.space_usage_components();
        let total = ch + il + off + res;
        let gb = |b: usize| b as f64 / 1_073_741_824.0;
        let pct = |b: usize| 100.0 * b as f64 / total as f64;
        println!("Index space usage:");
        println!(
            "  {:<20} {:6.2} GB  ({:5.1}%)",
            "centroids_hnsw",
            gb(ch),
            pct(ch)
        );
        println!(
            "  {:<20} {:6.2} GB  ({:5.1}%)",
            "inverted_lists",
            gb(il),
            pct(il)
        );
        println!(
            "  {:<20} {:6.2} GB  ({:5.1}%)",
            "offsets",
            gb(off),
            pct(off)
        );
        println!(
            "  {:<20} {:6.2} GB  ({:5.1}%)",
            "residuals",
            gb(res),
            pct(res)
        );
        println!("  {}", "─".repeat(38));
        println!("  {:<20} {:6.2} GB", "total", gb(total));
    }

    /// Reconstruct approximate token embeddings for a single document.
    ///
    /// Returns a 2D f32 array of shape `(n_tokens, dim)` obtained by decoding the
    /// stored PQ codes: `approx = coarse_centroid + norm * PQ_residual`.
    /// The result is approximate due to PQ lossy compression.
    ///
    /// Raises `ValueError` if `doc_id` is out of range.
    fn get_document_embeddings<'py>(
        &self,
        py: Python<'py>,
        doc_id: u32,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let n_docs = self.inner.residuals.len();
        if doc_id as usize >= n_docs {
            return Err(PyValueError::new_err(format!(
                "doc_id {doc_id} is out of range (index has {n_docs} documents)"
            )));
        }
        let encoded = self.inner.residuals.get(doc_id as u64);
        let decoded = self.inner.residuals.encoder().decode_vector(encoded);
        let n_tokens = decoded.num_vecs();
        let dim = decoded.dim();
        let array = ndarray::Array2::from_shape_vec((n_tokens, dim), decoded.values().to_vec())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array.into_pyarray(py).unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "<Tachiom: {} docs, dim={}, {} centroids>",
            self.len(),
            self.dim(),
            self.n_centroids()
        )
    }
}

// ============================================================================
// PyTac class
// ============================================================================

#[pyclass(name = "Tac", module = "tachiom", unsendable)]
pub struct PyTac {
    budget: usize,
    n_iter: usize,
    verbose: bool,
    max_sample_size: Option<usize>,
    result: Option<TacResult>,
}

impl PyTac {
    fn require_trained(&self) -> PyResult<&TacResult> {
        self.result.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Tac has not been trained yet; call train() first")
        })
    }
}

#[pymethods]
impl PyTac {
    #[new]
    #[pyo3(signature = (n_centroids, *, n_iter = 10, verbose = false, max_sample_size = None))]
    fn new(
        n_centroids: usize,
        n_iter: usize,
        verbose: bool,
        max_sample_size: Option<usize>,
    ) -> Self {
        PyTac {
            budget: n_centroids,
            n_iter,
            verbose,
            max_sample_size,
            result: None,
        }
    }

    /// Run Token-Aware Clustering on the given .npy inputs.
    ///
    /// May be called multiple times; each call overwrites the previous result.
    fn train(&mut self, py: Python<'_>, vectors_path: &str, token_ids_path: &str) -> PyResult<()> {
        let (flat_f16, dim) = read_f16_npy(vectors_path)?;
        let token_ids = read_token_ids_npy(token_ids_path)?;
        let n_tokens = flat_f16.len() / dim;
        if token_ids.len() != n_tokens {
            return Err(PyValueError::new_err(format!(
                "token_ids length ({}) != n_tokens ({})",
                token_ids.len(),
                n_tokens
            )));
        }

        let tac = TacBuilder::new()
            .n_iter(self.n_iter)
            .verbose(self.verbose)
            .max_sample_size(self.max_sample_size)
            .build();

        let budget = self.budget;
        let result = py.allow_threads(|| tac.train(&flat_f16, dim, &token_ids, budget));
        self.result = Some(result);
        Ok(())
    }

    // ── Properties (available after train()) ────────────────────────────────

    /// Coarse centroids as f32, shape `[n_centroids, dim]`.
    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f32>>> {
        let r = self.require_trained()?;
        let f32_vec: Vec<f32> = r.centroids.iter().map(|x| x.to_f32()).collect();
        let arr = ndarray::Array2::from_shape_vec((r.n_centroids, r.dim), f32_vec)
            .map_err(|e| PyRuntimeError::new_err(format!("centroids reshape: {e}")))?;
        Ok(arr.into_pyarray(py).unbind())
    }

    /// Coarse centroids as f16 (raw), shape `[n_centroids, dim]`.
    #[getter]
    fn centroids_f16<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray2<f16>>> {
        let r = self.require_trained()?;
        let arr = ndarray::Array2::from_shape_vec((r.n_centroids, r.dim), r.centroids.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("centroids_f16 reshape: {e}")))?;
        Ok(arr.into_pyarray(py).unbind())
    }

    /// Per-token centroid assignment, shape `[n_tokens]`.
    #[getter]
    fn assignments<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<u32>>> {
        let r = self.require_trained()?;
        Ok(r.assignments.clone().into_pyarray(py).unbind())
    }

    /// Actual number of centroids produced (equals the budget after reconciliation).
    #[getter]
    fn n_centroids(&self) -> PyResult<usize> {
        Ok(self.require_trained()?.n_centroids)
    }

    /// Token-vector dimensionality.
    #[getter]
    fn dim(&self) -> PyResult<usize> {
        Ok(self.require_trained()?.dim)
    }

    fn __repr__(&self) -> String {
        match &self.result {
            Some(r) => format!("<Tac: n_centroids={}, dim={}>", r.n_centroids, r.dim),
            None => format!("<Tac: budget={}, not yet trained>", self.budget),
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn warn_pq_subspaces(py: Python<'_>, pq_subspaces: usize) -> PyResult<()> {
    if pq_subspaces != M_FIXED {
        let msg = format!(
            "pq_subspaces={pq_subspaces} requested but only M={M_FIXED} is supported in this build; \
             proceeding with M={M_FIXED}.  Results will reflect M={M_FIXED}, not M={pq_subspaces}."
        );
        let warnings = py.import("warnings")?;
        warnings.call_method1("warn", (msg,))?;
    }
    Ok(())
}

fn require_contiguous_2d<'py, 'a>(
    arr: &'a PyReadonlyArray2<'py, f32>,
    expected_dim: usize,
    arg_name: &str,
) -> PyResult<&'a [f32]>
where
    'py: 'a,
{
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != expected_dim {
        return Err(PyValueError::new_err(format!(
            "{arg_name} must have shape (n_tokens, {expected_dim}); got {shape:?}"
        )));
    }
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{arg_name} must be C-contiguous; call np.ascontiguousarray({arg_name}) first"
        )));
    }
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{arg_name} could not be exposed as a slice")))
}

/// Pad a single-query result vector to length `k` with sentinels.
fn pad_result(result: Vec<(f32, u32)>, k: usize) -> (Vec<f32>, Vec<u32>) {
    let mut scores = Vec::with_capacity(k);
    let mut doc_ids = Vec::with_capacity(k);
    for (s, d) in result.iter().take(k) {
        scores.push(*s);
        doc_ids.push(*d);
    }
    while scores.len() < k {
        scores.push(f32::NEG_INFINITY);
        doc_ids.push(u32::MAX);
    }
    (scores, doc_ids)
}

/// Pad a batch result into rectangular `(n_queries, k)` ndarrays.
fn pad_results_batch(
    results: Vec<Vec<(f32, u32)>>,
    n_queries: usize,
    k: usize,
) -> (ndarray::Array2<f32>, ndarray::Array2<u32>) {
    let mut scores = ndarray::Array2::<f32>::from_elem((n_queries, k), f32::NEG_INFINITY);
    let mut doc_ids = ndarray::Array2::<u32>::from_elem((n_queries, k), u32::MAX);
    for (i, row) in results.into_iter().enumerate() {
        for (j, (s, d)) in row.into_iter().take(k).enumerate() {
            scores[(i, j)] = s;
            doc_ids[(i, j)] = d;
        }
    }
    (scores, doc_ids)
}

// ============================================================================
// Array-based dataset construction
// ============================================================================

/// Build a `TachiomInputDataset` directly from numpy array slices.
///
/// `vectors` is a `u16` array whose bit patterns are the raw IEEE-754 f16 values —
/// the same reinterpretation that `read_f16_npy` performs when reading from disk.
/// Works transparently with memory-mapped numpy arrays.
fn dataset_from_arrays(
    vectors: &PyReadonlyArray2<'_, u16>,
    token_ids_arr: &PyReadonlyArray1<'_, u32>,
    doclens_arr: &PyReadonlyArray1<'_, i32>,
) -> PyResult<(TachiomInputDataset, Vec<usize>)> {
    if !vectors.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "vectors must be C-contiguous; call np.ascontiguousarray(vectors) first",
        ));
    }
    let shape = vectors.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("vectors must be a 2D array [N, dim]"));
    }
    let (n_tokens, dim) = (shape[0], shape[1]);
    let flat_f16: Vec<f16> = vectors
        .as_slice()
        .map_err(|_| PyValueError::new_err("vectors could not be exposed as a slice"))?
        .iter()
        .map(|&bits| f16::from_bits(bits))
        .collect();

    if !token_ids_arr.is_c_contiguous() {
        return Err(PyValueError::new_err("token_ids must be C-contiguous"));
    }
    let token_ids_slice = token_ids_arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("token_ids could not be exposed as a slice"))?;
    if token_ids_slice.len() != n_tokens {
        return Err(PyValueError::new_err(format!(
            "token_ids length ({}) != n_tokens ({})",
            token_ids_slice.len(),
            n_tokens
        )));
    }
    let token_ids: Vec<usize> = token_ids_slice.iter().map(|&x| x as usize).collect();

    if !doclens_arr.is_c_contiguous() {
        return Err(PyValueError::new_err("doclens must be C-contiguous"));
    }
    let doclens_slice = doclens_arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("doclens could not be exposed as a slice"))?;
    for &d in doclens_slice {
        if d < 0 {
            return Err(PyValueError::new_err("doclens must be non-negative"));
        }
    }
    let doclens: Vec<usize> = doclens_slice.iter().map(|&x| x as usize).collect();
    let total: usize = doclens.iter().sum();
    if total != n_tokens {
        return Err(PyValueError::new_err(format!(
            "sum(doclens)={total} != n_tokens={n_tokens}"
        )));
    }

    let encoder = PlainMultiVecQuantizer::<f16>::new(dim);
    let mut offsets: Vec<usize> = Vec::with_capacity(doclens.len() + 1);
    offsets.push(0);
    for &n_tok in &doclens {
        offsets.push(offsets.last().unwrap() + n_tok * dim);
    }

    let dataset = MultiVectorDataset::from_raw(
        flat_f16.into_boxed_slice(),
        offsets.into_boxed_slice(),
        encoder,
    );
    Ok((dataset, token_ids))
}

// ============================================================================
// Input loading
// ============================================================================

/// Load vectors + token_ids + doclens, validate cross-consistency, build the
/// `TachiomInputDataset`, and return it together with the token_ids vector.
fn load_input_dataset(
    vectors_path: &str,
    token_ids_path: &str,
    doclens_path: &str,
) -> PyResult<(TachiomInputDataset, Vec<usize>)> {
    let (flat_f16, dim) = read_f16_npy(vectors_path)?;
    let n_tokens = flat_f16.len() / dim;

    let token_ids = read_token_ids_npy(token_ids_path)?;
    if token_ids.len() != n_tokens {
        return Err(PyValueError::new_err(format!(
            "token_ids length ({}) != n_tokens ({})",
            token_ids.len(),
            n_tokens
        )));
    }

    let doclens = read_doclens_npy(doclens_path)?;
    let total: usize = doclens.iter().sum();
    if total != n_tokens {
        return Err(PyValueError::new_err(format!(
            "sum(doclens)={total} != n_tokens={n_tokens}"
        )));
    }

    let encoder = PlainMultiVecQuantizer::<f16>::new(dim);
    let mut offsets: Vec<usize> = Vec::with_capacity(doclens.len() + 1);
    offsets.push(0);
    for &n_tok in &doclens {
        offsets.push(offsets.last().unwrap() + n_tok * dim);
    }
    if *offsets.last().unwrap() != flat_f16.len() {
        return Err(PyValueError::new_err(format!(
            "sum(doclens)*dim={} != flat_f16.len()={}",
            offsets.last().unwrap(),
            flat_f16.len()
        )));
    }
    let dataset: TachiomInputDataset = MultiVectorDataset::from_raw(
        flat_f16.into_boxed_slice(),
        offsets.into_boxed_slice(),
        encoder,
    );
    Ok((dataset, token_ids))
}

// ── .npy parsing (mirrors the CLI binaries) ────────────────────────────────

fn read_f16_npy(path: &str) -> PyResult<(Vec<f16>, usize)> {
    let mut reader = open_buf(path)?;
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err("Expected 2D array for token vectors"));
    }
    if elem_size != 2 {
        return Err(PyValueError::new_err("Expected f16 (2-byte) dtype"));
    }
    let (n_vecs, dim) = (shape[0], shape[1]);
    let mut raw = vec![0u8; n_vecs * dim * 2];
    reader
        .read_exact(&mut raw)
        .map_err(|e| PyIOError::new_err(format!("read_f16_npy: {e}")))?;
    let data = raw
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok((data, dim))
}

fn read_token_ids_npy(path: &str) -> PyResult<Vec<usize>> {
    let mut reader = open_buf(path)?;
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err("Expected 1D token-ID array"));
    }
    if elem_size != 4 && elem_size != 8 {
        return Err(PyValueError::new_err(format!(
            "Unsupported token-ID elem size: {elem_size}"
        )));
    }
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader
        .read_exact(&mut raw)
        .map_err(|e| PyIOError::new_err(format!("read_token_ids_npy: {e}")))?;
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

fn read_doclens_npy(path: &str) -> PyResult<Vec<usize>> {
    let mut reader = open_buf(path)?;
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err("Expected 1D doclens array"));
    }
    if elem_size != 4 && elem_size != 8 {
        return Err(PyValueError::new_err(format!(
            "Unsupported doclens elem size: {elem_size}"
        )));
    }
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader
        .read_exact(&mut raw)
        .map_err(|e| PyIOError::new_err(format!("read_doclens_npy: {e}")))?;
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

fn read_assignments_npy(path: &str, expected_len: usize) -> PyResult<Vec<usize>> {
    let mut reader = open_buf(path)?;
    let (shape, elem_size, _) = parse_npy_header(&mut reader)?;
    if shape.len() != 1 {
        return Err(PyValueError::new_err("assignments must be 1D"));
    }
    if shape[0] != expected_len {
        return Err(PyValueError::new_err(format!(
            "assignments length {} != n_tokens {}",
            shape[0], expected_len
        )));
    }
    let n = shape[0];
    let mut raw = vec![0u8; n * elem_size];
    reader
        .read_exact(&mut raw)
        .map_err(|e| PyIOError::new_err(format!("read_assignments_npy: {e}")))?;
    Ok(match elem_size {
        8 => raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        4 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
            .collect(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported assignments dtype (elem_size={elem_size})"
            )));
        }
    })
}

fn read_f32_2d_npy(path: &str) -> PyResult<(Vec<f32>, usize, usize)> {
    use ndarray::Array2;
    use ndarray_npy::ReadNpyExt;
    let arr: Array2<f32> = Array2::read_npy(open_buf(path)?)
        .map_err(|e| PyIOError::new_err(format!("read_f32_2d_npy: {e}")))?;
    let (rows, cols) = arr.dim();
    Ok((arr.into_raw_vec_and_offset().0, rows, cols))
}

fn open_buf(path: &str) -> PyResult<BufReader<File>> {
    let file = File::open(path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    Ok(BufReader::new(file))
}

fn parse_npy_header(reader: &mut impl Read) -> PyResult<(Vec<usize>, usize, usize)> {
    let mut n = 0usize;
    let mut magic = [0u8; 6];
    reader
        .read_exact(&mut magic)
        .map_err(|e| PyIOError::new_err(format!("npy header: {e}")))?;
    n += 6;
    if &magic != b"\x93NUMPY" {
        return Err(PyValueError::new_err("Not a NumPy file"));
    }
    let mut ver = [0u8; 2];
    reader
        .read_exact(&mut ver)
        .map_err(|e| PyIOError::new_err(format!("npy version: {e}")))?;
    n += 2;
    let header_len: usize = if ver[0] == 1 {
        let mut hl = [0u8; 2];
        reader
            .read_exact(&mut hl)
            .map_err(|e| PyIOError::new_err(format!("npy header len: {e}")))?;
        n += 2;
        u16::from_le_bytes(hl) as usize
    } else {
        let mut hl = [0u8; 4];
        reader
            .read_exact(&mut hl)
            .map_err(|e| PyIOError::new_err(format!("npy header len: {e}")))?;
        n += 4;
        u32::from_le_bytes(hl) as usize
    };
    let mut hdr_bytes = vec![0u8; header_len];
    reader
        .read_exact(&mut hdr_bytes)
        .map_err(|e| PyIOError::new_err(format!("npy header body: {e}")))?;
    n += header_len;
    let hdr = String::from_utf8_lossy(&hdr_bytes);
    if hdr.contains("'fortran_order': True") {
        return Err(PyValueError::new_err("Fortran-order arrays not supported"));
    }
    let descr = {
        let prefix = "'descr': '";
        let start = hdr
            .find(prefix)
            .ok_or_else(|| PyValueError::new_err("'descr' not found in npy header"))?
            + prefix.len();
        let rest = &hdr[start..];
        let end = rest
            .find('\'')
            .ok_or_else(|| PyValueError::new_err("descr end quote not found"))?;
        rest[..end].to_string()
    };
    if descr.starts_with('>') {
        return Err(PyValueError::new_err("Big-endian dtype not supported"));
    }
    let elem_size = dtype_elem_size(descr.trim_start_matches(['<', '=', '|']))
        .ok_or_else(|| PyValueError::new_err(format!("Unrecognised dtype '{descr}'")))?;
    let tok = "'shape': (";
    let si = hdr
        .find(tok)
        .ok_or_else(|| PyValueError::new_err("'shape' not found in npy header"))?;
    let rest = &hdr[si + tok.len()..];
    let ei = rest
        .find(')')
        .ok_or_else(|| PyValueError::new_err("shape ')' not found"))?;
    let shape: Vec<usize> = rest[..ei]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| PyRuntimeError::new_err(format!("shape parse: {e}")))
        })
        .collect::<PyResult<_>>()?;
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

"""Type stubs for the tachiom Python bindings.

Tachiom is an IVF-PQ index for late-interaction multivector retrieval
(ColBERT-style: documents are sequences of token vectors, scored against
query token vectors via max-sim sum).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class Tac:
    """Token-Aware Clustering for multivector data.

    Runs a separate k-means per token type and distributes a total centroid
    budget proportionally to each group's size and internal spread.

    Usage::

        tac = Tac(n_centroids=65536, n_iter=10, verbose=True)
        tac.train("vectors.npy", "token_ids.npy")

        # Feed directly into Tachiom (no temp files needed)
        index = Tachiom.build_from_tac(
            "vectors.npy", "token_ids.npy", "doclens.npy",
            centroids_path=..., assignments_path=...,
        )
    """

    def __init__(
        self,
        n_centroids: int,
        *,
        n_iter: int = 10,
        verbose: bool = False,
        max_sample_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_centroids: Total centroid budget distributed across all token types.
            n_iter: K-means iterations per token group.
            verbose: Print progress output.
            max_sample_size: Cap on training vectors per token group.
                None (default) — auto formula: sample when a group exceeds 1M vectors.
                Positive int   — hard cap regardless of group size.
        """
        ...

    def train(self, vectors_path: str, token_ids_path: str) -> None:
        """Run Token-Aware Clustering on the given .npy inputs.

        May be called multiple times; each call overwrites the previous result.

        Args:
            vectors_path:   .npy file, [N, dim] f16 token vectors.
            token_ids_path: .npy file, [N] i64/u32 token-type ids.
        """
        ...

    @property
    def centroids(self) -> NDArray[np.float32]:
        """Coarse centroids as f32, shape [n_centroids, dim]. Available after train()."""
        ...

    @property
    def centroids_f16(self) -> NDArray[np.float16]:
        """Coarse centroids as f16 (raw), shape [n_centroids, dim]. Available after train()."""
        ...

    @property
    def assignments(self) -> NDArray[np.uint32]:
        """Per-token centroid assignment, shape [n_tokens]. Available after train()."""
        ...

    @property
    def n_centroids(self) -> int:
        """Actual number of centroids produced. Available after train()."""
        ...

    @property
    def dim(self) -> int:
        """Token-vector dimensionality. Available after train()."""
        ...

    def __repr__(self) -> str: ...


class Tachiom:
    """IVF-PQ index for late-interaction multivector retrieval."""

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        vectors_path: str,
        token_ids_path: str,
        doclens_path: str,
        *,
        total_centroids: int = 4_194_304,
        tac_n_iter: int = 10,
        pq_sample_size: int = 10_000_000,
        pq_n_iter: int = 10,
        normalize: bool = False,
        pq_seed: int = 42,
        hnsw_m: int = 32,
        ef_construction: int = 1500,
        pq_subspaces: int = 32,
    ) -> Tachiom:
        """Build an index from .npy inputs (full pipeline: TAC → PQ → HNSW).

        Args:
            vectors_path:  .npy file, [N, dim] f16 token vectors.
            token_ids_path: .npy file, [N] i64/u32 token-type ids.
            doclens_path:   .npy file, [n_docs] i32/i64 document lengths.
            total_centroids: TAC coarse-centroid budget.
            normalize: if True, residuals are L2-normalised before PQ encoding
                       (per-token norms are embedded in the encoded payload).
            pq_subspaces: number of PQ subspaces.  Only 32 is currently
                          supported; other values trigger a warning and the
                          build proceeds with M=32.
        """
        ...

    @classmethod
    def build_from_arrays(
        cls,
        vectors: NDArray[np.uint16],
        token_ids: NDArray[np.uint32],
        doclens: NDArray[np.int32],
        *,
        total_centroids: int = 4_194_304,
        tac_n_iter: int = 10,
        pq_sample_size: int = 10_000_000,
        pq_n_iter: int = 10,
        normalize: bool = False,
        pq_seed: int = 42,
        hnsw_m: int = 32,
        ef_construction: int = 1500,
        pq_subspaces: int = 32,
    ) -> Tachiom:
        """Build an index from in-memory numpy arrays (full pipeline: TAC → PQ → HNSW).

        Equivalent to build() but accepts numpy arrays instead of file paths.
        Supports memory-mapped arrays (np.load(..., mmap_mode='r')) to minimise RAM
        usage — data is read from the buffer with a single copy into the index.

        Args:
            vectors:   [N, dim] uint16, C-contiguous.  The u16 bit patterns are
                       reinterpreted as IEEE-754 f16 values, matching the on-disk
                       format written by the indexing pipeline.
            token_ids: [N] u32, vocabulary id per token.
            doclens:   [n_docs] i32, tokens per document.
        """
        ...

    @classmethod
    def build_from_tac(
        cls,
        vectors_path: str,
        token_ids_path: str,
        doclens_path: str,
        centroids_path: str,
        assignments_path: str,
        *,
        pq_sample_size: int = 10_000_000,
        pq_n_iter: int = 10,
        normalize: bool = False,
        pq_seed: int = 42,
        hnsw_m: int = 32,
        ef_construction: int = 1500,
        pq_subspaces: int = 32,
    ) -> Tachiom:
        """Build an index using pre-computed coarse centroids and assignments.

        Skips Token-Aware Clustering and runs PQ training + encoding from
        scratch.  Useful for isolating retrieval differences between the
        clustering step and the residual/PQ encoding step.

        Args:
            centroids_path:   .npy file, [K, dim] f32 coarse centroids.
            assignments_path: .npy file, [N] u32/u64 centroid id per token.
        """
        ...

    @classmethod
    def load(cls, path: str) -> Tachiom:
        """Load a previously-saved index from disk."""
        ...

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the index to disk."""
        ...

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        *,
        k_centroids: int = 20,
        k_docs_to_score: int = 500,
        ef_search: int = 30,
        alpha: Optional[float] = 0.45,
        beta: Optional[int] = None,
        lambda_: Optional[float] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
        """Search a single multivector query.

        Args:
            query: 2D C-contiguous f32 array of shape (n_tokens, dim).
            k: number of results to return.

        Returns:
            (scores, doc_ids) — both 1D ndarrays of length k.  When fewer than
            k results are produced (e.g. beta-pruning), trailing positions are
            sentinel-padded: scores = -inf, doc_ids = u32::MAX.
        """
        ...

    def batch_search(
        self,
        tokens: NDArray[np.float32],
        n_queries: int,
        k: int = 10,
        *,
        offsets: Optional[NDArray[np.uint64]] = None,
        num_threads: int = 0,
        k_centroids: int = 20,
        k_docs_to_score: int = 500,
        ef_search: int = 30,
        alpha: Optional[float] = 0.45,
        beta: Optional[int] = None,
        lambda_: Optional[float] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
        """Search a batch of multivector queries.

        tokens is a flat [total_tokens, dim] f32 C-contiguous array with all query
        token vectors concatenated in query order.

        Uniform mode (offsets=None): all queries have the same token count.
            total_tokens must be divisible by n_queries.

        Ragged mode (offsets provided): [n_queries + 1] u64 boundary array.
            offsets[i]:offsets[i+1] is the row range in tokens for query i.
            n_queries is validated against len(offsets) - 1.

        Args:
            tokens:    [total_tokens, dim] f32, C-contiguous.
            n_queries: number of queries (always required).
            offsets:   [n_queries + 1] u64, or None for uniform token count.
            num_threads:
                0 — rayon's default thread pool (typically all available cores).
                1 — serial loop (reproducible single-thread benchmarks).
                n — temporary rayon pool of size n for this call.

        Returns:
            (scores, doc_ids) — both 2D ndarrays of shape (n_queries, k),
            sentinel-padded when fewer than k results are produced for a given query.
        """
        ...

    # ── Inspection ───────────────────────────────────────────────────────────

    @property
    def len(self) -> int:
        """Number of indexed documents."""
        ...

    @property
    def dim(self) -> int:
        """Token-vector dimensionality (before quantization)."""
        ...

    @property
    def n_tokens(self) -> int:
        """Total number of tokens across all documents."""
        ...

    @property
    def n_centroids(self) -> int:
        """Number of coarse centroids in the IVF."""
        ...

    def print_space_usage(self) -> None:
        """Print a per-component size breakdown of the index in GB with percentages."""
        ...

    def __repr__(self) -> str: ...

"""Type stubs for the tachiom Python bindings.

Tachiom is an IVF-PQ index for late-interaction multivector retrieval
(ColBERT-style: documents are sequences of token vectors, scored against
query token vectors via max-sim sum).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


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
        queries: NDArray[np.float32],
        k: int = 10,
        *,
        num_threads: int = 0,
        k_centroids: int = 20,
        k_docs_to_score: int = 500,
        ef_search: int = 30,
        alpha: Optional[float] = 0.45,
        beta: Optional[int] = None,
        lambda_: Optional[float] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
        """Search a batch of multivector queries.

        Args:
            queries: 3D C-contiguous f32 array of shape (n_queries, n_tokens, dim).
            num_threads:
                0 — rayon's default thread pool (typically all available cores).
                1 — serial loop (mirrors the CLI; reproducible single-thread benchmarks).
                n — temporary rayon pool of size n for this call.

        Returns:
            (scores, doc_ids) — both 2D ndarrays of shape (n_queries, k),
            sentinel-padded when fewer than k results are produced for a
            given query.
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

    def print_space_usage_bytes(self) -> None:
        """Print a per-component byte breakdown of the index."""
        ...

    def __repr__(self) -> str: ...

# Python API

## Installation

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --features python --release
```

```python
import tachiom  # exposes: tachiom.Tachiom, tachiom.Tac
```

---

## Input format

All `.npy` inputs use C-contiguous (row-major) layout.

| File | Shape | Dtype | Description |
|---|---|---|---|
| `vectors.npy` | `[N, dim]` | `f16` | One row per token across all documents |
| `token_ids.npy` | `[N]` | `i64` or `u32` | Vocabulary id of each token |
| `doclens.npy` | `[n_docs]` | `i32` or `i64` | Number of tokens per document |

Tokens must be concatenated in document order: the first `doclens[0]` rows in `vectors.npy` belong to document 0, the next `doclens[1]` to document 1, and so on.

---

## `Tachiom` — IVF-PQ index

### Building

#### Full pipeline (TAC + PQ + HNSW)

```python
index = tachiom.Tachiom.build(
    vectors_path,
    token_ids_path,
    doclens_path,
    total_centroids=4_194_304,   # coarse centroid budget
    tac_n_iter=10,               # k-means iterations inside TAC
    pq_sample_size=10_000_000,   # training vectors for the PQ encoder
    pq_n_iter=10,                # PQ k-means iterations
    normalize=False,             # L2-normalise residuals before PQ encoding
    pq_seed=42,
    hnsw_m=32,                   # HNSW neighbour count
    ef_construction=1500,        # HNSW build-time beam width
    pq_subspaces=32,             # PQ subspace count (only 32 supported)
)
```

#### From pre-computed TAC output

If you have already run TAC (e.g. to inspect centroids or tune the centroid budget separately), skip the clustering step:

```python
index = tachiom.Tachiom.build_from_tac(
    vectors_path,
    token_ids_path,
    doclens_path,
    centroids_path,    # [K, dim] f32 .npy
    assignments_path,  # [N]      u32 .npy
    pq_sample_size=10_000_000,
    pq_n_iter=10,
    normalize=False,
    pq_seed=42,
    hnsw_m=32,
    ef_construction=1500,
    pq_subspaces=32,
)
```

### Saving and loading

```python
index.save("index.bin")
index = tachiom.Tachiom.load("index.bin")
```

### Searching

#### Single query

```python
# query: [n_tokens, dim] f32 C-contiguous array
scores, doc_ids = index.search(
    query,
    k=10,
    k_centroids=20,       # coarse centroids retrieved per query token
    k_docs_to_score=500,  # candidates passed to PQ reranking
    ef_search=30,         # HNSW beam width during coarse scoring
    alpha=0.45,           # fraction of k-th coarse score used as pruning threshold
    beta=None,            # stop PQ reranking after this many candidates scored
    lambda_=None,         # distance-adaptive HNSW early-exit factor
)
# scores:   [k] f32   (−∞ sentinel for unfilled positions)
# doc_ids:  [k] u32   (u32::MAX sentinel for unfilled positions)
```

#### Batch search

```python
# queries: [n_queries, n_tokens, dim] f32 C-contiguous array
scores, doc_ids = index.batch_search(
    queries,
    k=10,
    num_threads=0,        # 0 = all cores, 1 = serial, n = custom pool
    k_centroids=20,
    k_docs_to_score=500,
    ef_search=30,
    alpha=0.45,
    beta=None,
    lambda_=None,
)
# scores:   [n_queries, k] f32
# doc_ids:  [n_queries, k] u32
```

#### Search parameters

Search runs in two phases: **Gather** (HNSW traversal over TAC centroids) then **Refine** (PQ reranking of surviving candidates).

| Parameter | Default | Phase | Description |
|---|---|---|---|
| `k_centroids` | `20` | Gather | Coarse centroids retrieved per query token via HNSW. Higher values increase recall and latency. |
| `ef_search` | `30` | Gather | HNSW beam width. Increase together with `k_centroids` for deeper search. |
| `alpha` | `0.45` | Gather→Refine | After accumulating coarse scores, only documents scoring above `alpha × score_k` are forwarded to Refine. Lower values prune more aggressively. Set to `None` to disable. |
| `k_docs_to_score` | `500` | Refine | Maximum candidates passed to PQ reranking (cap applied after alpha-pruning). |
| `beta` | `None` | Refine | Early-exit threshold: stop PQ reranking after `beta` candidates have been scored. Set to `None` to score all `k_docs_to_score` candidates. |
| `lambda_` | `None` | Gather | Distance-adaptive HNSW termination factor. Set to `None` to disable. |

### Inspection

```python
index.len          # number of indexed documents
index.dim          # token-vector dimensionality
index.n_tokens     # total tokens across all documents
index.n_centroids  # number of coarse centroids
index.print_space_usage()  # per-component size in GB
```

---

## `Tac` — Token-Aware Clustering

`Tac` runs a separate k-means per token type and distributes a total centroid budget proportionally across groups.
Use it when you want to inspect or reuse the clustering step independently of the full index build.

### Training

```python
tac = tachiom.Tac(
    n_centroids=2_000_000,  # total centroid budget
    n_iter=10,              # k-means iterations per token group
    verbose=True,
    max_sample_size=None,   # None = auto (cap at ~1M per group)
)
tac.train("vectors.npy", "token_ids.npy")
```

### Inspecting results

```python
tac.n_centroids        # actual centroids produced (may be < budget)
tac.dim                # dimensionality
tac.centroids          # [K, dim] f32
tac.centroids_f16      # [K, dim] f16
tac.assignments        # [N]      u32 — centroid id for each token
```

### Saving and feeding into `Tachiom`

```python
import numpy as np

np.save("centroids.npy",   tac.centroids)
np.save("assignments.npy", tac.assignments)

index = tachiom.Tachiom.build_from_tac(
    "vectors.npy", "token_ids.npy", "doclens.npy",
    "centroids.npy", "assignments.npy",
)
```

---

## End-to-end example

See [notebooks/tachiom_demo.ipynb](../notebooks/tachiom_demo.ipynb) for a complete walkthrough on the LOTTE dataset (2.4 M documents, 266 M tokens, dim=128, 2 M centroids, ~12.8 GB index, ~0.45 ms/query).

See [notebooks/tac_demo.ipynb](../notebooks/tac_demo.ipynb) for TAC centroid budget analysis and saving TAC output for later reuse.

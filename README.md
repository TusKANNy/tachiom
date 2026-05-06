<p align="center">
  <h1 align="center">Tachiom</h1>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.28142"><img src="https://img.shields.io/badge/arXiv-2604.28142-b31b1b.svg" alt="arXiv"></a>
</p>

**Tachiom** is the state-of-the-art data structure for **late-interaction multi-vector retrieval**.
Documents are sequences of token vectors; queries are scored against them via max-sim sum.

Standard k-means clustering scales poorly to millions of token vectors and tends to over-allocate centroids to frequent tokens while marginalizing rare, discriminative ones.
Tachiom addresses this with **Token-Aware Clustering (TAC)**, which decomposes the global clustering into independent per-token subproblems and distributes the centroid budget proportionally to each token type's frequency and semantic variance.
TAC is up to **247× faster** than Faiss k-means and produces coarse centroids that improve retrieval quality.

At search time, Tachiom uses a two-phase pipeline:
1. **Gather** — HNSW traversal over the TAC centroids to accumulate and alpha-prune per-document coarse scores.
2. **Refine** — cache-optimized PQ reranking of the surviving candidates via a hierarchical distance-table layout.

Tachiom achieves up to **5.5× faster retrieval** than EMVB at comparable effectiveness on MS MARCO-v1 and LoTTE.

## Installation

### Python

Tachiom is a Rust library with Python bindings built via [maturin](https://github.com/PyO3/maturin).

#### Prerequisites

SSH access to the private `vectorium` and `kannolo` dependencies is required.
Make sure your SSH key is registered on GitHub before building.

Install Rust via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

#### Build from source

1. Clone the repository:

```bash
git clone git@github.com:TusKANNy/tachiom.git
cd tachiom
```

2. Create a virtual environment (recommended):

```bash
python3 -m venv ./venv
source ./venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or with conda:

```bash
conda create -n tachiom python=3.11
conda activate tachiom
```

3. Install maturin:

```bash
pip install maturin
```

4. Build and install in editable mode:

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --features python --release
```

The `target-cpu=native` flag enables SIMD instructions optimized for your CPU and is strongly recommended for performance.

5. Verify the installation:

```bash
python -c "import tachiom; print('Successfully installed tachiom!')"
```

### Rust

To compile all the Rust binaries in `src/bin/`:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Details on how to use Tachiom's Rust CLI can be found in [docs/RustUsage.md](docs/RustUsage.md).

## Quick start

```python
import numpy as np
import tachiom

# ── Build ─────────────────────────────────────────────────────────────────────
# Inputs (all .npy files):
#   vectors.npy    — [N, dim]   f16  one row per token
#   token_ids.npy  — [N]        i64  vocabulary id of each token
#   doclens.npy    — [n_docs]   i32  number of tokens per document

index = tachiom.Tachiom.build(
    "vectors.npy",
    "token_ids.npy",
    "doclens.npy",
    total_centroids=2_000_000,
)
index.save("my_index.bin")

# ── Load & search ─────────────────────────────────────────────────────────────
index = tachiom.Tachiom.load("my_index.bin")

# queries: [n_queries, n_tokens, dim] f32 array
scores, doc_ids = index.batch_search(queries, k=10, num_threads=0)
# scores, doc_ids: [n_queries, k]
```

See [docs/PythonUsage.md](docs/PythonUsage.md) for the full API, all build and search parameters, and the two-step TAC workflow.

## Resources

| Document | Description |
|---|---|
| [Python API](docs/PythonUsage.md) | `Tachiom` and `Tac` classes, all parameters, search guide |
| [Rust CLI](docs/RustUsage.md) | `bench_tac`, `tachiom_build`, `tachiom_search` binaries, experiment runner, SIGIR 2026 reproduction |
| [Jupyter notebooks](notebooks/) | End-to-end demo on LoTTE; TAC inspection and budget analysis |
| [Experiments](experiments/sigir2026/) | TOML configs used for the SIGIR 2026 benchmarks |

## Bibliography

This paper has been accepted at **SIGIR 2026**. The full proceedings entry will be available after the conference.

```bibtex
@misc{martinico2026efficientmultivectorretrievaltokenaware,
      title={Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing}, 
      author={Silvio Martinico and Franco Maria Nardini and Cosimo Rulli and Rossano Venturini},
      year={2026},
      eprint={2604.28142},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2604.28142}, 
}
```

<h1 align="center">Tachiom</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.28142"><img src="https://badgen.net/static/arXiv/2604.28142/red" /></a>
</p>

Tachiom is a fast and scalable data structure for late-interaction multi-vector retrieval, written in Rust with Python bindings. It introduces **Token-Aware Clustering (TAC)**, which distributes the coarse-centroid budget proportionally across token types, and a hierarchical Product Quantization scheme for efficient candidate reranking.

## Installation

### Python

Tachiom is a Rust library with Python bindings built via [maturin](https://github.com/PyO3/maturin).

#### Prerequisites

Install Rust via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Activate the nightly toolchain (required):

```bash
rustup install nightly
rustup default nightly
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
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

The `target-cpu=native` flag enables SIMD instructions optimized for your CPU and is strongly recommended for performance.


### Rust

To compile all the Rust binaries in `src/bin/`:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Details on how to use Tachiom's Rust CLI can be found in [docs/RustUsage.md](docs/RustUsage.md).

## Quick start

```python
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
    total_centroids=2_097_152,
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
| [Jupyter notebooks](notebooks/) | End-to-end demo on TAC and TACHIOM |
| [Experiments](experiments/sigir2026/) | TOML configs used for the SIGIR 2026 benchmarks |

## License

This software is released under the **MIT License** (see [LICENSE](LICENSE)).

### Citation license

By downloading and using this software, you agree to cite the following paper in any material you produce where it was used to conduct a search or experimentation, whether it be a research paper, dissertation, article, poster, presentation, or documentation. By using this software, you have agreed to the citation license.

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

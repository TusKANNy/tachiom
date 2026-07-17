<h1 align="center">TACHIOM</h1>
<p align="center">
    <img width="250px" src="/imgs/tachiom_logo.png" />
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.28142"><img src="https://badgen.net/static/arXiv/2604.28142/red" /></a>
</p>

TACHIOM is a fast and scalable data structure for late-interaction multi-vector retrieval, written in Rust with Python bindings. It introduces **Token-Aware Clustering (TAC)**, which distributes the coarse-centroid budget proportionally across token types, and a hierarchical Product Quantization scheme for efficient candidate reranking.

## Installation

### Python

#### Quick start (prebuilt wheels)

For most users, this is the easiest option:

```bash
pip install tachiom
```

If a compatible wheel exists for your platform, pip will download and install it directly without compilation. If no compatible wheel exists, pip will automatically compile from source.

This installs the core library with its only required dependency (`numpy`). If you also need the benchmarking / experiment scripts (`scripts/run_experiments.py`, analysis notebooks), install the optional extras:

```bash
pip install tachiom[scripts]
```

#### Building from source (maximum performance)

For maximum performance optimized to your CPU, build from source.

**Shared prerequisites** — both approaches below require Rust nightly:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install nightly
rustup default nightly
```

**Approach 1 — compile from PyPI source:**

```bash
RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: tachiom
```

**Approach 2 — build from GitHub (development/editable mode):**

```bash
git clone https://github.com/TusKANNy/tachiom.git
cd tachiom
```

Create a virtual environment (recommended):

```bash
python3 -m venv ./venv
source ./venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or with conda:

```bash
conda create -n tachiom python=3.11
conda activate tachiom
```

Install maturin and build:

```bash
pip install maturin
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

Changes to Python code take effect immediately without reinstalling — ideal for development.

### Rust

The crate has two feature flags:

| Feature | What it enables |
|---|---|
| `python` | PyO3 bindings — used automatically by maturin |
| `cli` | CLI binaries in `src/bin/` (`tachiom_build`, `tachiom_search`, `bench_tac`, …) |

Neither feature is active by default, so a plain `cargo build --release` compiles only the library crate. To build the CLI binaries, enable the `cli` feature:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
```

The resulting binaries are placed in `target/release/`.

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
)
index.save("my_index.bin")

# ── Load & search ─────────────────────────────────────────────────────────────
index = tachiom.Tachiom.load("my_index.bin")

# queries: [n_queries, n_tokens, dim] f32 array
scores, doc_ids = index.batch_search(queries, k=10, num_threads=0)
# scores, doc_ids: [n_queries, k]
```

See [docs/PythonUsage.md](docs/PythonUsage.md) for the full API, all build and search parameters, and the two-step TAC workflow.

## Datasets

Pre-processed datasets and pre-built indexes are available on HuggingFace, ready to use with the experiment configs in `experiments/sigir2026/`.

| Dataset | HuggingFace | Index |
|---|---|---|
| MS MARCO-v1 (ColBERT v2) | [tuskanny/ms_marco_colbertv2](https://huggingface.co/datasets/tuskanny/ms_marco_colbertv2) | `tachiom_msmarco_4M_normalized` |
| LoTTE Pooled (ColBERT v2) | [tuskanny/lotte_pooled_colbertv2](https://huggingface.co/datasets/tuskanny/lotte_pooled_colbertv2) | `tachiom_lotte_2M_normalized` |

Each dataset contains `documents.npy`, `token_ids.npy`, `doclens.npy`, `queries.npy`, `doc_ids.npy`, `queries_ids.npy`, a qrels `.tsv` file, and a pre-built Tachiom index. Download with:

```bash
pip install huggingface_hub
huggingface-cli download tuskanny/ms_marco_colbertv2 --repo-type dataset --local-dir ./ms_marco
huggingface-cli download tuskanny/lotte_pooled_colbertv2 --repo-type dataset --local-dir ./lotte
```

## Resources

| Document | Description |
|---|---|
| [Python API](docs/PythonUsage.md) | `Tachiom` and `Tac` classes, all parameters, search guide |
| [Rust CLI](docs/RustUsage.md) | `bench_tac`, `tachiom_build`, `tachiom_search` binaries, experiment runner, SIGIR 2026 reproduction |
| [Jupyter notebooks](notebooks/) | End-to-end demo on TAC and TACHIOM |
| [Experiments](experiments/sigir2026/) | TOML configs used for the SIGIR 2026 benchmarks |

## License

This software is released under the **MIT License** (see [LICENSE](LICENSE)).

## Citation

If you use this software in your research, please cite our **SIGIR 2026** paper:


```bibtex
@inproceedings{10.1145/3805712.3809927,
author = {Martinico, Silvio and Nardini, Franco Maria and Rulli, Cosimo and Venturini, Rossano},
title = {Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing},
year = {2026},
isbn = {9798400725999},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3805712.3809927},
doi = {10.1145/3805712.3809927},
booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {3982–3987},
numpages = {6},
keywords = {multivector retrieval, late interaction, clustering, efficiency},
location = {Australia},
series = {SIGIR '26}
}
```

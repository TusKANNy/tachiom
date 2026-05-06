# Rust Usage

After building with:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

three binaries are available under `./target/release/`.

---

## `bench_tac` — Token-Aware Clustering

Trains TAC on a multivector dataset and saves centroids and assignments to disk.
Use this when you want to inspect or reuse the clustering step independently of the full index build.

```
./target/release/bench_tac \
  -i <vectors.npy> \
  --token-ids-file <token_ids.npy> \
  --total-centroids <K> \
  -o <output_dir>
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `-i`, `--vectors-file` | *(required)* | `[N, dim]` f16 token vectors |
| `--token-ids-file` | *(required)* | `[N]` token-type IDs (i64 or u32) |
| `--total-centroids` | *(required)* | Total centroid budget distributed across all token types |
| `--tac-n-iter` | `10` | K-means iterations per token group |
| `-o`, `--output-dir` | *(required)* | Directory where outputs are written |
| `--verbose` | off | Print per-token allocation details and spread scores |
| `--test-mode` | off | Load only the first 100 000 vectors (quick sanity check) |

### Output

Three files are written to `<output_dir>`:

| File | Description |
|---|---|
| `centroids_it{n}_k{K}_n{N}.npy` | `[K, dim]` f32 coarse centroids |
| `assignments_it{n}_k{K}_n{N}.npy` | `[N]` u32 centroid id per token |
| `timing.txt` | TAC training time in seconds |

The filename stem encodes the run parameters for easy traceability.

### Example

```bash
./target/release/bench_tac \
  -i /data/vectors.npy \
  --token-ids-file /data/token_ids.npy \
  --total-centroids 2097152 \
  --tac-n-iter 10 \
  --verbose \
  -o /data/tac_output/
```

---

## `tachiom_build` — Index Construction

Runs the full pipeline (TAC → PQ training → encoding → HNSW) and saves the index to disk.

```
./target/release/tachiom_build \
  -i <vectors.npy> \
  --token-ids-file <token_ids.npy> \
  --doclens-file <doclens.npy> \
  -o <index_file>
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `-i`, `--vectors-file` | *(required)* | `[N, dim]` f16 token vectors |
| `--token-ids-file` | *(required)* | `[N]` token-type IDs (i64 or u32) |
| `--doclens-file` | *(required)* | `[n_docs]` document lengths (i32 or i64) |
| `-o`, `--output-file` | *(required)* | Output path for the serialized index |
| `--total-centroids` | `4194304` | TAC coarse-centroid budget |
| `--tac-n-iter` | `10` | K-means iterations per token group in TAC |
| `--pq-sample-size` | `10000000` | Tokens sampled for PQ training |
| `--pq-n-iter` | `10` | K-means iterations for PQ subspace training |
| `--normalize` | off | L2-normalise residuals before PQ encoding |
| `--pq-seed` | `42` | RNG seed for reproducible PQ training |
| `--hnsw-m` | `32` | HNSW neighbours per node in the centroid graph |
| `--ef-construction` | `1500` | HNSW build-time beam width |
| `--pq-subspaces` | `32` | PQ subspace count (only 32 is currently supported) |

### Example

```bash
./target/release/tachiom_build \
  -i /data/vectors.npy \
  --token-ids-file /data/token_ids.npy \
  --doclens-file /data/doclens.npy \
  -o /indexes/tachiom_index \
  --total-centroids 2097152 \
  --normalize
```

---

## `tachiom_search` — Querying

Loads a built index and runs a batch of multivector queries, printing average latency and optionally writing ranked results to disk.

```
./target/release/tachiom_search \
  -i <index_file> \
  -q <queries.npy> \
  -o <results.tsv>
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `-i`, `--index-file` | *(required)* | Path to the serialized Tachiom index |
| `-q`, `--query-file` | *(required)* | `[n_queries, n_tokens, dim]` f32 query vectors |
| `-o`, `--output-path` | *(optional)* | Output TSV file (omit to benchmark without saving) |
| `--k` | `10` | Results returned per query |
| `--k-centroids` | `4` | Coarse centroids probed per query token (Gather phase) |
| `--k-docs-to-score` | `1000` | Candidates forwarded to PQ reranking (Refine phase) |
| `--ef-search` | `64` | HNSW beam width during coarse retrieval |
| `--alpha` | *(off)* | Alpha-pruning threshold (fraction of k-th coarse score); omit to disable |
| `--beta` | *(off)* | Early-exit staleness counter for PQ reranking; omit to disable |
| `--lambda` | *(off)* | Distance-adaptive HNSW termination factor; omit to disable |
| `--num-runs` | `1` | Timing runs; results from the last run are saved |

### Output format

The TSV file has one line per result with columns: `query_id`, `doc_id`, `rank`, `score`.

### Example

```bash
./target/release/tachiom_search \
  -i /indexes/tachiom_index \
  -q /data/queries.npy \
  -o results.tsv \
  --k 10 \
  --k-centroids 25 \
  --k-docs-to-score 2000 \
  --ef-search 40 \
  --alpha 0.4
```

---

## Experiment runner

`scripts/run_experiments.py` drives `tachiom_build` and `tachiom_search` from a TOML configuration file, logging machine info, git state, build output, search results, and evaluation metrics into a timestamped experiment folder.

```bash
python scripts/run_experiments.py --exp experiments/sigir2026/<config>.toml
```

### TOML structure

```toml
name          = "my_experiment"
build-command = "./target/release/tachiom_build"
query-command = "./target/release/tachiom_search"

[settings]
k        = 10       # results per query
num-runs = 1        # timing repetitions
build    = true     # set to false to skip build and use a prebuilt index
metric   = "RR@10"  # any ir_measures metric string (e.g. "Success@5")
NUMA     = "numactl --physcpubind='0' --localalloc"  # optional NUMA pinning

[folder]
data       = "/path/to/dataset"
index      = "/path/to/index/dir"
experiment = "."           # experiment logs are written here
qrels_path = "/path/to/qrels.tsv"

[filename]
vectors   = "vectors.npy"      # [N, dim] f16
token_ids = "token_ids.npy"    # [N] i64 or u32
doclens   = "doclens.npy"      # [n_docs] i32 or i64
queries   = "queries.npy"      # [n_queries, n_tokens, dim] f32
index     = "my_index"         # index filename (no extension)
doc_ids   = "doc_ids.npy"      # optional: integer → string doc ID mapping
query_ids = "query_ids.npy"    # optional: integer → string query ID mapping

[build_params]
total-centroids = 4194304
tac-n-iter      = 10
pq-sample-size  = 10000000
pq-n-iter       = 10
normalize       = true
pq-seed         = 42
hnsw-m          = 32
ef-construction = 1500
pq-subspaces    = 32

# One subsection per search configuration
[query.my_config]
ef-search       = 30
k-centroids     = 20
k-docs-to-score = 4000
alpha           = 0.4
```

The runner produces a `report.tsv` inside the experiment folder with one row per `[query.*]` subsection, reporting query latency (μs), the chosen metric, memory usage, and build time.

---

## Reproducing SIGIR 2026 results

The TOML configs used in the paper are in `experiments/sigir2026/`.

**Prerequisites**: single-core execution pinned via `numactl`, CPU governor set to `performance`.

```bash
# Check governor (should print the number of available CPUs)
cpufreq-info | grep "performance" | grep -v "available" | wc -l
```

### MS MARCO-v1

```bash
python scripts/run_experiments.py \
  --exp experiments/sigir2026/ms-marco.toml
```

Key parameters:

| Parameter | Value |
|---|---|
| `total-centroids` | 4 194 304 (4M) |
| `normalize` | true |
| `ef-search` | 30 |
| `k-centroids` | 20 |
| `k-docs-to-score` | 4000 |
| `alpha` | 0.4 |
| Metric | MRR@10 |

### LoTTE (pooled)

```bash
python scripts/run_experiments.py \
  --exp experiments/sigir2026/lotte.toml
```

Key parameters:

| Parameter | Value |
|---|---|
| `total-centroids` | 2 097 152 (2M) |
| `normalize` | true |
| `ef-search` | 40 |
| `k-centroids` | 25 |
| `k-docs-to-score` | 2000 |
| `alpha` | 0.4 |
| Metric | Success@5 |

Update the `[folder]` and `[filename]` paths in each TOML to point to your local copy of the dataset and the prebuilt index before running.

#!/usr/bin/env python3
"""
Tachiom Experiment Runner

Drives tachiom_build and tachiom_search binaries.
Set build = true in [settings] to build the index first,
or build = false to load a pre-built index and go straight to search.
"""

import re
import os
import sys
import socket
import argparse
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import ir_measures
import toml
import psutil
from termcolor import colored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_map_indices(series, arr):
    """Map 0-based integer indices in a Series to values from an id array."""
    s_num = pd.to_numeric(series, errors='coerce')
    if s_num.isna().any():
        return None
    s_int = s_num.astype(int)
    if s_int.empty or s_int.min() < 0 or s_int.max() >= len(arr):
        return None
    return pd.Series(arr[s_int], index=series.index)


def parse_toml(filename):
    try:
        return toml.load(filename)
    except Exception as e:
        print(f"Error reading the TOML file: {e}")
        return None


# ---------------------------------------------------------------------------
# Machine / git / compile helpers
# ---------------------------------------------------------------------------

def get_git_info(experiment_dir):
    print()
    print(colored("Git info", "green"))
    git_output_file = os.path.join(experiment_dir, "git.output")
    try:
        with open(git_output_file, "w") as f:
            branch = subprocess.Popen("git rev-parse --abbrev-ref HEAD",
                                      shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            branch_name = branch.stdout.read().decode().strip()
            branch.wait()
            commit = subprocess.Popen("git rev-parse HEAD",
                                      shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            commit_id = commit.stdout.read().decode().strip()
            commit.wait()
            f.write(f"Branch: {branch_name}\nCommit: {commit_id}\n")
            print(f"Branch: {branch_name}")
            print(f"Commit: {commit_id}")
    except Exception as e:
        print("Error retrieving Git information:", e)
        sys.exit(1)


def compile_rust_code(configs, experiment_dir):
    print()
    print(colored("Compiling the Rust code", "green"))
    cmd = configs.get("compile-command",
                      "RUSTFLAGS='-C target-cpu=native' cargo build --release")
    out_file = os.path.join(experiment_dir, "compiler.output")
    print("Command:", cmd)
    try:
        with open(out_file, "w") as f:
            proc = subprocess.Popen(cmd, shell=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(proc.stdout.readline, b''):
                decoded = line.decode()
                print(decoded, end='')
                f.write(decoded)
            proc.stdout.close()
            proc.wait()
        if proc.returncode != 0:
            print(colored("Rust compilation failed.", "red"))
            sys.exit(1)
        print("Compilation successful.")
    except Exception as e:
        print(colored("ERROR during compilation:", "red"), e)
        sys.exit(1)


def get_machine_info(configs, experiment_folder):
    out_file = os.path.join(experiment_folder, "machine.output")
    date = datetime.now()
    machine = socket.gethostname()
    cpu_pct = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    load = str(psutil.getloadavg())[1:-1]
    num_cpus = psutil.cpu_count()

    print()
    print(colored("Hardware configuration", "green"))
    lines = [
        f"Date: {date}",
        f"Machine: {machine}",
        f"CPU usage (%): {cpu_pct}",
        f"Machine load: {load}",
        f"Memory free (GiB): {mem.free // (1024**3)}",
        f"Memory available (GiB): {mem.available // (1024**3)}",
        f"Memory total (GiB): {mem.total // (1024**3)}",
    ]
    for l in lines:
        print(l)

    with open(out_file, "w") as f:
        f.write("----------------------\nHardware configuration\n----------------------\n")
        for l in lines:
            f.write(l + "\n")

        f.write("\n---------------------\ncpufreq configuration\n---------------------\n")
        gov_proc = subprocess.Popen(
            'cpufreq-info | grep "performance" | grep -v "available" | wc -l',
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        gov_proc.wait()
        cpus_perf = 0
        for line in iter(gov_proc.stdout.readline, b''):
            cpus_perf = int(line.decode().strip() or 0)
        f.write(f'CPUs with "performance" governor: {cpus_perf} / {num_cpus}\n')
        if num_cpus != cpus_perf:
            print(colored("WARNING: Not all CPUs are in performance governor mode.", "yellow"))

        f.write("\n-----------------\nCPU configuration\n-----------------\n")
        lscpu = subprocess.Popen("lscpu", shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lscpu.wait()
        for line in iter(lscpu.stdout.readline, b''):
            f.write(line.decode())

        numa_cmd = configs.get("settings", {}).get("NUMA")
        if numa_cmd:
            f.write(f"\n------------------\nNUMA configuration\n------------------\n")
            f.write(f'Shell command: "{numa_cmd}"\n\n')
            numactl = subprocess.Popen("numactl --hardware", shell=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            numactl.wait()
            for line in iter(numactl.stdout.readline, b''):
                f.write(line.decode())

    print(f"Hardware log: {out_file}")


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(configs, experiment_dir):
    """Build a Tachiom index using the tachiom_build binary."""
    build_cmd = configs.get("build-command")
    if not build_cmd:
        raise ValueError("build-command must be specified in the config.")

    data_dir = configs["folder"]["data"]
    index_dir = configs["folder"]["index"]
    os.makedirs(index_dir, exist_ok=True)

    index_file    = os.path.join(index_dir,  configs["filename"]["index"])
    vectors_file  = os.path.join(data_dir,   configs["filename"]["vectors"])
    token_ids_file = os.path.join(data_dir,  configs["filename"]["token_ids"])
    doclens_file  = os.path.join(data_dir,   configs["filename"]["doclens"])

    print()
    print(colored("Indexing", "green"))
    print(colored("Vectors file:  ", "blue"), vectors_file)
    print(colored("Index output:  ", "blue"), index_file)

    bp = configs.get("build_params", {})
    parts = [
        build_cmd,
        f"--vectors-file {vectors_file}",
        f"--token-ids-file {token_ids_file}",
        f"--doclens-file {doclens_file}",
        f"--output-file {index_file}",
    ]
    for flag in ["total-centroids", "tac-n-iter", "tac-micro-threshold", "tac-small-threshold",
                 "pq-sample-size", "pq-n-iter", "pq-seed", "hnsw-m", "ef-construction", "pq-subspaces"]:
        if flag in bp:
            parts.append(f"--{flag} {bp[flag]}")
    if bp.get("normalize", False):
        parts.append("--normalize")

    command = " ".join(parts)
    print(colored("Command:", "blue"), command)

    out_file = os.path.join(experiment_dir, "building.output")
    building_time = 0.0
    print(colored("Building index...", "yellow"))
    with open(out_file, "w") as f:
        proc = subprocess.Popen(command, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode()
            print(decoded, end='')
            f.write(decoded)
            if "Build time:" in decoded:
                m = re.search(r"Build time:\s*([\d.]+)", decoded)
                if m:
                    building_time = float(m.group(1))
        proc.stdout.close()
        proc.wait()

    if proc.returncode != 0:
        print(colored("ERROR: Index build failed!", "red"))
        sys.exit(1)

    print(colored(f"Index built in {building_time:.1f}s", "yellow"))
    return building_time


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metric(configs, output_file, metric):
    if not metric:
        print("No metric specified. Skipping evaluation.")
        return None

    column_names = ["query_id", "doc_id", "rank", "score"]
    res_pd = pd.read_csv(output_file, sep='\t', names=column_names)

    # Map integer indices to string IDs if id arrays are provided
    data_dir = configs.get('folder', {}).get('data', '')
    file_cfg = configs.get('filename', {})

    q_ids_name = file_cfg.get('query_ids')
    if q_ids_name and data_dir:
        path = os.path.join(data_dir, q_ids_name)
        if os.path.exists(path):
            arr = np.load(path, allow_pickle=True)
            mapped = _try_map_indices(res_pd['query_id'], arr)
            if mapped is not None:
                res_pd['query_id'] = mapped

    d_ids_name = file_cfg.get('doc_ids')
    if d_ids_name and data_dir:
        path = os.path.join(data_dir, d_ids_name)
        if os.path.exists(path):
            arr = np.load(os.path.realpath(path), allow_pickle=True)
            mapped = _try_map_indices(res_pd['doc_id'], arr)
            if mapped is not None:
                res_pd['doc_id'] = mapped

    qrels_path = configs['folder']['qrels_path']
    df_qrels = pd.read_csv(qrels_path, sep="\t",
                           names=["query_id", "useless", "doc_id", "relevance"])
    if len(pd.unique(df_qrels['useless'])) != 1:
        df_qrels = pd.read_csv(qrels_path, sep="\t",
                               names=["query_id", "doc_id", "relevance", "useless"])

    res_pd['doc_id'] = res_pd['doc_id'].astype(df_qrels.doc_id.dtype)
    res_pd['query_id'] = res_pd['query_id'].astype(df_qrels.query_id.dtype)

    # pytrec_eval requires string keys
    df_qrels['query_id'] = df_qrels['query_id'].astype(str)
    df_qrels['doc_id'] = df_qrels['doc_id'].astype(str)
    res_pd['query_id'] = res_pd['query_id'].astype(str)
    res_pd['doc_id'] = res_pd['doc_id'].astype(str)

    ir_metric = ir_measures.parse_measure(metric)
    metric_val = ir_measures.calc_aggregate([ir_metric], df_qrels, res_pd)[ir_metric]
    print(f"{ir_metric}: {metric_val:.6f}")
    return metric_val


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def query_execution(configs, query_config, experiment_dir,
                    subsection_name, subsection_index=None, total_subsections=None):
    """Run tachiom_search for one query configuration."""
    query_cmd = configs.get("query-command")
    if not query_cmd:
        raise ValueError("query-command must be specified in the config.")

    index_file  = os.path.join(configs["folder"]["index"], configs["filename"]["index"])
    query_file  = os.path.join(configs["folder"]["data"],  configs["filename"]["queries"])
    output_file = os.path.join(experiment_dir, f"results_{subsection_name}")
    log_file    = os.path.join(experiment_dir, f"log_{subsection_name}")

    numa     = configs.get("settings", {}).get("NUMA", "")
    k        = configs["settings"]["k"]
    num_runs = configs["settings"].get("num-runs", 1)

    parts = [
        numa,
        query_cmd,
        f"--index-file {index_file}",
        f"--query-file {query_file}",
        f"--output-path {output_file}",
        f"--k {k}",
        f"--num-runs {num_runs}",
        f"--ef-search {query_config['ef-search']}",
        f"--k-centroids {query_config['k-centroids']}",
        f"--k-docs-to-score {query_config['k-docs-to-score']}",
    ]
    for flag in ["alpha", "beta", "lambda", "pq-subspaces"]:
        val = query_config.get(flag)
        if val is not None:
            parts.append(f"--{flag} {val}")

    command = " ".join(p for p in parts if p)

    label = (f"{subsection_name} ({subsection_index}/{total_subsections})"
             if subsection_index else subsection_name)
    print()
    print(colored(f"Search: {label}", "green"))
    print(colored("Command:", "blue"), command)

    memory_usage = 0
    query_time = 0
    with open(log_file, "w") as log:
        proc = subprocess.Popen(command, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode()
            if "[######] Average Query Time" in decoded:
                m = re.search(r"Average Query Time: (\d+)", decoded)
                if m:
                    query_time = int(m.group(1))
            m = re.search(r"Total: (\d+) bytes", decoded)
            if m:
                memory_usage = int(m.group(1))
            print(decoded, end='')
            log.write(decoded)
        proc.stdout.close()
        proc.wait()

    if proc.returncode != 0:
        print(colored(f"ERROR: Search for '{subsection_name}' failed!", "red"))
        sys.exit(1)

    metric_name = configs['settings'].get('metric', '')
    metric_val = compute_metric(configs, output_file, metric_name)
    return query_time, metric_val, memory_usage


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def run_experiment(config_data):
    experiment_name = config_data.get("name")
    print(f"Running experiment:", colored(experiment_name, "green"))

    for k, v in config_data["folder"].items():
        if isinstance(v, str) and v.startswith("~"):
            config_data["folder"][k] = os.path.expanduser(v)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_folder = os.path.join(
        config_data["folder"]["experiment"], f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_folder, exist_ok=True)

    with open(os.path.join(experiment_folder, "experiment_config.toml"), 'w') as f:
        f.write(toml.dumps(config_data))

    get_machine_info(config_data, experiment_folder)
    get_git_info(experiment_folder)
    compile_rust_code(config_data, experiment_folder)

    building_time = 0.0
    if config_data['settings']['build']:
        building_time = build_index(config_data, experiment_folder)
    else:
        index_path = os.path.join(config_data["folder"]["index"], config_data["filename"]["index"])
        print(f"\nUsing prebuilt index: {index_path}")

    metric_name = config_data['settings'].get('metric', '')
    print(f"\nEvaluation metric: {metric_name}")

    report_path = os.path.join(experiment_folder, "report.tsv")
    with open(report_path, 'w') as report:
        report.write(f"Subsection\tQuery Time (μs)\t{metric_name}\tMemory (bytes)\tBuild Time (s)\n")
        if 'query' in config_data:
            subsections = config_data['query']
            total = len(subsections)
            for i, (name, qcfg) in enumerate(subsections.items(), 1):
                query_time, metric_val, memory = query_execution(
                    config_data, qcfg, experiment_folder, name, i, total)
                metric_str = f"{metric_val:.6f}" if metric_val is not None else "N/A"
                report.write(f"{name}\t{query_time}\t{metric_str}\t{memory}\t{building_time}\n")

    print()
    print(colored(f"Results written to: {report_path}", "green"))


def main(experiment_config_filename):
    config_data = parse_toml(experiment_config_filename)
    if not config_data:
        print(colored("ERROR: Configuration data is empty.", "red"))
        sys.exit(1)
    run_experiment(config_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Tachiom experiment.")
    parser.add_argument("--exp", required=True,
                        help="Path to the experiment configuration TOML file.")
    args = parser.parse_args()
    main(args.exp)
    sys.exit(0)

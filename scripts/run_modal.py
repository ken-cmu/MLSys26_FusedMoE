"""
Run your Triton/CUDA kernel on Modal B200 and get speedup vs FlashInfer baseline.

All flashinfer_bench work happens inside the Modal container (Linux), so this
script is safe to invoke from macOS.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/mlsys26-contest/
    modal run scripts/fix_volume_layout.py   # if dataset is in a subdirectory

Usage:
    modal run scripts/run_modal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import modal

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu132:20260401-2c675fb",
        add_python="3.12",
    )
    .pip_install("flashinfer-bench")
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def pack_and_run(sources: dict[str, str], config: dict) -> dict:
    """
    Runs entirely on the Modal Linux container.
    1. Writes source files to a temp directory.
    2. Packs them into a Solution via flashinfer_bench.
    3. Benchmarks against all workloads and returns results.
    """
    import os
    import tempfile

    from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files

    sol_cfg   = config["solution"]
    build_cfg = config["build"]

    language   = build_cfg["language"]
    entry_point = build_cfg["entry_point"]
    dps        = build_cfg.get("destination_passing_style", True)
    definition = sol_cfg["definition"]

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in sources.items():
            dest = os.path.join(tmpdir, filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                f.write(content)

        spec = BuildSpec(
            language=language,
            target_hardware=["cuda"],
            entry_point=entry_point,
            destination_passing_style=dps,
        )
        solution = pack_solution_from_files(
            path=tmpdir,
            spec=spec,
            name=sol_cfg["name"],
            definition=definition,
            author=sol_cfg["author"],
        )

    print(f"Packed: {solution.name}  (lang={solution.spec.language}, dps={dps})")

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    workloads = trace_set.workloads.get(definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for '{definition}' in {TRACE_SET_PATH}")

    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={definition: trace_set.definitions[definition]},
        solutions={definition: [solution]},
        workloads={definition: workloads},
        traces={definition: []},
    )

    bench_cfg = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    result_ts = Benchmark(bench_ts, bench_cfg).run_all(dump_traces=True)

    results = {}
    for trace in result_ts.traces.get(definition, []):
        if not trace.evaluation:
            continue
        entry = {"status": trace.evaluation.status.value}
        if trace.evaluation.performance:
            p = trace.evaluation.performance
            entry["latency_ms"]           = p.latency_ms
            entry["reference_latency_ms"] = p.reference_latency_ms
            entry["speedup"]              = p.speedup_factor
        if trace.evaluation.correctness:
            c = trace.evaluation.correctness
            entry["max_abs_error"] = c.max_absolute_error
            entry["max_rel_error"] = c.max_relative_error
        results[str(trace.workload.uuid)] = entry

    return results


@app.local_entrypoint()
def main():
    """
    Runs locally on macOS. Only reads files and config — no flashinfer_bench import.
    """
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    language = config["build"]["language"]
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    elif language == "python":
        source_dir = PROJECT_ROOT / "solution" / "python"
    else:
        raise ValueError(f"Unknown language: {language}")

    sources = {f.name: f.read_text() for f in source_dir.glob("*.py")}
    if language == "cuda":
        sources.update({f.name: f.read_text() for f in source_dir.glob("*.cu")})
        sources.update({f.name: f.read_text() for f in source_dir.glob("*.cuh")})

    if not sources:
        raise FileNotFoundError(f"No source files found in {source_dir}")

    print(f"Sending {len(sources)} file(s) to Modal: {list(sources.keys())}")
    print("Running on Modal B200 ...")

    results = pack_and_run.remote(sources, config)

    if not results:
        print("No results returned.")
        return

    print(f"\n{'Workload':<12} {'Status':<14} {'Latency (ms)':<16} {'Speedup':<12} {'abs_err'}")
    print("-" * 72)

    latencies, speedups = [], []
    for uuid, r in sorted(results.items()):
        status   = r.get("status", "?")
        lat      = r.get("latency_ms")
        speedup  = r.get("speedup")
        abs_err  = r.get("max_abs_error")

        lat_str     = f"{lat:.4f}"     if lat     is not None else "N/A"
        speedup_str = f"{speedup:.3f}x" if speedup is not None else "N/A"
        err_str     = f"{abs_err:.2e}" if abs_err  is not None else "N/A"

        print(f"{uuid[:8]:<12} {status:<14} {lat_str:<16} {speedup_str:<12} {err_str}")

        if lat     is not None: latencies.append(lat)
        if speedup is not None: speedups.append(speedup)

    if latencies:
        import statistics
        print(f"\nSummary ({len(latencies)} workloads):")
        print(f"  Latency — min: {min(latencies):.4f} ms  max: {max(latencies):.4f} ms  median: {statistics.median(latencies):.4f} ms")
    if speedups:
        import statistics
        print(f"  Speedup — min: {min(speedups):.3f}x  max: {max(speedups):.3f}x  mean: {statistics.mean(speedups):.3f}x")

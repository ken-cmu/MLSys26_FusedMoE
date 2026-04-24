"""
Run the Python reference implementation on Modal B200.

All flashinfer_bench work happens inside the Modal container (Linux), so this
script is safe to invoke from macOS.  The local entrypoint only reads source
files from disk; no CUDA / flashinfer_bench import is needed locally.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/mlsys26-contest/

Usage:
    modal run scripts/run_baseline_modal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-python-baseline")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

DEFINITION = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

# Use the official evaluation image and install flashinfer-bench on top.
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
def pack_and_run(sources: dict[str, str]) -> dict:
    """
    Runs entirely on the Modal Linux container.
    1. Writes source files to a temp directory.
    2. Packs them into a Solution via flashinfer_bench.
    3. Benchmarks against all workloads in the dataset.
    """
    import os
    import tempfile

    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet
    from flashinfer_bench.agents import pack_solution_from_files
    from flashinfer_bench import BuildSpec

    # --- write sources to a temp dir so pack_solution_from_files can read them ---
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, content in sources.items():
            dest = os.path.join(tmpdir, filename)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                f.write(content)

        spec = BuildSpec(
            language="python",
            target_hardware=["cuda"],
            entry_point="main.py::run",
            destination_passing_style=False,
        )

        solution = pack_solution_from_files(
            path=tmpdir,
            spec=spec,
            name="python-reference-baseline",
            definition=DEFINITION,
            author="reference",
        )

    print(f"Packed solution: {solution.name}  (lang={solution.spec.language})")

    # --- load dataset and run benchmark ---
    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    workloads = trace_set.workloads.get(DEFINITION, [])
    if not workloads:
        raise ValueError(f"No workloads found for '{DEFINITION}' in {TRACE_SET_PATH}")

    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={DEFINITION: trace_set.definitions[DEFINITION]},
        solutions={DEFINITION: [solution]},
        workloads={DEFINITION: workloads},
        traces={DEFINITION: []},
    )

    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    result_ts = Benchmark(bench_ts, config).run_all(dump_traces=True)

    results = {}
    for trace in result_ts.traces.get(DEFINITION, []):
        if not trace.evaluation:
            continue
        entry = {"status": trace.evaluation.status.value}
        if trace.evaluation.performance:
            p = trace.evaluation.performance
            entry["latency_ms"] = p.latency_ms
            entry["reference_latency_ms"] = p.reference_latency_ms
            entry["speedup"] = p.speedup_factor
        if trace.evaluation.correctness:
            c = trace.evaluation.correctness
            entry["max_abs_error"] = c.max_absolute_error
            entry["max_rel_error"] = c.max_relative_error
        results[str(trace.workload.uuid)] = entry

    return results


@app.local_entrypoint()
def main():
    """
    Runs locally on macOS.  Only does plain file I/O — no flashinfer_bench import.
    """
    source_dir = PROJECT_ROOT / "solution" / "python-baseline"
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Python baseline not found at {source_dir}\n"
            "Expected: solution/python-baseline/main.py"
        )

    # Read all .py files; pass as {filename: content} to the remote function.
    sources = {
        f.name: f.read_text()
        for f in source_dir.glob("*.py")
    }
    if not sources:
        raise FileNotFoundError(f"No .py files found in {source_dir}")

    print(f"Sending {len(sources)} source file(s) to Modal: {list(sources.keys())}")
    print("Running on Modal B200 ...")

    results = pack_and_run.remote(sources)

    if not results:
        print("No results returned.")
        return

    print(f"\n{'Workload':<12} {'Status':<12} {'Latency (ms)':<16} {'vs FlashInfer':<16} {'abs_err'}")
    print("-" * 70)

    latencies, speedups = [], []

    for uuid, r in sorted(results.items()):
        status = r.get("status", "?")
        lat = r.get("latency_ms")
        speedup = r.get("speedup")
        abs_err = r.get("max_abs_error")

        lat_str = f"{lat:.4f}" if lat is not None else "N/A"
        speedup_str = f"{speedup:.3f}x" if speedup is not None else "N/A"
        err_str = f"{abs_err:.2e}" if abs_err is not None else "N/A"

        print(f"{uuid[:8]:<12} {status:<12} {lat_str:<16} {speedup_str:<16} {err_str}")

        if lat is not None:
            latencies.append(lat)
        if speedup is not None:
            speedups.append(speedup)

    if latencies:
        import statistics
        print(f"\nSummary ({len(latencies)} workloads):")
        print(f"  Latency — min: {min(latencies):.4f} ms  max: {max(latencies):.4f} ms  median: {statistics.median(latencies):.4f} ms")
    if speedups:
        import statistics
        print(f"  Speedup — min: {min(speedups):.3f}x  max: {max(speedups):.3f}x  mean: {statistics.mean(speedups):.3f}x")
        print("  (speedup < 1.0 means slower than FlashInfer baseline — expected for pure Python)")

"""
evaluation/benchmark_robust.py

Benchmark ONNX models (FP32 and INT8) using ONNX Runtime with robust
warmup and statistical measurements.

Functions:
- benchmark_model(model_path, num_warmup=10, num_runs=100)

Command-line usage:
    python -m evaluation.benchmark_robust --fp32 models/fp32/yolov8n.onnx \
        --int8 models/int8/yolov8n_int8.onnx --warmup 10 --runs 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort


def _resolve_input_shape(shape: Tuple[Any, ...]) -> Tuple[int, ...]:
    """
    Resolve an ONNX model input shape into a concrete shape for creating
    a dummy input. Replace dynamic dimensions (None or strings) or non-
    positive numbers with defaults. The YOLOv8 models expect (N, 3, 640, 640).

    Args:
        shape: the raw input shape from the ONNX model

    Returns:
        A concrete shape tuple of ints.
    """
    resolved = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            # Common assumptions:
            # - Batch size -> 1
            # - Channel dimension -> 3
            # - Spatial dims -> 640
            if idx == 0:
                resolved.append(1)
            elif idx == 1:
                resolved.append(3)
            else:
                resolved.append(640)
    return tuple(resolved)


def _create_dummy_input(input_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Create a dummy input numpy array of dtype float32 compatible with
    YOLOv8 input requirements.

    Args:
        input_shape: concrete shape for the input tensor (N, C, H, W)

    Returns:
        A float32 numpy array with values in [0, 1).
    """
    # Use random values in a reproducible but not-seeded way; randomness is
    # fine for benchmarking but keep dtype right.
    return np.random.random_sample(input_shape).astype(np.float32)


def _print_results_table(name: str, metrics: Dict[str, float]) -> None:
    """
    Nicely print a small results table summarizing the metrics.

    Args:
        name: Label for the model (e.g., 'FP32', 'INT8').
        metrics: Dictionary containing mean_ms, std_ms, median_ms, min_ms,
                 max_ms, fps.
    """
    print(f"\nResults for {name}:")
    mean = metrics.get("mean_ms", 0.0)
    std = metrics.get("std_ms", 0.0)
    median = metrics.get("median_ms", 0.0)
    min_ms = metrics.get("min_ms", 0.0)
    max_ms = metrics.get("max_ms", 0.0)
    fps = metrics.get("fps", 0.0)

    print(f"  Mean ± Std : {mean:.3f} ms ± {std:.3f} ms")
    print(f"  Median     : {median:.3f} ms")
    print(f"  Range      : {min_ms:.3f} ms - {max_ms:.3f} ms")
    print(f"  FPS        : {fps:.2f} frames/sec")


def benchmark_model(model_path: str | Path,
                    num_warmup: int = 10,
                    num_runs: int = 100) -> Dict[str, Any]:
    """
    Benchmark an ONNX model using ONNX Runtime CPUExecutionProvider.

    The function will:
    - Load the ONNX model with SessionOptions using ORT_ENABLE_ALL optimizations.
    - Prepare a dummy input tensor of shape (1, 3, 640, 640) unless the model's
      input shape provides concrete positive dimensions.
    - Run a warmup phase (num_warmup iterations) which are not timed.
    - Run a measurement phase (num_runs iterations) timing each inference and
      reporting progress every 20 iterations.
    - Compute mean, std, median, min, max latencies (milliseconds) and FPS.

    Args:
        model_path: Path to the ONNX model file.
        num_warmup: Number of warmup iterations (default: 10).
        num_runs: Number of timed runs to collect statistics (default: 100).

    Returns:
        A dictionary containing measured metrics and meta information.

    Raises:
        RuntimeError: if the model cannot be loaded or inference fails.
    """
    model_p = Path(model_path)
    if not model_p.exists():
        raise FileNotFoundError(f"Model file not found: {model_p}")

    print(f"Loading model: {model_p}")
    try:
        so = ort.SessionOptions()
        # Use the highest graph optimization level available.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Create inference session using CPU
        sess = ort.InferenceSession(model_p.as_posix(),
                                    sess_options=so,
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to create ONNX Runtime session: {exc}") from exc

    # Inspect model input
    try:
        inputs = sess.get_inputs()
        if len(inputs) == 0:
            raise RuntimeError("Model has no inputs.")
        input_meta = inputs[0]
        input_name = input_meta.name
        raw_shape = tuple(input_meta.shape)
        input_shape = _resolve_input_shape(raw_shape)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to inspect model input: {exc}") from exc

    # Prepare dummy input
    dummy = _create_dummy_input(input_shape)

    # Warmup
    print(f"Warmup: {num_warmup} iterations...")
    try:
        for i in range(num_warmup):
            sess.run(None, {input_name: dummy})
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Error during warmup inference: {exc}") from exc

    # Measurement
    latencies_ms: List[float] = []
    print(f"Measurement: {num_runs} iterations...")
    try:
        for i in range(1, num_runs + 1):
            t0 = time.perf_counter()
            sess.run(None, {input_name: dummy})
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latencies_ms.append(latency_ms)

            if i % 20 == 0 or i == num_runs:
                print(f"Progress: {i}/{num_runs}")
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Error during timed inference: {exc}") from exc

    # Compute metrics
    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=0))
    median_ms = float(np.median(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    metrics: Dict[str, Any] = {
        "model_path": model_p.as_posix(),
        "num_warmup": num_warmup,
        "num_runs": num_runs,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps": fps,
        "raw_latencies_ms": latencies_ms,
        "input_shape": input_shape,
    }

    _print_results_table(model_p.name, metrics)

    return metrics


def _safe_size_mb(path: Path) -> float:
    """
    Return file size in megabytes. If file is missing, return 0.0.

    Args:
        path: Path to file.

    Returns:
        Size in megabytes.
    """
    try:
        return float(path.stat().st_size) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _save_results(results: Dict[str, Any], out_path: Path) -> None:
    """
    Save benchmark results to a JSON file with pretty printing.

    Args:
        results: Dictionary to save.
        out_path: Destination path.
    """
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"Saved benchmark results to: {out_path}")
    except Exception as exc:
        print(f"Warning: failed to save results to {out_path}: {exc}")


def main() -> None:
    """Command-line entry point for benchmarking FP32 and INT8 ONNX models."""
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX models (FP32 vs INT8) using ONNX Runtime."
    )
    parser.add_argument(
        "--fp32",
        type=str,
        default="models/fp32/yolov8n.onnx",
        help="Path to FP32 ONNX model (default: models/fp32/yolov8n.onnx)",
    )
    parser.add_argument(
        "--int8",
        type=str,
        default="models/int8/yolov8n_int8.onnx",
        help="Path to INT8 ONNX model (default: models/int8/yolov8n_int8.onnx)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed runs for measurement (default: 100)",
    )

    args = parser.parse_args()

    fp32_path = Path(args.fp32)
    int8_path = Path(args.int8)
    results: Dict[str, Any] = {}

    # Benchmark FP32 if available
    if fp32_path.exists():
        try:
            print("\n=== Benchmarking FP32 model ===")
            results["fp32"] = benchmark_model(fp32_path,
                                             num_warmup=args.warmup,
                                             num_runs=args.runs)
        except Exception as exc:
            print(f"Error benchmarking FP32 model: {exc}")
            results["fp32"] = {"error": str(exc)}
    else:
        print(f"FP32 model not found at {fp32_path}, skipping FP32 benchmark.")
        results["fp32"] = {"error": "model not found"}

    # Benchmark INT8 if available
    if int8_path.exists():
        try:
            print("\n=== Benchmarking INT8 model ===")
            results["int8"] = benchmark_model(int8_path,
                                             num_warmup=args.warmup,
                                             num_runs=args.runs)
        except Exception as exc:
            print(f"Error benchmarking INT8 model: {exc}")
            results["int8"] = {"error": str(exc)}
    else:
        print(f"INT8 model not found at {int8_path}, skipping INT8 benchmark.")
        results["int8"] = {"error": "model not found"}

    # Add model sizes
    fp32_size_mb = _safe_size_mb(fp32_path)
    int8_size_mb = _safe_size_mb(int8_path)
    results["sizes_mb"] = {"fp32_mb": fp32_size_mb, "int8_mb": int8_size_mb}

    # Comparison summary if both succeeded
    try:
        fp32_mean = float(results["fp32"].get("mean_ms", 0.0)) if isinstance(
            results.get("fp32"), dict) else 0.0
        int8_mean = float(results["int8"].get("mean_ms", 0.0)) if isinstance(
            results.get("int8"), dict) else 0.0

        print("\n=== Comparison Summary ===")
        if fp32_mean > 0 and int8_mean > 0:
            speedup = fp32_mean / int8_mean
            size_reduction_pct = ((fp32_size_mb - int8_size_mb) / fp32_size_mb
                                  * 100.0) if fp32_size_mb > 0 else 0.0
            print(f"FP32 mean latency: {fp32_mean:.3f} ms")
            print(f"INT8 mean latency : {int8_mean:.3f} ms")
            print(f"Speedup (FP32 / INT8): {speedup:.3f}x")
            print(f"FP32 size: {fp32_size_mb:.3f} MB")
            print(f"INT8 size: {int8_size_mb:.3f} MB")
            print(f"Size reduction: {size_reduction_pct:.2f}%")
            results["comparison"] = {
                "fp32_mean_ms": fp32_mean,
                "int8_mean_ms": int8_mean,
                "speedup": speedup,
                "fp32_size_mb": fp32_size_mb,
                "int8_size_mb": int8_size_mb,
                "size_reduction_pct": size_reduction_pct,
            }
        else:
            print("Could not compute comparison (missing or failed benchmarks).")
            results["comparison"] = {"error": "missing or failed benchmarks"}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: failed to compute comparison summary: {exc}")
        results["comparison"] = {"error": str(exc)}

    # Save results
    out_path = Path("results") / "benchmark_results.json"
    _save_results(results, out_path)


if __name__ == "__main__":  # pragma: no cover - executable script
    main()

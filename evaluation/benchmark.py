"""
Compares FP32 and INT8 models under identical conditions to evaluate
performance–accuracy trade-offs for edge deployment.

This script measures latency, FPS, approximate memory usage and model size
for both a PyTorch `ultralytics` FP32 model and an INT8 ONNX model.
It writes a CSV summary to `results/tables/` and saves comparison plots to
`results/plots/` for inclusion in reports or papers.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from ultralytics import YOLO


def measure_onnx(onnx_path: str, image_path: str, runs: int = 50):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    import cv2
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (640, 640)).astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))[None, :]

    # warmup
    for _ in range(5):
        sess.run(None, {input_name: im})

    times = []
    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: im})
        times.append(time.perf_counter() - t0)
    mem_after = proc.memory_info().rss

    return {
        'latency_ms': float(np.mean(times) * 1000),
        'fps': float(1.0 / np.mean(times)),
        'mem_rss_bytes': int(mem_after - mem_before),
        'model_size_bytes': Path(onnx_path).stat().st_size,
    }


def measure_fp32(weights: str, image_path: str, runs: int = 50):
    model = YOLO(weights)
    import cv2
    im = cv2.imread(image_path)
    # ultralytics handles preprocessing; pass path for consistency

    # warmup
    for _ in range(3):
        model.predict(source=image_path, imgsz=640, verbose=False)

    times = []
    proc = psutil.Process()
    mem_before = proc.memory_info().rss
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(source=image_path, imgsz=640, verbose=False)
        times.append(time.perf_counter() - t0)
    mem_after = proc.memory_info().rss

    return {
        'latency_ms': float(np.mean(times) * 1000),
        'fps': float(1.0 / np.mean(times)),
        'mem_rss_bytes': int(mem_after - mem_before),
        'model_size_bytes': Path(weights).stat().st_size if Path(weights).exists() else None,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fp32-weights', type=str, help='PyTorch weights file (e.g. yolov8n.pt)')
    p.add_argument('--int8-onnx', type=str, help='INT8 ONNX model path')
    p.add_argument('--image', type=str, required=True, help='Single image for benchmarking')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    out = {}
    if args.fp32_weights:
        logging.info('Measuring FP32 PyTorch model')
        out['fp32'] = measure_fp32(args.fp32_weights, args.image)
    if args.int8_onnx:
        logging.info('Measuring INT8 ONNX model')
        out['int8'] = measure_onnx(args.int8_onnx, args.image)

    # Prepare results directories
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    Path('results/plots').mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    rows = []
    # Try to load mAP metrics produced by the evaluation scripts
    import json
    fp32_map = None
    int8_map = None
    try:
        with open('results/tables/fp32_metrics.json') as f:
            d = json.load(f)
            fp32_map = d.get('mAP@0.5:0.95') or d.get('mAP@0.5:0.95')
    except Exception:
        pass
    try:
        with open('results/tables/int8_metrics.json') as f:
            d = json.load(f)
            int8_map = d.get('mAP@0.5:0.95') or d.get('mAP@0.5:0.95')
    except Exception:
        pass

    if 'fp32' in out:
        r = out['fp32'].copy()
        r.update({'model': 'YOLOv8-Nano', 'precision': 'FP32', 'mAP': fp32_map})
        rows.append(r)
    if 'int8' in out:
        r = out['int8'].copy()
        r.update({'model': 'YOLOv8-Nano', 'precision': 'INT8', 'mAP': int8_map})
        rows.append(r)

    df = pd.DataFrame(rows)
    csv_path = Path('results/tables/benchmark_summary.csv')
    df.to_csv(csv_path, index=False)
    logging.info('Saved benchmark summary to %s', csv_path)

    # Create simple plots: latency and fps comparison
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        if not df.empty:
            ax[0].bar(df['precision'], df['latency_ms'].astype(float), color=['C0', 'C1'][:len(df)])
            ax[0].set_title('Latency (ms)')
            ax[0].set_ylabel('ms')

            ax[1].bar(df['precision'], df['fps'].astype(float), color=['C0', 'C1'][:len(df)])
            ax[1].set_title('FPS')

        plt.tight_layout()
        plot_path = Path('results/plots/benchmark_latency_fps.png')
        fig.savefig(plot_path)
        logging.info('Saved benchmark plots to %s', plot_path)
    except Exception as e:
        logging.warning('Failed to create plots: %s', e)

    print(df.to_dict(orient='records'))

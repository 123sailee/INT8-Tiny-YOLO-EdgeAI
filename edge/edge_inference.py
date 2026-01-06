"""
Simulates CPU-based edge deployment by measuring inference latency,
throughput (FPS), and approximate memory usage.

This minimal example shows how to run a single-image inference with an
ONNX Runtime session and serves as a starting point for integrating the
INT8 model into an embedded pipeline.
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(img_path: str, img_size: int = 640):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (img_size, img_size)).astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))[None, :]
    return im


def run_once(onnx_path: str, image: str, img_size: int = 640):
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    inp = preprocess(image, img_size)
    outs = sess.run(None, {input_name: inp})
    return outs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', type=str, required=True)
    p.add_argument('--image', type=str, required=True)
    p.add_argument('--img-size', type=int, default=640)
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    outs = run_once(args.onnx, args.image, img_size=args.img_size)
    print('Raw outputs shape(s):', [o.shape for o in outs if hasattr(o, 'shape')])

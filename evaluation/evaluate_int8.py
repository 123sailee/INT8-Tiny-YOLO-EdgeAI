"""Evaluate INT8 ONNX model using ONNX Runtime and produce COCO-style results.

This script runs inference with ONNX Runtime and saves detections in COCO
results JSON format. If a COCO-format annotation file is provided via `--ann`,
it computes COCO mAP@0.5 and mAP@0.5:0.95 (using `pycocotools`) and writes a
metrics JSON to `results/tables/`.
"""
import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import json
import logging
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:
    COCO = None
    COCOeval = None


def preprocess(img_path: str, img_size: int):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (img_size, img_size)).astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, 0)
    return im


def infer_onnx(onnx_path: str, images_dir: str, out_json: str, img_size: int = 640):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    results = []
    images = sorted(Path(images_dir).glob('*.*'))
    for p in images:
        inp = preprocess(str(p), img_size)
        out = session.run(None, {input_name: inp})
        # ultralytics/YOLOv8 ONNX exports usually return a single output with shape (1, N, 6)
        pred = out[0]
        if pred is None or len(pred.shape) < 3:
            continue
        pred = np.squeeze(pred, 0)
        # pred: [x1,y1,x2,y2,score,class]
        for row in pred:
            if row[4] <= 0:
                continue
            x1, y1, x2, y2, score, cls = row[:6]
            w = x2 - x1
            h = y2 - y1
            image_id = int(p.stem) if p.stem.isdigit() else hash(str(p)) & 0xffffffff
            results.append({
                'image_id': image_id,
                'category_id': int(cls),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(score),
            })

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f)

    logging.info('Saved INT8 detections to %s', out_json)


def compute_map(gt_json: str, dt_json: str):
    if COCO is None:
        logging.warning('pycocotools not available; skipping mAP computation')
        return None, None
    gt = COCO(gt_json)
    dt = gt.loadRes(dt_json)
    eval = COCOeval(gt, dt, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    stats = eval.stats
    mAP = float(stats[0])
    mAP50 = float(stats[1])
    return mAP50, mAP


def save_metrics(prefix: str, mAP50, mAP):
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    out = {'mAP@0.5': mAP50, 'mAP@0.5:0.95': mAP}
    with open(f'results/tables/{prefix}_metrics.json', 'w') as f:
        json.dump(out, f)
    logging.info('Saved metrics to results/tables/%s_metrics.json', prefix)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', type=str, required=True)
    p.add_argument('--images', type=str, required=True)
    p.add_argument('--out', type=str, default='results/int8_detections.json')
    p.add_argument('--ann', type=str, default='data/annotations/instances_val.json',
                   help='COCO-format ground truth annotations for mAP computation')
    p.add_argument('--img-size', type=int, default=640)
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    infer_onnx(args.onnx, args.images, args.out, img_size=args.img_size)
    ann_path = Path(args.ann)
    if ann_path.exists():
        mAP50, mAP = compute_map(str(ann_path), args.out)
        save_metrics('int8', mAP50, mAP)
    else:
        logging.warning('Annotation file not found (%s). Skipping mAP computation.', ann_path)
        save_metrics('int8', None, None)

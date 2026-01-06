"""Evaluate FP32 PyTorch YOLOv8 model and produce COCO-style results.

This script performs inference using the `ultralytics` YOLO wrapper and
saves detections in COCO results JSON format. If a COCO-format annotation
file is provided via `--ann`, it will compute COCO mAP@0.5 and mAP@0.5:0.95
using `pycocotools` and write a small metrics JSON to `results/tables/`.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import logging
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:
    COCO = None
    COCOeval = None


def infer_and_save(weights: str, images_dir: str, out_json: str, img_size: int = 640):
    model = YOLO(weights)
    images = sorted(Path(images_dir).glob('*.*'))
    results = []

    for p in images:
        preds = model.predict(source=str(p), imgsz=img_size, conf=0.001, verbose=False)[0]
        boxes = preds.boxes.xyxy.cpu().numpy() if hasattr(preds, 'boxes') else np.zeros((0, 4))
        scores = preds.boxes.conf.cpu().numpy() if hasattr(preds, 'boxes') else np.zeros((0,))
        classes = preds.boxes.cls.cpu().numpy().astype(int) if hasattr(preds, 'boxes') else np.zeros((0,), dtype=int)

        image_id = int(p.stem) if p.stem.isdigit() else hash(str(p)) & 0xffffffff
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            results.append({
                'image_id': image_id,
                'category_id': int(cls),
                'bbox': [x1, y1, w, h],
                'score': float(score),
            })

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f)

    logging.info('Saved detections to %s', out_json)


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
    # stats[0] -> mAP (0.5:0.95), stats[1] -> mAP@0.5
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
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--images', type=str, required=True)
    p.add_argument('--out', type=str, default='results/fp32_detections.json')
    p.add_argument('--ann', type=str, default='data/annotations/instances_val.json',
                   help='COCO-format ground truth annotations for mAP computation')
    p.add_argument('--img-size', type=int, default=640)
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    infer_and_save(args.weights, args.images, args.out, img_size=args.img_size)
    # Attempt mAP computation if annotations are available
    ann_path = Path(args.ann)
    if ann_path.exists():
        mAP50, mAP = compute_map(str(ann_path), args.out)
        save_metrics('fp32', mAP50, mAP)
    else:
        logging.warning('Annotation file not found (%s). Skipping mAP computation.', ann_path)
        save_metrics('fp32', None, None)

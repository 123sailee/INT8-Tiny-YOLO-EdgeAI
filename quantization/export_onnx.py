"""Export a YOLOv8-Nano model to ONNX.

This script uses the `ultralytics` package for a PyTorch baseline model
and exports to ONNX for further quantization.

Example:
  python quantization/export_onnx.py --weights yolov8n.pt --output models/fp32/yolov8n.onnx
"""
import argparse
import logging
from pathlib import Path

from ultralytics import YOLO


def export(weights: str, output: str, img_size: int = 640, opset: int = 12, device: str = 'cpu'):
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info('Loading model: %s', weights)
    model = YOLO(weights)

    logging.info('Exporting to ONNX: %s', out_path)
    # ultralytics supports a convenient export wrapper
    model.export(format='onnx', imgsz=img_size, opset=opset, device=device)
    # ultralytics writes the ONNX file to the current working dir using the
    # weight filename (e.g., 'yolov8n.onnx'). Move it to the requested output
    # location to keep a clean project layout.
    src_candidate = Path(f"{Path(weights).stem}.onnx")
    if src_candidate.exists():
        import shutil
        shutil.move(str(src_candidate), str(out_path))
        logging.info('Moved exported ONNX to %s', out_path)
    else:
        logging.info('Export complete. Verify file in %s (ultralytics may place the file in the cwd).', out_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='yolov8n.pt')
    p.add_argument('--output', type=str, default='models/fp32/yolov8n.onnx')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--opset', type=int, default=12)
    p.add_argument('--device', type=str, default='cpu')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    export(args.weights, args.output, img_size=args.img_size, opset=args.opset, device=args.device)

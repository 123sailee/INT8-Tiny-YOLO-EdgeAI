"""Post-training static INT8 quantization for ONNX models using ORT.

This script expects a FP32 ONNX model and a small representative dataset
folder (images) for calibration. It produces an INT8 model under
`models/int8/`.

Notes:
- The script uses `onnxruntime`'s `quantize_static` API.
"""
import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import onnx
from PIL import Image

from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType


class ImageCalibrationReader(CalibrationDataReader):
    """Simple image data reader that yields batches for calibration.

    The model input preprocessing must match the exported ONNX model
    (image size, normalization). Adjust `img_size` and normalization as needed.
    """

    def __init__(self, image_paths: List[Path], input_name: str, img_size: int = 640):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.img_size = img_size
        self.data = None

    def _preprocess(self, path: Path):
        img = Image.open(path).convert('RGB').resize((self.img_size, self.img_size))
        arr = np.array(img).astype(np.float32)
        # normalize to 0-1 and transpose to NCHW
        arr = arr / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return {self.input_name: arr}

    def get_next(self):
        if not self.image_paths:
            return None
        path = self.image_paths.pop(0)
        return self._preprocess(path)


def quantize(onnx_model: str, calib_images_dir: str, output_path: str, img_size: int = 640):
    onnx_path = Path(onnx_model)
    if not onnx_path.exists():
        raise FileNotFoundError(f'FP32 ONNX model not found: {onnx_path}')

    model = onnx.load(str(onnx_path))
    input_name = model.graph.input[0].name

    images = list(Path(calib_images_dir).glob('*.*'))
    if not images:
        raise FileNotFoundError('No calibration images found in ' + calib_images_dir)

    reader = ImageCalibrationReader(images, input_name=input_name, img_size=img_size)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    logging.info('Starting static quantization: %s -> %s', onnx_path, out)
    quantize_static(model_input=str(onnx_path),
                    model_output=str(out),
                    calibration_data_reader=reader,
                    quant_format=None,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8)

    logging.info('Quantized model saved to %s', out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', type=str, required=True, help='FP32 ONNX model path')
    p.add_argument('--calib-dir', type=str, required=True, help='Representative images directory')
    p.add_argument('--output', type=str, default='models/int8/yolov8n_int8.onnx')
    p.add_argument('--img-size', type=int, default=640)
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    quantize(args.onnx, args.calib_dir, args.output, img_size=args.img_size)


#!/usr/bin/env bash
set -euo pipefail

# Orchestration script that runs the full FP32 -> INT8 -> evaluate -> benchmark flow.
# Execution order (exact):
# 1) Export FP32 YOLOv8-Nano to ONNX
# 2) Apply INT8 post-training quantization
# 3) Evaluate FP32 model
# 4) Evaluate INT8 model
# 5) Run benchmark comparison and save results (CSV + plots)

PYTHON=${PYTHON:-python}

# Ensure results dirs exist
mkdir -p results/tables results/plots

echo "1) Export FP32 ONNX (requires yolov8n.pt weights in cwd or specify path)"
$PYTHON quantization/export_onnx.py --weights yolov8n.pt --output models/fp32/yolov8n.onnx

echo "2) Quantize to INT8 (provide representative images under data/images/val)"
# The quantization script uses a representative calibration image set to produce INT8 model
$PYTHON quantization/int8_quantize.py --onnx models/fp32/yolov8n.onnx --calib-dir data/images/val --output models/int8/yolov8n_int8.onnx

echo "3) Evaluate FP32 model (saves detections as COCO-style JSON)"
$PYTHON evaluation/evaluate_fp32.py --weights yolov8n.pt --images data/images/val --out results/fp32_detections.json

echo "4) Evaluate INT8 model (saves detections as COCO-style JSON)"
$PYTHON evaluation/evaluate_int8.py --onnx models/int8/yolov8n_int8.onnx --images data/images/val --out results/int8_detections.json

echo "5) Run benchmark comparison and save results (CSV + plots)"
# Benchmark script writes CSV to results/tables/ and plots to results/plots/
$PYTHON evaluation/benchmark.py --fp32-weights yolov8n.pt --int8-onnx models/int8/yolov8n_int8.onnx --image data/images/sample.jpg

echo "Done. Results saved under results/tables and results/plots"

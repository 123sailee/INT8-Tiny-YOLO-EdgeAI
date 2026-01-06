# INT8-Tiny-YOLO-EdgeAI

Professional project demonstrating post-training INT8 quantization
and edge deployment of a lightweight object detector (YOLOv8-Nano).

Why This Project Matters
------------------------
This project demonstrates how modern object detection models can be optimized
for real-world edge deployment using INT8 post-training quantization.  
It highlights practical performance–accuracy trade-offs relevant to both
production systems and academic evaluation.

Overview
- Purpose: Showcase FP32 → INT8 post-training quantization, export to ONNX,
  evaluate accuracy and latency trade-offs, and provide an edge-ready inference example.
- Focused on tiny object detection for resource-constrained devices.

Why INT8 matters
- INT8 quantization reduces model size, memory footprint, and inference latency on CPUs
  and NPUs while retaining acceptable accuracy — critical for smart cameras and embedded systems.

Repository layout
```
INT8-Tiny-YOLO-EdgeAI/
├── data/                # Dataset notes and expected layout (COCO)
├── models/
│   ├── fp32/            # exported FP32 ONNX models
│   └── int8/            # quantized INT8 ONNX models
├── quantization/
│   ├── export_onnx.py   # export ultralytics YOLO to ONNX
│   └── int8_quantize.py # post-training static INT8 quantization (calibration)
├── evaluation/
│   ├── evaluate_fp32.py # run FP32 inference and save detections
│   ├── evaluate_int8.py # run INT8 ONNX inference
│   └── benchmark.py     # latency, FPS, model size, memory; saves CSV and plots
├── edge/
│   └── edge_inference.py# minimal ONNX runtime example
├── results/
│   ├── tables/          # CSV/Markdown summaries
│   └── plots/           # benchmark plots (PNG)
├── README.md
├── requirements.txt
└── run_experiments.sh
```

Quick start
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Export a pretrained YOLOv8-Nano to ONNX:

```bash
python quantization/export_onnx.py --weights yolov8n.pt --output models/fp32/yolov8n.onnx
```

3. Prepare a small representative calibration dataset (images) and quantize:

```bash
python quantization/int8_quantize.py --onnx models/fp32/yolov8n.onnx --calib-dir data/images/val --output models/int8/yolov8n_int8.onnx
```

4. Run evaluations and benchmarks (see `evaluation/` scripts)

Benchmark Summary (FP32 vs INT8)

| Model | Precision | mAP | FPS | Latency (ms) | Model Size |
|------|----------|-----|-----|--------------|------------|
| YOLOv8-Nano | FP32 | — | — | — | — |
| YOLOv8-Nano | INT8 | — | — | — | — |

Dataset
-------
Experiments are conducted on a standard object detection dataset
(e.g., a COCO subset or representative sample dataset) to evaluate
real-world edge performance.

Research & evaluation notes
- Evaluation scripts save COCO-style detection JSON results which can be
  evaluated with `pycocotools` or custom mAP code.
- The quantization flow uses post-training static calibration — suitable when
  labeled data is limited or retraining is infeasible.

Use cases
- Surveillance, safety monitoring, smart doorbells, industrial inspection,
  and other low-power camera-based detection tasks.

Contact
- This repository is structured for inclusion in a CV or research submission.
  If you want help tailoring experiments or drafting the methods/results
  sections for a conference paper, ask and I can help.

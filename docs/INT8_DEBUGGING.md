# INT8 Performance Debugging Checklist

If INT8 is slower than FP32, this is not necessarily wrong.
Follow this checklist to understand and explain the behavior.

## 1. ONNX Runtime Build Flags

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

- `CPUExecutionProvider` must be present.
- Additional providers may improve performance (e.g., OpenVINO, TensorRT, DNNL-specific providers).
- Absence of optimized providers can limit INT8 speedups; verify the runtime build was compiled with optimizations for your CPU.

## 2. CPU Capabilities

```bash
lscpu | grep -E 'avx|vnni'
```

- AVX2 is a practical minimum for reasonable INT8 performance on many runtimes.
- AVX512 with VNNI (Vector Neural Network Instructions) provides the best INT8 acceleration on supported CPUs.
- If the CPU lacks these features, INT8 code paths may not use fast vectorized kernels and can be slower than well-optimized FP32 code.

## 3. Verify Quantization Actually Happened

```python
import onnx
model = onnx.load('models/int8/yolov8n_int8.onnx')
ops = [node.op_type for node in model.graph.node]
print('QLinearConv' in ops or 'QuantizeLinear' in ops)
```

- The output should be `True` for a genuinely quantized model.
- If `False`, the file name may indicate INT8 but the model is effectively FP32; investigate the export pipeline.

## 4. Runtime Optimization Level

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    model_path,
    sess_options,
    providers=['CPUExecutionProvider']
)
```

- Graph optimizations can fuse operators, eliminate redundant computations, and enable runtime-specific INT8 kernels.
- Enabling the highest optimization level lets the runtime select faster execution plans which often benefit quantized models.

## 5. Thread Configuration

```python
sess_options.intra_op_num_threads = 4
sess_options.inter_op_num_threads = 1
```

- Threading has a large impact on latency and throughput; defaults are not universally optimal.
- `intra_op_num_threads` controls parallelism inside operators (important for per-op performance).
- `inter_op_num_threads` controls parallelism between independent operators.
- Tune these per-CPU and per-workload; some CPUs perform best with fewer threads for low-latency inference.

## 6. Expected Result Ranges

| Scenario | Expected Speedup | Notes |
|----------|------------------|-------|
| Modern CPU with VNNI | 1.2x – 2.0x | Typical for well-optimized INT8 kernels and medium-to-large models |
| CPU without VNNI | 0.9x – 1.1x | INT8 may be similar or slightly slower depending on kernel implementation |
| Small model on CPU | 0.8x – 1.0x | Overhead and FP32 microkernel efficiency can make FP32 faster for tiny models |

### Important Note

- INT8 being slower than FP32 is a VALID and PUBLISHABLE result.
- It demonstrates hardware and runtime dependency of quantization speedups.
- Results should be explained clearly, not hidden; report CPU features, runtime build, thread settings, and model sizes.
- Honest reporting strengthens research credibility.

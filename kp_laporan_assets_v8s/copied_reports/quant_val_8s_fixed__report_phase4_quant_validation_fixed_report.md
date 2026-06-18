# Validasi Quantization / Export YOLOv8s-seg Fixed

Dataset: `/home/pep-admin/croptic/croptic-segmentation/dataset/lahan_gabungan_yolo_standard_split/2026-04-24_split/dataset.yaml`

Image size: `416`

Catatan: validasi dipaksa menggunakan `task='segment'` dan `plots=False`.

## Ringkasan

| name | format | precision | quantization | model_size_mb | precision_M | recall_M | mAP50_M | mAP50_95_M | speed_inference_ms | speed_total_ms | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pt_baseline | PyTorch | FP32 | No | 19.564 | 0.796498 | 0.780795 | 0.807082 | 0.522445 | 1.13503 | 1.914275 | success |
| onnx_fp32 | ONNX | FP32 | No | 38.608 | 0.0 | 0.0 | 0.0 | 0.0 | 3.220639 | 4.305092 | success |
| onnx_fp16 | ONNX | FP16 | Yes | 19.384 | 0.792491 | 0.738723 | 0.782674 | 0.504036 | 2.340997 | 3.414584 | success |
| onnx_int8 | ONNX | INT8 | Yes | 19.698 | 0.744919 | 0.73298 | 0.768623 | 0.466212 | 6.860798 | 8.132538 | success |
| tensorrt_engine | TensorRT | Engine | Depends on export args | 22.171 | 0.724903 | 0.768587 | 0.759571 | 0.466371 | 0.83998 | 1.953346 | success |
| tflite_float32 | TFLite | FP32 | No | 38.587 | 0.791283 | 0.739095 | 0.781123 | 0.503841 | 110.866575 | 113.300836 | success |
| tflite_float16 | TFLite | FP16 | Yes | 19.376 | 0.781941 | 0.742859 | 0.781183 | 0.503916 | 111.168028 | 113.607117 | success |
| tflite_int8 | TFLite | INT8 | Yes | 10.017 | 0.779898 | 0.743371 | 0.782998 | 0.504955 | 38.938048 | 41.171542 | success |
| tflite_full_integer_quant | TFLite | INT8 | Yes - full integer | 10.145 | 0.015837 | 0.035338 | 0.00096 | 0.000386 | 31.734547 | 33.3681 | success |
| tflite_integer_quant | TFLite | INT8 | Yes - integer | 10.145 | 0.016189 | 0.035082 | 0.001105 | 0.000357 | 31.440701 | 32.955225 | success |

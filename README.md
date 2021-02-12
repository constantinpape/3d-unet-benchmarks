# 3D UNet Benchmarks

Training and inference benchmarks with a 3d-Unet.
Benchmarks results for EMBL cluster, for training lower is better, for inference higher is better.

| Gpu-Type             |   Cuda-Version | Pytorch-Version   | Training [s/iter]   | Inference [MVox/s]   |
|:---------------------|---------------:|:------------------|:--------------------|:---------------------|
| Tesla V100-SXM2-32GB |           11.0 | 1.7.1             | 1.87 +- 0.09        | 15.97 +- 0.25        |
| GeForce GTX 1080 Ti  |           11.0 | 1.7.1             | 3.08 +- 0.07        | 8.68 +- 0.07         |
| GeForce RTX 3090     |           11.0 | 1.7.1             | 1.84 +- 0.08        | 18.8 +- 0.53         |
| GeForce RTX 2080 Ti  |           11.0 | 1.7.1             | 2.96 +- 0.14        | 12.35 +- 0.32        |
| A100-PCIE-40GB       |           11.0 | 1.7.1             | 1.63 +- 0.06        | 19.52 +- 0.43        |

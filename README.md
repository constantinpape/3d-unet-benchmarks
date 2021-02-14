# 3D UNet Benchmarks

Training and inference benchmarks with a 3d-Unet.
For training lower is better, for inference higher is better.

## EMBL Cluster Results 

| Gpu-Type             |   Cuda-Version | Pytorch-Version   | Training [s/iter]   | Inference [MVox/s]   |
|:---------------------|---------------:|:------------------|:--------------------|:---------------------|
| A100-PCIE-40GB       |           11.0 | 1.7.1             | 1.63 +- 0.06        | 19.52 +- 0.43        |
| A100-PCIE-40GB       |           11.1 | 1.7.1             | 0.36 +- 0.19        | 32.94 +- 1.19        |
| GeForce GTX 1080 Ti  |           11.0 | 1.7.1             | 3.08 +- 0.07        | 8.68 +- 0.07         |
| GeForce GTX 1080 Ti  |           11.1 | 1.7.1             | 2.79 +- 0.07        | 12.56 +- 0.26        |
| GeForce RTX 2080 Ti  |           11.0 | 1.7.1             | 2.96 +- 0.14        | 12.35 +- 0.32        |
| GeForce RTX 2080 Ti  |           11.1 | 1.7.1             | 1.33 +- 0.13        | 46.64 +- 2.0         |
| GeForce RTX 3090     |           11.0 | 1.7.1             | 1.84 +- 0.08        | 18.8 +- 0.53         |
| GeForce RTX 3090     |           11.1 | 1.7.1             | 0.43 +- 0.22        | 57.97 +- 4.68        |
| Tesla V100-SXM2-32GB |           11.0 | 1.7.1             | 1.87 +- 0.09        | 15.97 +- 0.25        |
| Tesla V100-SXM2-32GB |           11.1 | 1.7.1             | 0.40 +- 0.12        | 61.68 +- 2.53        |

Note: for all benchmarks the same block size for training (48 x 384 x 384 voxels) and inference (64 x 448 x 448 voxels) was used, chosen s.t. it filled the RAM of the 2080Ti.
The torch for cuda version `11.0` is from conda, `11.1` is the version built with easybuild.

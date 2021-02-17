# 3D UNet Benchmarks

Pytorch training and inference benchmarks with a 3d-Unet.


## Results

The training benchmark measures the mean time per training iteration (lower is better).
The inference benchmark measures the throughput in megavoxel per second (higher is better).

For training, the benchmark supports single, half and mixed precision; for inference single and half precision.

### EMBL Cluster

**Configuration:**

The cuda 11.0 configuration uses the (latest) pytorch conda package:
```
pytorch                   1.7.1           py3.8_cuda11.0.221_cudnn8.0.5_0    pytorch
```

The cuda 11.1 configuration uses pytorch built with easybuild.


**Half-precision**:

Block shapes: 48 x 384 x 384 (training), 64 x 512 x 512 (inference)

| Gpu-Type             |   Cuda-Version | Pytorch-Version   | Training [s/iter]   | Inference [MVox/s]   |
|:---------------------|---------------:|:------------------|:--------------------|:---------------------|
| A100-PCIE-40GB       |           11   | 1.7.1             | 1.61 +- 0.09        | 19.28 +- 0.35        |
| A100-PCIE-40GB       |           11.1 | 1.7.1             | 0.32 +- 0.12        | 32.07 +- 1.06        |
| GeForce GTX 1080 Ti  |           11   | 1.7.1             | 2.97 +- 0.19        | 6.37 +- 0.03         |
| GeForce GTX 1080 Ti  |           11.1 | 1.7.1             | 2.68 +- 0.15        | 12.29 +- 0.21        |
| GeForce RTX 2080 Ti  |           11   | 1.7.1             | 2.62 +- 0.21        | 12.14 +- 0.25        |
| GeForce RTX 2080 Ti  |           11.1 | 1.7.1             | 0.55 +- 0.18        | 16.17 +- 0.5         |
| GeForce RTX 3090     |           11   | 1.7.1             | 1.83 +- 0.12        | 18.03 +- 0.49        |
| GeForce RTX 3090     |           11.1 | 1.7.1             | 0.39 +- 0.21        | 37.76 +- 1.64        |
| Tesla V100-SXM2-32GB |           11   | 1.7.1             | 1.78 +- 0.2         | 15.22 +- 0.21        |
| Tesla V100-SXM2-32GB |           11.1 | 1.7.1             | 0.36 +- 0.11        | 30.59 +- 1.04        |

**Mixed-precision**:

Block shapes: 48 x 384 x 384 (training), - (inference)

| Gpu-Type             |   Cuda-Version | Pytorch-Version   | Training [s/iter]   | Inference [MVox/s]   |
|:---------------------|---------------:|:------------------|:--------------------|:---------------------|
| A100-PCIE-40GB       |           11   | 1.7.1             | 1.66 +- 0.15        | -                    |
| A100-PCIE-40GB       |           11.1 | 1.7.1             | 0.37 +- 0.22        | -                    |
| GeForce GTX 1080 Ti  |           11   | 1.7.1             | 3.06 +- 0.08        | -                    |
| GeForce GTX 1080 Ti  |           11.1 | 1.7.1             | 2.84 +- 0.08        | -                    |
| GeForce RTX 2080 Ti  |           11   | 1.7.1             | 2.6 +- 0.09         | -                    |
| GeForce RTX 2080 Ti  |           11.1 | 1.7.1             | 0.6 +- 0.19         | -                    |
| GeForce RTX 3090     |           11   | 1.7.1             | 1.86 +- 0.15        | -                    |
| GeForce RTX 3090     |           11.1 | 1.7.1             | 0.43 +- 0.23        | -                    |
| Tesla V100-SXM2-32GB |           11   | 1.7.1             | 1.85 +- 0.06        | -                    |
| Tesla V100-SXM2-32GB |           11.1 | 1.7.1             | 0.4 +- 0.12         | -                    |

**Single-precision**:

Block shapes: 32 x 384 x 384 (training), 64 x 384 x 384 (inference)

| Gpu-Type             |   Cuda-Version | Pytorch-Version   | Training [s/iter]   | Inference [MVox/s]   |
|:---------------------|---------------:|:------------------|:--------------------|:---------------------|
| A100-PCIE-40GB       |           11   | 1.7.1             | 1.06 +- 0.09        | 17.95 +- 0.36        |
| A100-PCIE-40GB       |           11.1 | 1.7.1             | 0.22 +- 0.09        | 42.47 +- 1.38        |
| GeForce GTX 1080 Ti  |           11   | 1.7.1             | 2.22 +- 0.15        | 7.39 +- 0.08         |
| GeForce GTX 1080 Ti  |           11.1 | 1.7.1             | 2.51 +- 0.11        | 7.4 +- 0.12          |
| GeForce RTX 2080 Ti  |           11   | 1.7.1             | 1.12 +- 0.15        | 16.0 +- 0.43         |
| GeForce RTX 2080 Ti  |           11.1 | 1.7.1             | 1.0 +- 0.17         | 16.4 +- 0.49         |
| GeForce RTX 3090     |           11   | 1.7.1             | 1.1 +- 0.16         | 17.35 +- 0.46        |
| GeForce RTX 3090     |           11.1 | 1.7.1             | 1.38 +- 0.17        | 21.96 +- 0.78        |
| Tesla V100-SXM2-32GB |           11   | 1.7.1             | 0.88 +- 0.12        | 21.11 +- 0.4         |
| Tesla V100-SXM2-32GB |           11.1 | 1.7.1             | 0.88 +- 0.12        | 21.33 +- 0.58        |


## Running the benchmarks

**Setup:**

- The only dependencies are `pytorch` and `h5py`.
- The benchmark uses data from [the cremi challenge](https://cremi.org/): [training data](https://cremi.org/static/data/sample_A_20160501.hdf), [inference data](https://cremi.org/static/data/sample_A_padded_20160501.hdf)
- Before running the benchmark:
    - Create some directory `/path/to/gpu_benchmark_data`, copy the two hdf5 files there
    - Export the following environment variable: `export GPU_BENCHMARK_ROOT=/path/to//gpu_benchmark_data`

**Benchmark:**

The benchmarks are implemented in `training` and `inference`. Run them as follows:
- First, run `generate_config.py`. This will store the benchmark configuration with default values as json file in `GPU_BENCHMARK_ROOT`. The configuration values can be modified by passing arguments to `generate_config.py` or modifying the json. Different configuration for different precisions can be specified.
- After this, the benchmark script `benchmark.py` can be run. The precision can be selected by passing the `-p` option; e.g. `python benchmark.py -p single` to run the benchmark in half precision.
- The scripts `run_slurm.py` / `run_slurm_all.py` submits one / all of the benchmarks as slurm jobs.
- The script `evaluate_benchmark.py` collects evaluates all benchmark results and saves them as csv file: `python evaluate_benchmark.py -o results.csv`

Note that the inference benchmarks have to be executed after the training benchmarks becuse they rely on files saved to `GPU_BENCHMARK_ROOT` by the training benchmarks.

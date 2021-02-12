import argparse
import json
import os
import sys

import h5py
import torch

# import the utils module
extra_path = os.path.join(os.path.split(__file__)[0], '..')
sys.path.append(extra_path)
import bench_util

DEFAULT_MODEL_PATH = '/scratch/pape/gpu_benchmark/results/train_benchmark_2080Ti-conda-model.pt'
# DEFAULT_DATA_PATH = '/g/kreshuk/data/benchmark/sample_A_20160501.hdf'
DEFAULT_DATA_PATH = '/scratch/pape/gpu_benchmark/sample_A_padded_20160501.hdf'
DEFAULT_RAW = 'volumes/raw'

DEFAULT_BLOCK_SIZE = [32, 384, 384]
DEFAULT_HALO = [16, 32, 32]


def inference_benchmark(args):
    model = torch.load(args.model)

    print("Start inference benchmark with block shape", args.block_shape)
    with h5py.File(args.data, mode='r') as f:
        ds = f[args.raw]
        total_time, time_per_block = bench_util.run_inference(
            ds, model,
            args.block_shape, args.halo,
            preprocess=bench_util.normalize
        )

    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    output_prefix = args.output
    gpu_type = torch.cuda.get_device_name(torch.device('cuda'))
    bench_results = {
        'patch_size': args.block_shape,
        'halo': args.halo,
        'total_inference_time': total_time,
        'time_per_block': time_per_block,
        'cuda_version': cuda_version,
        'pytorch_version': torch_version,
        'gpu_type': gpu_type
    }

    res_dir = os.path.abspath(os.path.split(output_prefix)[0])
    os.makedirs(res_dir, exist_ok=True)

    with open(output_prefix + '-results.json', 'w') as f:
        json.dump(bench_results, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--data', '-d', default=DEFAULT_DATA_PATH)
    parser.add_argument('--raw', '-r', default=DEFAULT_RAW)
    parser.add_argument('--output', '-o', default='inference-benchmark')
    parser.add_argument('--block_shape', '-b', default=DEFAULT_BLOCK_SIZE,
                        type=int, nargs=3)
    parser.add_argument('--halo', default=DEFAULT_HALO, type=int, nargs=3)

    args = parser.parse_args()
    inference_benchmark(args)

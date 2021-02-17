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

ROOT = os.environ.get('GPU_BENCHMARK_ROOT', '/scratch/pape/gpu_benchmark')
MODEL_PATH = os.path.join(ROOT, 'results', 'model.pt')

DEFAULT_DATA_PATH = os.path.join(ROOT, 'sample_A_padded_20160501.hdf')
DEFAULT_RAW = 'volumes/raw'
DEFAULT_CONFIG = os.path.join(ROOT, 'inference_config.json')

PRECISION = ('single', 'half')


def inference_benchmark(args):
    precision = args.precision
    assert precision in PRECISION

    if not os.path.exists(MODEL_PATH):
        print("Could not find a saved model in", MODEL_PATH, ". Run a traning benchmark with argument '--save_model 1' to generate it.")
        exit(1)
    model = torch.load(MODEL_PATH)

    # TODO allow special configs for gpus
    with open(args.config) as f:
        config = json.load(f)
    # check if we have a special config for this precision,
    # otherwise load the default one
    config = config.get(precision, config['default'])

    block_size = config['block_size']
    halo = config['halo']

    print("Start inference benchmark with block size", block_size, "and halo", halo)
    with h5py.File(args.data, mode='r') as f:
        ds = f[args.raw]
        total_time, time_per_block = bench_util.run_inference(
            ds, model,
            block_size, halo,
            preprocess=bench_util.normalize,
            precision=precision
        )

    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    output_prefix = args.output
    gpu_type = torch.cuda.get_device_name(torch.device('cuda'))
    bench_results = {
        'block_size': block_size,
        'precision': precision,
        'halo': halo,
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
    parser.add_argument('--data', '-d', default=DEFAULT_DATA_PATH)
    parser.add_argument('--raw', '-r', default=DEFAULT_RAW)
    parser.add_argument('--output', '-o', default='inference-benchmark')
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG)
    parser.add_argument('--precision', '-p', default='half')

    args = parser.parse_args()
    inference_benchmark(args)

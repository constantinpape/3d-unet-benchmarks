import argparse
import json
import os
from glob import glob

import pandas as pd
import numpy as np

DEFAULT_TABLE = os.path.split(os.path.abspath(__file__))[0]
DEFAULT_TABLE = os.path.join(DEFAULT_TABLE, '..', 'results-embl.csv')


def evaluate_benchmark(args, input_):

    with open(input_) as f:
        result = json.load(f)

    time_per_block = result['time_per_block']
    block_shape = result['patch_size']
    halo = result['halo']
    block_shape = [bs + 2 * ha for bs, ha in zip(block_shape, halo)]

    mvox = np.prod(block_shape) / 1e6
    throughputs = [mvox / tb for tb in time_per_block]

    mean_thr = round(np.mean(throughputs), 2)
    std_thr = round(np.std(throughputs), 2)

    gpu_type = result['gpu_type']
    cuda_version = float(result['cuda_version'])
    pytorch_version = result['pytorch_version']
    thr = f'{mean_thr} +- {std_thr}'

    assert os.path.exists(args.output),\
        "Run inference benchmarks after inference benchmarks"
    df = pd.read_csv(args.output)

    selection = ((df['Gpu-Type'] == gpu_type) &
                 (df['Cuda-Version'] == cuda_version) &
                 (df['Pytorch-Version'] == pytorch_version))
    df.loc[selection, 'Inference [MVox/s]'] = thr

    df = df.sort_values(by=['Gpu-Type', 'Cuda-Version'])
    df.to_csv(args.output, index=False)


def evaluate_benchmarks(args, input_folder):
    inputs = glob(os.path.join(input_folder, 'inference_benchmark*.json'))
    for input_ in inputs:
        evaluate_benchmark(args, input_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default=DEFAULT_TABLE)

    args = parser.parse_args()
    input_ = args.input
    if os.path.isdir(input_):
        evaluate_benchmarks(args, input_)
    else:
        assert os.path.splitext(input_)[1] == '.json'
        evaluate_benchmark(args, input_)

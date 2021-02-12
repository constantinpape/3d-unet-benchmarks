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

    time_per_iter = result['time_per_iteration']
    mean_time = round(np.mean(time_per_iter), 2)
    std_time = round(np.std(time_per_iter), 2)

    data = np.array([result['gpu_type'],
                     result['cuda_version'],
                     result['pytorch_version'],
                     f'{mean_time} +- {std_time}',
                     '-'])
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        df = df.append(pd.DataFrame(data[None], columns=df.columns))
    else:
        columns = ['Gpu-Type', 'Cuda-Version', 'Pytorch-Version',
                   'Training [s/iter]', 'Inference [MVox/s]']
        df = pd.DataFrame(data[None], columns=columns)

    df.to_csv(args.output, index=False)


def evaluate_benchmarks(args, input_folder):
    inputs = glob(os.path.join(input_folder, 'train_benchmark*.json'))
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

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import the utils module
extra_path = os.path.join(os.path.split(__file__)[0], '..')
sys.path.append(extra_path)
import bench_util

ROOT = os.environ.get('GPU_BENCHMARK_ROOT', '/scratch/pape/gpu_benchmark')
DEFAULT_DATA_PATH = os.path.join(ROOT, 'sample_A_20160501.hdf')
DEFAULT_RAW = 'volumes/raw'
DEFAULT_LABELS = 'volumes/labels/clefts'

DEFAULT_CONFIG = os.path.join(ROOT, 'train_config.json')

PRECISIONS = ('single', 'half', 'mixed')


def train_benchmark(args):
    precision = args.precision
    assert precision in PRECISIONS

    # TODO allow special configs for gpus
    with open(args.config) as f:
        config = json.load(f)
    # check if we have a special config for this precision,
    # otherwise load the default one
    config = config.get(precision, config['default'])

    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    patch_size = config['patch_size']
    workers_per_batch = config['workers_per_batch']

    model = bench_util.UNet()
    if precision == 'half':
        dtype = torch.float16
        model = model.to(dtype=dtype)
    else:
        dtype = torch.float32

    loss = nn.BCEWithLogitsLoss()

    ds = bench_util.EMDataset(args.data, args.data,
                              args.raw, args.labels,
                              patch_size=patch_size,
                              raw_transform=bench_util.normalize,
                              target_transform=bench_util.seg_to_mask,
                              dtype=dtype)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=workers_per_batch*batch_size)

    print("Start training benchmark for", n_epochs, "epochs with epoch of length", len(loader))
    total_time, time_per_it = bench_util.train_loop(
        model, loader, loss, n_epochs,
        precision=precision
    )

    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    output_prefix = args.output
    gpu_type = torch.cuda.get_device_name(torch.device('cuda'))
    bench_results = {
        'batch_size': batch_size,
        'patch_size': patch_size,
        'n_epochs': n_epochs,
        'total_training_time': total_time,
        'time_per_iteration': time_per_it,
        'cuda_version': cuda_version,
        'pytorch_version': torch_version,
        'gpu_type': gpu_type,
        'precision': precision
    }

    res_dir = os.path.abspath(os.path.split(output_prefix)[0])
    os.makedirs(res_dir, exist_ok=True)

    with open(output_prefix + '-results.json', 'w') as f:
        json.dump(bench_results, f, indent=2, sort_keys=True)

    if args.save_model:
        torch.save(model, os.path.join(res_dir, 'model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', default=DEFAULT_DATA_PATH)
    parser.add_argument('--raw', '-r', default=DEFAULT_RAW)
    parser.add_argument('--labels', '-l', default=DEFAULT_LABELS)
    parser.add_argument('--output', '-o', default='train-benchmark')
    parser.add_argument('--config', '-c', default=DEFAULT_CONFIG)
    parser.add_argument('--precision', '-p', default='mixed')
    parser.add_argument('--save_model', default=0, type=int)

    args = parser.parse_args()
    train_benchmark(args)

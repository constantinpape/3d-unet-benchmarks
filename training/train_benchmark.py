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

# DEFAULT_DATA_PATH = '/g/kreshuk/data/benchmark/sample_A_20160501.hdf'
DEFAULT_DATA_PATH = '/scratch/pape/gpu_benchmark/sample_A_20160501.hdf'
DEFAULT_RAW = 'volumes/raw'
DEFAULT_LABELS = 'volumes/labels/clefts'

# this size roughly fills the 11 GB RAM of a 2080 TI
# DEFAULT_PATCH_SIZE = [64, 384, 384] for some reason this fits on the gpu6/7 nodes, but not on the cluster
DEFAULT_PATCH_SIZE = [48, 384, 384]
DEFAULT_BATCH_SIZE = 1
DEFAULT_WORKERS = 6


def train_benchmark(args):
    model = bench_util.UNet()

    loss = nn.BCEWithLogitsLoss()

    ds = bench_util.EMDataset(args.data, args.data,
                              args.raw, args.labels,
                              patch_size=args.patch_size,
                              raw_transform=bench_util.normalize,
                              target_transform=bench_util.seg_to_mask)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers_per_batch*args.batch_size)

    print("Start training benchmark for", args.n_epochs, "epochs with epoch of length", len(loader))
    total_time, time_per_it = bench_util.train_loop(
        model, loader, loss, args.n_epochs
    )

    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    output_prefix = args.output
    gpu_type = torch.cuda.get_device_name(torch.device('cuda'))
    bench_results = {
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'n_epochs': args.n_epochs,
        'total_training_time': total_time,
        'time_per_iteration': time_per_it,
        'cuda_version': cuda_version,
        'pytorch_version': torch_version,
        'gpu_type': gpu_type
    }

    res_dir = os.path.abspath(os.path.split(output_prefix)[0])
    os.makedirs(res_dir, exist_ok=True)

    with open(output_prefix + '-results.json', 'w') as f:
        json.dump(bench_results, f, indent=2, sort_keys=True)
    torch.save(model, output_prefix + '-model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', '-n', default=5, type=int)
    parser.add_argument('--data', '-d', default=DEFAULT_DATA_PATH)
    parser.add_argument('--raw', '-r', default=DEFAULT_RAW)
    parser.add_argument('--labels', '-l', default=DEFAULT_LABELS)
    parser.add_argument('--output', '-o', default='train-benchmark')
    parser.add_argument('--patch_size', '-c', default=DEFAULT_PATCH_SIZE, type=int, nargs=3)
    parser.add_argument('--batch_size', '-b', default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument('--workers_per_batch', '-w', default=DEFAULT_WORKERS, type=int)

    args = parser.parse_args()
    train_benchmark(args)

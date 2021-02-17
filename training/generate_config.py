import argparse
import os
import json

ROOT = os.environ.get('GPU_BENCHMARK_ROOT', '/scratch/pape/gpu_benchmark')
DEFAULT_OUT = os.path.join(ROOT, 'train_config.json')

# this patch size was chosen s.t. mixed precision training
# maxes out the memory of a 2080Ti.
# Note that this does not for single precision training the patch size needs to be lowered
# e.g. to [32, 384, 384]
DEFAULT_PATCH_SIZE = [48, 384, 384]
DEFAULT_BATCH_SIZE = 1
DEFAULT_WORKERS = 6
DEFAULT_EPOCHS = 5


def generate_train_config(args):
    assert args.config in ('default', 'single', 'half', 'mixed')

    if os.path.exists(args.output):
        with open(args.output) as f:
            config = json.load(f)
    else:
        config = {}

    config.update({
        args.config: {
            'n_epochs': args.n_epochs,
            'batch_size': args.batch_size,
            'patch_size': args.patch_size,
            'workers_per_batch': args.workers
        }
    })
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='default')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUT)
    parser.add_argument('--n_epochs', '-n', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--patch_size', type=int, nargs=3, default=DEFAULT_PATCH_SIZE)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    generate_train_config(args)

import argparse
import os
import json

ROOT = os.environ.get('GPU_BENCHMARK_ROOT', '/scratch/pape/gpu_benchmark')
DEFAULT_OUT = os.path.join(ROOT, 'inference_config.json')

# this block size was chosen s.t. half precision inference
# maxes out the memory of a 2080Ti.
DEFAULT_BLOCK_SIZE = [32, 384, 384]
DEFAULT_HALO = [16, 64, 64]


def generate_inference_config(args):
    assert args.config in ('default', 'single', 'half')

    if os.path.exists(args.output):
        with open(args.output) as f:
            config = json.load(f)
    else:
        config = {}

    config.update({
        args.config: {
            'block_size': args.block_size,
            'halo': args.halo
        }
    })
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='default')
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUT)
    parser.add_argument('--block_size', type=int, nargs=3, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument('--halo', type=int, nargs=3, default=DEFAULT_HALO)
    args = parser.parse_args()
    generate_inference_config(args)

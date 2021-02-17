import argparse
from run_slurm import main


class Args:
    def __init__(self, use_conda, gpu_type, precision, use_easybuild_torch):
        self.use_conda = use_conda
        self.gpu_type = gpu_type
        self.precision = precision

        self.n_gpus = 1
        self.use_easybuild_torch = use_easybuild_torch


def run_all(use_conda, use_easybuild_torch):
    gpu_types = ('2080Ti', 'V100', 'A100', '3090', '1080Ti')
    precisions = ('single', 'half')

    for gpu in gpu_types:
        for prec in precisions:
            args = Args(use_conda, gpu, prec, use_easybuild_torch)
            main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_conda', '-c', default=0)
    parser.add_argument('--use_easybuild_torch', '-e', default=0)
    args = parser.parse_args()
    run_all(args.use_conda, args.use_easybuild_torch)

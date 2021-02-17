import argparse
from run_slurm import main


class Args:
    def __init__(self, use_conda, gpu_type, precision, save_model,
                 use_easybuild_torch):
        self.use_conda = use_conda
        self.gpu_type = gpu_type
        self.precision = precision
        self.save_model = save_model

        self.n_gpus = 1
        self.use_easybuild_torch = use_easybuild_torch


def run_all(use_conda):
    gpu_types = ('2080Ti', 'V100', 'A100', '3090', '1080Ti')
    precisions = ('single', 'half', 'mixed')

    run_id = 0
    for gpu in gpu_types:
        for prec in precisions:
            # only save the model once
            save_model = int(run_id == 0)
            args = Args(use_conda, gpu, prec, save_model=save_model,
                        use_easybuild_torch=use_easybuild_torch)
            main(args)
            run_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_conda', '-c', default=0)
    parser.add_argument('--use_easybuild_torch', '-e', default=0)
    args = parser.parse_args()
    run_all(args.use_conda, args.use_easybuild_torch)

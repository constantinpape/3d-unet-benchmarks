import argparse
import json
import os
import stat
import subprocess

ROOT = os.environ.get('GPU_BENCHMARK_ROOT', '/scratch/pape/gpu_benchmark')
CONFIG = os.path.join(ROOT, 'train_config.json')
TMP = os.path.join(ROOT, 'slurm')
RES = os.path.join(ROOT, 'results')


# run benchmark from the current conda environment
def generate_batchscript_conda(
    script_file,
    gpu_type,
    output,
    n_gpus,
    use_easybuild_torch,
    precision,
    save_model
):
    with open(CONFIG) as f:
        config = json.load(f)
    config = config.get(precision, config['default'])
    n_workers = config['workers_per_batch']
    n_cores = n_workers * n_gpus
    mem = 12 * n_gpus

    this_path = os.path.abspath(os.path.split(__file__)[0])
    script_path = os.path.join(this_path, 'benchmark.py')
    time_lim = '0-1:00'

    # use a conda env with normal torch setup
    def _default_torch(f):
        env_name = os.environ['CONDA_DEFAULT_ENV']
        f.write("module purge\n")
        f.write("module load GCC\n")
        f.write(f"source activate {env_name}\n")

    # use a conda env which has the easybuild torch linked
    # see https://github.com/kreshuklab/gpu_envs for details
    def _easybuild_torch(f):
        f.write("source activate_gpu_env\n")

    with open(script_file, 'w') as f:
        f.write("#! /bin/bash\n")
        f.write("#SBATCH -A kreshuk\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -c %i\n" % n_cores)
        f.write("#SBATCH --mem %iG\n" % mem)
        f.write("#SBATCH -t %s\n" % time_lim)
        f.write("#SBATCH -p gpu\n")
        f.write("#SBATCH -C gpu=%s\n" % gpu_type)
        f.write("#SBATCH --gres=gpu:%i\n" % n_gpus)
        f.write("\n")
        if use_easybuild_torch:
            _easybuild_torch(f)
        else:
            _default_torch(f)
        f.write(f"export GPU_BENCHMARK_ROOT={ROOT} \n")
        f.write(f"python {script_path} -o {output} -p {precision}\n")
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)


# run benchmark with pytorch from easybuild
def generate_batchscript_easybuild(script_file, gpu_type, output,
                                   n_gpus, precision, save_model):

    with open(CONFIG) as f:
        config = json.load(f)
    config = config.get(precision, config['default'])
    n_workers = config['workers_per_batch']
    n_cores = n_workers * n_gpus
    mem = 12 * n_gpus

    this_path = os.path.abspath(os.path.split(__file__)[0])
    script_path = os.path.join(this_path, 'benchmark.py')
    time_lim = '0-1:00'

    with open(script_file, 'w') as f:
        f.write("#! /bin/bash\n")
        f.write("#SBATCH -A kreshuk\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -c %i\n" % n_cores)
        f.write("#SBATCH --mem %iG\n" % mem)
        f.write("#SBATCH -t %s\n" % time_lim)
        f.write("#SBATCH -p gpu\n")
        f.write("#SBATCH -C gpu=%s\n" % gpu_type)
        f.write("#SBATCH --gres=gpu:%i\n" % n_gpus)
        f.write("\n")
        f.write("module purge\n")
        f.write("module load PyTorch h5py\n")
        f.write(f"export GPU_BENCHMARK_ROOT={ROOT} \n")
        f.write(f"python {script_path} -o {output} -p {precision} --save_model {save_model}\n")
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)


def main(args):
    gpu_type = args.gpu_type
    assert gpu_type in ('2080Ti', 'V100', 'A100', '3090', '1080Ti')

    n_gpus = args.n_gpus
    assert n_gpus == 1

    tmp_dir = TMP
    res_dir = RES
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    script_file = os.path.join(tmp_dir, f"train_benchmark_{gpu_type}.batch")
    output = os.path.join(res_dir, f"train_benchmark_{gpu_type}_{args.precision}")

    if bool(args.use_conda):
        output += '-conda'
        generate_batchscript_conda(script_file, gpu_type, output,
                                   n_gpus=n_gpus,
                                   use_easybuild_torch=bool(args.use_easybuild_torch),
                                   precision=args.precision,
                                   save_model=args.save_model)
    else:
        output += '-easybuild'
        generate_batchscript_easybuild(script_file, gpu_type, output,
                                       n_gpus=n_gpus,
                                       precision=args.precision,
                                       save_model=args.save_model)
    subprocess.run(['sbatch', script_file])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_type', type=str)
    parser.add_argument('--use_conda', '-c', type=int, default=0)
    parser.add_argument('--n_gpus', '-n', type=int, default=1)
    parser.add_argument('--use_easybuild_torch', '-e', type=int, default=0)
    parser.add_argument('--precision', '-p', default='mixed')
    parser.add_argument('--save_model', default=0, type=int)
    args = parser.parse_args()

    main(args)

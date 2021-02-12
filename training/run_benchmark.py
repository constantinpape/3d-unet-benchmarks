import argparse
import os
import stat
import subprocess

DEFAULT_ROOT = '/scratch/pape/gpu_benchmark'
DEFAULT_TMP = os.path.join(DEFAULT_ROOT, 'slurm')
DEFAULT_RES = os.path.join(DEFAULT_ROOT, 'results')


# run benchmakr from the current conda environment
def generate_batchscript_conda(script_file, gpu_type, output,
                               qos='normal', n_gpus=1):
    # allocate 6 cores per gpu
    n_cores = 6 * n_gpus
    # allocate 12 GB per gpu
    mem = 12 * n_gpus

    this_path = os.path.abspath(os.path.split(__file__)[0])
    script_path = os.path.join(this_path, 'train_benchmark.py')
    time_lim = '0-1:00'

    env_name = os.environ['CONDA_DEFAULT_ENV']
    with open(script_file, 'w') as f:
        f.write("#! /bin/bash\n")
        f.write("#SBATCH -A kreshuk\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -n %i\n" % n_cores)
        f.write("#SBATCH --mem %iG\n" % mem)
        f.write("#SBATCH -t %s\n" % time_lim)
        f.write("#SBATCH -p gpu\n")
        f.write("#SBATCH -C gpu=%s\n" % gpu_type)
        f.write("#SBATCH --gres=gpu:%i\n" % n_gpus)
        f.write(f"#SBATCH --qos={qos}\n")
        f.write("\n")
        f.write("module purge\n")
        f.write(f"module load GCC\n")
        f.write(f"source activate {env_name}\n")
        f.write(f"python {script_path} -w 6 -o {output}\n")
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)


# run benchmark with pytorch from easybuild
def generate_batchscript_easybuild(script_file, gpu_type, output, n_gpus=1):
    # allocate 6 cores per gpu
    n_cores = 6 * n_gpus
    # allocate 12 GB per gpu
    mem = 12 * n_gpus

    this_path = os.path.abspath(os.path.split(__file__)[0])
    script_path = os.path.join(this_path, 'train_benchmark.py')
    time_lim = '0-1:00'

    with open(script_file, 'w') as f:
        f.write("#! /bin/bash\n")
        f.write("#SBATCH -A kreshuk\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -n %i\n" % n_cores)
        f.write("#SBATCH --mem %iG\n" % mem)
        f.write("#SBATCH -t %s\n" % time_lim)
        f.write("#SBATCH -p gpu\n")
        f.write("#SBATCH -C gpu=%s\n" % gpu_type)
        f.write("#SBATCH --gres=gpu:%i\n" % n_gpus)
        f.write("\n")
        f.write("module purge\n")
        f.write("module load PyTorch h5py\n")
        f.write(f"python {script_path} -w 6 -o {output}\n")
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_type', type=str)
    parser.add_argument('--use_conda', '-c', type=int, default=1)
    parser.add_argument('--n_gpus', '-n', type=int, default=1)
    parser.add_argument('--tmp_dir', '-t', type=str, default=DEFAULT_TMP)
    parser.add_argument('--res_dir', '-r', type=str, default=DEFAULT_RES)
    args = parser.parse_args()

    gpu_type = args.gpu_type
    assert gpu_type in ('2080Ti', 'V100', 'A100', '3090', '1080Ti')

    n_gpus = args.n_gpus
    assert n_gpus == 1

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.res_dir, exist_ok=True)

    script_file = os.path.join(args.tmp_dir, f"train_benchmark_{gpu_type}.batch")
    output = os.path.join(args.res_dir, f"train_benchmark_{gpu_type}")

    if bool(args.use_conda):
        output += '-conda'
        generate_batchscript_conda(script_file, gpu_type, output,
                                   qos='normal',
                                   n_gpus=n_gpus)
    else:
        output += '-easybuild'
        generate_batchscript_easybuild(script_file, gpu_type, output, n_gpus)
    subprocess.run(['sbatch', script_file])


if __name__ == '__main__':
    main()

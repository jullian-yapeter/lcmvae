import glob, os
try:
    os.mkdir("./slurm_outputs")
    os.mkdir("./slurm_jobs")
except:
    pass
exp_list = list(map(os.path.basename, glob.glob('experiment_configs/*.py')))
exp_list = [x[:-3] for x in exp_list]
for experiment in exp_list:
    with open(f"/project/schweigh_422/DOSE_project/lcmvae/slurm_jobs/{experiment}", 'w+') as f:   
        f.write(f"#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=24GB\n#SBATCH --time=24:00:00\n#SBATCH --account=schweigh_422\n"
                "#SBATCH --output=/project/schweigh_422/DOSE_project/lcmvae/slurm_outputs/%x_%j.out"
                "\nexport TMPDIR=/scratch1/dongzeye/tmp\ncd /project/schweigh_422/DOSE_project/lcmvae/\n\nmodule purge\nmodule load gcc/11.2.0 nvidia-hpc-sdk/21.7\n\n"
                f"eval \"$(conda shell.bash hook)\"\nconda activate /home1/dongzeye/.conda/envs/lcmvae\n\ndate\npython -u main.py experiment_configs/{experiment}.py\ndate\n")

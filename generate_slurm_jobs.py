import glob, os
exp_list = list(map(os.path.basename, glob.glob('experiment_configs/*.py')))
exp_list = [x[:-3] for x in exp_list]
for experiment in exp_list:
    gpu_type = "v100" if ("noCap" in experiment) and ("noPreC" in experiment) else 'a40'
    if ("base" in experiment) or ("noMask" in experiment):
        gpu_type = 'a100' 
    with open(f"/scratch1/dongzeye/lcmvae/slurm_jobs/{experiment}", 'w+') as f:   
        f.write(f"#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:{gpu_type}:1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=7\n#SBATCH --mem=24GB\n#SBATCH --time=40:00:00\n#SBATCH --account=xiangren_818\n"
                "#SBATCH --output=/scratch1/dongzeye/lcmvae/slurm_jobs/outputs/%x_%j.out"
                "\nexport TMPDIR=/scratch1/dongzeye/tmp\ncd /scratch1/dongzeye/lcmvae\n\nmodule purge\nmodule load gcc/11.2.0 nvidia-hpc-sdk/21.7\n\n"
                f"eval \"$(conda shell.bash hook)\"\nconda activate /home1/dongzeye/.conda/envs/lcmvae\n\ndate\npython -u main.py experiment_configs/{experiment}.py\ndate\n")

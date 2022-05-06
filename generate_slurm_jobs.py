import glob, os, sys
try:
    os.mkdir("./slurm_outputs")
    os.mkdir("./slurm_jobs")
except:
    pass

SEG = ('seg' in sys.argv[1])
if SEG:
    exp_list = list(map(os.path.basename, glob.glob('saved_models/*0505*')))
    exp_list = [x[:-3] for x in exp_list]
else: 
    exp_list = list(map(os.path.basename, glob.glob('experiment_configs/*.py')))
    exp_list = [x[:-3] for x in exp_list]
for experiment in exp_list:
    with open(f"/project/schweigh_422/DOSE_project/lcmvae/slurm_jobs/{'seg_' if SEG else ''}{experiment}", 'w+') as f:  
        gpu = 'a100' if 'base' in experiment \
            else ('a40' if 'noMask' in experiment else 'v100')
        exp_config = f"saved_models/{experiment}/params_{experiment}" if SEG else f"experiment_configs/{experiment}.py" 
        f.write(f"#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:{gpu}:1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=16\n#SBATCH --mem=24GB\n#SBATCH --time=30:00:00\n#SBATCH --account=schweigh_422\n"
                "#SBATCH --output=/project/schweigh_422/DOSE_project/lcmvae/slurm_outputs/%x_%j.out"
                "\nexport TMPDIR=/scratch1/dongzeye/tmp\ncd /project/schweigh_422/DOSE_project/lcmvae/\n\nmodule purge\nmodule load gcc/11.2.0 nvidia-hpc-sdk/21.7\n\n"
                f"eval \"$(conda shell.bash hook)\"\nconda activate /home1/dongzeye/.conda/envs/lcmvae\n\ndate\n"
                f"python -u {sys.argv[1]} {exp_config}\ndate\n")

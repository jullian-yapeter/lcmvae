import glob, os, sys

try:
    os.mkdir("slurm_outputs")
    os.mkdir("slurm_jobs")
except:
    pass

for mask_type in ['Patch', 'Pixel']:
    for noVar in [True, False]:
        with open(f"slurm_jobs/svae_{mask_type}{'_noVar' if noVar else ''}", 'w+') as f:
            f.write(f"#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=16\n#SBATCH --mem=24GB\n#SBATCH --time=30:00:00\n#SBATCH --account=schweigh_422\n"
                    "#SBATCH --output=/project/schweigh_422/DOSE_project/lcmvae/slurm_outputs/%x_%j.out"
                    "\nexport TMPDIR=/scratch1/dongzeye/tmp\ncd /project/schweigh_422/DOSE_project/lcmvae/\n\nmodule purge\nmodule load gcc/11.2.0 nvidia-hpc-sdk/21.7\n\n"
                    f"eval \"$(conda shell.bash hook)\"\nconda activate /home1/dongzeye/.conda/envs/lcmvae\n\ndate\n"
                    f"python -u vae_baseline_script.py svae_{mask_type}{'_noVar' if noVar else ''} {mask_type} {'noVar' if noVar else ' '}\ndate\n"
                    )

# for mask_ratio in [0, 0.25, 0.5, 0.75]:
#     with open(f"slurm_jobs/rec_losses_{int(mask_ratio*100)}", 'w+') as f:  
#         f.write(f"#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:v100:1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=16\n#SBATCH --mem=24GB\n#SBATCH --time=30:00:00\n#SBATCH --account=schweigh_422\n"
#                 "#SBATCH --output=/project/schweigh_422/DOSE_project/lcmvae/slurm_outputs/%x_%j.out"
#                 "\nexport TMPDIR=/scratch1/dongzeye/tmp\ncd /project/schweigh_422/DOSE_project/lcmvae/\n\nmodule purge\nmodule load gcc/11.2.0 nvidia-hpc-sdk/21.7\n\n"
#                 f"eval \"$(conda shell.bash hook)\"\nconda activate /home1/dongzeye/.conda/envs/lcmvae\n\ndate\n"
#                 f"python -u get_rec_losses.py {mask_ratio} \ndate\n"
#                 )


# SEG = ('seg' in sys.argv[1])
# if SEG:
#     exp_list = list(map(os.path.basename, glob.glob('saved_models/*0505*')))
# else: 
#     exp_list = list(map(os.path.basename, glob.glob('experiment_configs/*.py')))
#     exp_list = [x[:-3] for x in exp_list]
# for experiment in exp_list:
#     with open(f"slurm_jobs/{'seg_' if SEG else ''}{experiment}", 'w+') as f:  
#         gpu = 'a100' if 'base' in experiment \
#             else ('a40' if 'noMask' in experiment else 'v100')
#         exp_config = f'experiment_configs/{experiment}.py'
#         f.write(f"#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:{gpu}:1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=16\n#SBATCH --mem=24GB\n#SBATCH --time=30:00:00\n#SBATCH --account=schweigh_422\n"
#                 "#SBATCH --output=/project/schweigh_422/DOSE_project/lcmvae/slurm_outputs/%x_%j.out"
#                 "\nexport TMPDIR=/scratch1/dongzeye/tmp\ncd /project/schweigh_422/DOSE_project/lcmvae/\n\nmodule purge\nmodule load gcc/11.2.0 nvidia-hpc-sdk/21.7\n\n"
#                 f"eval \"$(conda shell.bash hook)\"\nconda activate /home1/dongzeye/.conda/envs/lcmvae\n\ndate\n"
#                 f"python -u {sys.argv[1]} {experiment if SEG else exp_config} \ndate\n"
#                )
#!/bin/bash 
#SBATCH --job-name=exp_misr_joint_ldm_lcc_train_clip_diffsat_fz_mtd_sf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/exp_misr_joint_ldm_lcc_train_clip_diffsat_fz_mtd_sf_%j.out
#SBATCH --error logs/exp_misr_joint_ldm_lcc_train_clip_diffsat_fz_mtd_sf_%j.err
source load_modules.sh
export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$BIGWORK
conda activate /software/NHGN20600/nhgnkany/flair_venv
which python
cd $HOME/MISR_JOINT_LDM_LCC_CLIP_DiffSat_FZ_MTD_SF
srun python trainer.py --config configs/diffsr_maxvit_ltae.yaml --config_file flair-config-server.yml --exp_name srdiff_maxvit_ltae_ckpt --reset



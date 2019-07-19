#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=field1-9Traces
#SBATCH --mail-type=END
#SBATCH --mail-user=jds814@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --array=1-6,8,9

h5s=(/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_01_D2019_7_15T10_38_12_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_02_D2019_7_15T10_57_35_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_03_D2019_7_15T11_23_50_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_04_D2019_7_15T11_43_38_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_05_D2019_7_15T12_0_46_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_06_D2019_7_15T12_21_8_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_07_D2019_7_15T12_42_6_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_08_D2019_7_15T12_57_34_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_09_D2019_7_15T13_0_13_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_10_D2019_7_15T13_19_15_odor.h5
/gpfs/scratch/jds814/2P-data/HN1953/190715/1953_1_11_D2019_7_15T13_37_15_odor.h5)


cd /gpfs/home/jds814/Git/ROI_Functions

pipenv run ./analyzeSession.py traces --h5 ${h5s[$SLURM_ARRAY_TASK_ID]} -T /gpfs/scratch/nakayh01/2P_data/HN1953/190715/aligned/HN1953_190715_field${SLURM_ARRAY_TASK_ID}_00001_000*.tif -M /gpfs/scratch/jds814/2P-data/HN1953/190715/aligned/190715_field${SLURM_ARRAY_TASK_ID}_masks/*.bmp --saveDir /gpfs/scratch/jds814/2P-data/HN1953/190715/figures/field$SLURM_ARRAY_TASK_ID --savePrefix 190715_field$SLURM_ARRAY_TASK_ID -A

exit

#!/bin/bash -vx

echo 'downloading pascal'
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf VOCtrainval_11-May-2012.tar

# pip3.9 install gdown 
!pip install gdown #python 3.8 at least 

echo 'downloading segmentation class of pascal'
!gdown 10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2
!unzip SegmentationClassAug.zip

# mkdir ./VOCdevkit/VOC2012/
!mv ./SegmentationClassAug ./VOCdevkit/VOC2012/

!mkdir Datasets_PATNET
!mv ./VOCdevkit ./Datasets_PATNET

echo 'downloading Fss dataset'
!gdown 16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI
!unzip fewshot_data.zip

!mv ./fewshot_data/* ./Datasets_PATNET
!rm -r ./fewshot_data
!mv ./Datasets_PATNET/fewshot_data ./Datasets_PATNET/FSS-1000


python train.py --backbone resnet50  --fold 4  --benchmark pascal --lr 1e-3 --bsz 20 --logpath "my-logs"

https://pytorch.org/get-started/previous-versions/



pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

x = torch.tensor([1,2]).cuda()
torch.cuda.device_count()


pip3 install torch torchvision
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113




#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-menna
#SBATCH --gpus-per-node=v100:1      # Request GPU "generic resources"
#SBATCH --cpus-per-task=16           # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=5G                 # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=00:05:00              # hh:mm:ss
#SBATCH --job-name=nvidia_smi
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# Run your simulation step here...

export CUDA_AVAILABLE_DEVICES=0  # 0,1 if you want to use say 2 gpus you ask for
module load python/3.9                        # load the version of your choice
virtualenv --no-download ~/env                # To create a virtual environment, where ENV is the name of the environment
source ~/env/bin/activate
pip3 install torch torchvision
pip install --upgrade pip

pip3 install --no-index --upgrade pip          # You should also upgrade pip
pip3 install --no-index --upgrade setuptools   # You should also upgrade setuptools

python
import torch 
x = torch.tensor([1,2]).cuda()
#dataset=$1
#python ~/scratch/my_job.py  #--dataset $dataset

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------

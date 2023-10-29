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


pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


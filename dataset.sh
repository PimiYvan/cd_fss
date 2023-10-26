#!/bin/bash

echo 'downloading pascal'
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf VOCtrainval_11-May-2012.tar

!pip install gdown

echo 'downloading segmentation class of pascal'
!gdown 10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2
!unzip SegmentationClassAug.zip

!mv SegmentationClassAug ./VOCdevkit/VOC2012/

!mkdir Datasets_PATNET
!mv ./VOCdevkit ./Datasets_PATNET

echo 'downloading Fss dataset'
!gdown 16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI
!unzip fewshot_data.zip

!mv ./fewshot_data/* ./Datasets_PATNET
!rm -r ./fewshot_data
!mv ./Datasets_PATNET/fewshot_data ./Datasets_PATNET/FSS-1000


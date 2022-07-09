# Data Efficient 3D Learner via Knowledge Transferred from 2D Model 
This is the implementation of our ECCV'22 "[Data Efficient 3D Learner via Knowledge Transferred from 2D Model](https://arxiv.org/abs/2203.08479)"

## Overview
We use a strong and well-trained image scene parser to augment single view RGB-D datasets with pseudo-labels, which is used to pre-train a 3D model in an architecture agnostic manner. Our pre-training improves the results of the limited annotation training.
![](https://i.imgur.com/i4NsezB.png)



## Preparation & Installation 

### Dataset 
Download the 3D dataset [ScanNet](http://www.scan-net.org/) and pre-process by following command 
``python tools/scannet.py --run process_scannet --path_in <scannet_folder> --path_out data/scannet``

You can download the filelist to split train, valid, and test subset by command 

``python tools/scannet.py --run download_filelists``

### Pre-traind model weights
We utilize RGB-D dataset [Matterport3D](https://niessner.github.io/Matterport/) to pre-train 3D models. The pre-trained weight is provided in [link](https://drive.google.com/file/d/189dr0abasreJrO_LoidDz_niNNLqfmij/view?usp=sharing). To fine-tune or evaluate the following cases, the pre-trained weight needs to be downloaded and placed in ``ckpt``

### Installation
The codes have been run on Ubuntu20 and PyTorch1.7. 
```shell
conda create --name 3DLearner python=3.7
conda activate 3DLearner``
conda install -c conda-forge cudatoolkit-dev==10.2
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
```

```shell
sudo apt-get install openexr
sudo apt-get install libopenexr-dev
pip install -r requirement.txt
```

Follow [O-CNN](https://github.com/microsoft/O-CNN/blob/master/docs/installation.md) tutorial to install ``ocnn`` package, or run the commands below: 

```shell
git clone https://github.com/microsoft/O-CNN.git
cd octree/external && git clone --recursive https://github.com/wang-ps/octree-ext.git
cd .. && mkdir build && cd build
cmake ..  && cmake --build . --config Release
export PATH=`pwd`:$PATH
```

```shell
cd pytorch
python setup.py install --build_octree
python -W ignore test/test_all.py -v
```

## Training and Testing 

### Fine-tuning 
#### Limited Annotations training
``python LA_segmentation.py --config config/LA_scannet.yaml``
#### Limited Recontructions training
``python LR_segmentation.py --config config/LR_scannet.yaml``


- Key argument in yaml file

    - **train.semi**: If True, use the unlabeled data for semi-supervied training. 
    - **train.limit**: Points of LA configuration
    - **train.limited_reconst**: File-lists of LR configuration 

### Evaluation 
To evaluate our pretraining model on validation set, we provide the trained model weights [here](https://drive.google.com/drive/folders/1JGXG6k5yQDc0s9HohfJpq9T8aC-Xt7lZ?usp=sharing). You can replace ``SOLVER.ckpt`` with the directory of checkpoint file in ``config/scannet_val.yaml``. 

`` python segmentation.py --config config/scannet_val.yaml``

## Customized Pre-training
You can choose any RGB-D dataset as you want, and all you need is to convert the image type into point-cloud forms by pre-processing``tool/preprocess.py``. We provide the soft-label pre-training scripts by following command.

### Data preparation 
Download RGB-D dataset [Matterport3D document]((https://niessner.github.io/Matterport/)) and pre-process by following command 
``python tools/mp3d.py --run process_mp3d --path_in <mp3d_folder> --path_out data/mp3d``

### Image scene parser 
Download the well-trained scene parser weights in [DPT repository](https://github.com/isl-org/DPT). We utilize [dpt_hybrid-ade20k-53898607.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-ade20k-53898607.pt) in our paper.  

### Pre-training
``python softlabel_segmentation --config config/mp3d.yaml``

## Acknowledgement
Many thanks to these practical open source projects:
- [O-CNN](https://github.com/microsoft/O-CNN)
- [DPT](https://github.com/isl-org/DPT)
# Multi-GPU training tutorial based on PyTorch

## Environment Installation

### 1. GPU environment
Ubuntu 18.04 \
CUDA 10.1 \
Python==3.7.3 \
PyTorch==1.8.1

### 2. Create a new conda environment
conda create -n pytorch-multi-GPU-training-tutorial python=3.7.3\
conda activate pytorch-multi-GPU-training-tutorial

### 3. Download some packages
pip install https://download.pytorch.org/whl/cu101/torch-1.8.1%2Bcu101-cp37-cp37m-linux_x86_64.whl \
pip install https://download.pytorch.org/whl/cu101/torchvision-0.9.1%2Bcu101-cp37-cp37m-linux_x86_64.whl

### 4. Run demo

To run the tutorial, please click the link below:

1. Run [single-machine-and-single-GPU.py](https://github.com/HongxinXiang/pytorch-multi-GPU-training-tutorial/blob/master/RUN.md#run-with-single-machine-and-multi-gpu-dataparallelpy)

2. Run [single-machine-and-multi-GPU-DataParallel.py](https://github.com/HongxinXiang/pytorch-multi-GPU-training-tutorial/blob/master/RUN.md#run-with-single-machine-and-single-gpupy)

3. Run [single-machine-and-multi-GPU-DistributedDataParallel-launch.py](https://github.com/HongxinXiang/pytorch-multi-GPU-training-tutorial/blob/master/RUN.md#run-with-single-machine-and-multi-gpu-distributeddataparallel-launchpy)

4. Run [single-machine-and-multi-GPU-DistributedDataParallel-launch.py](https://github.com/HongxinXiang/pytorch-multi-GPU-training-tutorial/blob/master/RUN.md#run-with-single-machine-and-multi-gpu-distributeddataparallel-launchpy)

# Multi-GPU training tutorial based on PyTorch

## News!

[2022/07/23] Chinese blogs are starting to be updated on [闪闪红星闪闪@知乎](https://www.zhihu.com/people/xiang-hong-xin-6)

[2022/07/20] Repository installation completed



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

4. Run [single-machine-and-multi-GPU-DistributedDataParallel-mp.py](https://github.com/HongxinXiang/pytorch-multi-GPU-training-tutorial/blob/master/RUN.md#run-with-single-machine-and-multi-gpu-distributeddataparallel-mppy)



## Chinese Tutorial ([闪闪红星闪闪@知乎](https://www.zhihu.com/people/xiang-hong-xin-6))
The following tutorials are being published: 
1. [PyTorch 多GPU训练实践 (1) - 单机单 GPU](https://zhuanlan.zhihu.com/p/542584557)
2. [PyTorch 多GPU训练实践 (2) - DP 代码修改](https://zhuanlan.zhihu.com/p/542622592)
3. PyTorch 多GPU训练实践 (3) - DDP 入门
4. PyTorch 多GPU训练实践 (4) - DDP 进阶
5. PyTorch 多GPU训练实践 (5) - DDP-torch.distributed.launch 代码修改
6. PyTorch 多GPU训练实践 (6) - DDP-torch.multiprocessing 代码修改
7. PyTorch 多GPU训练实践 (7) - slurm 集群安装
8. PyTorch 多GPU训练实践 (8) - DDP- slurm 代码修改
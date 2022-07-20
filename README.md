# Multi-GPU training tutorial based on PyTorch

## Environment Installation

#### 1. GPU environment
CUDA 10.1
Python==3.7.3
PyTorch==1.8.1


#### 2. create a new conda environment
conda create -n pytorch-multi-GPU-training-tutorial python=3.7.3\
conda activate pytorch-multi-GPU-training-tutorial

#### 3. download some packages



#### Run with single-machine-and-single-GPU.py
```bash
> CUDA_AVAILABLE_DEVICES=0 python single-machine-and-single-GPU.py
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
Epoch 1
-------------------------------
loss: 2.298414  [    0/60000]
loss: 2.282716  [ 6400/60000]
loss: 2.265407  [12800/60000]
loss: 2.265287  [19200/60000]
loss: 2.243809  [25600/60000]
loss: 2.229841  [32000/60000]
...
```

#### Run with single-machine-and-multi-GPU-DataParallel.py
```bash
> CUDA_AVAILABLE_DEVICES=0,1,2,3 python single-machine-and-multi-GPU-DataParallel.py

```

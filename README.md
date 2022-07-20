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
> CUDA_VISIBLE_DEVICES=0 python single-machine-and-single-GPU.py
Using cuda:0 device
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
loss: 2.301383  [    0/60000]
loss: 2.295246  [ 6400/60000]
loss: 2.276515  [12800/60000]
loss: 2.269020  [19200/60000]
loss: 2.255433  [25600/60000]
loss: 2.228429  [32000/60000]
loss: 2.239701  [38400/60000]
loss: 2.209971  [44800/60000]
loss: 2.211788  [51200/60000]
loss: 2.186936  [57600/60000]
Test Error: 
 Accuracy: 40.1%, Avg loss: 2.176335 

Epoch 2
-------------------------------
loss: 2.190498  [    0/60000]
loss: 2.180624  [ 6400/60000]
loss: 2.132775  [12800/60000]
loss: 2.141592  [19200/60000]
loss: 2.098609  [25600/60000]
loss: 2.047074  [32000/60000]
loss: 2.078812  [38400/60000]
loss: 2.011228  [44800/60000]
loss: 2.022019  [51200/60000]
loss: 1.951299  [57600/60000]
Test Error: 
 Accuracy: 53.9%, Avg loss: 1.944823 

Epoch 3
-------------------------------
loss: 1.983202  [    0/60000]
loss: 1.948037  [ 6400/60000]
loss: 1.844122  [12800/60000]
loss: 1.870956  [19200/60000]
loss: 1.765571  [25600/60000]
loss: 1.721192  [32000/60000]
loss: 1.745432  [38400/60000]
loss: 1.654244  [44800/60000]
loss: 1.677890  [51200/60000]
loss: 1.565888  [57600/60000]
Test Error: 
 Accuracy: 59.3%, Avg loss: 1.578995 

Epoch 4
-------------------------------
loss: 1.649577  [    0/60000]
loss: 1.605777  [ 6400/60000]
loss: 1.460091  [12800/60000]
loss: 1.516236  [19200/60000]
loss: 1.398032  [25600/60000]
loss: 1.399217  [32000/60000]
loss: 1.410613  [38400/60000]
loss: 1.345814  [44800/60000]
loss: 1.375249  [51200/60000]
loss: 1.267737  [57600/60000]
Test Error: 
 Accuracy: 62.4%, Avg loss: 1.292355 

Epoch 5
-------------------------------
loss: 1.374084  [    0/60000]
loss: 1.351198  [ 6400/60000]
loss: 1.185489  [12800/60000]
loss: 1.276066  [19200/60000]
loss: 1.152795  [25600/60000]
loss: 1.188806  [32000/60000]
loss: 1.201429  [38400/60000]
loss: 1.153891  [44800/60000]
loss: 1.187127  [51200/60000]
loss: 1.096417  [57600/60000]
Test Error: 
 Accuracy: 64.1%, Avg loss: 1.116014 

Done!
Saved PyTorch Model State to model.pth
```

#### Run with single-machine-and-multi-GPU-DataParallel.py
```bash
> CUDA_VISIBLE_DEVICES=0,1,2,3 python single-machine-and-multi-GPU-DataParallel.py
n_gpu: 4
DataParallel(
  (module): NeuralNetwork(
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (linear_relu_stack): Sequential(
      (0): Linear(in_features=784, out_features=512, bias=True)
      (1): ReLU()
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=10, bias=True)
    )
  )
)
Epoch 1
-------------------------------
loss: 2.309276  [    0/60000]
loss: 2.290961  [ 6400/60000]
loss: 2.278524  [12800/60000]
loss: 2.272659  [19200/60000]
loss: 2.253739  [25600/60000]
loss: 2.233879  [32000/60000]
loss: 2.235425  [38400/60000]
loss: 2.205171  [44800/60000]
loss: 2.200310  [51200/60000]
loss: 2.169638  [57600/60000]
Test Error: 
 Accuracy: 47.2%, Avg loss: 2.165897 

Epoch 2
-------------------------------
loss: 2.173666  [    0/60000]
loss: 2.161774  [ 6400/60000]
loss: 2.110973  [12800/60000]
loss: 2.131320  [19200/60000]
loss: 2.078964  [25600/60000]
loss: 2.024526  [32000/60000]
loss: 2.052748  [38400/60000]
loss: 1.970439  [44800/60000]
loss: 1.975696  [51200/60000]
loss: 1.909384  [57600/60000]
Test Error: 
 Accuracy: 57.7%, Avg loss: 1.903836 

Epoch 3
-------------------------------
loss: 1.928953  [    0/60000]
loss: 1.899612  [ 6400/60000]
loss: 1.783553  [12800/60000]
loss: 1.838050  [19200/60000]
loss: 1.723950  [25600/60000]
loss: 1.664515  [32000/60000]
loss: 1.696275  [38400/60000]
loss: 1.578931  [44800/60000]
loss: 1.612579  [51200/60000]
loss: 1.513682  [57600/60000]
Test Error: 
 Accuracy: 61.9%, Avg loss: 1.525344 

Epoch 4
-------------------------------
loss: 1.584913  [    0/60000]
loss: 1.551337  [ 6400/60000]
loss: 1.395768  [12800/60000]
loss: 1.487152  [19200/60000]
loss: 1.364975  [25600/60000]
loss: 1.348510  [32000/60000]
loss: 1.371712  [38400/60000]
loss: 1.277878  [44800/60000]
loss: 1.325432  [51200/60000]
loss: 1.229618  [57600/60000]
Test Error: 
 Accuracy: 63.4%, Avg loss: 1.253269 

Epoch 5
-------------------------------
loss: 1.329453  [    0/60000]
loss: 1.310928  [ 6400/60000]
loss: 1.138112  [12800/60000]
loss: 1.259237  [19200/60000]
loss: 1.132849  [25600/60000]
loss: 1.149312  [32000/60000]
loss: 1.177834  [38400/60000]
loss: 1.097068  [44800/60000]
loss: 1.144713  [51200/60000]
loss: 1.068396  [57600/60000]
Test Error: 
 Accuracy: 65.0%, Avg loss: 1.087378 

Done!
Saved PyTorch Model State to model.pth
```

#### Run with single-machine-and-multi-GPU-DistributedDataParallel-launch.py
**Machine 0 (master, IP: 192.168.1.105):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='192.168.1.105' --master_port='12345' single-machine-and-multi-GPU-DistributedDataParallel-launch.py

```

**Machine 1 (IP: 192.168.1.106):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr='192.168.1.105' --master_port='12345' single-machine-and-multi-GPU-DistributedDataParallel-launch.py

```
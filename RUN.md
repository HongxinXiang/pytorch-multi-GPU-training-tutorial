```python


> 
`````
# Run with single-machine-and-single-GPU.py
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

# Run with single-machine-and-multi-GPU-DataParallel.py
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

# Run with single-machine-and-multi-GPU-DistributedDataParallel-launch.py
We use 2 machines to run, the IP is 192.168.1.105 (master), 192.168.1.106. Each machine has 4 GPUs.

**Note:** In order to display more NCCL information, we can set it with the following script, 
which helps us to find the bug of DDP when writing code.
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

**Machine 0 (master, IP: 192.168.1.105):**
```bash
> CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='192.168.1.105' --master_port='12345' single-machine-and-multi-GPU-DistributedDataParallel-launch.py 
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
Using device: cuda:3
local rank: 3, global rank: 3, world size: 8
Using device: cuda:2
Using device: cuda:1
local rank: 1, global rank: 1, world size: 8
local rank: 2, global rank: 2, world size: 8
Using device: cuda:0
local rank: 0, global rank: 0, world size: 8
tesla-105:1475:1475 [0] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.105<0>
tesla-105:1475:1475 [0] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-105:1475:1475 [0] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-105:1475:1475 [0] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.105<0>
tesla-105:1475:1475 [0] NCCL INFO Using network Socket
NCCL version 2.7.8+cuda10.1
tesla-105:1477:1477 [1] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.105<0>
tesla-105:1477:1477 [1] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-105:1477:1477 [1] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-105:1477:1477 [1] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.105<0>
tesla-105:1477:1477 [1] NCCL INFO Using network Socket
tesla-105:1481:1481 [3] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.105<0>
tesla-105:1481:1481 [3] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-105:1481:1481 [3] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-105:1481:1481 [3] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.105<0>
tesla-105:1481:1481 [3] NCCL INFO Using network Socket
tesla-105:1480:1480 [2] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.105<0>
tesla-105:1480:1480 [2] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-105:1480:1480 [2] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-105:1480:1480 [2] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.105<0>
tesla-105:1480:1480 [2] NCCL INFO Using network Socket
tesla-105:1481:2165 [3] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 8/8/64
tesla-105:1481:2165 [3] NCCL INFO Trees [0] -1/-1/-1->3->2|2->3->-1/-1/-1 [1] -1/-1/-1->3->2|2->3->-1/-1/-1
tesla-105:1481:2165 [3] NCCL INFO Setting affinity for GPU 3 to ff,c00ffc00
tesla-105:1475:2146 [0] NCCL INFO Channel 00/02 :    0   1   2   3   4   5   6   7
tesla-105:1477:2150 [1] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 8/8/64
tesla-105:1477:2150 [1] NCCL INFO Trees [0] 2/4/-1->1->0|0->1->2/4/-1 [1] 2/-1/-1->1->0|0->1->2/-1/-1
tesla-105:1477:2150 [1] NCCL INFO Setting affinity for GPU 1 to 3ff003ff
tesla-105:1475:2146 [0] NCCL INFO Channel 01/02 :    0   1   2   3   4   5   6   7
tesla-105:1475:2146 [0] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 8/8/64
tesla-105:1480:2169 [2] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 8/8/64
tesla-105:1480:2169 [2] NCCL INFO Trees [0] 3/-1/-1->2->1|1->2->3/-1/-1 [1] 3/-1/-1->2->1|1->2->3/-1/-1
tesla-105:1480:2169 [2] NCCL INFO Setting affinity for GPU 2 to ff,c00ffc00
tesla-105:1475:2146 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1|-1->0->1/-1/-1 [1] 1/-1/-1->0->5|5->0->1/-1/-1
tesla-105:1475:2146 [0] NCCL INFO Setting affinity for GPU 0 to 3ff003ff
tesla-105:1481:2165 [3] NCCL INFO Channel 00 : 3[b1000] -> 4[18000] [send] via NET/Socket/0
tesla-105:1480:2169 [2] NCCL INFO Channel 00 : 2[af000] -> 3[b1000] via P2P/IPC
tesla-105:1477:2150 [1] NCCL INFO Channel 00 : 1[1a000] -> 2[af000] via direct shared memory
tesla-105:1475:2146 [0] NCCL INFO Channel 00 : 7[b1000] -> 0[18000] [receive] via NET/Socket/0
tesla-105:1480:2169 [2] NCCL INFO Channel 00 : 2[af000] -> 1[1a000] via direct shared memory
tesla-105:1475:2146 [0] NCCL INFO Channel 00 : 0[18000] -> 1[1a000] via P2P/IPC
tesla-105:1477:2150 [1] NCCL INFO Channel 00 : 4[18000] -> 1[1a000] [receive] via NET/Socket/0
tesla-105:1477:2150 [1] NCCL INFO Channel 00 : 1[1a000] -> 0[18000] via P2P/IPC
tesla-105:1475:2146 [0] NCCL INFO Channel 01 : 7[b1000] -> 0[18000] [receive] via NET/Socket/0
tesla-105:1475:2146 [0] NCCL INFO Channel 01 : 0[18000] -> 1[1a000] via P2P/IPC
tesla-105:1481:2165 [3] NCCL INFO Channel 00 : 3[b1000] -> 2[af000] via P2P/IPC
tesla-105:1477:2150 [1] NCCL INFO Channel 00 : 1[1a000] -> 4[18000] [send] via NET/Socket/0
tesla-105:1481:2165 [3] NCCL INFO Channel 01 : 3[b1000] -> 4[18000] [send] via NET/Socket/0
tesla-105:1480:2169 [2] NCCL INFO Channel 01 : 2[af000] -> 3[b1000] via P2P/IPC
tesla-105:1477:2150 [1] NCCL INFO Channel 01 : 1[1a000] -> 2[af000] via direct shared memory
tesla-105:1481:2165 [3] NCCL INFO Channel 01 : 3[b1000] -> 2[af000] via P2P/IPC
tesla-105:1480:2169 [2] NCCL INFO Channel 01 : 2[af000] -> 1[1a000] via direct shared memory
tesla-105:1481:2165 [3] NCCL INFO 2 coll channels, 2 p2p channels, 1 p2p channels per peer
tesla-105:1481:2165 [3] NCCL INFO comm 0x7fd634001060 rank 3 nranks 8 cudaDev 3 busId b1000 - Init COMPLETE
tesla-105:1475:2146 [0] NCCL INFO Channel 01 : 0[18000] -> 5[1a000] [send] via NET/Socket/0
tesla-105:1477:2150 [1] NCCL INFO Channel 01 : 1[1a000] -> 0[18000] via P2P/IPC
tesla-105:1477:2150 [1] NCCL INFO 2 coll channels, 2 p2p channels, 1 p2p channels per peer
tesla-105:1477:2150 [1] NCCL INFO comm 0x7f30b4001060 rank 1 nranks 8 cudaDev 1 busId 1a000 - Init COMPLETE
tesla-105:1480:2169 [2] NCCL INFO 2 coll channels, 2 p2p channels, 1 p2p channels per peer
tesla-105:1480:2169 [2] NCCL INFO comm 0x7f37d4001060 rank 2 nranks 8 cudaDev 2 busId af000 - Init COMPLETE
tesla-105:1475:2146 [0] NCCL INFO Channel 01 : 5[1a000] -> 0[18000] [receive] via NET/Socket/0
tesla-105:1475:2146 [0] NCCL INFO 2 coll channels, 2 p2p channels, 1 p2p channels per peer
tesla-105:1475:2146 [0] NCCL INFO comm 0x7f9e54001060 rank 0 nranks 8 cudaDev 0 busId 18000 - Init COMPLETE
tesla-105:1475:1475 [0] NCCL INFO Launch mode Parallel
DistributedDataParallel(
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
loss: 2.294374  [    0/60000]
loss: 2.301075  [  800/60000]
loss: 2.315739  [ 1600/60000]
loss: 2.299692  [ 2400/60000]
loss: 2.258646  [ 3200/60000]
loss: 2.252302  [ 4000/60000]
loss: 2.218223  [ 4800/60000]
loss: 2.126724  [ 5600/60000]
loss: 2.174220  [ 6400/60000]
loss: 2.177455  [ 7200/60000]
Test Error: 
 Accuracy: 4.1%, Avg loss: 2.166388 

Epoch 2
-------------------------------
loss: 2.136480  [    0/60000]
loss: 2.127040  [  800/60000]
loss: 2.118551  [ 1600/60000]
loss: 2.051364  [ 2400/60000]
loss: 2.076279  [ 3200/60000]
loss: 2.002108  [ 4000/60000]
loss: 2.075573  [ 4800/60000]
loss: 1.959522  [ 5600/60000]
loss: 1.861534  [ 6400/60000]
loss: 1.872814  [ 7200/60000]
Test Error: 
 Accuracy: 7.2%, Avg loss: 1.908959 

Epoch 3
-------------------------------
loss: 2.081742  [    0/60000]
loss: 1.841850  [  800/60000]
loss: 1.939971  [ 1600/60000]
loss: 1.684577  [ 2400/60000]
loss: 1.648371  [ 3200/60000]
loss: 1.774270  [ 4000/60000]
loss: 1.552769  [ 4800/60000]
loss: 1.508346  [ 5600/60000]
loss: 1.516589  [ 6400/60000]
loss: 1.481997  [ 7200/60000]
Test Error: 
 Accuracy: 7.8%, Avg loss: 1.533547 

Epoch 4
-------------------------------
loss: 1.625404  [    0/60000]
loss: 1.543570  [  800/60000]
loss: 1.428792  [ 1600/60000]
loss: 1.446484  [ 2400/60000]
loss: 1.841029  [ 3200/60000]
loss: 1.320562  [ 4000/60000]
loss: 1.511142  [ 4800/60000]
loss: 1.444456  [ 5600/60000]
loss: 1.570060  [ 6400/60000]
loss: 1.482602  [ 7200/60000]
Test Error: 
 Accuracy: 8.0%, Avg loss: 1.256674 

Epoch 5
-------------------------------
loss: 1.064455  [    0/60000]
loss: 1.233810  [  800/60000]
loss: 1.168940  [ 1600/60000]
loss: 1.227281  [ 2400/60000]
loss: 1.437644  [ 3200/60000]
loss: 1.195065  [ 4000/60000]
loss: 1.305991  [ 4800/60000]
loss: 1.258441  [ 5600/60000]
loss: 0.970569  [ 6400/60000]
loss: 1.698888  [ 7200/60000]
Test Error: 
 Accuracy: 8.2%, Avg loss: 1.083617 

Done!
Saved PyTorch Model State to model.pth
```

**Machine 1 (IP: 192.168.1.106):**
```bash
> CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr='192.168.1.105' --master_port='12345' single-machine-and-multi-GPU-DistributedDataParallel-launch.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
Using device: cuda:0
Using device: cuda:1

local rank: 1, global rank: 5, world size: 8
local rank: 0, global rank: 4, world size: 8
Using device: cuda:2
local rank: 2, global rank: 6, world size: 8
Using device: cuda:3
local rank: 3, global rank: 7, world size: 8
tesla-106:1942:1942 [1] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.106<0>
tesla-106:1942:1942 [1] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-106:1942:1942 [1] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-106:1942:1942 [1] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.106<0>
tesla-106:1942:1942 [1] NCCL INFO Using network Socket
tesla-106:1988:1988 [3] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.106<0>
tesla-106:1988:1988 [3] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-106:1988:1988 [3] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-106:1988:1988 [3] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.106<0>
tesla-106:1988:1988 [3] NCCL INFO Using network Socket
tesla-106:1943:1943 [2] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.106<0>
tesla-106:1943:1943 [2] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-106:1943:1943 [2] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-106:1943:1943 [2] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.106<0>
tesla-106:1943:1943 [2] NCCL INFO Using network Socket
tesla-106:1940:1940 [0] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.106<0>
tesla-106:1940:1940 [0] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-106:1940:1940 [0] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-106:1940:1940 [0] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.106<0>
tesla-106:1940:1940 [0] NCCL INFO Using network Socket
tesla-106:1988:2787 [3] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 8/8/64
tesla-106:1988:2787 [3] NCCL INFO Trees [0] -1/-1/-1->7->6|6->7->-1/-1/-1 [1] -1/-1/-1->7->6|6->7->-1/-1/-1
tesla-106:1988:2787 [3] NCCL INFO Setting affinity for GPU 3 to ff,c00ffc00
tesla-106:1943:2821 [2] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 8/8/64
tesla-106:1943:2821 [2] NCCL INFO Trees [0] 7/-1/-1->6->5|5->6->7/-1/-1 [1] 7/-1/-1->6->5|5->6->7/-1/-1
tesla-106:1943:2821 [2] NCCL INFO Setting affinity for GPU 2 to ff,c00ffc00
tesla-106:1942:2786 [1] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 8/8/64
tesla-106:1942:2786 [1] NCCL INFO Trees [0] 6/-1/-1->5->4|4->5->6/-1/-1 [1] 6/0/-1->5->4|4->5->6/0/-1
tesla-106:1942:2786 [1] NCCL INFO Setting affinity for GPU 1 to 3ff003ff
tesla-106:1940:2831 [0] NCCL INFO threadThresholds 8/8/64 | 64/8/64 | 8/8/64
tesla-106:1940:2831 [0] NCCL INFO Trees [0] 5/-1/-1->4->1|1->4->5/-1/-1 [1] 5/-1/-1->4->-1|-1->4->5/-1/-1
tesla-106:1942:2786 [1] NCCL INFO Channel 00 : 5[1a000] -> 6[af000] via direct shared memory
tesla-106:1940:2831 [0] NCCL INFO Setting affinity for GPU 0 to 3ff003ff
tesla-106:1943:2821 [2] NCCL INFO Channel 00 : 6[af000] -> 7[b1000] via P2P/IPC
tesla-106:1988:2787 [3] NCCL INFO Channel 00 : 7[b1000] -> 0[18000] [send] via NET/Socket/0
tesla-106:1940:2831 [0] NCCL INFO Channel 00 : 3[b1000] -> 4[18000] [receive] via NET/Socket/0
tesla-106:1940:2831 [0] NCCL INFO Channel 00 : 4[18000] -> 5[1a000] via P2P/IPC
tesla-106:1988:2787 [3] NCCL INFO Channel 00 : 7[b1000] -> 6[af000] via P2P/IPC
tesla-106:1942:2786 [1] NCCL INFO Channel 00 : 5[1a000] -> 4[18000] via P2P/IPC
tesla-106:1940:2831 [0] NCCL INFO Channel 00 : 4[18000] -> 1[1a000] [send] via NET/Socket/0
tesla-106:1940:2831 [0] NCCL INFO Channel 00 : 1[1a000] -> 4[18000] [receive] via NET/Socket/0
tesla-106:1988:2787 [3] NCCL INFO Channel 01 : 7[b1000] -> 0[18000] [send] via NET/Socket/0
tesla-106:1943:2821 [2] NCCL INFO Channel 00 : 6[af000] -> 5[1a000] via direct shared memory
tesla-106:1942:2786 [1] NCCL INFO Channel 01 : 5[1a000] -> 6[af000] via direct shared memory
tesla-106:1943:2821 [2] NCCL INFO Channel 01 : 6[af000] -> 7[b1000] via P2P/IPC
tesla-106:1988:2787 [3] NCCL INFO Channel 01 : 7[b1000] -> 6[af000] via P2P/IPC
tesla-106:1988:2787 [3] NCCL INFO 2 coll channels, 2 p2p channels, 1 p2p channels per peer
tesla-106:1988:2787 [3] NCCL INFO comm 0x7fbb14001060 rank 7 nranks 8 cudaDev 3 busId b1000 - Init COMPLETE
tesla-106:1943:2821 [2] NCCL INFO Channel 01 : 6[af000] -> 5[1a000] via direct shared memory
tesla-106:1940:2831 [0] NCCL INFO Channel 01 : 3[b1000] -> 4[18000] [receive] via NET/Socket/0
tesla-106:1940:2831 [0] NCCL INFO Channel 01 : 4[18000] -> 5[1a000] via P2P/IPC
tesla-106:1942:2786 [1] NCCL INFO Channel 01 : 0[18000] -> 5[1a000] [receive] via NET/Socket/0
tesla-106:1943:2821 [2] NCCL INFO 2 coll channels, 2 p2p channels, 1 p2p channels per peer
tesla-106:1943:2821 [2] NCCL INFO comm 0x7f6fec001060 rank 6 nranks 8 cudaDev 2 busId af000 - Init COMPLETE
tesla-106:1942:2786 [1] NCCL INFO Channel 01 : 5[1a000] -> 4[18000] via P2P/IPC
tesla-106:1940:2831 [0] NCCL INFO 2 coll channels, 2 p2p channels, 1 p2p channels per peer
tesla-106:1940:2831 [0] NCCL INFO comm 0x7f5550001060 rank 4 nranks 8 cudaDev 0 busId 18000 - Init COMPLETE
tesla-106:1942:2786 [1] NCCL INFO Channel 01 : 5[1a000] -> 0[18000] [send] via NET/Socket/0
tesla-106:1942:2786 [1] NCCL INFO 2 coll channels, 2 p2p channels, 1 p2p channels per peer
tesla-106:1942:2786 [1] NCCL INFO comm 0x7f75d4001060 rank 5 nranks 8 cudaDev 1 busId 1a000 - Init COMPLETE
```

# Run with single-machine-and-multi-GPU-DistributedDataParallel-mp.py
```bash
> CUDA_VISIBLE_DEVICES=0,1,2,3 python single-machine-and-multi-GPU-DistributedDataParallel-mp.py --nodes 1 --ngpus_per_node 4
Using device: cuda:3
local rank: 3, global rank: 3, world size: 4
Using device: cuda:1
Using device: cuda:2
local rank: 2, global rank: 2, world size: 4
Using device: cuda:0
local rank: 1, global rank: 1, world size: 4
local rank: 0, global rank: 0, world size: 4
tesla-106:13395:13395 [0] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.106<0>
tesla-106:13395:13395 [0] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-106:13395:13395 [0] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-106:13395:13395 [0] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.106<0>
tesla-106:13395:13395 [0] NCCL INFO Using network Socket
tesla-106:13398:13398 [3] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.106<0>
tesla-106:13397:13397 [2] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.106<0>
tesla-106:13397:13397 [2] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-106:13397:13397 [2] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-106:13397:13397 [2] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.106<0>
tesla-106:13398:13398 [3] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-106:13397:13397 [2] NCCL INFO Using network Socket
tesla-106:13398:13398 [3] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-106:13398:13398 [3] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.106<0>
tesla-106:13398:13398 [3] NCCL INFO Using network Socket
NCCL version 2.7.8+cuda10.1
tesla-106:13396:13396 [1] NCCL INFO Bootstrap : Using [0]eno2:192.168.1.106<0>
tesla-106:13396:13396 [1] NCCL INFO NET/Plugin : No plugin found (libnccl-net.so), using internal implementation

tesla-106:13396:13396 [1] misc/ibvwrap.cc:63 NCCL WARN Failed to open libibverbs.so[.1]
tesla-106:13396:13396 [1] NCCL INFO NET/Socket : Using [0]eno2:192.168.1.106<0>
tesla-106:13396:13396 [1] NCCL INFO Using network Socket
tesla-106:13395:14211 [0] NCCL INFO Channel 00/04 :    0   1   2   3
tesla-106:13395:14211 [0] NCCL INFO Channel 01/04 :    0   3   2   1
tesla-106:13395:14211 [0] NCCL INFO Channel 02/04 :    0   1   2   3
tesla-106:13395:14211 [0] NCCL INFO Channel 03/04 :    0   3   2   1
tesla-106:13395:14211 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 8/8/64
tesla-106:13395:14211 [0] NCCL INFO Trees [0] 2/-1/-1->0->-1|-1->0->2/-1/-1 [1] 2/-1/-1->0->1|1->0->2/-1/-1 [2] 2/-1/-1->0->-1|-1->0->2/-1/-1 [3] 2/-1/-1->0->1|1->0->2/-1/-1
tesla-106:13395:14211 [0] NCCL INFO Setting affinity for GPU 0 to 3ff003ff
tesla-106:13398:14208 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 8/8/64
tesla-106:13398:14208 [3] NCCL INFO Trees [0] 1/-1/-1->3->2|2->3->1/-1/-1 [1] 1/-1/-1->3->-1|-1->3->1/-1/-1 [2] 1/-1/-1->3->2|2->3->1/-1/-1 [3] 1/-1/-1->3->-1|-1->3->1/-1/-1
tesla-106:13398:14208 [3] NCCL INFO Setting affinity for GPU 3 to ff,c00ffc00
tesla-106:13397:14207 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 8/8/64
tesla-106:13397:14207 [2] NCCL INFO Trees [0] 3/-1/-1->2->0|0->2->3/-1/-1 [1] -1/-1/-1->2->0|0->2->-1/-1/-1 [2] 3/-1/-1->2->0|0->2->3/-1/-1 [3] -1/-1/-1->2->0|0->2->-1/-1/-1
tesla-106:13397:14207 [2] NCCL INFO Setting affinity for GPU 2 to ff,c00ffc00
tesla-106:13396:14213 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 8/8/64
tesla-106:13398:14208 [3] NCCL INFO Channel 00 : 3[b1000] -> 0[18000] via direct shared memory
tesla-106:13397:14207 [2] NCCL INFO Channel 00 : 2[af000] -> 3[b1000] via P2P/IPC
tesla-106:13396:14213 [1] NCCL INFO Trees [0] -1/-1/-1->1->3|3->1->-1/-1/-1 [1] 0/-1/-1->1->3|3->1->0/-1/-1 [2] -1/-1/-1->1->3|3->1->-1/-1/-1 [3] 0/-1/-1->1->3|3->1->0/-1/-1
tesla-106:13396:14213 [1] NCCL INFO Setting affinity for GPU 1 to 3ff003ff
tesla-106:13396:14213 [1] NCCL INFO Channel 00 : 1[1a000] -> 2[af000] via direct shared memory
tesla-106:13397:14207 [2] NCCL INFO Channel 00 : 2[af000] -> 0[18000] via direct shared memory
tesla-106:13395:14211 [0] NCCL INFO Channel 00 : 0[18000] -> 1[1a000] via P2P/IPC
tesla-106:13396:14213 [1] NCCL INFO Channel 00 : 1[1a000] -> 3[b1000] via direct shared memory
tesla-106:13398:14208 [3] NCCL INFO Channel 00 : 3[b1000] -> 2[af000] via P2P/IPC
tesla-106:13398:14208 [3] NCCL INFO Channel 00 : 3[b1000] -> 1[1a000] via direct shared memory
tesla-106:13398:14208 [3] NCCL INFO Channel 01 : 3[b1000] -> 2[af000] via P2P/IPC
tesla-106:13396:14213 [1] NCCL INFO Channel 01 : 1[1a000] -> 0[18000] via P2P/IPC
tesla-106:13395:14211 [0] NCCL INFO Channel 00 : 0[18000] -> 2[af000] via direct shared memory
tesla-106:13397:14207 [2] NCCL INFO Channel 01 : 2[af000] -> 1[1a000] via direct shared memory
tesla-106:13397:14207 [2] NCCL INFO Channel 01 : 2[af000] -> 0[18000] via direct shared memory
tesla-106:13395:14211 [0] NCCL INFO Channel 01 : 0[18000] -> 3[b1000] via direct shared memory
tesla-106:13396:14213 [1] NCCL INFO Channel 01 : 1[1a000] -> 3[b1000] via direct shared memory
tesla-106:13398:14208 [3] NCCL INFO Channel 01 : 3[b1000] -> 1[1a000] via direct shared memory
tesla-106:13395:14211 [0] NCCL INFO Channel 01 : 0[18000] -> 1[1a000] via P2P/IPC
tesla-106:13395:14211 [0] NCCL INFO Channel 01 : 0[18000] -> 2[af000] via direct shared memory
tesla-106:13398:14208 [3] NCCL INFO Channel 02 : 3[b1000] -> 0[18000] via direct shared memory
tesla-106:13396:14213 [1] NCCL INFO Channel 02 : 1[1a000] -> 2[af000] via direct shared memory
tesla-106:13395:14211 [0] NCCL INFO Channel 02 : 0[18000] -> 1[1a000] via P2P/IPC
tesla-106:13397:14207 [2] NCCL INFO Channel 02 : 2[af000] -> 3[b1000] via P2P/IPC
tesla-106:13396:14213 [1] NCCL INFO Channel 02 : 1[1a000] -> 3[b1000] via direct shared memory
tesla-106:13397:14207 [2] NCCL INFO Channel 02 : 2[af000] -> 0[18000] via direct shared memory
tesla-106:13398:14208 [3] NCCL INFO Channel 02 : 3[b1000] -> 2[af000] via P2P/IPC
tesla-106:13398:14208 [3] NCCL INFO Channel 02 : 3[b1000] -> 1[1a000] via direct shared memory
tesla-106:13395:14211 [0] NCCL INFO Channel 02 : 0[18000] -> 2[af000] via direct shared memory
tesla-106:13398:14208 [3] NCCL INFO Channel 03 : 3[b1000] -> 2[af000] via P2P/IPC
tesla-106:13395:14211 [0] NCCL INFO Channel 03 : 0[18000] -> 3[b1000] via direct shared memory
tesla-106:13397:14207 [2] NCCL INFO Channel 03 : 2[af000] -> 1[1a000] via direct shared memory
tesla-106:13396:14213 [1] NCCL INFO Channel 03 : 1[1a000] -> 0[18000] via P2P/IPC
tesla-106:13397:14207 [2] NCCL INFO Channel 03 : 2[af000] -> 0[18000] via direct shared memory
tesla-106:13395:14211 [0] NCCL INFO Channel 03 : 0[18000] -> 1[1a000] via P2P/IPC
tesla-106:13395:14211 [0] NCCL INFO Channel 03 : 0[18000] -> 2[af000] via direct shared memory
tesla-106:13396:14213 [1] NCCL INFO Channel 03 : 1[1a000] -> 3[b1000] via direct shared memory
tesla-106:13398:14208 [3] NCCL INFO Channel 03 : 3[b1000] -> 1[1a000] via direct shared memory
tesla-106:13395:14211 [0] NCCL INFO 4 coll channels, 4 p2p channels, 2 p2p channels per peer
tesla-106:13395:14211 [0] NCCL INFO comm 0x7fdc40001060 rank 0 nranks 4 cudaDev 0 busId 18000 - Init COMPLETE
tesla-106:13395:13395 [0] NCCL INFO Launch mode Parallel
DistributedDataParallel(
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
tesla-106:13397:14207 [2] NCCL INFO 4 coll channels, 4 p2p channels, 2 p2p channels per peer
tesla-106:13397:14207 [2] NCCL INFO comm 0x7f04c8001060 rank 2 nranks 4 cudaDev 2 busId af000 - Init COMPLETE
tesla-106:13396:14213 [1] NCCL INFO 4 coll channels, 4 p2p channels, 2 p2p channels per peer
tesla-106:13398:14208 [3] NCCL INFO 4 coll channels, 4 p2p channels, 2 p2p channels per peer
tesla-106:13398:14208 [3] NCCL INFO comm 0x7fd174001060 rank 3 nranks 4 cudaDev 3 busId b1000 - Init COMPLETE
tesla-106:13396:14213 [1] NCCL INFO comm 0x7f3330001060 rank 1 nranks 4 cudaDev 1 busId 1a000 - Init COMPLETE
loss: 2.301109  [    0/60000]
loss: 2.293916  [ 1600/60000]
loss: 2.302621  [ 3200/60000]
loss: 2.259469  [ 4800/60000]
loss: 2.252404  [ 6400/60000]
loss: 2.239343  [ 8000/60000]
loss: 2.185202  [ 9600/60000]
loss: 2.140033  [11200/60000]
loss: 2.158100  [12800/60000]
loss: 2.119617  [14400/60000]
Test Error: 
 Accuracy: 12.0%, Avg loss: 2.138575 

Epoch 2
-------------------------------
loss: 2.120426  [    0/60000]
loss: 2.120949  [ 1600/60000]
loss: 2.135741  [ 3200/60000]
loss: 2.021280  [ 4800/60000]
loss: 2.076435  [ 6400/60000]
loss: 1.989785  [ 8000/60000]
loss: 2.057000  [ 9600/60000]
loss: 1.840006  [11200/60000]
loss: 1.772224  [12800/60000]
loss: 1.738061  [14400/60000]
Test Error: 
 Accuracy: 13.5%, Avg loss: 1.840712 

Epoch 3
-------------------------------
loss: 1.917364  [    0/60000]
loss: 1.730065  [ 1600/60000]
loss: 1.906000  [ 3200/60000]
loss: 1.718702  [ 4800/60000]
loss: 1.486567  [ 6400/60000]
loss: 1.610462  [ 8000/60000]
loss: 1.431992  [ 9600/60000]
loss: 1.478280  [11200/60000]
loss: 1.497222  [12800/60000]
loss: 1.386750  [14400/60000]
Test Error: 
 Accuracy: 15.1%, Avg loss: 1.478247 

Epoch 4
-------------------------------
loss: 1.452221  [    0/60000]
loss: 1.571878  [ 1600/60000]
loss: 1.406897  [ 3200/60000]
loss: 1.460781  [ 4800/60000]
loss: 1.586754  [ 6400/60000]
loss: 1.300083  [ 8000/60000]
loss: 1.295014  [ 9600/60000]
loss: 1.321493  [11200/60000]
loss: 1.395649  [12800/60000]
loss: 1.349784  [14400/60000]
Test Error: 
 Accuracy: 15.9%, Avg loss: 1.227023 

Epoch 5
-------------------------------
loss: 1.091690  [    0/60000]
loss: 1.106918  [ 1600/60000]
loss: 1.163208  [ 3200/60000]
loss: 1.215325  [ 4800/60000]
loss: 1.357648  [ 6400/60000]
loss: 1.262445  [ 8000/60000]
loss: 1.171132  [ 9600/60000]
loss: 1.208320  [11200/60000]
loss: 0.778282  [12800/60000]
loss: 1.311920  [14400/60000]
Test Error: 
 Accuracy: 16.4%, Avg loss: 1.068742 

Done!
Saved PyTorch Model State to model.pth
```
# Start with single machine and multiple GPU
```bash
> CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
before running dist.init_process_group()
MASTER_ADDR: 127.0.0.1	MASTER_PORT: 29500
LOCAL_RANK: 0	RANK: 0	WORLD_SIZE: 2
before running dist.init_process_group()
MASTER_ADDR: 127.0.0.1	MASTER_PORT: 29500
LOCAL_RANK: 1	RANK: 1	WORLD_SIZE: 2
after running dist.init_process_group()
after running dist.init_process_group()

> CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
before running dist.init_process_group()
MASTER_ADDR: 127.0.0.1	MASTER_PORT: 29500
LOCAL_RANK: 0	RANK: 0	WORLD_SIZE: 4
before running dist.init_process_group()
MASTER_ADDR: 127.0.0.1	MASTER_PORT: 29500
LOCAL_RANK: 2	RANK: 2	WORLD_SIZE: 4
before running dist.init_process_group()
MASTER_ADDR: 127.0.0.1	MASTER_PORT: 29500
LOCAL_RANK: 3	RANK: 3	WORLD_SIZE: 4
before running dist.init_process_group()
MASTER_ADDR: 127.0.0.1	MASTER_PORT: 29500
LOCAL_RANK: 1	RANK: 1	WORLD_SIZE: 4
after running dist.init_process_group()
after running dist.init_process_group()
after running dist.init_process_group()
```
**note:** You can control the number of GPUs using `CUDA_VISIBLE_DEVICES`. `nproc_per_node` should be equal to the number of GPUs (`CUDA_VISIBLE_DEVICES`). 

# Start with multiple machine and multiple GPU
**Machine 0:**
```bash
> python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='192.168.1.105' --master_port='12345' train.py

```

**Machine 1:**
```bash
> python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr='192.168.1.106' --master_port='12345' train.py

```
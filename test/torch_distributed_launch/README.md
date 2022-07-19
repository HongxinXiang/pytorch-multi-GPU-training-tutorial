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
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 0	RANK: 0	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 1	RANK: 1	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 3	RANK: 3	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 2	RANK: 2	WORLD_SIZE: 8
after running dist.init_process_group()
after running dist.init_process_group()
after running dist.init_process_group()
after running dist.init_process_group()

```

**Machine 1:**
```bash
> python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr='192.168.1.105' --master_port='12345' train.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 0	RANK: 4	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 1	RANK: 5	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 3	RANK: 7	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 2	RANK: 6	WORLD_SIZE: 8
after running dist.init_process_group()
after running dist.init_process_group()
after running dist.init_process_group()
after running dist.init_process_group()
```

**Note:** When `nnodes`>1, the server will block in dist.init_process_group() until all servers (nnodes) complete init_process_group() .

# Some wrong attempts
## Run with single machine and multiple GPU

```bash
> CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 wrong_attemp.py
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
run dist.destroy_process_group()
complete dist.destroy_process_group()
Traceback (most recent call last):
  File "wrong_attemp.py", line 15, in <module>
    dist.init_process_group('nccl')
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 525, in init_process_group
    _store_based_barrier(rank, store, timeout)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 201, in _store_based_barrier
    worker_count = store.add(store_key, 0)
RuntimeError: Broken pipe
Traceback (most recent call last):
  File "wrong_attemp.py", line 15, in <module>
    dist.init_process_group('nccl')
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 525, in init_process_group
    _store_based_barrier(rank, store, timeout)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 201, in _store_based_barrier
    worker_count = store.add(store_key, 0)
RuntimeError: Broken pipe

run dist.destroy_process_group()
complete dist.destroy_process_group()
Killing subprocess 31050
Killing subprocess 31052
Killing subprocess 31053
Killing subprocess 31054
Traceback (most recent call last):
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 340, in <module>
    main()
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 326, in main
    sigkill_handler(signal.SIGTERM, None)  # not coming back
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 301, in sigkill_handler
    raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
subprocess.CalledProcessError: Command '['/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/bin/python', '-u', 'wrong_attemp.py', '--local_rank=3']' returned non-zero exit status 1.
```
## Run with multiple machine and multiple GPU
**Machine 0:**
```bash
> python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='192.168.1.105' --master_port='12345' wrong_attemp.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 0	RANK: 0	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 2	RANK: 2	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 3	RANK: 3	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 1	RANK: 1	WORLD_SIZE: 8
after running dist.init_process_group()
after running dist.init_process_group()
run dist.destroy_process_group()
run dist.destroy_process_group()
complete dist.destroy_process_group()
complete dist.destroy_process_group()
Traceback (most recent call last):
  File "wrong_attemp.py", line 15, in <module>
    dist.init_process_group('nccl')
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 525, in init_process_group
Traceback (most recent call last):
  File "wrong_attemp.py", line 15, in <module>
    dist.init_process_group('nccl')
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 525, in init_process_group
    _store_based_barrier(rank, store, timeout)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 201, in _store_based_barrier
        _store_based_barrier(rank, store, timeout)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 201, in _store_based_barrier
worker_count = store.add(store_key, 0)
    RuntimeError: Connection reset by peer
worker_count = store.add(store_key, 0)
RuntimeError: Broken pipe
Killing subprocess 33095
Killing subprocess 33099
Killing subprocess 33108
Killing subprocess 33109
Traceback (most recent call last):
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 340, in <module>
    main()
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 326, in main
    sigkill_handler(signal.SIGTERM, None)  # not coming back
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 301, in sigkill_handler
    raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
subprocess.CalledProcessError: Command '['/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/bin/python', '-u', 'wrong_attemp.py', '--local_rank=3']' returned non-zero exit status 1.
```

**Machine 1:**
```bash
> python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 1 --master_addr='192.168.1.105' --master_port='12345' wrong_attemp.py
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 3	RANK: 7	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 2	RANK: 6	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 0	RANK: 4	WORLD_SIZE: 8
before running dist.init_process_group()
MASTER_ADDR: 192.168.1.105	MASTER_PORT: 12345
LOCAL_RANK: 1	RANK: 5	WORLD_SIZE: 8
Traceback (most recent call last):
  File "wrong_attemp.py", line 15, in <module>
    dist.init_process_group('nccl')
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 525, in init_process_group
    Traceback (most recent call last):
Traceback (most recent call last):
  File "wrong_attemp.py", line 15, in <module>
    dist.init_process_group('nccl')
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 525, in init_process_group
      File "wrong_attemp.py", line 15, in <module>
    _store_based_barrier(rank, store, timeout)_store_based_barrier(rank, store, timeout)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 201, in _store_based_barrier

  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 201, in _store_based_barrier
    dist.init_process_group('nccl')
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 525, in init_process_group
    worker_count = store.add(store_key, 0)    
worker_count = store.add(store_key, 0)RuntimeError
RuntimeError: Connection reset by peer: 
Connection reset by peer
_store_based_barrier(rank, store, timeout)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 201, in _store_based_barrier
    worker_count = store.add(store_key, 0)
RuntimeError: Connection reset by peer
Traceback (most recent call last):
  File "wrong_attemp.py", line 15, in <module>
    dist.init_process_group('nccl')
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 525, in init_process_group
    _store_based_barrier(rank, store, timeout)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 196, in _store_based_barrier
    worker_count = store.add(store_key, 0)
RuntimeError: Connection reset by peer
Killing subprocess 29184
Killing subprocess 29185
Killing subprocess 29186
Killing subprocess 29187
Traceback (most recent call last):
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 340, in <module>
    main()
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 326, in main
    sigkill_handler(signal.SIGTERM, None)  # not coming back
  File "/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/lib/python3.7/site-packages/torch/distributed/launch.py", line 301, in sigkill_handler
    raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
subprocess.CalledProcessError: Command '['/home/hxxiang/anaconda3/envs/pytorch-multi-GPU-training-tutorial/bin/python', '-u', 'wrong_attemp.py', '--local_rank=3']' returned non-zero exit status 1.
```

## Error analysis
The main reason for these errors is communication errors caused by different processes having inconsistent end times.
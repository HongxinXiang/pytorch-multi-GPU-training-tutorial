import os
import time
import torch.distributed as dist

print("before running dist.init_process_group()")
MASTER_ADDR = os.environ["MASTER_ADDR"]
MASTER_PORT = os.environ["MASTER_PORT"]
LOCAL_RANK = os.environ["LOCAL_RANK"]
RANK = os.environ["RANK"]
WORLD_SIZE = os.environ["WORLD_SIZE"]

print("MASTER_ADDR: {}\tMASTER_PORT: {}".format(MASTER_ADDR, MASTER_PORT))
print("LOCAL_RANK: {}\tRANK: {}\tWORLD_SIZE: {}".format(LOCAL_RANK, RANK, WORLD_SIZE))

dist.init_process_group('nccl')
print("after running dist.init_process_group()")
# time.sleep(60)  # In this example, we do not perform sleep operations
print("run dist.destroy_process_group()")
dist.destroy_process_group()
print("complete dist.destroy_process_group()")
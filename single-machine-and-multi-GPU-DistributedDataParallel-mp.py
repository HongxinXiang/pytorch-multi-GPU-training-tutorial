import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import NeuralNetwork

# [*] Packages required to import distributed data parallelism
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

"""Start DDP code with "python -m torch.distributed.launch"
"""

# [*] Initialize the distributed process group and distributed device
def setup_DDP_mp(init_method, local_rank, rank, world_size, backend="nccl", verbose=False):
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print("Using device: {}".format(device))
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return device


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # copy data from cpu to gpu

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # [*] only print log on rank 0
        if dist.get_rank() == 0 and batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)  # copy data from cpu to gpu
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # [*] only print log on rank 0
    print_only_rank0(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def print_only_rank0(log):
    if dist.get_rank() == 0:
        print(log)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--ngpus_per_node", default=2, type=int, help="number of GPUs per node for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:12355", type=str, help="url used to set up distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")


def main(local_rank, ngpus_per_node, args):
    """
    :param local_rank: the local_rank is automatically passed in by mp.spawn()
    :param ngpus_per_node:
    :param args:
    :return:
    """
    args.local_rank = local_rank
    args.rank = args.node_rank * ngpus_per_node + local_rank

    # [*] initialize the distributed process group and device
    device = setup_DDP_mp(init_method=args.dist_url, local_rank=args.local_rank, rank=args.rank,
                          world_size=args.world_size, verbose=True)

    # initialize dataset
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    # initialize data loader
    # [*] using DistributedSampler
    batch_size = 64 // args.world_size  # [*] // world_size
    train_sampler = DistributedSampler(training_data, shuffle=True)  # [*]
    test_sampler = DistributedSampler(test_data, shuffle=False)  # [*]
    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)  # [*] sampler=...
    test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)  # [*] sampler=...

    # initialize model
    model = NeuralNetwork().to(device)  # copy model from cpu to gpu
    # [*] using DistributedDataParallel
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)  # [*] DDP(...)
    print_only_rank0(model)  # [*]

    # initialize optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train on multiple-GPU
    epochs = 5
    for t in range(epochs):
        # [*] set sampler
        train_dataloader.sampler.set_epoch(t)
        test_dataloader.sampler.set_epoch(t)

        print_only_rank0(f"Epoch {t + 1}\n-------------------------------")  # [*]
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

    print_only_rank0("Done!")  # [*]

    # [*] save model on rank 0
    if dist.get_rank() == 0:
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, "model.pth")
        print("Saved PyTorch Model State to model.pth")


if __name__ == '__main__':
    # [*] initialize some arguments
    args = parse_args()
    args.world_size = args.ngpus_per_node * args.nodes
    # [*] run with torch.multiprocessing
    mp.spawn(main, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))

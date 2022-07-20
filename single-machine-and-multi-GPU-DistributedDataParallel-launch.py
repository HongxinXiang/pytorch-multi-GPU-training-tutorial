import os
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
def setup_DDP(backend="nccl", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...

    You don't set rank and world_size in dist.init_process_group() explicitly.

    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    word_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print("Using device: {}".format(device))
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {word_size}")
    return rank, local_rank, word_size, device


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
    if dist.get_rank() == 0:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # [*] initialize the distributed process group and device
    rank, local_rank, word_size, device = setup_DDP(verbose=True)

    # initialize dataset
    training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

    # initialize data loader
    # [*] using DistributedSampler
    batch_size = 64 // word_size  # [*] // world_size
    train_sampler = DistributedSampler(training_data, shuffle=True)  # [*]
    test_sampler = DistributedSampler(test_data, shuffle=False)  # [*]
    train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)  # [*] sampler=...
    test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)  # [*] sampler=...

    # initialize model
    model = NeuralNetwork().to(device)  # copy model from cpu to gpu
    # [*] using DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # [*] DDP(...)
    print(model)

    # initialize optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train on multiple-GPU
    epochs = 5
    for t in range(epochs):
        # [*] set sampler
        train_dataloader.sampler.set_epoch(t)
        test_dataloader.sampler.set_epoch(t)

        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)

    print("Done!")

    # [*] save model on rank 0
    if dist.get_rank() == 0:
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, "model.pth")
        print("Saved PyTorch Model State to model.pth")

import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

    
def save_images_1D_to_2D(signals, path, **kwargs):
    signals = signals.to('cpu').detach().numpy()
    dim = signals.shape[1]
    imgs = signals.reshape(signals.shape[0], 1, 28, 28)
    print(imgs.shape)
    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    for i in range(2):
        for j in range(5):
            for k in range(dim):
                axs[i, j].imshow(imgs[i*5+j][k], cmap='gray')
    plt.savefig(path, format="jpeg")
    

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    if args.dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(root=args.dataset_path,
           train=True, 
           transform=torchvision.transforms.ToTensor(),
           download=True)
    elif args.dataset_name == "Cifar10":
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("checkpoint", run_name), exist_ok=True)

    
def save_signals(signals, path, **kwargs):
    signals = signals.to('cpu').detach().numpy()
    dim = signals.shape[1]
    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    for i in range(2):
        for j in range(5):
            for k in range(dim):
                axs[i, j].plot(signals[i*5+j][k][:])
    plt.savefig(path, format="jpeg")
    

def save_signals_cls_free(signals, labels, path, **kwargs):
    signals = signals.to('cpu').detach().numpy()
    dim = signals.shape[1]
    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    for i in range(2):
        for j in range(5):
            for k in range(dim):
                axs[i, j].plot(signals[i*5+j][k][:])
            axs[i, j].set_title(f'{labels[i*5+j]}')
    plt.savefig(path, format="jpeg")
    

    
    
def save_signals_cond_cls_free(sampled_signals, org_signals, cond_signals, labels, path, **kwargs):
    sampled_signals = sampled_signals.to('cpu').detach().numpy()
    org_signals = org_signals.to('cpu').detach().numpy()
    cond_signals = cond_signals.to('cpu').detach().numpy()
#     print(f'sampled_signals shape is {sampled_signals.shape}') # shape = (5, 1, 128)
#     print(f'org_signals shape is {org_signals.shape}') # shape = (5, 1, 128)
#     print(f'cond_signals shape is {cond_signals.shape}') # shape = (5, 1, 128)
    
    dim = sampled_signals.shape[1]
            
    fig, axs = plt.subplots(3, 5, figsize=(20,5))
    for i in range(5):
        for d in range(dim):
            axs[0, i].plot(org_signals[i][d][:])
            axs[1, i].plot(cond_signals[i][d][:])
            axs[2, i].plot(sampled_signals[i][d][:])
            axs[0, i].set_title(f'{labels[i]}')
    plt.savefig(path, format="jpeg")    
    
def save_images_1D_to_2D_cls_free(signals, labels, path, **kwargs):
    signals = signals.to('cpu').detach().numpy()
    dim = signals.shape[1]
    imgs = signals.reshape(signals.shape[0], 1, 28, 28)
    print(imgs.shape)
    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    for i in range(2):
        for j in range(5):
            for k in range(dim):
                axs[i, j].imshow(imgs[i*5+j][k], cmap='gray')
            axs[i, j].set_title(f'{labels[i*5+j]}')
    plt.savefig(path, format="jpeg")
            

def save_checkpoint(states, is_best, output_dir, filename="checkpoint.pt"):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, "checkpoint_best.pt"))
import os
import copy
import sys
sys.path.insert(0, './modules/')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules.modules1D_cls_free import Unet1D_cls_free, GaussianDiffusion1D_cls_free
import logging
from torch.utils.tensorboard import SummaryWriter
from MITBIH import *
from torch.utils import data

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
data_path = "./heartbeat/mitbih_train.csv"


def train(args):
    setup_logging(args.run_name)
    device = args.device
    fiveClassECG = mitbih_allClass(filename = data_path)
    dataloader = data.DataLoader(fiveClassECG, batch_size=32, num_workers=4, shuffle=True)
    classes = 5
    model = Unet1D_cls_free(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = args.num_classes,
        cond_drop_prob = 0.5,
        channels = 1).to(device)

    diffusion = GaussianDiffusion1D_cls_free(
        model,
        seq_length = 128,
        timesteps = 1000).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (signals, labels) in enumerate(pbar):
            signals = signals.to(device).to(torch.float)
            labels = labels.to(device).to(torch.long)
            loss = diffusion(signals, classes = labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)

        labels = torch.randint(0, args.num_classes, (10,)).to(device)
        sampled_signals = diffusion.sample(
            classes = labels,
            cond_scale = 3.)
        sampled_signals.shape # (10, 1, 128)
        
        is_best = False
        
        save_signals_cls_free(sampled_signals, labels, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join("checkpoint", args.run_name))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM1D_cls_free_MITBIH"
    args.epochs = 300
    args.batch_size = 64
    args.seq_length = 128
    args.num_classes = 5
    args.device = "cuda:0"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()


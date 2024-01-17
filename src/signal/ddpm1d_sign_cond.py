import os
import sys
sys.path.insert(0, './modules/')
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils_cond import *
from modules.modules1D_cond import Unet1D, GaussianDiffusion1D
import logging
from torch.utils.tensorboard import SummaryWriter
from MITBIH import *
from torch.utils import data

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
data_path = "./heartbeat/mitbih_train.csv"

# def train(args):
#     setup_logging(args.run_name)
#     device = args.device
#     twoClassECG = mitbih_twoClass(filename = data_path, class_id1 = 1, class_id2 = 0)
#     dataloader = data.DataLoader(twoClassECG, batch_size=32, num_workers=4, shuffle=True)
#     model = Unet1D(
#         dim = 64,
#         self_condition = True,
#         dim_mults = (1, 2, 4, 8),
#         channels = 1).to(device)
#     # seq_length must be able to divided by dim_mults
#     diffusion = GaussianDiffusion1D(
#         model,
#         seq_length = 128,
#         timesteps = 1000,
#         objective = 'pred_v').to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr)
#     logger = SummaryWriter(os.path.join("runs", args.run_name))
#     l = len(dataloader)

#     for epoch in range(args.epochs):
#         logging.info(f"Starting epoch {epoch}:")
#         pbar = tqdm(dataloader)
#         for i, (sig1, _, sig2, _) in enumerate(pbar):
#             sig1 = sig1.to(device).to(torch.float)
#             sig2 = sig2.to(device).to(torch.float)
#             loss = diffusion(sig1, sig2) # GaussianDiffusion1D doesn't have x_self_cond parameter

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)

#         sampled_signals = diffusion.sample(batch_size = 10)
#         sampled_signals.shape # (10, 1, 128)
        
#         is_best = False
        
#         save_signals(sampled_signals, os.path.join("results", args.run_name, f"{epoch}.jpg"))
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'model': model,
#             'model_state_dict': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#         }, is_best, os.path.join("checkpoint", args.run_name))


def train(args):
    setup_logging(args.run_name)
    device = args.device
    cond_ECG = mitbih_masked(filename = data_path, class_id = 0)
    dataloader = data.DataLoader(cond_ECG, batch_size=32, num_workers=4, shuffle=True)
    model = Unet1D(
        dim = 64,
        self_condition = True,
        dim_mults = (1, 2, 4, 8),
        channels = 1).to(device)
    # seq_length must be able to divided by dim_mults
    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 128,
        timesteps = 1000,
        objective = 'pred_v').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    sample_size = args.sample_size
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data_dict in enumerate(pbar):
            sig1 = data_dict['org_data'].to(device).to(torch.float)
            sig2 = data_dict['cond_data'].to(device).to(torch.float)
            loss = diffusion(sig1, sig2) # GaussianDiffusion1D doesn't have x_self_cond parameter

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("loss", loss.item(), global_step=epoch * l + i)
        
        index_list = [i for i in range(len(cond_ECG))]
        random.shuffle(index_list)
        cond_data = torch.from_numpy(cond_ECG.cond_data[index_list][:sample_size])
        cond_data = cond_data.to(device).to(torch.float)
        sampled_signals = diffusion.sample(batch_size = sample_size, input_cond =cond_data)
        sampled_signals.shape # (sample_size, 1, 128)
        
        is_best = False
        
        save_signals(sampled_signals, os.path.join("results", args.run_name, f"{epoch}.jpg"))
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
    args.run_name = "DDPM1D_Selfcondtional_maskedCond"
    args.epochs = 100
    args.batch_size = 32
    args.device = "cuda:0"
    args.lr = 3e-4
    args.sample_size = 10
    train(args)


if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
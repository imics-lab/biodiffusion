# Import necessary libraries
import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext
import os  # Add missing import statement

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

import wandb
from utils import *
from modules.modules import UNet_conditional, EMA

torch.cuda.empty_cache()

# Define configuration using SimpleNamespace
config = SimpleNamespace(    
    run_name="DDPM_conditional",
    epochs=100,
    noise_steps=1000,
    seed=42,
    batch_size=10,
    img_size=32,
    num_classes=10,
    dataset_path=get_cifar(img_size=32),
    train_folder="train",
    val_folder="test",
    device="cuda:3",
    slice_size=1,
    do_validation=True,
    fp16=True,
    log_every_epoch=10,
    num_workers=10,
    lr=5e-3)

# Set up logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# Define the Diffusion class
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, device="cuda:3", **kwargs):
        """
        Initializes the Diffusion class.

        Args:
            noise_steps (int): Number of noise steps.
            beta_start (float): Starting value for beta in noise schedule.
            beta_end (float): Ending value for beta in noise schedule.
            img_size (int): Image size.
            num_classes (int): Number of classes.
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            device (str): Device for computation.
            **kwargs: Additional keyword arguments.
        """
        # Initialize parameters
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        print(device)
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        """
        Prepares the noise schedule using beta_start and beta_end.

        Returns:
            torch.Tensor: Noise schedule.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        """
        Samples random timesteps.

        Args:
            n (int): Number of timesteps to sample.

        Returns:
            torch.Tensor: Sampled timesteps.
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        """
        Adds noise to images at a specific timestep.

        Args:
            x (torch.Tensor): Input images.
            t (torch.Tensor): Timestep.

        Returns:
            tuple: Tuple containing noisy images and noise.
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        """
        Generates samples using the diffusion model.

        Args:
            use_ema (bool): Whether to use the EMA model.
            labels (torch.Tensor): Labels for conditional sampling.
            cfg_scale (int): Configuration scale.

        Returns:
            torch.Tensor: Generated samples.
        """
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        """
        Performs a training step.

        Args:
            loss (torch.Tensor): Loss value.
        """
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        """
        Performs one epoch of training or validation.

        Args:
            train (bool): Whether to perform training.

        Returns:
            float: Average loss for the epoch.
        """
        avg_loss = 0.
        if train:
            self.model.train()
        else:
            self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                wandb.log({"train_mse": loss.item(),
                           "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"
        return avg_loss.mean().item()

    def log_images(self):
        """
        Logs sampled images to WandB and saves them to disk.
        """
        labels = torch.arange(self.num_classes).long().to(self.device)
        sampled_images = self.sample(use_ema=False, labels=labels)
        wandb.log({"sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in sampled_images]})

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        plot_images(sampled_images)  # to display on Jupyter if available
        wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1, 2, 0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        """
        Loads model weights from checkpoint files.

        Args:
            model_cpkt_path (str): Path to the model checkpoint.
            model_ckpt (str): Model checkpoint filename.
            ema_model_ckpt (str): EMA model checkpoint filename.
        """
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        """
        Saves the model locally and on WandB.

        Args:
            run_name (str): Name of the run.
            epoch (int): Epoch number.
        """
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        at.add_dir(os.path.join("models", run_name))
        wandb.log_artifact(at)

    def prepare(self, args):
        """
        Prepares the data, optimizer, scheduler, loss function, and other components for training.

        Args:
            args: Arguments containing hyperparameters.
        """
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        """
        Trains the model for the specified number of epochs.

        Args:
            args: Arguments containing hyperparameters.
        """
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _ = self.one_epoch(train=True)
            
            # Validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                wandb.log({"val_mse": avg_loss})
            
            # Log predictions
            if epoch % args.log_every_epoch == 0:
                self.log_images()

        # Save model
        self.save_model(run_name=args.run_name, epoch=epoch)


def parse_args(config):
    """
    Parses command line arguments and updates the configuration.

    Args:
        config: Configuration object.
    """
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # Update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    parse_args(config)

    # Seed everything
    set_seed(config.seed)
    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    with wandb.init(project="train_sd", group="train", config=config):
        diffuser.prepare(config)
        diffuser.fit(config)

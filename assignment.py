import torch

from models import VAE
from models import DDPM
from models import DDIM
from models import UNet
from models import VarianceScheduler


def prepare_ddpm() -> DDPM:
    """
    EXAMPLE OF INITIALIZING DDPM. Feel free to change the following based on your needs and implementation.
    """
    # TODO: define the configurations of the Variance Scheduler
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    # TODO: define the confifurations of the UNet
    in_channels = 1
    down_channels = [64, 128, 128, 128, 128]
    up_channels = [128, 128, 128, 128, 64]
 
    
    time_embed_dim = 128
    num_classes = 10

    # TODO: define the variance scheduler
    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

    # TODO: define the noise estimating UNet
    network = UNet(in_channels=in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim,
                   num_classes=num_classes)
    
    ddpm = DDPM(network=network, var_scheduler=var_scheduler)

    return ddpm

def prepare_ddim() -> DDIM:
    """
    EXAMPLE OF INITIALIZING DDIM. Feel free to change the following based on your needs and implementation.
    """
    # TODO: define the configurations of the Variance Scheduler
    beta1 = 0.0001
    beta2 = 0.02
    num_steps = 1000
    interpolation = 'quadratic'

    # TODO: define the confifurations of the UNet
    in_channels = 1
    down_channels = [64, 128, 128, 128, 128]
    up_channels = [128, 128, 128, 128, 64]
    time_embed_dim = 128
    num_classes = 10

    # TODO: define the variance scheduler
    var_scheduler = VarianceScheduler(beta_start=beta1, beta_end=beta2, num_steps=num_steps, interpolation=interpolation)

    # TODO: define the noise estimating UNet
    network = UNet(in_channels=in_channels, 
                   down_channels=down_channels, 
                   up_channels=up_channels, 
                   time_emb_dim=time_embed_dim,
                   num_classes=num_classes)
    
    ddim = DDIM(network=network, var_scheduler=var_scheduler)

    return ddim

def prepare_vae() -> VAE:
    """
    EXAMPLE OF INITIALIZING VAE. Feel free to change the following based on your needs and implementation.
    """
    # TODO: vae configs
    in_channels = 1
    mid_channels = [64, 128, 256, 512]
    height = width = 32
    latent_dim = 1
    num_classes = 10

    # TODO: defining the diffusion model component nets
    vae = VAE(in_channels=in_channels, 
              height=height, 
              width=width, 
              mid_channels=mid_channels, 
              latent_dim=latent_dim,
              num_classes=num_classes)
    
    return vae


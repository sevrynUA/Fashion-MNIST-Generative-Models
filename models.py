import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple


class VarianceScheduler:
    def __init__(self, beta_start: float = 0.0001, beta_end: float = 0.02, num_steps: int = 1000, interpolation: str = 'linear') -> None:
        self.num_steps = num_steps

        # Find the beta values by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif interpolation == 'quadratic':
            self.betas = torch.linspace(beta_start**(1/2), beta_end**(1/2), num_steps) ** 2
        else:
            raise ValueError('[!] Error: invalid beta interpolation encountered...')
        
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x: torch.Tensor, time_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to the input based on the timestep."""
        device = x.device
        time_step = time_step.to(device)

        a_bar = self.alpha_bars.to(device)
        a_bar_t = a_bar[time_step]
        a_bar_t = a_bar_t.view(-1, 1, 1, 1)

        noise = torch.randn_like(x).to(device) 
        noisy_input = torch.sqrt(a_bar_t) * x + torch.sqrt(1 - a_bar_t) * noise

        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()

      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2

        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = time[:, None] * torch.exp(-embeddings * torch.arange(half_dim, device=device)[None, :])
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        return embeddings


class MyBlock(nn.Module):
            def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True, num_groups=8):
                super().__init__()
                self.ln = nn.LayerNorm(shape) 
                self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_c)
                self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_c)
                self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
                self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
                self.activation = nn.ReLU() if activation is None else activation
                self.normalize = normalize

            def forward(self, x):
                out = self.ln(x) if self.normalize else x
                out = self.conv1(x)
                out = self.activation(out)
                out = self.gn1(out)
                out = self.conv2(out)
                out = self.activation(out)
                out = self.gn2(out)
                return out
            

class UNet(nn.Module):
    def __init__(self, in_channels: int=1, 
                 down_channels: List=[64, 128, 128, 128, 128], 
                 up_channels: List=[128, 128, 128, 128, 64], 
                 time_emb_dim: int=128,
                 num_classes: int=10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.in_channels = in_channels

        encoding = [32, 64, 128]
        decoding = [256, 128, 64, 32]
        
        # Sinusoidal embedding
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim)
        self.label_embed = nn.Embedding(num_classes, time_emb_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((in_channels, 32, 32), in_channels, encoding[0]),
            MyBlock((encoding[0], 32, 32), encoding[0], encoding[0]),
            MyBlock((encoding[0], 32, 32), encoding[0], encoding[0])
        )
        self.down1 = nn.Conv2d(encoding[0], encoding[0], 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, encoding[0])
        self.b2 = nn.Sequential(
            MyBlock((encoding[0], 16, 16), encoding[0], encoding[1]),
            MyBlock((encoding[1], 16, 16), encoding[1], encoding[1]),
            MyBlock((encoding[1], 16, 16), encoding[1], encoding[1])
        )
        self.down2 = nn.Conv2d(encoding[1], encoding[1], 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, encoding[1])
        self.b3 = nn.Sequential(
            MyBlock((encoding[1], 8, 8), encoding[1], encoding[2]),
            MyBlock((encoding[2], 8, 8), encoding[2], encoding[2]),
            MyBlock((encoding[2], 8, 8), encoding[2], encoding[2])
        )
        self.down3 = nn.Conv2d(encoding[2], encoding[2], 4, 2, 1)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, encoding[2])

        self.b_mid = nn.Sequential(
            MyBlock((encoding[2], 4, 4), encoding[2], encoding[1]),
            MyBlock((encoding[1], 4, 4), encoding[1], encoding[1]),
            MyBlock((encoding[1], 4, 4), encoding[1], encoding[2])
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(encoding[2], encoding[2], 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(encoding[2], encoding[2], 4, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, decoding[0])
        self.b4 = nn.Sequential(
            MyBlock((decoding[0], 8, 8), decoding[0], decoding[1]),
            MyBlock((decoding[1], 8, 8), decoding[1], decoding[2]),
            MyBlock((decoding[2], 8, 8), decoding[2], decoding[2])
        )

        self.up2 = nn.ConvTranspose2d(decoding[2], decoding[2], 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, decoding[1])
        self.b5 = nn.Sequential(
            MyBlock((decoding[1], 16, 16), decoding[1], decoding[2]),
            MyBlock((decoding[2], 16, 16), decoding[2], decoding[3]),
            MyBlock((decoding[3], 16, 16), decoding[3], decoding[3]) 
        )

        self.up3 = nn.ConvTranspose2d(decoding[3], decoding[3], 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, decoding[2])
        self.b_out = nn.Sequential(
            MyBlock((decoding[2], 32, 32), decoding[2], decoding[3]),
            MyBlock((decoding[3], 32, 32), decoding[3], decoding[3]),
            MyBlock((decoding[3], 32, 32), decoding[3], decoding[3])
        )

        self.conv_out = nn.Conv2d(decoding[3], 1, 3, 1, 1)

    def forward(self, x, t, labels):
        l = self.label_embed(labels)
        t = self.time_embed(t)
        t = t + l

        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1)) 

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1)) 

        out = torch.cat((out1, self.up3(out5)), dim=1) 
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1)) 

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int = 32, 
                 width: int = 32, 
                 mid_channels: List[int] = [64, 128, 256, 512], 
                 latent_dim: int = 1, 
                 num_classes: int = 10) -> None:
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.mid_size = [mid_channels[-1], height // (2 ** len(mid_channels)), width // (2 ** len(mid_channels))]

        self.class_emb = nn.Embedding(num_classes, mid_channels[-1])

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(mid_channels[1], mid_channels[2], kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(mid_channels[2], mid_channels[3], kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
        )

        test_tensor = torch.zeros(1, in_channels, height, width)
        with torch.no_grad():
            flattened = self.encoder(test_tensor).view(1, -1).size(1)

        self.fc_mean = nn.Linear(flattened, latent_dim)
        self.fc_logvar = nn.Linear(flattened, latent_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + mid_channels[-1], mid_channels[3], kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[3], mid_channels[2], kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[2], mid_channels[1], kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[1], mid_channels[0], kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(mid_channels[0], in_channels, kernel_size=3, stride=1, padding=1),  
            nn.Sigmoid(), 
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)

        mean = self.fc_mean(encoded)
        logvar = self.fc_logvar(encoded)

        sample = self.reparameterize(mean, logvar)

        decoded = self.decode(sample, label)

        return decoded, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mean + noise * std

    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.class_emb(labels).unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(-1, -1, self.mid_size[1], self.mid_size[2])


        sample = sample.unsqueeze(-1).unsqueeze(-1)
        sample = sample.expand(-1, -1, self.mid_size[1], self.mid_size[2])
        combined = torch.cat([sample, label_emb], dim=1)

        return self.decoder(combined)

    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(pred, target, reduction='sum')

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor = None) -> torch.Tensor:
        if labels is not None:
            assert len(labels) == num_samples, "Error: number of labels must match number of samples!"
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=device)

        noise = torch.randn(num_samples, self.latent_dim, device=device)

        samples = self.decode(noise, labels)

        return samples


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # uniformly sample as many timesteps as the batch size
    
        t = torch.randint(0, self.var_scheduler.num_steps, (x.shape[0],), device=x.device)

        # generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # compute the loss (either L1, or L2 loss)
        loss = F.mse_loss(estimated_noise, noise) 

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Recover a sample from the noisy sample using the backward process.
        """
        device = noisy_sample.device
        t = int(timestep[0].item())
        alpha = self.var_scheduler.alphas
        alpha_bar = self.var_scheduler.alpha_bars
        beta = self.var_scheduler.betas
        
        sample = (1.0/alpha[t].sqrt()) * (noisy_sample - (beta[t] / torch.sqrt(1 - alpha_bar[t])) * estimated_noise) 

        if t > 0:
            noise = torch.randn_like(noisy_sample)
            sample += (torch.sqrt(((1.0 - alpha_bar[t-1]) / (1 - alpha_bar[t])) * beta[t]) * noise)

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None

        # apply the iterative sample generation of the DDPM
        xt = torch.randn((num_samples, 1, 32, 32), device=device)
        for t in reversed(range(self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(xt, t_tensor, labels)
            xt = self.recover_sample(xt, estimated_noise, t_tensor)
        sample = xt

        return sample


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # uniformly sample as many timesteps as the batch size
        t =  torch.randint(0, self.var_scheduler.num_steps, (x.shape[0],), device=x.device)

        # generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # compute the loss
        loss = F.mse_loss(estimated_noise, noise)

        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # apply the sample recovery strategy of the DDIM
        t = int(timestep[0].item())

        if t == 0:
            return noisy_sample
        
        alpha_bar = self.var_scheduler.alpha_bars
        beta = self.var_scheduler.betas
        variance = ((1 - alpha_bar[t-1]) / (1 - alpha_bar[t])) * beta[t]

        noise = torch.randn_like(noisy_sample)

        prediction = ((noisy_sample - torch.sqrt(1-alpha_bar[t]) * estimated_noise) / torch.sqrt(alpha_bar[t]))
        direction = torch.sqrt(1 - alpha_bar[t-1] - variance) * estimated_noise
        sample = torch.sqrt(alpha_bar[t-1]) * prediction + direction + (torch.sqrt(variance) * noise)
        
        return sample
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None
        # apply the iterative sample generation of DDIM (similar to DDPM)
        xt = torch.randn((num_samples, 1, 32, 32), device=device)
        for t in reversed(range(self.var_scheduler.num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            estimated_noise = self.network(xt, t_tensor, labels)
            xt = self.recover_sample(xt, estimated_noise, t_tensor)
        sample = xt

        return sample
    
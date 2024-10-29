"""
Convolutional Variational Autoencoder (VAE) Implementation

This module implements a convolutional VAE architecture for music generation, along with
conditional neural networks for multi-track generation. The VAE learns a latent representation
of musical sequences that can be used to generate new music.

The module contains three main classes:
- ConvVAE: The core VAE architecture with convolutional layers
- ConditionalNN: Neural network for conditional generation of harmony tracks
- MelodyNN: Neural network for generating melody sequences

Requirements:
    - torch
    - vae_helpers
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from vae_helpers import *

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for music generation.
    
    This VAE uses convolutional layers in both encoder and decoder to process musical sequences.
    It learns a compressed latent representation that captures musical patterns and structure.
    
    Attributes:
        encoder (nn.Sequential): Convolutional encoder network
        fc_mean (nn.Linear): Linear layer for latent mean
        fc_logvar (nn.Linear): Linear layer for latent log variance
        decoder_input (nn.Linear): Linear layer for decoder input
        decoder (nn.Sequential): Convolutional decoder network
        log_sig_x (nn.Parameter): Learnable output standard deviation
    """
    
    def __init__(self, K: int, num_filters: int = 32, filter_size: int = 5):
        """
        Initialize the Convolutional VAE.
        
        Args:
            K (int): Dimension of the latent space
            num_filters (int, optional): Number of filters in conv layers. Defaults to 32
            filter_size (int, optional): Size of conv filters. Defaults to 5
        """
        super(ConvVAE, self).__init__()

        # Encoder (recognition model)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(2, 8), stride=(2, 8)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mean = nn.Linear(256, K)
        self.fc_logvar = nn.Linear(256, K)

        # Decoder (generative model)
        self.decoder_input = nn.Linear(K, 256)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 8), stride=(2, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4)
        )

        self.log_sig_x = nn.Parameter(torch.zeros(()))
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input x to latent space parameters.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, time_steps, features]
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and log variance tensors of shape [batch_size, K]
        """
        h = self.encoder(x.unsqueeze(1))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu (torch.Tensor): Mean tensor of shape [batch_size, K]
            logvar (torch.Tensor): Log variance tensor of shape [batch_size, K]
        
        Returns:
            torch.Tensor: Sampled latent vectors of shape [batch_size, K]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z to reconstructed x.
        
        Args:
            z (torch.Tensor): Latent vectors of shape [batch_size, K]
        
        Returns:
            torch.Tensor: Reconstructed output of shape [batch_size, time_steps, features]
        """
        return self.decoder(self.decoder_input(z)).squeeze(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, time_steps, features]
        
        Returns:
            torch.Tensor: Reconstructed output of same shape as input
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Infer latent space parameters for input x.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, time_steps, features]
        
        Returns:
            torch.Tensor: Concatenated mean and last log variance of shape [batch_size, K+1]
        """
        mu, logvar = self.encode(x)
        return torch.cat([mu, logvar[:, -1].unsqueeze(1)], dim=1)

    def generate(self, zs: torch.Tensor) -> torch.Tensor:
        """
        Generate outputs from latent samples.
        
        Args:
            zs (torch.Tensor): Latent samples of shape [batch_size, n_samples, K]
        
        Returns:
            torch.Tensor: Generated outputs of shape [batch_size, n_samples, time_steps, features]
        """
        b, n, k = zs.size()
        return self.decode(zs.view(b*n, k)).view(b, n, -1)

    def elbo(self, x: torch.Tensor, n: int = 1) -> torch.Tensor:
        """
        Compute the Evidence Lower Bound (ELBO).
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, time_steps, features]
            n (int, optional): Number of samples for Monte Carlo estimation. Defaults to 1
        
        Returns:
            torch.Tensor: ELBO value (scalar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar).unsqueeze(1).expand(-1, n, -1)
        x_recon = self.generate(z)
        
        recon_loss = log_p_x(x, x_recon, self.log_sig_x.exp())
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss - kl_div


class ConditionalNN(nn.Module):
    """
    Neural network for conditional generation of harmony tracks.
    
    Takes previous harmony and melody latent vectors to predict next harmony latent vector.
    
    Attributes:
        model (nn.Sequential): Feed-forward neural network
    """
    
    def __init__(self, K: int, hidden_dim: int = 128):
        """
        Initialize the conditional neural network.
        
        Args:
            K (int): Dimension of latent vectors
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128
        """
        super(ConditionalNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2*K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, prev_harmony: torch.Tensor, melody: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict next harmony latent vector.
        
        Args:
            prev_harmony (torch.Tensor): Previous harmony latent vector [batch_size, K]
            melody (torch.Tensor): Current melody latent vector [batch_size, K]
            
        Returns:
            torch.Tensor: Predicted next harmony latent vector [batch_size, K]
        """
        x = torch.cat((prev_harmony, melody), dim=1)
        return self.model(x)


class MelodyNN(nn.Module):
    """
    Neural network for generating melody sequences.
    
    Uses previous melody's latent vectors to predict next melody's latent vectors.
    
    Attributes:
        model (nn.Sequential): Feed-forward neural network with dropout
    """
    
    def __init__(self, K: int, hidden_dim: int = 128, dropout_rate: float = 0.2):
        """
        Initialize the melody generation network.
        
        Args:
            K (int): Dimension of latent vectors
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2
        """
        super(MelodyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict next melody latent vector.
        
        Args:
            x (torch.Tensor): Previous melody latent vector [batch_size, K]
            
        Returns:
            torch.Tensor: Predicted next melody latent vector [batch_size, K]
        """
        return self.model(x)
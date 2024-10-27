import torch
import torch.nn as nn
from torch.nn import functional as F
from vae_helpers import *

class ConvVAE(nn.Module):
    def __init__(self, K, num_filters=32, filter_size=5):
        """
        Initialize the Convolutional Variational Autoencoder.
        
        Args:
            K (int): Dimension of the latent space.
            num_filters (int): Number of filters in convolutional layers (not used in current implementation).
            filter_size (int): Size of filters in convolutional layers (not used in current implementation).
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
    
    def encode(self, x):
        """
        Encode input x to latent space parameters (mean and log variance).
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Mean and log variance of the latent distribution.
        """
        h = self.encoder(x.unsqueeze(1))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log variance of the latent distribution.
        
        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent z to reconstructed x.
        
        Args:
            z (torch.Tensor): Latent vector.
        
        Returns:
            torch.Tensor: Reconstructed input.
        """
        return self.decoder(self.decoder_input(z)).squeeze(1)
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Reconstructed input.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def infer(self, x):
        """
        Infer latent space parameters for input x.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Concatenated mean and last log variance.
        """
        mu, logvar = self.encode(x)
        return torch.cat([mu, logvar[:, -1].unsqueeze(1)], dim=1)

    def generate(self, zs):
        """
        Generate outputs from latent samples.
        
        Args:
            zs (torch.Tensor): Latent samples.
        
        Returns:
            torch.Tensor: Generated outputs.
        """
        b, n, k = zs.size()
        return self.decode(zs.view(b*n, k)).view(b, n, -1)

    def elbo(self, x, n=1):
        """
        Compute the Evidence Lower Bound (ELBO).
        
        Args:
            x (torch.Tensor): Input tensor.
            n (int): Number of samples for Monte Carlo estimation.
        
        Returns:
            torch.Tensor: ELBO value.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar).unsqueeze(1).expand(-1, n, -1)
        x_recon = self.generate(z)
        
        recon_loss = log_p_x(x, x_recon, self.log_sig_x.exp())
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss - kl_div


class ConditionalNN(nn.Module):
    def __init__(self, K, hidden_dim=128):
        super(ConditionalNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2*K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, prev_harmony, melody):
        x = torch.cat((prev_harmony, melody), dim=1)
        return self.model(x)

# Melody NN - uses previous melody's LATENT vectors to predict next melody's LATENT VECTORS
class MelodyNN(nn.Module):
    def __init__(self, K, hidden_dim=128, dropout_rate=0.2):
        super(MelodyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )

    def forward(self, x):
        return self.model(x)
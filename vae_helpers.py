import torch
import torch.nn as nn
import math


def kl_q_p(zs, phi):
    """
    Compute KL divergence KL(q||p) given samples from q and parameters of q.
    
    Args:
        zs (torch.Tensor): Samples from q, shape [b,n,k]
        phi (torch.Tensor): Parameters of q, shape [b,k+1]
    
    Returns:
        torch.Tensor: Estimated KL divergence
    """
    b, n, k = zs.size()
    mu_q, log_sig_q = phi[:, :-1], phi[:, -1]
    
    # Compute log probabilities
    log_p = -0.5 * (zs**2)
    log_q = -0.5 * ((zs - mu_q.unsqueeze(1))**2 / log_sig_q.exp().unsqueeze(1).unsqueeze(2)**2) - log_sig_q.unsqueeze(1)
    
    # Sum over latent dimensions, mean over batch and samples
    return (log_q - log_p).sum(dim=2).mean()

def log_p_x(x, mu_xs, sig_x):
    """
    Compute log probability of x under the model.
    
    Args:
        x (torch.Tensor): Input data, shape [batch, ...]
        mu_xs (torch.Tensor): Reconstructed means, shape [batch, n_samples, ...]
        sig_x (float): Standard deviation of the output distribution
    
    Returns:
        torch.Tensor: Log probability
    """
    x_flat = x.reshape(x.size(0), 1, -1)
    mu_xs_flat = mu_xs.reshape(mu_xs.size(0), mu_xs.size(1), -1)
    
    squared_error = (x_flat - mu_xs_flat)**2 / (2 * sig_x**2)
    log_prob = -(squared_error + torch.log(sig_x))
    
    return log_prob.sum(dim=2).mean()

def rsample(phi, n_samples):
    """
    Sample z from q(z|x) using the reparameterization trick.
    
    Args:
        phi (torch.Tensor): Parameters of q, shape [b, K+1]
        n_samples (int): Number of samples to draw
    
    Returns:
        torch.Tensor: Samples from q(z|x), shape [b, n_samples, K]
    """
    mu, log_sig = phi[:, :-1], phi[:, -1]
    eps = torch.randn(mu.size(0), n_samples, mu.size(1), device=phi.device)
    return eps * log_sig.exp().unsqueeze(1) + mu.unsqueeze(1)
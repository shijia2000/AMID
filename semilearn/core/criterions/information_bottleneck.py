
import torch
import torch.nn as nn


class IBLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class vaeLoss(nn.Module):
    def forward(self, mu, logvar):
        return - 0.01 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
# def cal_vae_loss(mu, logvar):
#     return - 0.01 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
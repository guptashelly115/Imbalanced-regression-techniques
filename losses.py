import torch
import torch.nn.functional as F
import math

def focal_R_loss(inputs, targets, gamma = 5):
    # This is my implementation for focal-r loss based on the article - https://czxttkl.com/2020/10/26/focal-loss-for-classification-and-regression/
    loss = torch.abs(inputs - targets)
    loss = torch.pow(loss, 2+gamma)
    loss = torch.mean(loss)
    return loss

def exponentially_weighted_mse_loss(inputs, targets):
    # This is my implementation for custom weighted loss. Here, labels 0 or closer to zero are abundant and are penalized more as compared to rarer outcomes that are further away from 0 and closer to 1.
    loss = F.mse_loss(inputs, targets, reduce=False).squeeze()
    weights = torch.tensor([math.exp(math.exp(t)) for t in targets]).T
    loss *= weights
    loss = torch.mean(loss)
    return loss

def shrinkage_loss(inputs, targets, a = 12, c = 0.1):
    # This is my implementation of the shrinkage loss based on the article - https://czxttkl.com/2020/10/26/focal-loss-for-classification-and-regression/ 
    l = torch.abs(inputs - targets)
    numerator = torch.pow(l, 2)
    denominator = 1 + torch.exp(a * (c - l))
    loss = numerator / denominator
    loss = torch.mean(loss)
    return loss
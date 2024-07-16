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
    weights = torch.tensor([math.exp(t) for t in targets]).T
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

################################################################################################################################################
# Following code is from the mentioned github repository - https://github.com/YyzHarry/imbalanced-regression/blob/main/tutorial/tutorial.ipynb #
################################################################################################################################################

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def bmse_loss(inputs, targets, noise_sigma=8.):
    return bmc_loss(inputs, targets, noise_sigma ** 2)

def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))
    loss = loss * (2 * noise_var)
    return loss

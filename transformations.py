import torch
from scipy import stats

def log_transform(y, epsilon=1e-6):
    """
    Apply log transformation to the tensor y with a small constant epsilon to avoid log(0).

    Parameters:
    y (torch.Tensor): The tensor to be transformed.
    epsilon (float): A small constant added to avoid log(0).

    Returns:
    torch.Tensor: The log-transformed tensor.
    """
    return torch.log(y + epsilon)

def inverse_log_transform(y_transformed, epsilon=1e-6):
    """
    Inverse the log transformation i.e. exp(y_transformed)

    Parameters:
    y_transformed (torch.Tensor): The log-transformed tensor.
    epsilon (float): The same small constant used during log transformation.

    Returns:
    torch.Tensor: The inverse log-transformed tensor.
    """
    return torch.exp(y_transformed) - epsilon

def box_cox_transform(y, lambda_=None):
    """
    Apply the Box-Cox transformation to the tensor y.

    Parameters:
    y (torch.Tensor): The tensor to be transformed.
    lambda_ (float, optional): The lambda parameter for Box-Cox transformation. 
                               If None, the optimal lambda is calculated.

    Returns:
    torch.Tensor: The Box-Cox transformed tensor.
    float: The lambda used for the transformation.
    """
    y_np = y.numpy()
    
    # Find the optimal lambda if not provided
    if lambda_ is None:
        y_transformed_np, lambda_ = stats.boxcox(y_np + 1e-6)  # Adding a small constant to avoid issues with zero values
    else:
        y_transformed_np = stats.boxcox(y_np + 1e-6, lmbda=lambda_)

    y_transformed = torch.tensor(y_transformed_np)
    
    return y_transformed, lambda_

def inverse_box_cox_transform(y_transformed, lambda_ = 0):
    """
    Inverse the Box-Cox transformation.

    Parameters:
    y_transformed (torch.Tensor): The Box-Cox transformed tensor.
    lambda_ (float): The lambda parameter used for the Box-Cox transformation.

    Returns:
    torch.Tensor: The inverse Box-Cox transformed tensor.
    """
    y_transformed_np = y_transformed.numpy()
    
    # Apply the inverse Box-Cox transformation
    if lambda_ == 0:
        y_original_np = np.exp(y_transformed_np) - 1e-6
    else:
        y_original_np = (y_transformed_np * lambda_ + 1) ** (1 / lambda_) - 1e-6

class YeoJohnsonTransform(torch.nn.Module):
    def __init__(self, lmbda=1.0):
        super(YeoJohnsonTransform, self).__init__()
        self.lmbda = lmbda

    def forward(self, x):
        pos_mask = (x >= 0).float()
        neg_mask = (x < 0).float()
        
        if self.lmbda != 0:
            pos_transformed = ((x + 1).pow(self.lmbda) - 1) / self.lmbda * pos_mask
        else:
            pos_transformed = torch.log(x + 1) * pos_mask
        
        if self.lmbda != 2:
            neg_transformed = (-(torch.abs(x) + 1).pow(2 - self.lmbda) + 1) / (2 - self.lmbda) * neg_mask
        else:
            neg_transformed = -torch.log(torch.abs(x) + 1) * neg_mask
        
        return pos_transformed + neg_transformed

    y_original = torch.tensor(y_original_np)
    
    return y_original

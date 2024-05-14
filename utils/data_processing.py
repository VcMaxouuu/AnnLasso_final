import numpy as np
import torch
import pandas as pd


class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler for tensors

        Param:
            mean: The mean of the features. The property will be set after a call to fit.
            std: The standard deviation of the features. The property will be set after a call to fit.
            epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


def X_to_tensor(X) -> torch.Tensor:
    """Transform input data X to a PyTorch tensor and performs standardization.

    Args:
        X (array-like or pandas.DataFrame): Input data.

    Returns:
        X (torch.Tensor): Standardized input data as a PyTorch tensor.
    """
    if isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values, dtype= torch.float)
    else:
        X = torch.tensor(X, dtype=torch.float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X
        

def y_to_tensor(y) -> torch.Tensor:
    """Convert target data y to a PyTorch tensor.

    Args:
        y (array-like or pandas.DataFrame): Target data.

    Returns:
        y (torch.Tensor): Target data as a PyTorch tensor.
    """ 
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = torch.tensor(y.values.squeeze(), dtype=torch.float)
    else:
        y = torch.tensor(y, dtype=torch.float)
    return y

def data_to_tensor(X, y) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert input data X and target data y to a PyTorch tensor.

    Args:
        X (array-like or pandas.DataFrame): Input data.
        y (array-like or pandas.DataFrame): Target data.

    Returns:
        X (torch.Tensor): Standardized input data as a PyTorch tensor.
        y (torch.Tensor): Target data as a PyTorch tensor.
    """
    X = X_to_tensor(X)
    y = y_to_tensor(y)
    return X, y


def generate_linear_data(X, s):
    """Genrate y values and features given data X.

    Args:
        X (array-like or pandas.DataFrame): Input data
        s (int): Number of features
    
    Returns:
        y (pandas.DataFrame): Target data
        features (array-like): Features indices
    """
    n, p = X.shape
    SNR = 10/np.sqrt(n)
    beta = SNR * np.ones(s)
    inds = np.random.choice(range(p), s, replace=False)
    if isinstance(X, pd.DataFrame):
        y = np.dot(X.iloc[:, inds], beta) + np.random.normal(size=(n))
    else:
        y = np.dot(X[:, inds], beta) + np.random.normal(size=(n))

    y = pd.DataFrame(y)
    features = inds 
    return y, features
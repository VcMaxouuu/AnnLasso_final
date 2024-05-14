import torch
import torch.nn as nn
import numpy as np
import utils
from .ModelArchitecture import ModelArchitecture

class AnnLassoClassification(ModelArchitecture):
    def __init__(self, p2=20, lambda_qut=None, device=None):
        super.__init__(model_type = 1, p2=p2, lambda_qut=lambda_qut, device=device)

    def forward(self, X):
        pass
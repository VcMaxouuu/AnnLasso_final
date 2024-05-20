import torch
import torch.nn as nn
import numpy as np
import utils
from torch.nn.functional import linear, normalize, softmax
from .ModelArchitecture import ModelArchitecture

class AnnLassoClassification(ModelArchitecture):
    def __init__(self, penalty=0, p2=20, lambda_qut=None, device=None):
        super().__init__(penalty=penalty, model_type = 1, p2=p2, lambda_qut=lambda_qut, device=device)


    def forward(self, X):
        layer1_output = self.act_fun(self.layer1(X))
        w2_weights_normalized = normalize(self.layer2.weight, p=2, dim=1)
        logits = linear(layer1_output, w2_weights_normalized, self.layer2.bias)
        return logits # No softamx here cuz that's done with the crossentropy loss

    
    def train(self, nu, X, y, lambda_, initial_lr, rel_err, loss_fn, verbose, param_history):
        layer1, layer2 = self.layer1, self.layer2

        epoch, last_loss = 0, np.inf
        optimizer_l1 = utils.FISTA(params=layer1.parameters(), lr=initial_lr, lambda_=lambda_, nu=nu)
        optimizer_l2 = utils.FISTA(params=layer2.parameters(), lr=initial_lr, lambda_=torch.tensor(0, dtype=torch.float, device = self.device), nu=nu)

        lr_factor = 0.9
        max_epochs = 10000

        if param_history: self.layer1_history[lambda_.item()] = [layer1.weight.data.clone()]

        while epoch < max_epochs:
            optimizer_l1.zero_grad()
            optimizer_l2.zero_grad()

            y_pred = self.forward(X)
            loss, bare_loss = loss_fn(y_pred, y, layer1)
            loss = loss.detach()

            if verbose and epoch % 20 ==0:
                print(f"\tEpoch: {epoch} | Loss: {loss.item():.5f} | learning rate : {optimizer_l1.get_lr():.6f}")

            if loss > last_loss: 
                optimizer_l1.update(optimizer_l1.get_lr()*lr_factor)
                optimizer_l2.update(optimizer_l2.get_lr()*lr_factor)

            if epoch % 10 == 0:
                if epoch > 0 and abs(loss - last_loss) / loss < rel_err:
                    if verbose: print(f"\n\t Descent stopped: loss is no longer decreasing significantly.\n")
                    break
                last_loss = loss
                
            epoch += 1
            bare_loss.backward()
            optimizer_l1.step()
            optimizer_l2.step()

            if param_history: self.layer1_history[lambda_.item()].append(layer1.weight.data.clone())

        if epoch == max_epochs and verbose: print("FISTA descent stopped: maximum iterations reached") 

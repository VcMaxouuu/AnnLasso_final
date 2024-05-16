import torch 
import torch.nn as nn
import utils
import numpy as np
import pandas as pd
from torch.nn.functional import normalize, linear

class ModelArchitecture:
    def __init__(self, model_type=0, p2=20, lambda_qut=None, device=None):
        self.model_type = model_type
        self.p2 = p2
        self.lambda_qut = lambda_qut
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') if device is None else device
        self.act_fun = utils.Custom_act_fun()

        # Attributes to be modified after training
        self.trained = False
        self.layer1, self.layer2 = None, None
        self.important_features = None
        self.layer1_simplified = None
        self.layer1_history = {}


    def fit(self, X, y, verbose=False, param_history=False):
        if self.trained:
            raise Exception("Model already trained, call 'reset' method first.")
            
        X, y = utils.data_to_tensor(X, y)
        X, y = X.to(self.device), y.to(self.device)
        
        n_features, p1 = X.shape
        self.layer1 = nn.Linear(p1, self.p2, dtype=torch.float, device=self.device)
        self.layer2 = nn.Linear(self.p2, 1, dtype=torch.float, device=self.device)
        nn.init.normal_(self.layer1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.layer2.weight, mean=0.0, std=0.01)

        if self.lambda_qut is None:
            if self.model_type == 0:
                self.lambda_qut = utils.lambda_qut_regression(X, self.act_fun)
            elif self.model_type == 1:
                self.lambda_qut = utils.lambda_qut_classification(X, self.act_fun)
        else:
            if isinstance(self.lambda_qut, torch.Tensor):
                self.lambda_qut = self.lambda_qut.to(self.device, dtype=torch.float)
            else:
                self.lambda_qut = torch.tensor(self.lambda_qut, dtype=torch.float, device=self.device)

        self.train_loop(X, y, verbose, param_history)

        self.important_features = self.imp_feat()
        self.layer1_simplified = nn.Linear(self.important_features[0], self.p2, device=self.device)
        self.layer1_simplified.weight.data, self.layer1_simplified.bias.data = self.layer1.weight.data[:, self.important_features[1]].clone(), self.layer1.bias.data.clone()

        if verbose: print("MODEL FITTED !")
        self.trained = True

    def train_loop(self, X, y, verbose, param_history):
        for i, nu in zip(range(-1, 6), [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]):
            lambi = self.lambda_qut * (np.exp(i) / (1 + np.exp(i)) if i < 5 else 1)
            rel_err = 1e-9 if i == 5 else 1e-5
            loss_fn = utils.CustomRegressionLoss(lambi, nu).to(self.device) if self.model_type == 0 else utils.CustomClassificationLoss(lambi, nu).to(self.device)
            
            if verbose:
                print(f"Lambda = {lambi.item():.4f} -- Nu = {nu}")
                
            self.train(nu, X, y, lambi, 0.1, rel_err, loss_fn, verbose, param_history)

    def reset(self):
        self.trained = False
        self.important_features = None
        self.layer1_simplified = None
        self.layer1, self.layer2 = None, None
        
    def imp_feat(self):
        weight = self.layer1.weight.data
        non_zero_columns = torch.any(weight != 0, dim=0)
        indices = torch.where(non_zero_columns)[0]
        count = torch.sum(non_zero_columns).item()
        return count, sorted(indices.tolist())

    def apply_feature_selection(self, X):
        input_type = type(X)
        X_tensor = utils.X_to_tensor(X).to(self.device)
        X_selected = X_tensor[:, self.important_features[1]]

        if input_type == pd.DataFrame:
            return pd.DataFrame(X_selected.cpu().numpy(), columns=[X.columns[i] for i in self.important_features[1]])
        if input_type == torch.Tensor:
            return X_selected
        else:
            return X_selected.cpu().numpy()

    def fit_and_apply(self, X, y, verbose=False):
        self.fit(X, y, verbose)
        X = self.apply_feature_selection(X)
        if verbose: print("Features selection applied !")
        return X
            
    def predict(self, X):
        if not self.trained:
            raise Exception("Model not trained, call 'fit' method first.")

        X = utils.X_to_tensor(X).to(self.device)
        X = self.apply_feature_selection(X)
        with torch.inference_mode():
            layer1_output = self.act_fun(self.layer1_simplified(X))
            w2_weights_normalized = normalize(self.layer2.weight, p=2, dim=1)
            logits = linear(layer1_output, w2_weights_normalized, self.layer2.bias)
        return logits.squeeze().numpy()

    def lasso_path(self):
        if not self.trained:
            raise Exception("Model not trained, call 'fit' method first and set 'param_history' to True.")
        if self.trained and not bool(self.layer1_history):
            raise Exception("Model has been trained with 'param_history' set to False.")

        return utils.lasso_path(self.layer1_history, self.important_features[1])
    
    def layer1_evolution(self):
        if not self.trained:
            raise Exception("Model not trained, call 'fit' method first and set 'param_history' to True.")
        if self.trained and not bool(self.layer1_history):
            raise Exception("Model has been trained with 'param_history' set to False.")

        return utils.draw_layer1_evolution(self.layer1_history)

    def info(self):
        print("MODEL INFORMATIONS:")
        print('=' * 20)
        print("General:")
        print('―' * 20)
        print(f"  Training Status: {'Trained' if self.trained else 'Not Trained'}")
        if self.trained:
            print(f"  Lambda_qut: {np.round(self.lambda_qut.item(), 4)}\n")
            print("Layers:")
            print('―' * 20)
            print("  Layer 1: ")
            print(f"\t Shape = {list(self.layer1.weight.shape)}")
            print(f"\t Number of non zero entries in weights: {self.important_features[0]}")
            print(f"\t Non zero entries indexes: {self.important_features[1]}")
            print(f"\t Call 'layer1_simplified' attribute to get full, non zero, first layer.")
            print("  Layer 2:")
            print(f"\t Shape = {list(self.layer2.weight.shape)}")





    
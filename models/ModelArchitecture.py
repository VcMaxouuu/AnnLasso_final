import torch 
import torch.nn as nn
import utils
import numpy as np
import pandas as pd
from torch.nn.functional import normalize, linear

class ModelArchitecture:
    def __init__(self, penalty=0, model_type=0, p2=20, lambda_qut=None, device=None):
        self.penalty = penalty # 0 for custom and 1 for l1
        self.model_type = model_type # 0 for regression and 1 for classification
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

        if self.model_type == 1:
            y = y.long()
            hat_p = utils.get_hat_p(y)

        X, y = X.to(self.device), y.to(self.device)

        n_features, p1 = X.shape

        output_dim = 1 if self.model_type == 0 else len(hat_p)
        self.layer1 = nn.Linear(p1, self.p2, dtype=torch.float, device=self.device)
        self.layer2 = nn.Linear(self.p2, output_dim, dtype=torch.float, device=self.device)
        nn.init.normal_(self.layer1.weight, mean=0.0, std=10) #! std = y.std() ??!!
        nn.init.normal_(self.layer2.weight, mean=0.0, std=10)

        if self.lambda_qut is None:
            if self.model_type == 0:
                self.lambda_qut = utils.lambda_qut_regression(X, self.act_fun)
            elif self.model_type == 1:
                self.lambda_qut = utils.lambda_qut_classification(X, hat_p, self.act_fun)
        else:
            if isinstance(self.lambda_qut, torch.Tensor):
                self.lambda_qut = self.lambda_qut.to(self.device, dtype=torch.float)
            else:
                self.lambda_qut = torch.tensor(self.lambda_qut, dtype=torch.float, device=self.device)

        self.train_loop(X, y, verbose, param_history)
        
        # Check if returning mean(y) gives better results
        if self.model_type == 0 :
            if self.penalty == 0:
                loss_fn = utils.CustomRegressionLoss(self.lambda_qut, 0.1).to(self.device)
            else:
                loss_fn = utils.CustomRegressionLoss(self.lambda_qut, 1).to(self.device)
            model_error = loss_fn(self.forward(X), y, self.layer1)[0]

            metric = torch.nn.MSELoss(reduction='sum')
            baseline_error = torch.sqrt(metric(y.mean(), y))

            if model_error > baseline_error:
                self.layer1.weight.data.fill_(0)
                self.layer1.bias.data.fill_(0)
                self.layer2.weight.data.fill_(0)
                self.layer2.bias.data.fill_(y.mean().item())

        self.important_features = self.imp_feat()
        self.layer1_simplified = nn.Linear(self.important_features[0], self.p2, device=self.device)
        self.layer1_simplified.weight.data, self.layer1_simplified.bias.data = self.layer1.weight.data[:, self.important_features[1]].clone(), self.layer1.bias.data.clone()

        if verbose: print("MODEL FITTED !")
        self.trained = True

    def train_loop(self, X, y, verbose, param_history):
        for i in range(-1, 6):
            nu = 1 if self.penalty == 1 else [1, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1][i + 1]
            lambi = self.lambda_qut * (np.exp(i) / (1 + np.exp(i)) if i < 5 else 1)
            rel_err = 1e-9 if i == 5 else 1e-5
            loss_fn = utils.CustomRegressionLoss(lambi, nu).to(self.device) if self.model_type == 0 else utils.CustomClassificationLoss(lambi, nu).to(self.device)
            init_lr = 0.1*0.8**(i+1)

            if verbose:
                print(f"Lambda = {lambi.item():.4f} -- Nu = {nu}" if self.penalty == 0 else f"Lambda = {lambi.item():.4f}")
            
            self.train(nu, X, y, lambi, init_lr, rel_err, loss_fn, verbose, param_history)

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

        if self.model_type == 0:
            return logits.squeeze().numpy() # Regression

        softmax = nn.Softmax(dim=1)
        return torch.argmax(softmax(logits), dim=1).numpy() # Classification

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





    
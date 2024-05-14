import torch

def function_derivative(func, u):
    y = func(u)
    y.backward()
    return u.grad.item()


def lambda_qut_regression(X, act_fun, n_samples=10000, mini_batch_size=500, alpha=0.05, option='quantile'):
    offset = 0 if n_samples % mini_batch_size == 0 else 1
    n_samples_per_batch = n_samples // mini_batch_size + offset
    
    n, p1 = X.shape    
    fullList = torch.zeros((mini_batch_size*n_samples_per_batch,), device= X.device)

    for index in range(n_samples_per_batch):
        y_sample = torch.normal(mean=0., std=1, size=(n, 1, mini_batch_size)).type(torch.float).to(X.device)
        y = (torch.mean(y_sample, dim=0)- y_sample)
        xy = y * X.unsqueeze(2).expand(-1, -1, mini_batch_size)
        xy_max = torch.amax(torch.abs(torch.sum(xy, dim=0)), dim=0)
        norms = torch.norm(y, p=2, dim=0)
        fullList[index * mini_batch_size:(index+1) * mini_batch_size]= xy_max/norms
    
    fullList = fullList * function_derivative(act_fun, torch.tensor(0, dtype=torch.float, requires_grad=True, device = X.device))
 
    if option=='full':
        return fullList
    elif option=='quantile':
        return torch.quantile(fullList, 1-alpha)
    else:
        pass

def lambda_qut_classification(X, act_fun, n_samples=10000, mini_batch_size=500, alpha=0.05, option='quantile'):
    pass


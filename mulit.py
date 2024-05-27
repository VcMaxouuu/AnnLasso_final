from models import AnnLassoRegression, AnnLassoClassification, LinearModel
import utils
import pandas as pd
import numpy as np
import torch
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

def simulate(index, X, s, lambda_qut, cont):
    y, features = utils.generate_linear_data(X, s)
    pesr = {}
    
    if cont['linear old']:
        model = LinearModel(penalty=1, lambda_qut=lambda_qut)
        model.fit(X, y)
        pesr["linear old"] = set(model.important_features[1]) == set(features)
    else:
        pesr["linear old"] = 0.0

    if cont['neural old']:
        model = AnnLassoRegression(penalty=1, lambda_qut=lambda_qut)
        model.fit(X, y)
        pesr["neural old"] = set(model.important_features[1]) == set(features)
    else:
        pesr["neural old"] = 0.0

    if cont['linear new']:
        model = LinearModel(penalty=0, lambda_qut=lambda_qut)
        model.fit(X, y)
        pesr["linear new"] = set(model.important_features[1]) == set(features)
    else:
        pesr["linear new"] = 0.0

    if cont['neural new']:
        model = AnnLassoRegression(penalty=0, lambda_qut=lambda_qut)
        model.fit(X, y)
        pesr["neural new"] = set(model.important_features[1]) == set(features)
    else:
        pesr["neural new"] = 0.0

    return pesr

if __name__ == '__main__':
    for n in [70, 100, 130]:
        X = pd.read_csv(f"data/linear/n{n}/X-n{n}-p300.csv")
        X_tensor = utils.X_to_tensor(X)
        lambda_qut = 0.8*utils.lambda_qut_regression(X_tensor, utils.Custom_act_fun())

        names = ["linear old", "neural old", "linear new", "neural new"]
        cont = {name: True for name in names}
        pesr_means = {name: [] for name in names}

        if n==70:
            df = pd.read_csv('data/linear/n70/PESR-0.8lambda.csv')

            for name in names:
                pesr_means[name] = df[name].tolist()
            s_values = np.arange(15, 35)
        else:
            s_values = np.arange(0, 35)

        
        for s in s_values:
            pesr = {name: [] for name in names}
            tasks = [(m, X.copy(), s, lambda_qut, cont.copy()) for m in range(200)]

            if not any(cont.values()):
                for name in names:
                    pesr_means[name].append(0.0)

                df = pd.DataFrame(pesr_means)
                df.to_csv(f"data/linear/n{n}/PESR-0.8lambda.csv", index=False)
                continue
            
            with Pool(processes=8) as pool:  
                results = pool.starmap(simulate, tasks)

            for result in results:
                for name in names:
                    pesr[name].append(result[name])

            for name in names:
                pesr_means[name].append(np.mean(pesr[name]))
                cont[name] = pesr_means[name][-1] != 0.0

            df = pd.DataFrame(pesr_means)
            df.to_csv(f"data/linear/n{n}/PESR-0.8lambda.csv", index=False)

    
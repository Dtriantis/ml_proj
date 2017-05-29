import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LassoCV

names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
         "Viscera weight", "Shell weight", "Rings"]

df = pd.read_csv("abalone.data", names=names, header=None)

print df.head()

mapping = dict(zip(["I", "F", "M"], [0, 1, 2]))
df.replace({"Sex": mapping}, inplace=True)

targets = df["Rings"]
del df["Rings"]

# lasso print ta varh gia na doume ti kanei 0
# oti kanei 0 shmainei oti ta featues den exoyn smasia
alphas = np.asarray([0.000001, 0.01, 0.05, 0.1, 0.2, 0.5, 1])

lasso_cv = LassoCV(alphas=alphas, cv=20)
lasso_cv.fit(df, targets)

print lasso_cv.mse_path_.mean(axis = 1)

print lasso_cv.alpha_

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
         "Viscera weight", "Shell weight", "Rings"]

df = pd.read_csv("abalone.data", names=names, header=None)

mapping = dict(zip(["I", "F", "M"], [0, 1, 2]))
df.replace({"Sex": mapping}, inplace=True)
df.head()

targets = df["Rings"]
del df["Rings"]

regr = linear_model.LinearRegression()
print(cross_val_score(regr, df, targets, cv = 10, scoring=make_scorer(mean_squared_error)).mean())

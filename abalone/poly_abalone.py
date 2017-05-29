import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, make_scorer

names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
         "Viscera weight", "Shell weight", "Rings"]

df = pd.read_csv("abalone.data", names=names, header=None)

df.head()

mapping = dict(zip(["I", "F", "M"], [0, 1, 2]))
df.replace({"Sex": mapping}, inplace=True)
targets = df["Rings"]
del df["Rings"]

regr = linear_model.LinearRegression()

for degr in range(1,10):

	poly = PolynomialFeatures(degree = degr)
	x_trans = poly.fit_transform(df)

	print(cross_val_score(regr, x_trans, targets, cv = 10, scoring=make_scorer(mean_squared_error)).mean())

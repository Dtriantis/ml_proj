from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, make_scorer


names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight",
         "Viscera weight", "Shell weight", "Rings"]

df = pd.read_csv("abalone.data", names=names, header=None)

mapping = dict(zip(["I", "F", "M"], [0, 1, 2]))
df.replace({"Sex": mapping}, inplace=True)

targets = df["Rings"]
del df["Rings"]

df=np.asarray(df)
targets=np.asarray(targets)

# define 10-fold cross validation
kfold = KFold(n_splits=10, shuffle=True)
cvscores = []

for train, test in kfold.split(df,targets):
	model = Sequential()
	model.add(Dense(170, input_dim=8))
	model.add(Activation('relu'))
	model.add(Dense(1, activation='linear'))
	
	#mean squared error regression
	model.compile(optimizer='adam',
	              loss='mae')
	
	model.fit(df[train], targets[train], epochs=100, batch_size=32)
	scores = model.evaluate(df[test], targets[test], verbose=0)
	cvscores.append(scores)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

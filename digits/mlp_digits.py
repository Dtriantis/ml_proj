from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV 
from sklearn.neural_network import MLPClassifier
import scipy.io
import numpy as np

digits = scipy.io.loadmat("digits.mat")
X = digits["learn"][0,0][0]
X = np.swapaxes(X,0,1)
y = digits["learn"][0,0][1]
y =np.squeeze(y)

tuned_parameters = {'hidden_layer_sizes': [100, 150, 170, 200, 220], 'alpha': [0.00001, 0.0001], 'learning_rate_init': [0.001, 0.01]}

classifier_ = MLPClassifier(activation='relu', batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

classifier = GridSearchCV(classifier_, cv=10, param_grid=tuned_parameters)

classifier_.fit(X,y)

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y, classifier_.predict(X)))

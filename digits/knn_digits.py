from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier
import scipy.io
import numpy as np

digits = scipy.io.loadmat("digits.mat")
X = digits["learn"][0,0][0]
X = np.swapaxes(X,0,1)
y = digits["learn"][0,0][1]
y =np.squeeze(y)

tuned_parameters = {'metric': ['hamming', 'euclidean'], 'n_neighbors': [3, 5, 10, 15, 20, 30]}

classifier_ = KNeighborsClassifier(algorithm='brute', leaf_size=30,
           metric_params=None, n_jobs=1, p=2,
           weights='uniform')

classifier = GridSearchCV(classifier_, cv=10, param_grid=tuned_parameters)

classifier.fit(X,y)

print "The best accuracy was: {0:f} \n".format(classifier.best_score_)
print "The best distance metric was: {0:s} \n".format(classifier.best_params_['metric'])
print "The best number of neighbors was: {0:f} \n".format(classifier.best_params_['n_neighbors'])

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y, classifier.predict(X)))

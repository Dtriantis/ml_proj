from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score, GridSearchCV 
from sklearn import svm
import scipy.io
import numpy as np

digits = scipy.io.loadmat("digits.mat")
X = digits["learn"][0,0][0]
X = np.swapaxes(X,0,1)
y = digits["learn"][0,0][1]
y =np.squeeze(y)

# Set the parameters by cross-validation
tuned_parameters = {'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}

classifier_ = svm.SVC()

classifier = GridSearchCV(classifier_, cv=10, param_grid=tuned_parameters)

classifier.fit(X,y)

print "The best accuracy was: {0:f} \n".format(classifier.best_score_)
print "The best kernel was: {0:s} \n".format(classifier.best_params_['kernel'])
print "The best gamma was: {0:f} \n".format(classifier.best_params_['gamma'])
print "The best C was: {0:f} \n".format(classifier.best_params_['C'])

classifier_ = svm.SVC(kernel="rbf", gamma=0.001, C=10)
classifier_.fit(X,y)

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y, classifier_.predict(X)))

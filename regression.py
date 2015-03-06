import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Returns a model fitted to (X,y) using Logistic Regression
def LogRegression(X, y):
  """
  input: 
    X is the design matrix
    y = training output (i.e. t_train)
  return: 
    LogisticRegression object that is fitted to the data
  """
  model = LogisticRegression(penalty="l2") # using L2 norm penalty
  
  return model.fit(X, y)

def DecisionTree(X, y):
  return DecisionTreeClassifier().fit(X.toarray(),y)

def BayRidge(X, y):
  model = linear_model.BayesianRidge()
  return model.fit(X.toarray(), y)

def Percept(X, y):
  model = linear_model.Perceptron()
  return model.fit(X, y)

def SGD():
  model = linear_model.SGDClassifier(shuffle=True)
  return model

def SGD_hinge(X, y):
  model = linear_model.SGDClassifier(loss="hinge", penalty="l2", shuffle=True)
  return model.fit(X, y)

def SGD_huber(X, y):
  model = linear_model.SGDClassifier(loss="modified_huber", penalty="l2", shuffle=True)
  return model.fit(X, y)

def SGD_log(X, y):
  scaler = StandardScaler(with_mean=False)
  scaler.fit(X)
  X = scaler.transform(X)
  model = linear_model.SGDClassifier(loss="log", penalty="l2", shuffle=True)
  return model.fit(X, y)

def SVM_linear(X, y):
  model = svm.SVC(kernel="linear")
  return model.fit(X, y)

def SVM_polynomial(X, y):
  model = svm.SVC(kernel="polynomial")
  return model.fit(X, y)

def SVM_rbf(X, y):
  model = svm.SVC(kernel="rbf")
  return model.fit(X, y)

def SVM_sigmoid(X, y):
  model = svm.SVC(kernel="sigmoid")
  return model.fit(X, y)

#One-vs-the-rest with linear kernel
def SVM_Linear_One_v_All(X, y):
  model = svm.LinearSVC()
  return model.fit(X, y)
# Returns models prediction as list on the data X

def makePrediction(model, X):
  """
  args:
    model = a sklearn 'regression' object that is fitted to data
    X = test data to make predictions on
  return:
    a numpy array of class assignments/predictions to each datum in X
  """
  return model.predict(X)

def DecisionPrediction(model, X):
  return model.predict(X.toarray())
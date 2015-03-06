import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import svm

class RegressionModel:
  def __init__(self, X, y, reg):
    self.model = reg.fit(X, y)

  def makePrediction(X_test):
    return self.model.predict(X_test)

  def score(self, X, y):
    return self.model.score(X, y)

class LogRegression(RegressionModel):
  def __init__(self, X, y):
    RegressionModel.__init__(self, X, y, LogisticRegression(penalty="l2"))

class DecisionTree(RegressionModel):
  def __init__(self, X, y):
    self.model = DecisionTreeClassifier().fit(X.toarray(), y)

  def makePrediction(self, X):
    return self.model.predict(X.toarray())

  def score(self, X, y):
    return self.model.score(X.toarray(), y)

class BayRidge(RegressionModel):
  def __init__(self, X, y):
    self.model = linear_model.BayesianRidge().fit(X.toarray(), y)

  def makePrediction(self, X_test):
    return self.model.predict(X.toarray())

class Percept(RegressionModel):
  def __init__(self, X, y):
    self.model = linear_model.Perceptron().fit(X, y)

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
# def makePrediction(model, X):
#   """
#   args:
#     model = a sklearn 'regression' object that is fitted to data
#     X = test data to make predictions on
#   return:
#     a numpy array of class assignments/predictions to each datum in X
#   """
#   return model.predict(X)

# def DecisionPrediction(model, X):
#   return model.predict(X.toarray())
import numpy as np
from sklearn.linear_model import LogisticRegression

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
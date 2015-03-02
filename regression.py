import numpy as np
from sklearn.linear_model import LogisticRegression

def LogRegression(X, y):
  model = LogisticRegression(penalty="l2")
  return model.fit(X, y)

def makePrediction(model, X):
  return model.predict(X)
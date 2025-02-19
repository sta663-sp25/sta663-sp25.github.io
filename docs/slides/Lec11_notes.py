## Setup

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sklearn

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold



## Digits

from sklearn.datasets import load_digits
digits = load_digits(as_frame=True)

## Classification Tree

X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=1234
)

digits_tree = GridSearchCV(
  DecisionTreeClassifier(),
  param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": range(2,16)
  },
  cv = KFold(5, shuffle=True, random_state=12345),
  n_jobs = 4
).fit(
  X_train, y_train
)

digits_tree.best_estimator_
digits_tree.best_score_

accuracy_score(y_test, digits_tree.best_estimator_.predict(X_test))
confusion_matrix(
  y_test, digits_tree.best_estimator_.predict(X_test)
)


### GridSearchCV w/ multiple models

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

p = Pipeline([
  ("model", DecisionTreeClassifier())
])


digits_tree = GridSearchCV(
  p,
  param_grid = {
    "model": [
      DecisionTreeClassifier(),
      RandomForestClassifier()
    ],
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": range(2,10)
  },
  cv = KFold(5, shuffle=True, random_state=12345),
  n_jobs = 4
).fit(
  X_train, y_train
)

digits_tree.best_estimator_
digits_tree.best_score_

accuracy_score(y_test, digits_tree.best_estimator_.predict(X_test))
confusion_matrix(
  y_test, digits_tree.best_estimator_.predict(X_test)
)

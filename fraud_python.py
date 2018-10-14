import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier, RidgeCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

finance_data = pd.read_csv('creditcard.csv')
X_data = finance_data.iloc[:, 0:-1]
Xtranspose = X_data.T
labels = finance_data.iloc[:, -1]
print(finance_data)

classifier = RidgeClassifier(alpha = 1.5, max_iter=100, class_weight='balanced').fit(X_data, labels)
classifier2 = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=10).fit(X_data, labels)
adaboost = AdaBoostClassifier( n_estimators=250, learning_rate=1.5, algorithm='SAMME')

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X_data):
    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    adaboost.fit(X_train, y_train)
    predictions = adaboost.predict(X_test)
    print(accuracy_score(y_test, predictions))
    print(adaboost.score(X_test, y_test))


'''from the following results, it can be clearly shown that result one
is consistent with overfitting'''
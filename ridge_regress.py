import linear_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import minmax_scale

def seperate(nparray):
    return nparray[:, 0], nparray[:, 1]


training_data = pd.read_csv('MNIST_15_15.csv')
labels = pd.read_csv('MNIST_LABEL.csv')
normed_data = minmax_scale(training_data)
normed_data = pd.DataFrame(normed_data)

#combined = np.concatenate((labels, training_data), axis=1)

pos_neg = []
vectorized_function = np.vectorize(lambda x: 1 if x > 5 else -1)
pos_neg = vectorized_function(labels)
pos_neg = pd.DataFrame(pos_neg)


lambdas = [float("-inf"), -1, -0.1, 0.1, 1, float("Inf")]

predicted = []
roc_data = []
tpr = []
fpr = []
positive_sum = []
negative_sum = []
Accuracy = []


kf = KFold(n_splits=10)
for train_index, test_index in kf.split(normed_data):
    X_train, X_test = normed_data.iloc[train_index], normed_data.iloc[test_index]
    y_train, y_test = pos_neg.iloc[train_index], pos_neg.iloc[test_index]
    alpha = 1.5
    clf = linear_regression.ridge_regress(X_train, y_train, alpha, linear_regression.find_coef)
    for i in lambdas:
        for row in X_test.iterrows():
            rows = row[1]
            results = linear_regression.Rpredict(rows, clf.slope)
            vectorized_function = np.vectorize(lambda x: 1 if x >= i else -1)
            results = vectorized_function(results)
            predicted.append(results)
    predicted = np.array(predicted)
    predicted = np.vsplit(predicted, 6)

    for i in predicted:
        matrix = confusion_matrix(i, y_test)
        roc_data.append(np.array(matrix))
        i = pd.DataFrame(i)
        acc = accuracy_score(i, y_test)
        Accuracy.append(acc)
        #print(acc)
    roc_data = np.concatenate(roc_data, axis=0)
    positive, negative = seperate(roc_data)

    for i in range(1, len(positive), 2):
        sum = positive[i] + positive[i-1]
        positive_sum.append(sum)
    for i in range(1, len(negative), 2):
        sum = negative[i] + negative[i-1]
        negative_sum.append(sum)


    roc = np.delete(roc_data, list(range(1, roc_data.shape[0], 2)), axis=0)
    nroc = roc[:, 1].tolist()
    proc = roc[:, 0].tolist()


    for i, j in zip(proc, positive_sum):
        tpr.append(i/j)
    for i, j in zip(nroc, negative_sum):
        fpr.append(i/j)
    plt.plot(fpr, tpr)
    plt.show()
    predicted = []
    roc_data = []
    tpr = []
    fpr = []
    positive_sum = []
    negative_sum = []
Accuracy = np.array(Accuracy)
print(np.average(Accuracy))
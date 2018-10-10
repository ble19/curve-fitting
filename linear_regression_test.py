import linear_regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

if __name__ == '__main__':
    file_regtrain = 'housing_training.csv'
    file_regtest = 'housing_test.csv'
    regtrain = pd.read_csv(file_regtrain, header=None)
    regtest = pd.read_csv(file_regtest, header=None)

    ''' For the first part of the homework, make sure the training and test sets are in the proper format
    Next, make sure the output of the functions are in the correct format. In the case of the output for 
    predictions, it should contain a vector of the same shape as the observed values.
    The first 13 columns of the dataset are the independent variable and the last column is the dependent
    vars. 
    
    Expected outputs of the model are the optimal coefficients, and a plot of the ground truth
    versus the predictions.
    
    UPDATE MODEL DETERMINING WHICH COEF IS THE BEST, IE LOOP THROUGH DATASET AND KEEP THE DATA WITH THE
    LEAST ERROR'''
    independent_var = regtrain.iloc[0:206, 0:13]
    dependent_var = regtrain.iloc[0:206, -1]
    observed_values = regtest.iloc[:, -1]
    test_inputs = regtest.iloc[:, 0:13]
    linearModel = linear_regression.linear_model(independent_var, dependent_var, linear_regression.linear_prediction)
    predictions = linear_regression.predict(linearModel, test_inputs)
    print(np.mean(linearModel.intercept))
    LR_predictions = linear_regression.Linear_Regression(test_inputs, observed_values)
    display_predictions = predictions.T
    LR = plt.plot(observed_values, predictions.T, 'bo')
    plt.show()
    ax = sns.heatmap(linearModel.slope)
    plt.show()
    print(np.sum(linearModel.intercept))
    '''For part two of this assignment, we simple use a ridge regression model to test the outputs
    of various thresholds and create an ROC curve for each threshold to see how the model performs.
    The expected outputs it to see sparsity of the output matrix minimize but as the theshold increases
    toward infinity, you can expect the values to be zeroed out
    
    OUTPUTS: TPR/FPR, ROC for each threshold'''
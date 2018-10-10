import numpy as np
import pandas as pd
import math


'''UPDATE THE CLASS TO LOOP THROUGH THE DATASET AND KEEP LOWEST ERROR SLOPE?INTERCEPT'''
class linear_model:
    def __init__(self, ind_var=None, dep_var=None, function=None):
        self.ind_var = ind_var
        self.dep_var = dep_var
        self.function = function
        self.slope, self.intercept = function(self.ind_var, self.dep_var)

class ridge_regress:
    def __init__(self, ind_var=None, dep_var=None,  Lambda=None, function=None):
        self.ind_var = ind_var
        self.dep_var = dep_var
        self.Lambda = Lambda
        self.slope, self.intercept = function(self.ind_var, self.dep_var, Lambda)



def Root_MSE(observed_values, predictions, num_of_pred):
    return np.sqrt(np.dot((1/num_of_pred), np.sum(np.square(observed_values-predictions))));




def Linear_Regression(independent_variable, observed_variable): #observed = dependent var,
    coefficients = np.dot(np.linalg.pinv(np.dot(independent_variable.T, independent_variable)), independent_variable.T* observed_variable)
    prediction = independent_variable.T*coefficients
    error = Root_MSE(observed_variable, prediction, np.size(prediction))
    coef = pd.DataFrame(coefficients)
    coef_row = sum(coef[row] for row in coef)
    print( coef)
    final_prediction = np.sum(prediction + error)
    return final_prediction

def predict(linear_model, x_input):
    return np.sum(x_input.T*linear_model.slope+linear_model.intercept)

def predictR(linear_model, x_input):
    return x_input.dot(linear_model.slope)+linear_model.intercept

def predict1(linear_model, x_input):
    predictions = []
    for column in x_input:
        predictions.append(linear_model.slope*x_input[column]+linear_model.slope)
    predictions = pd.DataFrame(predictions)
    predictions = predictions.T
    return predictions


def linear_prediction(independent_variable, observed_variable):
    slope = np.dot(np.linalg.pinv(np.dot(independent_variable.T, independent_variable)),
                          independent_variable.T * observed_variable)
    prediction = independent_variable.T * slope
    intercept = Root_MSE(observed_variable, prediction, np.size(prediction))
    return slope, intercept


def linear_prediction1(independent_vect, dependent_vec):

    indepMean = np.mean(independent_vect)
    depMean = np.mean(dependent_vec)
    co1, co2, co3, co4 = np.cov(independent_vect, dependent_vec, bias=1).flat
    numerator = co2
    #denominator = np.sqrt(co1, co2)
    slope = numerator/co1
    intercept = depMean - slope*indepMean
    means = []
    for column in intercept:
        means.append(np.mean(intercept[column]))
    return slope, intercept

def ridge_regression(independent_variable, observed_variable, Lambda):
        coef = independent_variable.T.dot(independent_variable) + Lambda
        slope = np.linalg.inv(coef).dot(independent_variable.T.dot(observed_variable))
        prediction = np.dot(independent_variable, slope)
        intercept = Root_MSE(observed_variable, prediction, np.size(prediction))
        return slope, intercept


def find_coef(ind_var, dep_var, Lambda):
    if Lambda == 0:
        coefficients = np.dot(ind_var.T, ind_var)
    else:
        coefficients = np.dot(ind_var.T, ind_var) + Lambda*np.eye(ind_var.shape[1])
    weights = np.dot(np.linalg.inv(coefficients), (np.dot(ind_var.T, dep_var)))
    predictions = np.dot(ind_var, weights)
    intercept = Root_MSE(dep_var, predictions, np.size(predictions))
    return weights, intercept

def Rpredict(ind_var, weights):
    return ind_var.dot(weights)
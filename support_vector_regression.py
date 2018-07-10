import os
import sys
import numpy as np
import pandas as pd
import datetime
import sklearn as sk
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score



'''
                Documentation for Load forecasting of household power usage. 
                
    Data-set: The data was gathered as part of the London metropolitan area smart meter initiative. The main data is 
    energy usage gather from smart meters over the span of 3-4 years with a maximum household participation number of 
    ~5000 households. Data concerning weather and holidays are also included. 
    
    Purpose: The purpose of this project is to forecast the energy usage of household power consumption. This is the
    foundation of algorithms accounting for demand response in distribution systems. 
        Possible improvements beyond basic regression is adding a nonlinear kernal or module to account for stochastic 
        variables such as weather. The latest research in the area has developed reservoir programming that predicts 
        stochastic systems out to 8-12 units of lyapunov time. 
    Algorithms used:
        Pre-processing: scale and standard scaler from sklearn
        Cross-Validation: first attempt kfold. 
        Machine Learning Algorithm: Support vector regression. 
        
    Data Input: Household power meter data. Done within date range. Training set is done by household. 
                Test data is a set of data from a new household. Use off SVR to predict future data usage. 
                questions: do I need to provide future data usage? Data validation can be done by
                comparing the predicted data to actual data. 
                
    Expected trends: as a generalization I expect to see electricity usage to conform to the standard demand response 
    curve created due to regular work hours. The exceptions for the most part would occur due to holidays, weather 
    events, possibly vacation, or moving (among some of the possibilities. 
                
    Program Structure: 
        Import dataset function
        
        sklearn SVR
        
        accuracy of predicted vs actual
        
        SVR plot
        
'''

def check_none(indexminusone):
    if indexminusone is None:
        return True

def load_dataset():
    dataset = 'dataset file path here'
    list_dailyselection = []
    for file in os.listdir(dataset):
        df_daily = pd.read_csv(dataset+'\\'+file)
        df_daily_selection = df_daily[["LCLid", "day", "energy_sum"]]
        list_dailyselection.append(df_daily_selection)


    df_dailyselection = pd.concat(list_dailyselection, axis=0) 
    df_dailyselection["day"] = pd.to_datetime(df_dailyselection["day"])
    df_count = df_dailyselection.groupby(["day"]).count()


    start_date = datetime.datetime(year=2013, month=1, day=1)
    end_date = datetime.datetime(year=2014, month=1, day=1)
    df_dailyselection_zoom = df_dailyselection[
        (df_dailyselection["day"] >= start_date) & (df_dailyselection["day"] < end_date)]


    '''data = df_dailyselection_zoom.groupby('LCLid')
    [data.get_group(x) for x in data.groups]
    data = data[["LCLid", "energy_sum"]]
    for index in data:
        data[1][index]'''

    # create usable data format. In this case, label = household id, and the rest of the entry is the measure
    # this is using java logic, does not work because row values not tabled

    reformatted_datalist = []


    # here is going to be the code to save the file for future usage to avoid reformatting every time I run the program
    # numpy.savetxt("housingdailypowerusage.csv", reformatted_data, delimiter=",")
    # need to change this code to be more pythonic/ code is below, only way to debug is run block through it,
    # it's a pain to debug
    HouseholdIDS = pd.unique(df_dailyselection_zoom.LCLid)
    # reformatted_data = reformatted_datalist.append([df_dailyselection_zoom.loc[df_dailyselection_zoom['LCLid'] == index] for index in HouseholdIDS])
    # pd.DataFrame(reformatted_data).pivot_table(index='LCLid', columns='day', values='energy_sum')
    for index in HouseholdIDS:
        entrytable = df_dailyselection_zoom.loc[df_dailyselection_zoom['LCLid'] == index]
        # entry_data = pd.DataFrame(entrytable['energy_sum'])
        entrytable = pd.DataFrame(entrytable).pivot(index='LCLid', columns='day', values='energy_sum')
        reformatted_datalist.append(entrytable)
    result = pd.concat(reformatted_datalist)

    result.to_csv('~/Documents/reformatted_data.csv')
    return result

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

def main():
    dataset =  pd.read_csv('~/Documents/reformatted_data.csv')
    #dropped rows which had NANs to avoid data issues. Side effect is lowering households to high 4000's
    dataset.dropna(axis=0, how='any',inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    #trying to predict the next day's energy consumption based on past history, this splits the data appropriately
    input_data = dataset.iloc[:, 1:-2]
    HHids = dataset.iloc[:, -1]
    input_data.as_matrix()
    HHids.as_matrix()



    #Sets the models and cross validation. Allows for both linear regression and support vector regression.
    svmregression = SVR(cache_size=1000)
    LR = LinearRegression()
    kf = KFold(n_splits=10)
    #compared svr with linear regression
    model = input('Enter Model, LR or SVR: ')

    # used kfold cross validation
    # add  MSPE, MSAE, R square, adjusted R square
    for train, test in kf.split(dataset):
        X_train, X_test = input_data.iloc[train], input_data.iloc[test]
        Y_train, Y_test = HHids[train], HHids[test]

        if model == 'SVR':

            svr_ytrain = svmregression.fit(X_train, Y_train).predict(X_train)
            svr_ytest = svmregression.fit(X_test, Y_test).predict(X_test)
            svrtrain_mse = mean_squared_error(Y_train, svr_ytrain)
            svrtest_mse = mean_squared_error(Y_test, svr_ytest)
            svrtest_mae = sk.metrics.mean_absolute_error(Y_test, svr_ytest)
            explained_var = sk.metrics.explained_variance_score(Y_test, svr_ytest)
            r2 = sk.metrics.r2_score(Y_test, svr_ytest)
            print("MSE train:",  np.sqrt(svrtrain_mse))
            print("MSE test:",  np.sqrt(svrtest_mse))
            print("MAE:",  svrtest_mae)
            print("Explained Variance:", explained_var)
            print("r^2", r2)

        elif model == 'LR':

            lr_ytrain = LR.fit(X_train, Y_train).predict(X_train)
            lr_ytest = LR.fit(X_test, Y_test).predict(X_test)
            lrtrain_mse = mean_squared_error(Y_train, lr_ytrain)
            lrtest_mse = mean_squared_error(Y_test, lr_ytest)
            lrtest_mae = sk.metrics.mean_absolute_error(Y_test, lr_ytest)
            lr_explained_var = sk.metrics.explained_variance_score(Y_test, lr_ytest)
            lr2 = sk.metrics.r2_score(Y_test, lr_ytest)
            print("LR MSE train:", np.sqrt(lrtrain_mse))
            print("LR MSE test:", np.sqrt(lrtest_mse))
            print("LR MAE:", lrtest_mae)
            print("LR Explained Variance:", lr_explained_var)
            print("LR r2", lr2)
    lw = 1
    '''
    #need to tweak the code here to visualize the results. Currently non-functional
    # this is where I'll show the plots to give a visual idea of what is going on 
    plt.plot(X_test, Y_test, color='darkorange', lw=lw, label='data')
    plt.plot(X_train, svr_ytrain, color='navy', lw=lw, label='RBF model')
    plt.plot(X_test, svr_ytest, color='c', lw=lw, label='Linear model')

    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    plt.plot(X_train, lr_ytrain, color='navy', lw=lw, label='RBF model')
    plt.plot(X_test, lr_ytest, color='c', lw=lw, label='Linear model')
    plt.tight_layout()
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()'''


    if __name__ == "__main__":
        main()

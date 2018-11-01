import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

import tensorflow as tf
import keras

''' TO DO LIST
should have kept any answers from the training data, adjusted training data creation
adjusted data seemed to have corrected initial curve discrepancies but the small amount of 
hidden nodes seems to have cause underfitting to the nonlinear nature'''

def root_mean_squared_error(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))


def return_vectorIndex(vector, value):
    for row in vector.iterrows():
        if (row[1] != value).bool():
            continue
        else:
            return row[0]

#Code courtesy of Ravindra Kompella's Medium post
def forecast(n_future_predictions):
    #the num of future predictions we want to make
    forecasted_preds = []
    new_input1 = new_input.T
    expanded_test_data_list = np.array([new_input1[series.shape[1]-1, :, :].tolist()])
    #expanded_test_data_list = np.squeeze(expanded_test_data_list, axis=0)
    for index in range(n_future_predictions):
        one_step_predictions = model_LSTM.predict(expanded_test_data_list)
        forecasted_preds.append(one_step_predictions[0, 0])
        one_step_predictions = one_step_predictions.reshape(1, 1, 1)
        expanded_test_data_list = np.concatenate((expanded_test_data_list[:, 1:, :], one_step_predictions), axis=1)

    forecasted_preds = np.reshape(np.array(forecasted_preds), (-1, 1))
    forecasted_preds = scaler.inverse_transform(forecasted_preds)

    return forecasted_preds
''' This module uses the transformed data set containing the total power used 
at each time point over the year. From this we get the demand response curve 
where we should be able to use machine learning and linear optimization 
techniques upon in. 

NOTE: the sample size is relatively small ~4700 households and results must 
take that into account.

Things to check for is the power usage of the households for holidays and
seasonal changes. 

Current goal for this set is to predict the difference between the local 
minimum in the afternoon and the global max in the afternoon. Also to predict 
when the local minimum occurs (within 15 minutes on this model)'''


'''CHECK WHETHER THIS IS A RANDOM WALK:The time series shows a strong temporal dependence (autocorrelation) that decays linearly or in a similar pattern.
The time series is non-stationary and making it stationary shows no obviously learnable structure in the data.
The persistence model (using the observation at the previous time step as what will happen in the next time step) provides the best source of reliable predictions.'''

#use first values from 1-364 to use as predictive values

#the following visualizes the power curves which conform to expectations
dr_data = pd.read_csv('quarter_hr_pwr.csv', header=None)

for row in dr_data.iterrows():
    plt.plot(range(1, 49), row[1])
plt.show()

#this part is where the data is shaped into the training and test cases.
'''Model comparisons follow: LSTM and nonlinear regression on the full set
 of the imported data, and linear regression on a half day set and full day
  set.
  Neural Network: RNN with LSTM
  Nonlinear regression: SVR, adaboost regressor
  '''
scaler = MinMaxScaler()
#scaled = scaler.fit_transform(dr_data.values)
#series = pd.DataFrame(scaled).shape(48, 1)

powerCurve_X = dr_data.iloc[0:364, :]
powerCurve_y = dr_data.iloc[364, :]

answer = dr_data.iloc[1:, 0]
answer = np.reshape(answer, (-1, 1))
scaled_answer = scaler.fit_transform(answer)
answer = pd.DataFrame(scaled_answer)
# powerCurve_X = powerCurve_X.T
scaled_power_X = scaler.fit_transform(powerCurve_X.values)
powerCurve_y = np.reshape(powerCurve_y, (-1, 1)) # feature range must differ, next reshape to sample
scaled_power_y = scaler.fit_transform(powerCurve_y[:, :])


powerCurve_X = pd.DataFrame(scaled_power_X)

#hyperparameters

window_size = 528
units = 10
batch_size = 1
epoch = 5
'''This where I copy the data and copy it so that it iterates alone each data point in order to predict the next point
effectively, which will take quite a bit more time. '''
series = []
answers = []
series_copy = pd.DataFrame(powerCurve_X.values.flatten())
for i in range(0, len(series_copy)-((window_size*2)-1)):
        series.append(series_copy.iloc[i:i+window_size, ])
        answers.append(series_copy.iloc[i+window_size+1, ])

answers = pd.DataFrame(answers)

#serial = pd.concat(series, axis=0)
series = np.hstack(series)
#powerCurve_y = pd.DataFrame(scaled_power_y)

#for fitting the model below
#powerCurve_y1 = np.reshape(powerCurve_y, (48, 1))



                                    #below returns the values for the 2nd and 3rd objectives: local min time/load demand
                                    #format is [min time value, local min, max, max_time value, difference]
'''objectives = [] 
for row in afternoon.iterrows():
    row_val = pd.DataFrame(row[1])
    min = np.min(row_val)
    max = np.max(row_val)
    #return time. In .25 hr increments. +24 because time segment is after noon
    min_time = return_vectorIndex(row_val, min)
    max_time = return_vectorIndex(row_val, max)
    difference = max-min
    objectives.append([min, max, min_time, max_time, difference])

objectives = pd.DataFrame(objectives, columns=['min', 'max', 'min_time', 'max_time', 'difference'])'''

#RNN-LSTM implementation: first model uses history to predict the next day
#the next model will show a moving prediction scenario
criteria = keras.callbacks.EarlyStopping(monitor='loss', patience= 24, baseline=100,  mode='auto', restore_best_weights=True) #min_delta=0,
criteria1 = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, min_delta=.004)

inputs = keras.Input(shape=(1, window_size, 1), batch_shape=(batch_size, window_size, 1))
x = keras.layers.Dense(units, activation='sigmoid')(inputs)
x = keras.layers.CuDNNLSTM(units, kernel_initializer='RandomNormal', return_sequences=True, unit_forget_bias=True, stateful=True)(x)
x = keras.layers.CuDNNLSTM(units, return_sequences=True, stateful=True)(x)
x = keras.layers.CuDNNLSTM(units, stateful=True)(x)
outputs = keras.layers.Dense(1, activation="tanh", activity_regularizer=keras.regularizers.l2(0.005))(x)
model_LSTM = keras.models.Model(inputs, outputs)
model_LSTM.compile(optimizer='adam', loss=root_mean_squared_error) #'mean_absolute_percentage_error' caterror worked


#lstm_input = np.array(powerCurve_X.values.flatten()).reshape(17472, 1)
#numpy reshape produces nans when used in model -> used wrong error (classification)
new_input = np.expand_dims(series, axis=0)
powerCurve_y = np.reshape(scaled_power_y, (1, 48))
powerCurve_y = np.reshape(powerCurve_y[0, 0], (1, 1))
model_LSTM.fit(new_input.T, answers, epochs=epoch, callbacks=[criteria1], batch_size=batch_size)


result = forecast(48)
#result = model_LSTM.predict(new_input)
powerCurve_y = dr_data.iloc[364, :]
#powerCurve_y = np.reshape(powerCurve_y, (-1, 1))
#result = np.reshape(result, (-1, 1))
plt.plot(range(0, 48), powerCurve_y)
plt.plot(range(0, 48), result)
plt.show()
#nonlinear regression




#the next bit of code will analyze the effects of weather and holidays on the power curves





#following is visualizations for the results.


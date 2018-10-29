This folder contains my work and program that I've used to experiment with LSTM neural networks in keras.
I want to make a jupyter notebook so that problem visualization is simpler and can be displayed without compiling the code. 

This is the current state of predicting the next day shown below. I've tuned the model where it gets the trends but further tuning is required to 
predict the curve better. To that end I need to get a few more days worth of measurements to compare though that comes with an issue with a difference in measurements.

In order to get the data for that, I'll have to make a script that compares house IDs in power_shape with the new data. Only return data that matches and then run empty measurement filtering again to the get the final set. 

![alt text](https://user-images.githubusercontent.com/20343931/47660032-8f2f6900-db6c-11e8-98fb-3097fe212885.png)

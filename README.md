# curve-fitting
experimenting with machine learning techniques and visualizations

This repository has various scripts and programs that I have made for class or messed around with in my own time. 
The goal for me is to make the files as transparent as possible in the commenting and variable names as possible. 

The SVR file contains a comparison of support vector regression and linear regression. It does use the sklearn implementation so it won't
give insight into the actual algorithm method for Support Vector Regression. 
    The data set used is the Kaggle London Smart Meter data set. It was conducted over several years and gathered datat at its peak of 
    over 5000 households. 
    The purpose of the program was to predict next day energy usage using historical data. This was done using the last day of the year as 
    the 'label' and the rest of the data as the input. K-fold regression is used upon the data. 
    The next project with the data set would be trying to model and predict demand response for the households.
    
The neural network is based off the neuralnetworksanddeeplearning.com implementation revised by https://github.com/MichalDanielDobrzanski/DeepLearningPython35
The code has been revised to incorperate a softmax output function. 
    The main change is that in order to use a different function during forward propogation, you have to detect which layer is currently
    being calculated. In this case, the output layer. The change can be seen in the backpropogation function. Although an array that holds
    activation functions is already coded, in practice it only works if the neural network has a single activation function. Thus, the 
    solution I came to was to determine if it was the hidden layer, and apply softmax to the output layer. 
    
The next project I wish to work on concerning neural networks would be a capsnet implementation. I also read an interesting paper on 
a reservoir progamming algorithm to simulate chaotic systems. Yet another would be creating a neural network class that builds the network
layer by layer including number of nodes and what function is applied to the nodes.
    

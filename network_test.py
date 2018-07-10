import homework_4_NN
'''This is the program to run the Neural network code from my machine learning assignemnt
   The next thing that should be done to this section of code would be to tune the hyperparameters'''


training_data, validation_data, test_data = homework_4_NN.reformat_data()
training_data = list(training_data)

# bigger mini-batches increase accuracy by 5-10%
# learning rate of 1.5 is optimal for sigmoid, but not for softmax
net = homework_4_NN.Network([784, 120, 10])
net.SGD(training_data, 100, 16, .85, test_data=test_data)


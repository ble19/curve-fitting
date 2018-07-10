# %load network.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as iter

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features. such as multiple activation functions

MUST ADD Softmax and softmax gradient
"""

# Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # first of the feedforward functions, doesn't account for multiple layers
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for bias, weights in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weights, a)+bias)
        return a

    def forwardprop(self, z):
        activations = [z]
        activation = np.array(2, dtype=object)
        np.append(activation, sigmoid(z))
        np.append(activation, softmax(z))
        for index in range(1, self.num_layers-1):
            for bias, weights in zip(self.biases, self.weights):
                z = np.dot(activations[index], weights) + bias
                activations.append(z)

        return activations

        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        costs = []
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, costs)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
                num_correct_class = self.evaluate(test_data)
                costs = self.get_class_error(num_correct_class, n_test, costs)
            else:
                print("Epoch {} complete".format(j))

        # accuracy = np.linspace(1, epochs, num=epochs)
        # Plot the cost over the iterations
        # plt.plot(accuracy, j, 'k-', linewidth=0.5, label='cost minibatches')

        # Add labels to the plot
        plt.xlabel('Epoch')
        plt.ylabel('error rate', fontsize=15)
        plt.title('training curve')
        plt.plot(costs)

        plt.show()

    def update_mini_batch(self, mini_batch, eta, costs):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, costs = self.backprop(x, y, costs)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y, costs):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]# list to store all the activations, layer by layer, in practice it fails
        zs = [] # list to store all the z vectors, layer by layer
        # use case: first -prop to hidden layer(sig), then prop to output(softmax) then backprop
        # MAIN CHANGE HERE: used size to determine which layer was being calculated.
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            if b.size == 30:
                activation = softmax(z)
            else:
                activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * softmax_prime(zs[-1])  # sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w, costs

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""

        return output_activations-y

# here is where I tried different cost functions instead of standard mean squared error
    def cross_entropy(self, predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        n = predictions.shape[0]
        ce = -np.sum(np.sum(targets * np.log(predictions + 1e-9))) / n
        return ce

    def get_logloss(self, test_data, costs):
        test_results = [((self.feedforward(x)), y)
                        for (x, y) in test_data]
        # test_results = [(np.argmax(self.forwardprop(x)), y)
        #                for (x, y) in test_data]
        (x, y) = test_results[-1]
        cost = logloss(x)
        costs.append(cost)
        return costs

    def get_class_error(self, num_correct_class, n_test, costs):
        # 200 is set for the number of samples, need to replace this with a dynamic variable
        error_rate = 1-(num_correct_class/n_test)
        costs.append(error_rate)
        return costs

    def linearschedule(self, eta, j):
        tao = 400
        alpha = float(j/tao)
        learning_rate = ((1 - alpha) * eta + alpha * (.01 * eta))
        return learning_rate


# Miscellaneous functions
def logloss(predicted, eps=1e-15):
  logloss = []
  predicted = np.array(predicted)
  for index in range(0, 9):
        p = np.clip(predicted[index], eps, 1 - eps)
        logloss.append(np.log(p))
  cost = -np.sum(logloss)
  return cost


def sigmoid(z):
    """The sigmoid function."""

    return 1.0/(1.0+(np.exp(-z)))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def softmax(z):

    ex = z - np.max(z)
    output = ex / np.sum(ex)
    return output


def log_softmax(z):
    return z - np.log(np.sum(np.exp(z)))


def softmax_prime(z):
    z = softmax(z)*(1-softmax(z))
    return z
    # input s is softmax value of the original input x. Its shape is (1,n)
    # e.i. s = np.array([0.3,0.7]), x = np.array([0,1])

    # make the matrix whose size is n^2.


def softmax_derivative(z):
    z = np.clip(z, -500, 500)
    Jacobian = -z[..., None] * z[:, None, :]
    ii, ij = np.diag_indices_from(Jacobian[0])
    if ii == ij:
        Jacobian[:, ii, ij] = z * (1 - z)
    else:
        Jacobian[:, ii, ij] = -ii * ij
    return Jacobian.sum(axis=1)


def ReLu(z):
    return np.maximum(z, 0)


def ReLu_prime(z):
    return float(z > 0)


def load_mnist_data():
    # read csv uses the complete path
    mnist_data = pd.read_csv('dataset complete file path')

    training_data = []
    validation_data = []
    test_data = []
    training_data_nontuple = [] #nparray containing nparrays of images
    training_labels_nontuple = [] # these allow iteration in the reformat func
    pre_training_data = mnist_data.iloc[:, 1:]
    training_labels = mnist_data.iloc[:, 0:1]

    for i, row in pre_training_data.iterrows():
        entry = np.array(row)
        training_data_nontuple.append(entry)
    for i, row in training_labels.iterrows():
        entry = row
        training_labels_nontuple.append(entry)
    '''This was my solution to making an np array of np arrays adding to the list is
       easier with a loop than using np append in a loop which is notoriously problematic'''
    training_data_nontuple = np.asarray(training_data_nontuple)
    training_data.append((training_data_nontuple, training_labels))

    validation_data_nontuple = training_data_nontuple[0:200:]
    validation_labels = training_labels_nontuple[0:200:]

    validation_data.append((validation_data_nontuple, validation_labels))

    test_data_nontuple = training_data_nontuple[799:999:]
    test_data_labels = training_labels_nontuple[799:999:]

    test_data.append((test_data_nontuple, test_data_labels))

    return training_data, validation_data, test_data


def reformat_data():
    train_data, valid_data, test_data = load_mnist_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in train_data[0][0]]
    training_results = [vectorized_result(row) for y, row in train_data[0][1].iterrows()]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in valid_data[0][0]]
    validation_data = zip(validation_inputs, valid_data[0][1])
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0][0]]
    test_data = zip(test_inputs, test_data[0][1])
    return training_data, validation_data, test_data


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



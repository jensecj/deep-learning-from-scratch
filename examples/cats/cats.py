import os
import numpy as np
import h5py

import matplotlib.pyplot as plt
from PIL import Image
import scipy
from scipy import ndimage

import dlfs.plotting as plot
import dlfs.network as nn
from dlfs.core import Layer, Network, save_model, load_model
import dlfs.cost_functions as C
import dlfs.optimizers as O
import dlfs.activations as A

root_dir = "examples/cats/"

def load_data():
    train_dataset = h5py.File(root_dir + 'datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # training set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # training set labels

    test_dataset = h5py.File(root_dir + 'datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

    plt.show()

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

net, parameters = load_model(root_dir + "cats-model.pkl")

if not (net and parameters):
    layers = [
        Layer(train_x.shape[0]),
        Layer(20, A.relu),
        Layer(7, A.relu),
        Layer(5, A.relu),
        Layer(1, A.sigmoid)
    ]

    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    iterations = 1500
    learning_rate = 0.01

    parameters, costs = nn.train(net, train_x, train_y, learning_rate, iterations)
    save_model(net, parameters, root_dir + "cats-model.pkl")

    plot.costs(costs)
    plt.title(f"Costs for {iterations} iterations with learning rate = {learning_rate}")
    plt.show()

predictions = nn.predict(net, test_x, parameters)
accuracy = np.sum((predictions == test_y) / test_x.shape[1])
print(f"Accuracy on test data: {(accuracy*100):.3}%")

my_image = Image.open(root_dir + "images/cat.jpg").resize((num_px, num_px))
my_image = np.array(my_image).reshape((num_px*num_px*3,1))
my_image = my_image/255.

my_predicted_image = np.squeeze(nn.predict(net, my_image, parameters))

print("y = " + str(my_predicted_image))
print(f"your {len(net.layers)}-layer model predicts: \"" + classes[int(my_predicted_image),].decode("utf-8") +  "\" picture.")

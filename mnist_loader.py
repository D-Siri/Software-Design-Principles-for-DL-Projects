import tensorflow.keras.datasets as datasets
import numpy as np

class MNISTLoader:
    def __init__(self):
        pass

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
        return (X_train, y_train), (X_test, y_test)

    def describe_dataset(self, X_train, y_train, X_test, y_test):
        print("MNIST Dataset Statistics:")
        print("Number of training samples:", X_train.shape[0])
        print("Number of test samples:", X_test.shape[0])
        print("Image shape:", X_train[0].shape)
        print("Number of classes:", np.unique(y_train).shape[0])
        print("Class distribution in training set:")
        for i in range(np.max(y_train) + 1):
            count = np.sum(y_train == i)
            print("Class {}: {} samples".format(i, count))
        print("Class distribution in test set:")
        for i in range(np.max(y_test) + 1):
            count = np.sum(y_test == i)
            print("Class {}: {} samples".format(i, count))

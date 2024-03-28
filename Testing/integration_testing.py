import unittest
import numpy as np
from mnist_loader import MNISTLoader
from mnist_visualizer import MNISTVisualizer
from data_preprocessor import DataPreprocessor
from models.nn_model import NNModel
from models.cnn_model import CNNModel
from model_evaluator import ModelEvaluator

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize necessary components
        self.mnist_loader = MNISTLoader()
        (self.X_train, self.y_train), (self.X_test, self.y_test) = self.mnist_loader.load_data()
        self.data_preprocessor = DataPreprocessor()
        self.nn_model = NNModel(self.X_train, self.y_train, self.X_test, self.y_test)
        self.cnn_model = CNNModel(self.X_train, self.y_train, self.X_test, self.y_test)
        self.model_evaluator_nn = ModelEvaluator(self.nn_model, self.X_test, self.y_test)
        self.model_evaluator_cnn = ModelEvaluator(self.cnn_model, self.X_test, self.y_test)

    def test_nn_model_integration(self):
        # Train NN model
        self.nn_model.train()
        # Evaluate NN model
        accuracy_nn, _, _, _, _ = self.model_evaluator_nn.evaluate_model()
        self.assertTrue(accuracy_nn > 0.5)  # Assert that accuracy is reasonable

    def test_cnn_model_integration(self):
        # Train CNN model
        self.cnn_model.train()
        # Evaluate CNN model
        accuracy_cnn, _, _, _, _ = self.model_evaluator_cnn.evaluate_model()
        self.assertTrue(accuracy_cnn > 0.5)  # Assert that accuracy is reasonable

if __name__ == '__main__':
    unittest.main()

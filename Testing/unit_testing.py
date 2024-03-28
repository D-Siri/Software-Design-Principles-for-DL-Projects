import unittest
import numpy as np
from mnist_loader import MNISTLoader
from mnist_visualizer import MNISTVisualizer
from data_preprocessor import DataPreprocessor
from nn_model import NNModel
from cnn_model import CNNModel
from model_evaluator import ModelEvaluator


class TestMNISTLoader(unittest.TestCase):
    def setUp(self):
        self.mnist_loader = MNISTLoader()

    def test_load_data(self):
        (X_train, y_train), (X_test, y_test) = self.mnist_loader.load_data()
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_test)
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])


class TestMNISTVisualizer(unittest.TestCase):
    def setUp(self):
        # Mock data for testing
        self.X_train = np.random.randint(0, 255, size=(100, 28, 28))
        self.y_train = np.random.randint(0, 9, size=(100,))
        self.visualizer = MNISTVisualizer(self.X_train, self.y_train)

    def test_visualize_sample_images(self):
        # Test if method runs without errors
        self.visualizer.visualize_sample_images()

    def test_visualize_class_distribution(self):
        # Test if method runs without errors
        self.visualizer.visualize_class_distribution()


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.data_preprocessor = DataPreprocessor()

    def test_preprocess_data(self):
        X_train = np.random.randint(0, 255, size=(100, 28, 28))
        X_test = np.random.randint(0, 255, size=(50, 28, 28))
        X_train_processed, X_test_processed = self.data_preprocessor.preprocess_data(X_train, X_test)
        self.assertGreaterEqual(X_train_processed.min(), 0)

        # Assert that the maximum value is less than or equal to 1
        self.assertLessEqual(X_train_processed.max(), 1)
        self.assertGreaterEqual(X_test_processed.min(), 0)

        # Assert that the maximum value is less than or equal to 1
        self.assertLessEqual(X_test_processed.max(), 1)



class TestNNModel(unittest.TestCase):
    def setUp(self):
        self.X_train = np.random.rand(100, 28, 28)
        self.y_train = np.random.randint(0, 10, size=(100,))
        self.X_test = np.random.rand(50, 28, 28)
        self.y_test = np.random.randint(0, 10, size=(50,))
        self.nn_model = NNModel(self.X_train, self.y_train, self.X_test, self.y_test)

    def test_train(self):
        # Test if method runs without errors
        self.nn_model.train()


class TestCNNModel(unittest.TestCase):
    def setUp(self):
        self.X_train = np.random.rand(100, 28, 28)
        self.y_train = np.random.randint(0, 10, size=(100,))
        self.X_test = np.random.rand(50, 28, 28)
        self.y_test = np.random.randint(0, 10, size=(50,))
        self.cnn_model = CNNModel(self.X_train, self.y_train, self.X_test, self.y_test)

    def test_train(self):
        # Test if method runs without errors
        self.cnn_model.train()


class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.X_test = np.random.rand(50, 28, 28)
        self.y_test = np.random.randint(0, 10, size=(50,))

        # Create an instance of CNNModel
        self.cnn_model = CNNModel(np.random.rand(100, 28, 28), np.random.randint(0, 10, size=(100,)), self.X_test,
                                  self.y_test)

        # Pass the instance of CNNModel to ModelEvaluator
        self.model_evaluator = ModelEvaluator(self.cnn_model, self.X_test, self.y_test)

    def test_evaluate_model(self):
        # Test if method runs without errors
        self.model_evaluator.evaluate_model()


if __name__ == '__main__':
    unittest.main()

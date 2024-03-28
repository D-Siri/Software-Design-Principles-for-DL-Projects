import unittest
import numpy as np

from model_evaluator import ModelEvaluator
from models.nn_model import NNModel  # Assuming this is the module containing your neural network model


class TestModelFunctionality(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.X_train = np.random.rand(100, 28, 28)
        self.y_train = np.random.randint(0, 10, size=(100,))
        self.X_test = np.random.rand(50, 28, 28)
        self.y_test = np.random.randint(0, 10, size=(50,))

        # Initialize the model
        self.model = NNModel(self.X_train, self.y_train, self.X_test, self.y_test)
        self.model_evaluator = ModelEvaluator(self.model, self.X_test, self.y_test)
    def test_train_model(self):
        # Test model training
        self.model.train(epochs=2, batch_size=32)
        # Assert that the model has been trained (you may add more specific assertions here)

    def test_predict(self):
        # Test model prediction
        predictions = self.model.predict(self.X_test)
        # Assert that predictions have been made and have the correct shape/content (you may add more specific assertions here)

    def test_evaluate(self):
        # Test model evaluation
        accuracy, precision, recall, f1, loss = self.model_evaluator.evaluate_model()        # Assert that evaluation metrics have been calculated and have reasonable values (you may add more specific assertions here)


if __name__ == '__main__':
    unittest.main()

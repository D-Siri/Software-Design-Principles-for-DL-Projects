import time
import numpy as np
from nn_model import NNModel  # Assuming this is the module containing your neural network model

class TestModelPerformance:
    def setUp(self):
        # Set up test data
        X_train = np.random.rand(100, 28, 28)
        y_train = np.random.randint(0, 10, size=(100,))
        X_test = np.random.rand(50, 28, 28)
        y_test = np.random.randint(0, 10, size=(50,))

        # Initialize the model with training and test data
        self.model = NNModel(X_train, y_train, X_test, y_test)

    def test_inference_time(self):
        # Measure the inference time for the model
        start_time = time.time()
        predictions = self.model.predict(self.model.X_test)
        end_time = time.time()
        inference_time = end_time - start_time
        print("Inference Time:", inference_time)

    def test_memory_usage(self):
        # Measure the memory usage of the model
        # Note: This example only measures the memory usage of the predictions array
        predictions = self.model.predict(self.model.X_test)
        memory_usage = predictions.nbytes / (1024 * 1024)  # Convert bytes to MB
        print("Memory Usage:", memory_usage, "MB")

    def test_throughput(self):
        # Measure the throughput of the model
        start_time = time.time()
        for _ in range(5):  # Perform inference on 100 batches
            predictions = self.model.predict(self.model.X_test)
        end_time = time.time()
        elapsed_time = end_time - start_time
        throughput = 1000 / elapsed_time  # Number of samples processed per second
        print("Throughput:", throughput, "samples/second")

if __name__ == '__main__':
    test = TestModelPerformance()
    test.setUp()
    test.test_inference_time()
    test.test_memory_usage()
    test.test_throughput()

from mnist_loader import MNISTLoader
from mnist_visualizer import MNISTVisualizer
from data_preprocessor import DataPreprocessor
from models.nn_model import NNModel
from models.cnn_model import CNNModel
from model_evaluator import ModelEvaluator


def print_results(accuracy, precision, recall, f1):
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)


# Load MNIST dataset
mnist_loader = MNISTLoader()
(X_train, y_train), (X_test, y_test) = mnist_loader.load_data()

# Visualize MNIST dataset
mnist_visualizer = MNISTVisualizer(X_train, y_train)
mnist_visualizer.visualize_sample_images()
mnist_visualizer.visualize_class_distribution()

# Preprocess data
data_preprocessor = DataPreprocessor()
X_train_processed, X_test_processed = data_preprocessor.preprocess_data(X_train, X_test)

# Train and evaluate NN model
nn_model = NNModel(X_train_processed, y_train, X_test_processed, y_test)
nn_model.train()
evaluator = ModelEvaluator(nn_model.model, X_test_processed, y_test)
accuracy, precision, recall, f1 = evaluator.evaluate_model()
print("NN Model:")
print_results(accuracy, precision, recall, f1)

# Train and evaluate CNN model
cnn_model = CNNModel(X_train, y_train, X_test, y_test)
cnn_model.train()
evaluator = ModelEvaluator(cnn_model.model, X_test, y_test)
accuracy, precision, recall, f1 = evaluator.evaluate_model()
print("CNN Model:")
print_results(accuracy, precision, recall, f1)

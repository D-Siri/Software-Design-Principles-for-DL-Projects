import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.cnn_model import CNNModel


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test


    def evaluate_model(self):


        y_pred = np.argmax(self.model.predict(self.X_test), axis=-1)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)[0]

        return accuracy, precision, recall, f1, loss

from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from models.base_model import BaseModel


class NNModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.model = self._build_nn_model()

    def _build_nn_model(self):
        model = Sequential()
        model.add(Input(shape=self.X_train.shape[1:]))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(set(self.y_train)), activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, epochs=2, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        # Assuming the model attribute contains the trained CNN model
        return self.model.predict(X)


    def evaluate(self, X, y, verbose=0):
        # Perform evaluation using the trained model
        return self.model.evaluate(X, y, verbose=verbose)

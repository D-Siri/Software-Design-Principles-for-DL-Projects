import numpy as np
from keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from models.base_model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self._preprocess_images()
        self.model = self._build_cnn_model()

    def _preprocess_images(self):
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)

    def _build_cnn_model(self):
        model = Sequential()
        model.add(Input(shape=self.X_train.shape[1:]))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(len(set(self.y_train)), activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, epochs=2, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        # Perform predictions using the trained model
        return self.model.predict(X)

    def evaluate(self, X, y, verbose=0):
        # Perform evaluation using the trained model
        return self.model.evaluate(X, y, verbose=verbose)
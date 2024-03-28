

class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess_data(self, X_train, X_test):
        X_train_processed = self._preprocess_images(X_train)
        X_test_processed = self._preprocess_images(X_test)
        return X_train_processed, X_test_processed

    def _preprocess_images(self, images):
        images_processed = images.astype('float32') / 255.0  # Normalize pixel values
        return images_processed

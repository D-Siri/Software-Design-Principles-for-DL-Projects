import matplotlib.pyplot as plt


class MNISTVisualizer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def visualize_sample_images(self, num_samples=25):
        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            plt.imshow(self.X_train[i], cmap='gray')
            plt.title("Label: {}".format(self.y_train[i]))
            plt.axis('off')
        plt.show()

    def visualize_class_distribution(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.y_train, bins=10, edgecolor='black')
        plt.title('Class Distribution in Training Set')
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.xticks(range(10))
        plt.grid(axis='y', alpha=0.5)
        plt.show()

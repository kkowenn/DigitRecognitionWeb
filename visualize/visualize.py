import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Load and preprocess the MNIST dataset
def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

# Load the model
def load_trained_model(filepath='best_model.h5'):
    return load_model(filepath)

# Predict and visualize
def visualize_predictions(model, x_test, y_test, num_samples=10):
    predictions = model.predict(x_test[:num_samples])
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {predicted_labels[i]}\nTrue: {y_test[i]}")
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Load trained model
    model = load_trained_model()

    # Visualize predictions
    visualize_predictions(model, x_test, y_test)

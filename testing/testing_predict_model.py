import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load a digit image and preprocess it
def preprocess_image(image_path):
    digit_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    digit_image = cv2.resize(digit_image, (28, 28))
    digit_image = digit_image / 255.0
    digit_image = digit_image.reshape(1, 28, 28, 1)
    return digit_image

# Predict the digit from an image
def predict_digit(model, image):
    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions)
    return predicted_digit

if __name__ == "__main__":
    # Load the trained model
    model = load_model('model/best_model.h5')

    # Load and preprocess an example image
    example_image_path = 'testing/9.png'
    digit_image = preprocess_image(example_image_path)

    # Predict the digit
    predicted_digit = predict_digit(model, digit_image)
    print(f'Predicted digit: {predicted_digit}')

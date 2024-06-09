from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import re
import io
from PIL import Image
import base64

app = Flask(__name__)

# Load the trained model
model = load_model('model/best_model.h5')

# Preprocess the image for prediction
def preprocess_image(image_data):
    image_str = re.search(r'base64,(.*)', image_data).group(1)
    image_bytes = io.BytesIO(base64.b64decode(image_str))
    image = Image.open(image_bytes).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    print(f"Processed image shape: {image.shape}, values: {image}")
    return image

# Predict the digit from the image
def predict_digit(image):
    prediction = model.predict(image)
    predicted_digit = int(np.argmax(prediction))  # Convert int64 to int
    print(f"Prediction: {prediction}, Predicted digit: {predicted_digit}")
    return predicted_digit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.get_json()['image']
    print(f"Received image data: {image_data[:30]}...")  # Print only the beginning of the data
    processed_image = preprocess_image(image_data)
    predicted_digit = predict_digit(processed_image)
    return jsonify({'digit': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)

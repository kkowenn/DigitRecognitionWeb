# Digit Classifier Web Application

This project is a web application that allows users to draw digits on a canvas and classify them using a pre-trained convolutional neural network (CNN) model. The model is trained on the MNIST dataset.

![Visualization](visualize/visualize1.png)
![Interface](visualize/interface.gif)

## Features

- Draw digits on a canvas.
- Predict the drawn digit using a pre-trained CNN model.
- Visualize training metrics and results.

## Prerequisites

- Python 3.x
- Flask
- TensorFlow
- Matplotlib
- Pandas

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/digit-classifier.git
   cd digit-classifier
   ```

2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional):**

   If you want to train the model from scratch, run:

   ```bash
   python train_model.py
   ```

   This will train the model and save the best model as `best_model.h5`.

4. **Run the application:**

   ```bash
   python app.py
   ```

   Open a web browser and go to `http://127.0.0.1:5000`.

5. **Visualize training metrics:**

   To visualize the training metrics, run:

   ```bash
   python visualize.py
   ```

## Directory Structure

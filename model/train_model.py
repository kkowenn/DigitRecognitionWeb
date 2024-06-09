import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Load and preprocess the MNIST dataset
def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

# Create a more complex CNN model for digit classification
def create_improved_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Train the model with data augmentation and learning rate scheduler
def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)

    # Learning rate scheduler
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Early stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # Model checkpoint
    model_checkpoint = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', verbose=1)

    # Train the model
    history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                        epochs=50,
                        validation_data=(x_test, y_test),
                        callbacks=[lr_scheduler, early_stopping, model_checkpoint])

    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv('training_log.csv', index=False)

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')

# Save the model
def save_model(model, filepath='digit_classifier.h5'):
    model.save(filepath)

if __name__ == "__main__":
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Create and train the improved model
    model = create_improved_model()
    train_model(model, x_train, y_train, x_test, y_test)

    # Load the best model
    model = models.load_model('best_model.h5')

    # Evaluate the model
    evaluate_model(model, x_test, y_test)

    # Save the trained model
    save_model(model)

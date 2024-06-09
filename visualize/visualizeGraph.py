import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(logfile='training_log.csv'):
    # Load the training log
    log = pd.read_csv(logfile)

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(log['epoch'], log['accuracy'], label='Train Accuracy')
    plt.plot(log['epoch'], log['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(log['epoch'], log['loss'], label='Train Loss')
    plt.plot(log['epoch'], log['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_metrics()


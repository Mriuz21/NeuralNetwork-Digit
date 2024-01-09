import numpy as np
import os
from Layer_Dense import Layer_Dense
from Activation_Relu import Activation_ReLU
from LossFunction import Loss_CategoricalCrossentropy
from Softmax import Activation_Softmax
from Optimizer import Optimizer_SGD
from tensorflow import keras as data
from PIL import Image
from DigitRecGUI import DigitRecGUI
from Model import Model
import tkinter as tk

layers = [(784, 256), (256, 128), (128, 10)]  # Example layers
batch_size = 100
epochs = 28
model = Model(layers, batch_size, epochs)
model.load_data(data.datasets.mnist.load_data())
model.train()
model.evaluate(model.X_test, model.Y_test)  # Evaluate on test set
model.save_model(version='_bestmodel')
model.load_model(version='_bestmodel')
def predict_callback(image, label):
        prediction = model.predict(gui.image)
        gui.display_prediction(prediction)

    # Define the train model callback
def train_model_callback():
    try:
        model.load_TrainData('data.npz')

        if model.X_train is None or model.Y_train is None:
            print("No Data Found!")
            return

        # Find the minimum count of samples for any digit
        min_samples = min(np.bincount(model.Y_train))
        print(min_samples)

        # Initialize variables to store balanced training data
        balanced_images = []
        balanced_labels = []

        # Counters to track the number of samples for each digit
        digit_count = [0] * 10

        # Iterate through the existing training data and balance the samples
        for image, label in zip(gui.trainImages, gui.trainLabels):
            if digit_count[label] < min_samples:
                balanced_images.append(image)
                balanced_labels.append(label)
                digit_count[label] += 1
        #Debug
        print("Shapes before conversion:")
        print("balanced_images:", [np.array(img).shape for img in balanced_images])
        print("balanced_labels:", [label for label in balanced_labels])
        if len(set(np.array(img).shape for img in balanced_images)) != 1:
            print("Error: Images have inconsistent shapes.")
            return
        # Convert to NumPy arrays
        balanced_images = np.array(balanced_images)
        balanced_labels = np.array(balanced_labels)

        # Update the training data in the model
        model.X_train, model.Y_train = balanced_images, balanced_labels
       

        # Train the model
        model.train()
        model.save_model(version='_Dudui')

    except Exception as e:
        print(f"Error during training: {e}")

window = tk.Tk()
gui = DigitRecGUI(window, model, predict_callback, train_model_callback)
window.mainloop()

# Neural Network - Handwritten digit recognizer
Handwritten digit recognizer is a personal project in collaboration with Ivan Andrei (.), with the purpose of experimenting with machine learning. The program is a feed-forward neural network made from scratch, using mainly numpy.
## Training  
The NN has been trained on the MNIST dataset which consist of 28x28 images of handwritten digits.
Data augumentation was a must because The MNIST dataset has the images centered and scaled to fit in a 20x20 box
## GUI
![GUI](https://github.com/Mriuz21/NeuralNetwork-Digit/assets/136023924/bafd9f73-a9a2-4a0b-8666-f03c8c7c3cbf)
360x360 canvas for drawing , 10 labels for each digit.
## Dinamic Training
In the GUI the Train model button dinamically trains the current model with the images you draw. After drawing you can press the button of the corresponding digit and after pressing predict, it saves the image and the label into a npz file.

Equalizing Training Samples:
For each training iteration, the program selects the digit with the least samples and trains the model using the same number of samples for every digit.

Handling Digits with No Samples:
If a digit has no training samples (count equals 0), the program excludes it from the training iteration.
## Instalation
Python version : 3.8.2
install these dependencies:
numpy,
tkinter, opencv, tenserflow (to load the MNIST dataset), pillow

Use: pip install dependencies name


import numpy as np

class Activation_ReLU:
  def forward(self,inputs):
    self.output = np.maximum(0,inputs)
    self.inputs = inputs
  def backward(self, dvalues):
    # Reshape dvalues to match self.inputs
    dvalues = dvalues.reshape(self.inputs.shape)

    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0

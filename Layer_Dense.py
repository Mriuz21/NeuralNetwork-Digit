import numpy as np

class Layer_Dense:
  def __init__(self,n_inputs,n_neurons):
    self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
    self.biases = np.zeros((1,n_neurons))
  def forward(self,inputs):
    self.inputs = inputs
    self.output = np.dot(inputs,self.weights) + self.biases
  def backward(self, dvalues):
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)   
        
  #Handle save and load
  def save(self, file):
     np.savez(file, weights=self.weights, biases=self.biases)
     
  def load(self, file):
     data = np.load(file)
     self.weights = data['weights']
     self.biases = data['biases']

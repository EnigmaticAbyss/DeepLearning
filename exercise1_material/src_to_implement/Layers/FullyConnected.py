import numpy as np
from Layers.Base import BaseLayer
from Optimization import *
"""
X: every row is a batch of data and 
 columns are the number of features
 
W: every column is the set of weights for a batch of data 
output is the number of batches or test cases
 
"""
class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True 
        self.input_size = input_size
        self.output_size = output_size
        #wT
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        
        self.input_tensor = None
        self._optimizer = None
    def forward(self, input_tensor):
      
        self.input_tensor = input_tensor
        
        # Compute the output tensor consuder
        #input in xT
        # input to be b*n rather than n*b by default  thus XT.WT
        output_tensor = np.dot(input_tensor, self.weights) 
        return output_tensor
    
    def backward(self, error_tensor):
        # Compute gradients weight X.ET
        
        grad_weights = np.dot(self.input_tensor.T, error_tensor)
       
        # Propagate the error tensor to the previous layer
        #error tensor is gradient of loss with respect to output of layer
        # it is ET.W
        error_tensor_prev = np.dot(error_tensor, self.weights.T)
        
        #update the weights
        if self._optimizer:
            self.weights = self._optimizer.update(self.weights, grad_weights)
            
        return error_tensor_prev
              
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
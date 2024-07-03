# import numpy as np
# from Layers.Base import BaseLayer
# from Optimization import *
# """
# X: every row is a batch of data and 
#  columns are the number of features
 
# W: every column is the set of weights for a batch of data 
# output is the number of batches or test cases
 
# """
# class FullyConnected(BaseLayer):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.trainable = True 
#         self.input_size = input_size
#         self.output_size = output_size
#         #wT
#         self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        
#         self.input_tensor = None
#         self._optimizer = None
#         self.gradient_weights = None
        
#         self.bias=None
        
#     def forward(self, input_tensor):
#         col_of_ones = np.ones(( input_tensor.shape[0],1))
#         self.input_tensor  = np.hstack((input_tensor, col_of_ones))
        
#         # Compute the output tensor consuder
#         #input in xT
#         # input to be b*n rather than n*b by default  thus XT.WT
#         output_tensor = np.dot(self.input_tensor, self.weights) 
#         return output_tensor
#     def backward(self, error_tensor):
#         # Compute gradients weight X.ET
        
#         self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
       
#         # Propagate the error tensor to the previous layer
#         #error tensor is gradient of loss with respect to output of layer
#         # it is ET.W
#         error_tensor_prev = np.dot(error_tensor, self.weights.T)
        
#         #update the weights
#         if self._optimizer:
#             self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
#         #the error has one more redundent column due to w having one more row!
#         error_tensor_prev = error_tensor_prev[:, :-1]    
#         return error_tensor_prev
    
#     def initialize(self,weights_initializer, bias_initializer):
#         self.weights=weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)    
#         self.bias = bias_initializer.initialize(self.output_size, 1, self.output_size)
    
#     @property
#     def optimizer(self):
#         return self._optimizer

#     @optimizer.setter
#     def optimizer(self, opt):
#         self._optimizer = opt

#     @property
#     def gradient_weights(self):  # getter
#         return self._gradient_weights

#     @gradient_weights.setter
#     def gradient_weights(self, value):
#         self._gradient_weights = value












import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # input_size = number of input neurons
        # output_size = number of output neurons
        self.input_size = int(input_size)  # dimensionality of the columns of input_tensor
        self.output_size = int(output_size)
        self._optimizer = None  # properties
        self._gradient_weights = None
        self.input_store = None
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(self.input_size + 1, self.output_size))

    @property
    def optimizer(self):  # getter
        #  https://docs.python.org/2/library/functions.html#property
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):  # getter
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def initialize(self, weights_initializer, bias_initializer):
        # re-initialize weights
        self.weights[:self.input_size, :] = weights_initializer.initialize((self.input_size, self.output_size),
                                                                           fan_in=self.input_size,
                                                                           fan_out=self.output_size)
        self.weights[self.input_size, :] = bias_initializer.initialize((1, self.output_size),
                                                                       fan_in=1,
                                                                       fan_out=self.output_size)

    # computes the output y of a layer for given input x
    def forward(self, input_tensor):
        tmp = np.ones((input_tensor.shape[0], 1))
        # add bias column
        self.input_store = np.concatenate((input_tensor, tmp), axis=1)  # # batch_size x input_size + 1
        # append ones for biases as row input_size + 1 and then do matrix multiplication
        # dimensions: batch_size x (input_size+1) * (input_size+1) x output_size = batch_size x output_size
        y_hat = np.dot(self.input_store, self.weights)
        return y_hat

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX
    def backward(self, error_tensor):
        # error_tensor shape: batch_size x output_size
        gradient_x = np.dot(error_tensor, self.weights.T)  # gradient w.r.t to input
        self._gradient_weights = np.dot(self.input_store.T, error_tensor)
        if self._optimizer:
            # update weights if optimizer has been set
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        # output shape: batch_size x input_size
        return gradient_x[:, :self.input_size]  # remove last bias column

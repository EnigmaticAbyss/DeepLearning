from Layers.Base import BaseLayer
import numpy as np
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self,input_tensor):
        # print(input_tensor.shape)
        # print(input_tensor.reshape(input_tensor.shape[0], -1).shape)
        self.shape=input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1) # Ndarray.flatten() does not work
    def backward(self,error_tensor):
        # print(error_tensor)
        return error_tensor.reshape(self.shape)
import numpy as np
from Layers.Base import BaseLayer
class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        self.fResult=np.tanh(input_tensor)
        return self.fResult
    def backward(self,error_tensor):
        return (1-np.power(self.fResult,2))*error_tensor
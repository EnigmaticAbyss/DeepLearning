import numpy as np
from Layers.Base import BaseLayer
class TanH(BaseLayer):
    def __init__(self):
        # storing activations, since gradient only depends on them (not inputs)
        self.activation_store = None
        super().__init__()
    def forward(self, input_tensor):
        self.activation_store=np.tanh(input_tensor)
        return self.activation_store
    def backward(self,error_tensor):
        return (1-np.power(self.activation_store,2))*error_tensor
    

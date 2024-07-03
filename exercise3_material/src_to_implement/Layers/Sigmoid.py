from Layers.Base import BaseLayer
import scipy
class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        # storing activations, since gradient only depends on them (not inputs)
        self.activation_store = None

    def forward(self, input_tensor):
        self.activation_store=scipy.special.expit(input_tensor)
        return self.activation_store
    def backward(self,error_tensor):
        return (1-self.activation_store)*self.activation_store* error_tensor
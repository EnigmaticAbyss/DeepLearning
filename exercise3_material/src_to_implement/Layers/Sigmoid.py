from Layers.Base import BaseLayer
import scipy
class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        self.fResult=scipy.special.expit(input_tensor)
        return self.fResult
    def backward(self,error_tensor):
        return (1-self.fResult)*self.fResult* error_tensor
import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self, ):
        super().__init__()
        self.input_tensor = None
 
 
    def forward(self, input_tensor):
      
        self.input_tensor = input_tensor
        return   np.maximum(0, input_tensor)
    
    def backward(self, error_tensor):
        """
        The logic behind is to calculate dL/dx which through chain rule will be 
        dL/dy * dy/dx
 
        """



        grad_input = np.where(self.input_tensor > 0, 1, 0)
        return grad_input * error_tensor
              

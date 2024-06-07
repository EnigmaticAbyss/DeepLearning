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
              
# def softmax_backward(dL_dy, y):
#     # dL_dy: Gradient of the loss with respect to the output of the softmax layer (E_n)
#     # y: Output of the softmax layer (predicted probabilities)
    
#     # Compute the correction term for each element in the batch
#     correction = np.sum(dL_dy * y, axis=1, keepdims=True)
    
#     # Compute the gradient w.r.t. input to softmax using the formula
#     grad_input = dL_dy - correction * y
    
#     return grad_input

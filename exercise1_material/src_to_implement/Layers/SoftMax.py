import numpy as np
from Layers.Base import BaseLayer
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self,input_tensor):
        result=[]
        for i in input_tensor:
            i-=i.max()
            exponential = np.exp(i)
            result.append(exponential / np.sum(exponential))
        self.ouput=np.array(result)
        return self.ouput
    def backward(self,error_tensor):
        return self.ouput * (error_tensor - (error_tensor * self.ouput).sum(axis = 1)[np.newaxis].T)
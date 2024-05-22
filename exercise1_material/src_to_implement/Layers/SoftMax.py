import numpy as np
from Layers.Base import BaseLayer
class SoftMax(BaseLayer):
    def forward(self,input_tensor):
        result=[]
        for i in input_tensor:
            i-=i.max()
            exponential = np.exp(i)
            result.append(exponential / np.sum(exponential))
        return np.array(result)
    def backward(self,error_tensor):
        rest=np.zeros_like(error_tensor)
        for i in error_tensor:
            i-=i.max()
            exponential = np.exp(i)     
            y=exponential / np.sum(exponential)
            rest[i]=(y*(i- i*y))
        return rest
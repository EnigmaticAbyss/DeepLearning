import numpy as np
class Constant():
    def __init__(self,constant=0.1):
        self.constant=constant
    def initialize(self,weights_shape, fan_in, fan_out):
        return np.zeros(weights_shape)+self.constant
class UniformRandom():
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(size=weights_shape)
class Xavier():
    def initialize(self,weights_shape, fan_in, fan_out):
        sigmoid=np.sqrt( 2/(fan_in+fan_out))
        return np.random.randn(*weights_shape)*sigmoid
class He():
    def initialize(self,weights_shape,fan_in,fan_out):
        sigmoid=np.sqrt(2/fan_in)
        
        if isinstance(weights_shape, int):
            weights_shape = (weights_shape,)
        return np.random.randn(*weights_shape)*sigmoid

    
    
    
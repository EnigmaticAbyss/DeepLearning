import numpy as np
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
    def forward(self, input_tensor):
        if self.testing_phase:
            # in test mode
            self.mask = np.ones(input_tensor.shape) # keep all neuron
        else:
            # in train mode
            self.mask = ((np.random.rand(*input_tensor.shape) < self.probability).astype(float))/self.probability # create masking and select dropout randomly
        return input_tensor * self.mask # apply the masking
    
    def backward(self, error_tensor):
        return error_tensor * self.mask
        
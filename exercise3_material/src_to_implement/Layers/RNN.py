import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
import copy

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.FC_H = FullyConnected(hidden_size + input_size, hidden_size)
        self.FC_Y = FullyConnected(hidden_size, output_size)
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._memorize = False
        self.weights = self.FC_H.weights
        self.optimizer = None
        self.weightsY = None
        self.weightsH = None
        self.hT = None
        self.prevH_T = None
        
    
    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value
    def initialize(self, weights_initializer, bias_initializer):
        self.FC_Y.initialize(copy.deepcopy(weights_initializer),copy.deepcopy( bias_initializer))
        self.FC_H.initialize(copy.deepcopy(weights_initializer), copy.deepcopy(bias_initializer))
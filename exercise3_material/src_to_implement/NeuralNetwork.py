import copy
import numpy as np
class NeuralNetwork:
    def __init__(self, optimizer,weight_init,bias_init):
        self.optimizer=optimizer
        self.weight_init=weight_init
        self.bias_init=bias_init
        self.loss=[]
        self.layers=[]
        
    def forward(self):
        data, self.label_tensor = copy.deepcopy(self.data_layer.next())
        for layer in self.layers:
            data = layer.forward(data)
        return self.loss_layer.forward(data, copy.deepcopy(self.label_tensor))
    def backward(self):
        y = copy.deepcopy(self.label_tensor)
        y = self.loss_layer.backward(y)
        for layer in reversed(self.layers):
            y = layer.backward(y)
    def append_layer(self,layer):
        if layer.trainable:
            layer.initialize(self.weight_init, self.bias_init)
            layer.optimizer =copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    def train(self,iterations):
        self.phase='train'
        for epoch in range(iterations):
            self.loss.append(self.forward()) # calculate the froward pass for loss
            self.backward()
            
    
    def test(self,input_tensor):
        self.phase='test'
        for layer in self.layers:
            input_tensor=layer.forward(input_tensor)
        return input_tensor
    
    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self, phase):
        self._phase = phase
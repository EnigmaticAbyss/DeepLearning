import copy
import numpy as np
import pickle

def save(filename, net):
    pickle.dump(net, open(filename, 'wb'))

def load(filename, data_layer):
    net = pickle.load(open(filename, 'rb'))
    net.data_layer = data_layer
    return net
class NeuralNetwork:
    def __init__(self, optimizer,weight_init,bias_init):
        self.optimizer=optimizer
        self.weight_init=weight_init
        self.bias_init=bias_init
        self._phase=None
        self.loss=[]
        self.layers=[]
        
    def forward(self):
        data, self.label_tensor = copy.deepcopy(self.data_layer.next())
        regLoss=0
        for layer in self.layers:
            layer.testing_phase = False
            data = layer.forward(data)
            if self.optimizer.regularizer is not None and layer.weights is not None:
                regLoss += self.optimizer.regularizer.norm(layer.weights)
        return (self.loss_layer.forward(data, copy.deepcopy(self.label_tensor)))+ regLoss
    def backward(self):
        y = copy.deepcopy(self.label_tensor)
        y = self.loss_layer.backward(y)
        for layer in reversed(self.layers):
            y = layer.backward(y)
    def append_layer(self,layer):
        if layer.trainable:
            layer.initialize(copy.deepcopy(self.weight_init),copy.deepcopy( self.bias_init))
            layer.optimizer =copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    def train(self,iterations):
        self.phase='train'
        for epoch in range(iterations):
            self.loss.append(self.forward()) # calculate the froward pass for loss
            self.backward()
            
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data_layer']
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
    def test(self,input_tensor):
        self.phase='test'
        for layer in self.layers:
            layer.testing_phase=True
            input_tensor=layer.forward(input_tensor)
        return input_tensor
    
    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self, phase):
        self._phase = phase

from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients
import copy
import numpy as np
class BatchNormalization(BaseLayer):
    def __init__(self,channels,decay=0.7):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self._optimizer=None
        self.initialize()
        self.decay=decay
        self.moving_mean=None
        self.moving_var=None
    
    def initialize(self):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)
    def forward(self,input_tensor):
        isCnn= (input_tensor.ndim==4)
        self.input_tensor=copy.deepcopy(input_tensor)
        if isCnn:
            input_tensor=self.reformat(input_tensor)
        self.mean=np.mean(input_tensor,0)
        self.var=np.var(input_tensor,0)
        if self.testing_phase:
            self.mean=self.moving_mean
            self.var=self.moving_var
        else:
            if self.moving_mean is None and self.moving_var is None:
                self.moving_mean=self.mean.copy()
                self.moving_var=self.var.copy()
            self.moving_mean = self.moving_mean * self.decay + self.mean * (1 - self.decay)
            self.moving_var = self.moving_var * self.decay + self.var * (1 - self.decay)
        
        
        self.Xtilda=(input_tensor-self.mean) / np.sqrt(self.var+np.finfo(float).eps)
        y_hat=self.gamma*self.Xtilda+self.beta
        if isCnn:
            y_hat=self.reformat(y_hat)
        return y_hat
    def reformat(self, tensor):
        isNewComer=(tensor.ndim==4)
        if isNewComer:
            B, H, M, N=self.reformat_shape = tensor.shape # Store it for when I want press ctl + z
            return tensor.reshape(B, H, M * N).transpose(0, 2, 1).reshape(B * M * N, H)
        else:
            B, H, M, N = self.reformat_shape
            return tensor.reshape(B, M * N, H).transpose(0, 2, 1).reshape(B, H, M, N)
    def reformatHot(self, tensor):
        B, H, M, N = tensor.shape # Store it for when I want press ctl + z
        return tensor.reshape(B, H, M * N).transpose(0, 2, 1).reshape(B * M * N, H)
    def backward(self,error_tensor):
        isCnn= (error_tensor.ndim==4)
        if isCnn:
            error_tensor=self.reformat(error_tensor)
        
        self.gradient_bias=np.sum(error_tensor,0)
        self.gradient_weights=np.sum(error_tensor*self.Xtilda,0)
        if self._optimizer is not None:
            self._optimizer.weight.calculate_update(self.gamma, self.gradient_weights)
            self._optimizer.bias.calculate_update(self.beta, self.gradient_bias)
        gradients=compute_bn_gradients(error_tensor,self.reformatHot(self.input_tensor) if isCnn else self.input_tensor, self.gamma, self.mean, self.var)
        if isCnn:
            gradients=self.reformat(gradients)
        return gradients
    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta
        
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)
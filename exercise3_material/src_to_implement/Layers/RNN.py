# import numpy as np
# from Layers.Base import BaseLayer
# from Layers.FullyConnected import FullyConnected
# import copy

# class RNN(BaseLayer):
#     def __init__(self, input_size, hidden_size, output_size):
#         self.FC_H = FullyConnected(hidden_size + input_size, hidden_size)
#         self.FC_Y = FullyConnected(hidden_size, output_size)
#         self.trainable = True
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self._memorize = False
#         self.weights = self.FC_H.weights
#         self.optimizer = None
#         self.weightsY = None
#         self.weightsH = None
#         self.hT = None
#         self.prevH_T = None
        
    
#     @property
#     def weights(self):
#         return self._weights
    
#     @weights.setter
#     def weights(self, weights):
#         self._weights = weights
#     @property
#     def optimizer(self):
#         return self._optimizer
    
#     @optimizer.setter
#     def optimizer(self, optimizer):
#         self._optimizer = copy.deepcopy(optimizer)

#     @property
#     def memorize(self):
#         return self._memorize

#     @memorize.setter
#     def memorize(self, value):
#         self._memorize = value
#     def initialize(self, weights_initializer, bias_initializer):
#         self.FC_Y.initialize(copy.deepcopy(weights_initializer),copy.deepcopy( bias_initializer))
#         self.FC_H.initialize(copy.deepcopy(weights_initializer), copy.deepcopy(bias_initializer))













import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

# Elman cell implementation


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros((1, self.hidden_size))
        # whether to regard subsequent sequences belonging to the same long sequence or not
        self._memorize = False
        # Fully connected layer containing Whh and Wxh and bh
        self.hidden_layer = FullyConnected(input_size + hidden_size, hidden_size)
        # Why and by
        self.output_layer = FullyConnected(hidden_size, output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self._optimizer = None
        self._gradient_weights = self.hidden_layer.gradient_weights
        self._weights = None
        self.hidden_state_store = None  # store h_t for output_layer's backward pass
        self.hidden_input_store = None  # store stacked h_(t-1) and input for hidden layer's backward pass
        self.sigmoid_activation_store = None
        self.tanh_activation_store = None
        self.trainable = True

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):  # getter
        return self._gradient_weights

    @gradient_weights.setter
    # the gradient_weights need to be independent of self.hidden_layer._gradient_weights
    # since the hidden layer's gradient weights differ in every time step in the backward pass
    # and self._gradient_weights corresponds to the sum over all time steps
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def weights(self):  # getter
        return self.hidden_layer.weights

    @weights.setter
    # important for the unittests to work
    # if the weights are set from outside, the hidden layer weights need to be set, too
    # self._weights and self.hidden_layer.weights is always the same
    def weights(self, value):
        self.hidden_layer.weights = value

    def initialize(self, weights_initializer, bias_initializer):
        # re-initialize weights
        self.hidden_layer.initialize(weights_initializer, bias_initializer)
        self.output_layer.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):  # getter
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def forward(self, input_tensor):
        # batch dimension (rows) = time dimension
        # hidden_previous = h_(t-1)
        if not self._memorize:  # first hidden state
            hidden_previous = np.zeros((1, self.hidden_size))
        else:
            hidden_previous = self.hidden_state  # restore state from previous iteration
        time = input_tensor.shape[0]
        output_tensor = np.zeros((time, self.output_size))
        self.hidden_input_store = np.zeros((time, self.hidden_size + self.input_size + 1))
        self.hidden_state_store = np.zeros((time, self.hidden_size))
        self.sigmoid_activation_store = np.zeros((time, self.output_size))
        # forward pass through time
        for t in range(time):
            # get current input
            # (inpput_size,)
            x_t = input_tensor[t][np.newaxis, :]  # shape: (1, input_size)
            # use as batch with one element
            # input to hidden_layer: 1 x (hidden_size + input_size)
            input_hidden = np.concatenate((hidden_previous, x_t), axis=1)
            h_t = self.hidden_layer.forward(input_hidden)
            self.hidden_input_store[t] = self.hidden_layer.input_store

            h_t = self.tanh.forward(h_t)
            hidden_previous = h_t
            self.hidden_state_store[t] = h_t

            sigmoid_input = self.output_layer.forward(h_t)

            output_tensor[t] = self.sigmoid.forward(sigmoid_input)  # y_t
            self.sigmoid_activation_store[t] = output_tensor[t]
        self.hidden_state = self.hidden_state_store[time-1][np.newaxis, :]
        return output_tensor

    def backward(self, error_tensor):
        time = error_tensor.shape[0]
        error_tensor_previous = np.zeros((time, self.input_size))

        # backpropagate trough time -> reverse process from above
        self._gradient_weights = np.zeros(self.hidden_layer.weights.shape)
        gradient_previous_hidden = 0
        output_gradient_weights = 0
        for t in reversed(range(time)):
            # Sigmoid
            self.sigmoid.activation_store = self.sigmoid_activation_store[t][np.newaxis, :]
            y_t = self.sigmoid.backward(error_tensor[t][np.newaxis, :])

            # Output layer
            tmp = self.hidden_state_store[t][np.newaxis, :]
            # bias needs to be added: like for input_store in forward pass of fully connected layer
            self.output_layer.input_store = np.concatenate((tmp, np.ones((1, 1))), axis=1)
            # gradient of copy procedure = sum
            hidden_state = self.output_layer.backward(y_t) + gradient_previous_hidden

            output_gradient_weights += self.output_layer.gradient_weights

            # TanH: activation = input for output layer
            self.tanh.activation_store = self.hidden_state_store[t][np.newaxis, :]
            tanh_input = self.tanh.backward(hidden_state)

            # Hidden layer
            self.hidden_layer.input_store = self.hidden_input_store[t][np.newaxis, :]
            h_x = self.hidden_layer.backward(tanh_input)  # get stacked hidden state and input
            gradient_previous_hidden = h_x[:, :self.hidden_size]
            # sum over all time steps
            self._gradient_weights += self.hidden_layer.gradient_weights

            error_tensor_previous[t] = h_x[:, self.hidden_size:]
        # update weights after backward pass is finished
        # (such that the weights stay the same until one time sequence is finished)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.output_layer.weights = self._optimizer.calculate_update(self.output_layer.weights, output_gradient_weights)
        return error_tensor_previous

    def calculate_regularization_loss(self):
        # as in forward in neural network
        # define calculate_regularization_loss in BaseLayer to integrate into forward pass of NN?
        reg_loss = 0
        if self._optimizer.regularizer:
            reg_loss += self._optimizer.regularizer.norm(self.hidden_layer.weights)
            reg_loss += self._optimizer.regularizer.norm(self.output_layer.weights)
        return reg_loss

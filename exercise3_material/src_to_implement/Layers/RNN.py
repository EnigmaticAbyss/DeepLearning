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
# import numpy as np
# from Layers.Base import BaseLayer
# from Layers.FullyConnected import FullyConnected
# from Layers.TanH import TanH
# import copy

# derive back propagation on simpole RNN/LSTM


# class RNN(BaseLayer):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.input_size = input_size      # 13
#         self.hidden_size = hidden_size    # 7
#         self.output_size = output_size    # 5
#         self.len_tbptt = 0    # batch size = 9
#         self.FC_h = FullyConnected(hidden_size + input_size, hidden_size)
#         self.FC_y = FullyConnected(hidden_size, output_size)
#         self.tan_h = TanH()
#         self.h_t = None
#         self.memory = False
#         self.last_iter_h_t = None
#         self.optimizer = None
        
#         self.batch_size = None
#         self.hidden_FC_mem = []

#     # whether the RNN regards subsequent sequences as a belonging to the same long sequence
#     @property
#     def memorize(self):
#         return self.memory
    
#     @memorize.setter
#     def memorize(self, value):
#         self.memory = value


#     """
#     Implement a method forward(input tensor) which returns the input tensor for the next layer.
#     Consider the ”batch” dimension as the ”time” dimension of a sequence over
#     which the recurrence is performed. The first hidden state for this iteration is all zero if
#     the boolean member variable is False, otherwise restore the hidden state from the last
#     iteration. You can choose to compose parts of the RNN from other layers you already
#     implemented.
#     """
#     # input_tensor = (input_size, batch_size).T
#     def forward(self, input_tensor):
#         self.batch_size = input_tensor.shape[0]
#         # prepare a matrix of the output so that no need of extra saving of vectors
#         if self.memory:
#             if self.h_t is None:
#                 self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))    # (9+1, 7)
#             else:
#                 print('********************')
#                 self.h_t[0] = self.last_iter_h_t
#         else:   # take previous value
#             self.h_t = np.zeros((self.batch_size + 1, self.hidden_size))

#         y_t = np.zeros((self.batch_size, self.output_size))
        
#         # concatenating x,ht-1 and 1 to do forwarding to obtain new hidden state ht
#         # 1: for t from 1 to T do:
#         # 2:    ut = W hh · h t − 1 + W xh · x t + b h --> h t = tanh (x̃ t · W h )
#         # 3:    h t = tanh ( u t )
#         # 4:    o t = W hy · h t + b y
#         # 5:    ŷ t = σ( o t )
#         #self.batch_size = 1
#         for batch in range(self.batch_size):    # batch = time
#             axis_h_t = self.h_t[batch][np.newaxis, :]   # add row
#             axis_input_t = input_tensor[batch][np.newaxis, :]
#             new_input = np.concatenate((axis_h_t, axis_input_t), axis=1)    # x̃_t
#             #print(new_input.shape)
            
#             self.hidden_FC_mem.append(new_input)
#             # print(self.hidden_FC_mem)
#             #if self.memory:
#             #    wt = self.FC_h.forward(self.hidden_FC_mem[batch - 1])
#             #else:
#             wt = self.FC_h.forward(new_input)
#             new_input = np.concatenate((np.expand_dims(self.h_t[batch], 0), np.expand_dims(input_tensor[batch], 0)), axis=1)
#             self.h_t[batch+1] = self.tan_h.forward(wt)    # h t = tanh (x̃ t · W h )
#             # o_t = W_hy · h_t + b_y ---> no need of sigmoid and bias added afterwards
#             # ŷ_t = W_hy · h_t --> batch+1 = h_t and batch = h_t-1
#             y_t[batch] = (self.FC_y.forward(self.h_t[batch + 1][np.newaxis, :]))

#         self.last_iter_h_t = self.h_t[-1]
#         print(self.h_t.shape)
#         #print(self.last_iter_h_t)
#         self.input_tensor = input_tensor
#         return y_t

#     # Remember that optimizers are decoupled from our layers.
#     def backward(self, error_tensor):
        
#         #print('error_tensor', error_tensor.shape)
#         self.error_tensor_out = np.zeros((self.batch_size, self.input_size))
#         hx_size = self.hidden_size + self.input_size    # (20,7)
#         steps = 1
#         self.gradient_weights_y = np.zeros((self.hidden_size+1, self.output_size))
#         self.gradient_weights_hx = np.zeros((hx_size+1, self.hidden_size))
#         #print(self.h_t.shape)
#         gradient_tanh = 1 - self.h_t[1::] ** 2
#         error_h = np.zeros((1, self.hidden_size))  # backward
        
#         # 1: for t from 1 to T do:
#         # 2:    Run RNN for one step, computing h_t and y_t
#         # 3:    if t mod k_1 == 0:
#         # 4:        Run BPTT from t down to t-k_2

#         for batch in reversed(range(self.batch_size)):
#             one_batch_error = error_tensor[batch]
#             error_y_h = self.FC_y.backward(one_batch_error[np.newaxis, :])
#             #print(error_y_h.shape)
#             self.FC_y.input_tensor = np.hstack((self.h_t[batch+1], 1))[np.newaxis, :]

#             gra_y_ht = error_h+error_y_h
#             # print('ht,gradient_tanh', error_y_h.shape, error_h.shape, gra_y_ht.shape, gradient_tanh[batch].shape)
#             gradient_hidden_t = gradient_tanh[batch]*gra_y_ht
#             error_hx = self.FC_h.backward(gradient_hidden_t)
#             error_h = error_hx[:, 0:self.hidden_size]   # hidden
#             error_x = error_hx[:, self.hidden_size:hx_size + 1]
#             self.error_tensor_out[batch] = error_x
#             concat = np.hstack((self.h_t[batch], self.input_tensor[batch], 1))
#             self.FC_h.input_tensor = concat[np.newaxis, :]

#             print(steps, ' ', self.len_tbptt)
#             #self.weights_y = self.FC_y.getter()  # get_weights()
#             if steps <= self.len_tbptt:
#                 self.weights_y = self.FC_y.getter() #get_weights()
#                 self.weights_h = self.FC_h.getter()  #get_weights()
#                 self.gradient_weights()

#             steps += 1

#         if self.optimizer is not None:
#             self.weights_y = self.optimizer.calculate_update(self.weights_y, self.gradient_weights_y)
#             self.weights_h = self.optimizer.calculate_update(self.weights_h, self.gradient_weights_hx)
#             self.FC_y.setter(self.weights_y)      # .set_weights(self.weights_y)
#             self.FC_h.setter(self.weights_h)      # .set_weights(self.weights_h)

#         return self.error_tensor_out
    
#     """
#     if the hidden state is computed with a single Fully Connected layer, which receives a
# stack of the hidden state and the input tensor, the weights of this particular Fully
# Connected Layer, are the weights considered to be weights for the whole class. In order
# to provide access to the weights of the RNN layer, implement a getter and a setter with
# a property for the weights member.
# """
#     @property
#     def gradient_weights(self):
#         return self.gradient_weights_hx
#         #self.gradient_weights_y += self.FC_y.gradient_weights()      # .get_gradient_weights()
#         #self.gradient_weights_hx += self.FC_h.gradient_weights()     # .get_gradient_weights()
#         #return self.gradient_weights_hx

#     """@property
#     def weights(self):
#         weights = self.FC_h.getter()   # .get_weights()
#         return weights

#     @weights.setter
#     def weights(self, weights):
#         self.FC_h.setter(weights)"""

#     def setter(self, optimizer):
#         self._optimizer = copy.deepcopy(optimizer)

#     def getter(self):
#         return self._optimizer

#     optimizer = property(getter, setter)

#     def initialize(self, weights_initializer, bias_initializer):
#         self.FC_y.initialize(weights_initializer, bias_initializer)
#         self.FC_h.initialize(weights_initializer, bias_initializer)
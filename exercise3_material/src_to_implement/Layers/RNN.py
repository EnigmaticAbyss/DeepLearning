



import numpy as np
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import copy
# Elman cell implementation


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros((1, self.hidden_size))
        # whether to regard subsequent sequences belonging to the same long sequence or not so we can see if there is a previous cell or not
        self._memorize = False
        # this is the  out put of first part in cell: Fully connected layer containing Whh and Wxh and bh  first part for tanh
        self.hidden_layer = FullyConnected(input_size + hidden_size, hidden_size)
        #  this is the the out put yt Why and by second part for sigmoid
        self.output_layer = FullyConnected(hidden_size, output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

        self._optimizer = None
        # gradient layer weight differ at every time step 
 
        #here it is only a refrencing to whole rnn
        self._gradient_weights = self.hidden_layer.gradient_weights
        self._weights = None
        
        self.trainable = True
        self.hidden_state_next_store = None  # store h_t  it is required for backward pass outputs layer this is after tanh so will be used for tanh
        self.hidden_input_prevc_store = None  # here bothe the previous h_t-1 and input layer yet the bias is  added  
        self.sigmoid_activation_store = None # sigmoid at each time, used for backward calculating gradient for output layer final output with bias in it




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



    def forward(self, input_tensor):
        # batch dimension (rows) = time dimension
        # hidden_previous = h_(t-1)
        
        if not self._memorize:  # first hidden state
            hidden_previous = np.zeros((1, self.hidden_size))
        else:
            hidden_previous = self.hidden_state  # restore state from previous iteration
            
            
            
        time = input_tensor.shape[0]
        
        self.hidden_state_next_store = np.zeros((time, self.hidden_size))
        self.sigmoid_activation_store = np.zeros((time, self.output_size))
        self.hidden_input_prevc_store = np.zeros((time, self.hidden_size + self.input_size + 1))# size of combine and bias
        output_tensor = np.zeros((time, self.output_size))
        
        #for backward comming state
        self.hidden_state = self.hidden_state_next_store[time-1].reshape(1, -1)     

        # forward pass through time
        for t in range(time):
            # get current input
            # (inpput_size,)
            x_t = input_tensor[t].reshape(1, -1)  # shape: (1, input_size)
            # use as batch with one element
            # input to hidden_layer: (hidden_size + input_size) this is without bias
            input_hidden = np.hstack((hidden_previous, x_t))

            h_t = self.hidden_layer.forward(input_hidden)
            # this is after the adding of bias
            self.hidden_input_prevc_store[t] = self.hidden_layer.input_store

            h_t = self.tanh.forward(h_t)
            self.hidden_state_next_store[t] = copy.deepcopy(h_t)
            hidden_previous = copy.deepcopy(h_t)
            sigmoid_input = self.output_layer.forward(h_t)

            output_tensor[t] = self.sigmoid.forward(sigmoid_input)  # y_t
            #this one has bias and output
            self.sigmoid_activation_store[t] = output_tensor[t]
            
            
            
  
        return output_tensor

    def backward(self, error_tensor):
        time = error_tensor.shape[0]
        error_tensor_previous = np.zeros((time, self.input_size))

        # backpropagate trough time -> reverse process from above
        self._gradient_weights = np.zeros(self.hidden_layer.weights.shape)
        gradient_previous_hidden = 0
        output_gradient_weights = 0
        for t in reversed(range(time)):
            # Sigmoid here we get the error tensor of sigmoid
            
            self.sigmoid.activation_store = self.sigmoid_activation_store[t].reshape(1, -1)
            

            # do oposite of sigmoid to get the without sigmoid y_t, since backprop removes the bias itself its fine
            y_t_de = self.sigmoid.backward(error_tensor[t].reshape(1, -1))

        
            # Output layer
            tmp = self.hidden_state_next_store[t].reshape(1, -1)
            # bias needs to be added: this is the method we choosed so that in backwar for haviing correct layer must be with ones
            # as you know the connected layer will not remove the bias so we have to add it to make it compatible with our connected layer
            tmp_with_ones = np.ones((tmp.shape[0], tmp.shape[1] + 1))
            tmp_with_ones[:, :-1] = tmp
            
            # we should consider that in backwar we use the ones in our hidden layer hear we need ones for back to work
            self.output_layer.input_store = tmp_with_ones

    
            # gradient of copy procedure is like doing a sum the green part
            hidden_state = self.output_layer.backward(y_t_de) + gradient_previous_hidden
            #first gradient for output function (red part)
            output_gradient_weights += self.output_layer.gradient_weights
            #second part
            # TanH: activation = input for output layer this one is infact the hidden state info previously saved
            self.tanh.activation_store = self.hidden_state_next_store[t].reshape(1, -1)
            tanh_input_de = self.tanh.backward(hidden_state)

            # Hidden layer
            #here we add the input that is saved, remember our alghoritm uses the input with bias combined for calculating the backprop
 
            self.hidden_layer.input_store = self.hidden_input_prevc_store[t].reshape(1, -1)

            h_x = self.hidden_layer.backward(tanh_input_de)  # get stacked hidden state and input
            
            
            #sperating the two elements of input and hx
            
            # this one is the gradient for the ht-1 so that to be passed for the previous one
            #remember this one is just passed  in this state  it doesn't do anything since it is not a part of a layer
            gradient_previous_hidden = h_x[:, :self.hidden_size]
            
            # this one is the error tensor back
            error_tensor_previous[t] = h_x[:, self.hidden_size:]
            
            
            # sum over all time steps this is the main gradient which is  the sum of all layers
            self._gradient_weights += self.hidden_layer.gradient_weights


            
            
            
        # update weights after backward pass is finished
        # both the layers are being updated hidden and output
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.output_layer.weights = self._optimizer.calculate_update(self.output_layer.weights, output_gradient_weights)
        return error_tensor_previous


    @property
    def memorize(self):  # getter
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value
        
    
    @weights.setter
    # self._weights and self.hidden_layer.weights is always the same
    def weights(self, value):
        self.hidden_layer.weights = value
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value



    def initialize(self, weights_initializer, bias_initializer):
        # re-initialize weights
        self.hidden_layer.initialize(weights_initializer, bias_initializer)
        self.output_layer.initialize(weights_initializer, bias_initializer)

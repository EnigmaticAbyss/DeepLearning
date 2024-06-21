import numpy as np
from scipy import signal
from Layers import Base
from scipy.signal import correlate2d, convolve2d
import copy
from enum import Enum


#padding make it same size but stride can yet reduce the size 

class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels:int):
        self.trainable = True
        # Handle stride_shape
        self.stride_shape= self.__handleStrideShape(stride_shape)
        #getting the conv mode
        self.convMode:ConvMode=ConvMode(len(convolution_shape))
        #creating the weight: it shoud be of size (K,C,X,Y)
        self.weights = np.random.uniform(0,1,size=(num_kernels, *convolution_shape))
        #every kernel with own bias
        self.bias = np.random.uniform(0,1,size=(num_kernels,))
        if self.convMode==ConvMode.Conv2:
            self.convolution_shape = convolution_shape
        else:
            # if it is not 2d we transform it to 2d
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]
        self.num_kernels:int = num_kernels
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if input_tensor.ndim == 3:
            #all the inputs will transform to the same size
            input_tensor = input_tensor[:, :, :, np.newaxis]
            # creating padding size which will be able to handle the final pixel -1 is for the fact that final element will be in kernel and then left out
        padded_image = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
                                 input_tensor.shape[2] + self.convolution_shape[1] - 1,
                                 input_tensor.shape[3] + self.convolution_shape[2] - 1))
        p1 = int(self.convolution_shape[1] // 2 == self.convolution_shape[1] / 2)
        p2 = int(self.convolution_shape[2] // 2 == self.convolution_shape[2] / 2)
        if self.convolution_shape[1] // 2 == 0 and self.convolution_shape[2] // 2 == 0:
            padded_image = input_tensor
        else:
            #if it is even it will add one more to end making the padding smaller due to symetry 
            #fill the image in the given location with real input tensor
            padded_image[:, :, (self.convolution_shape[1] // 2):-(self.convolution_shape[1] // 2) + p1,
            (self.convolution_shape[2] // 2):-(self.convolution_shape[2] // 2) + p2] = input_tensor
        #replace the input tensor with the previous local input tensor
        input_tensor = padded_image
        self.padded = padded_image.copy()
        # dims output here it is same padding which creates the exact form before padding
        # one other consideration is the fact that stride will make it smaller as well! so if stride is one
        # the size is similar
        h_cnn = np.ceil((padded_image.shape[2] - self.convolution_shape[1] + 1) / self.stride_shape[0])
        v_cnn = np.ceil((padded_image.shape[3] - self.convolution_shape[2] + 1) / self.stride_shape[1])

        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, int(h_cnn), int(v_cnn)))
        

                
                
        for n in range(input_tensor.shape[0]):
            # Ro filter ha
            for f in range(self.num_kernels):
                # ro hight of output
                for i in range(int(h_cnn)):
                    # ro weights of output
                    for j in range(int(v_cnn)):
                        # check weights limits
                        #multipying the stride make it jump  base on 
                        #conv through all channels
                        if ((i * self.stride_shape[0]) + self.convolution_shape[1] <= input_tensor.shape[2]) and (
                                (j * self.stride_shape[1]) + self.convolution_shape[2] <= input_tensor.shape[3]):
                            output_tensor[n, f, i, j] = np.sum(input_tensor[n, :,
                                                               i * self.stride_shape[0]:i * self.stride_shape[0] +
                                                                                        self.convolution_shape[1],
                                                               j * self.stride_shape[1]:j * self.stride_shape[1] +
                                                                                        self.convolution_shape[
                                                                                            2]] * self.weights[f, :, :,
                                                                                                  :])
                            output_tensor[n, f, i, j] += self.bias[f]
                        else:
                            output_tensor[n, f, i, j] = 0
        # moshkele conv1d ro hal kone 
        if self.convMode==ConvMode.Conv1:
            output_tensor = output_tensor.squeeze(axis=3)  
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)

    def backward(self, error_tensor):
        #making it as it should be
        self.error_T = error_tensor.reshape(self.output_shape)
        #add for cinsistency
        if self.convMode==ConvMode.Conv1:
            self.input_tensor = self.input_tensor[:, :, :, np.newaxis]
        # b,k,X,y upsampled error  tensor which are results/ there can be reduction in tensor size becuase of stride thats the reason behind it
        self.up_error_T = np.zeros((self.input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))

        # b,c,x,y final error tensor 
        return_tensor = np.zeros(self.input_tensor.shape)
        
        # b,c,x,y return tensor but this time padded
        de_padded = np.zeros(
            (*self.input_tensor.shape[:2], self.input_tensor.shape[2] + self.convolution_shape[1] - 1,
             self.input_tensor.shape[3] + self.convolution_shape[2] - 1))
        
        self.gradient_bias = np.zeros(self.num_kernels)
    # gradient is based on k,c,x,y 
        self.gradient_weights = np.zeros(self.weights.shape)
        #calculate padding
        pad_up = int(np.floor(self.convolution_shape[2] / 2))
        pad_left = int(np.floor(self.convolution_shape[1] / 2))
        # creating previos error tensor
        for batch in range(self.up_error_T.shape[0]):
            for kernel in range(self.up_error_T.shape[1]):

                self.gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :])
                # now in backprop we want to fill the strided matrix so we use stride jumps to fill them as due to stride the size might have got smaller
                for h in range(self.error_T.shape[2]):
                    for w in range(self.error_T.shape[3]):

                        self.up_error_T[batch, kernel, h * self.stride_shape[0], w * self.stride_shape[1]] = \
                        self.error_T[batch, kernel, h, w]
                # convolve to give same size making it same padding/ we have deleted the stride effect
                #  En-1= WT. En   the error should be considered without stride
                for ch in range(self.input_tensor.shape[1]):  
                    return_tensor[batch, ch, :] += convolve2d(self.up_error_T[batch, kernel, :],
                                                              self.weights[kernel, ch, :], 'same')  


            for n in range(self.input_tensor.shape[1]):
                for h in range(de_padded.shape[2]):
                    for w in range(de_padded.shape[3]):
                        #stride was previously removed from error_T and was put in error_up
                        # shape the padded in a form that it will take the input but in correct form and padding
                        if (h > pad_left - 1) and (h < self.input_tensor.shape[2] + pad_left):
                            if (w > pad_up - 1) and (w < self.input_tensor.shape[3] + pad_up):
                                de_padded[batch, n, h, w] = self.input_tensor[batch, n, h - pad_left, w - pad_up]
            # for getting the gradient-w every step has a gradient decent value for itslef
            # DW = En. XT
            for kernel in range(self.num_kernels):
                for c in range(self.input_tensor.shape[1]):
                    # here we have error without stride and input tensor with the padding arround it
                    self.gradient_weights[kernel, c, :] += correlate2d(de_padded[batch, c, :],
                                                                       self.up_error_T[batch, kernel, :],
                                                                       'valid')  

        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        if self.convMode==ConvMode.Conv1:
            return_tensor = return_tensor.squeeze(axis=3) 
        return return_tensor
# through prod we get the number of elem in matrix
    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])

        self.weights = weights_initializer.initialize(self.weights.shape, fan_in,
                                                      fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)
    def __handleStrideShape(self,stride_shape):
        if isinstance(stride_shape, int):
            stride_shapeRes = (stride_shape,stride_shape)
        elif isinstance(stride_shape, tuple):
            stride_shapeRes = stride_shape
        elif isinstance(stride_shape,list):
            stride_shapeRes = (stride_shape[0], stride_shape[0])
        else:
            print(type(stride_shape))
            raise ValueError("Invalid stride_shape, must be int or tuple")
        return stride_shapeRes
class ConvMode(Enum):
    Conv1 = 2
    Conv2 = 3
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
        # self.output_shape = output_tensor.shape
        # for b in range(input_tensor.shape[0]):
        #     for k in range(self.num_kernels):
        #         for c in range(input_tensor.shape[1]):
                   
        #             output_tensor[b, k] += signal.correlate2d(input_tensor[b, c], self.weights[k, c], mode='valid')[::self.stride_shape[0], ::self.stride_shape[1]]
        #         output_tensor[b, k] += self.bias[k] 
    
    
    
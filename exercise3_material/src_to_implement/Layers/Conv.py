import numpy as np
from scipy import signal
from Layers.Base import BaseLayer
from scipy.signal import correlate2d, convolve2d
import copy
from enum import Enum

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels:int):
        self.trainable = True
        # Handle stride_shape
        self.stride_shape= self.__handleStrideShape(stride_shape)
        self.convMode:ConvMode=ConvMode(len(convolution_shape))
        self.weights = np.random.uniform(0,1,size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0,1,size=(num_kernels,))
        if self.convMode==ConvMode.Conv2:
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]
        self.num_kernels:int = num_kernels
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[:, :, :, np.newaxis]
        padded_image = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
                                 input_tensor.shape[2] + self.convolution_shape[1] - 1,
                                 input_tensor.shape[3] + self.convolution_shape[2] - 1))
        p1 = int(self.convolution_shape[1] // 2 == self.convolution_shape[1] / 2)
        p2 = int(self.convolution_shape[2] // 2 == self.convolution_shape[2] / 2)
        if self.convolution_shape[1] // 2 == 0 and self.convolution_shape[2] // 2 == 0:
            padded_image = input_tensor
        else:
            padded_image[:, :, (self.convolution_shape[1] // 2):-(self.convolution_shape[1] // 2) + p1,
            (self.convolution_shape[2] // 2):-(self.convolution_shape[2] // 2) + p2] = input_tensor

        input_tensor = padded_image
        self.padded = padded_image.copy()
        # dims output
        h_cnn = np.ceil((padded_image.shape[2] - self.convolution_shape[1] + 1) / self.stride_shape[0])
        v_cnn = np.ceil((padded_image.shape[3] - self.convolution_shape[2] + 1) / self.stride_shape[1])

        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, int(h_cnn), int(v_cnn)))
        self.output_shape = output_tensor.shape

        for n in range(input_tensor.shape[0]):
            # Ro filter ha
            for f in range(self.num_kernels):
                # ro hight of output
                for i in range(int(h_cnn)):
                    # ro weights of output
                    for j in range(int(v_cnn)):
                        # check weights limits
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
        self.error_T = error_tensor.reshape(self.output_shape)
        if self.convMode==ConvMode.Conv1:
            self.input_tensor = self.input_tensor[:, :, :, np.newaxis]

        self.up_error_T = np.zeros((self.input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))
        return_tensor = np.zeros(self.input_tensor.shape)
        
        
        de_padded = np.zeros(
            (*self.input_tensor.shape[:2], self.input_tensor.shape[2] + self.convolution_shape[1] - 1,
             self.input_tensor.shape[3] + self.convolution_shape[2] - 1))
        
        self.gradient_bias = np.zeros(self.num_kernels)
    
        self.gradient_weights = np.zeros(self.weights.shape)

        pad_up = int(np.floor(self.convolution_shape[2] / 2))
        pad_left = int(np.floor(self.convolution_shape[1] / 2))

        for batch in range(self.up_error_T.shape[0]):
            for kernel in range(self.up_error_T.shape[1]):

                self.gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :])

                for h in range(self.error_T.shape[2]):
                    for w in range(self.error_T.shape[3]):

                        self.up_error_T[batch, kernel, h * self.stride_shape[0], w * self.stride_shape[1]] = \
                        self.error_T[batch, kernel, h, w]

                for ch in range(self.input_tensor.shape[1]):  
                    return_tensor[batch, ch, :] += convolve2d(self.up_error_T[batch, kernel, :],
                                                              self.weights[kernel, ch, :], 'same')  


            for n in range(self.input_tensor.shape[1]):
                for h in range(de_padded.shape[2]):
                    for w in range(de_padded.shape[3]):
                        if (h > pad_left - 1) and (h < self.input_tensor.shape[2] + pad_left):
                            if (w > pad_up - 1) and (w < self.input_tensor.shape[3] + pad_up):
                                de_padded[batch, n, h, w] = self.input_tensor[batch, n, h - pad_left, w - pad_up]

            for kernel in range(self.num_kernels):
                for c in range(self.input_tensor.shape[1]):

                    self.gradient_weights[kernel, c, :] += correlate2d(de_padded[batch, c, :],
                                                                       self.up_error_T[batch, kernel, :],
                                                                       'valid')  

        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        if self.convMode==ConvMode.Conv1:
            return_tensor = return_tensor.squeeze(axis=3) 
        return return_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
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
# import numpy as np
# import scipy.signal
# class Conv:
#     def __init__(self, stride_shape, convolution_shape, num_kernels):
#         self.trainable = True
#         if isinstance(stride_shape, tuple):
#             self.stride_shape = stride_shape
#         elif isinstance(stride_shape, list):
#             self.stride_shape = tuple(stride_shape)
#         else:
#             self.stride_shape = (stride_shape,)
        
#         self.convolution_shape = convolution_shape
#         self.num_kernels = num_kernels
#         self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
#         self.bias = np.random.uniform(0, 1, num_kernels)
#         self._optimizer_weights = None
#         self._optimizer_bias = None
#         self.gradient_weights = None
#         self.gradient_bias = None

#     @property
#     def optimizer(self):
#         return self._optimizer_weights, self._optimizer_bias

#     @optimizer.setter
#     def optimizer(self, opt):
#         self._optimizer_weights, self._optimizer_bias = opt if isinstance(opt, tuple) else (opt, opt)


#     def forward(self, input_tensor):
#         self.input_tensor = input_tensor
#         batch_size, channels, *spatial_dims = input_tensor.shape
#         conv_dims = self.convolution_shape[1:]  # Get spatial dimensions of the convolution filter

#         if len(conv_dims) == 1:  # 1D convolution
#             pad = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)
          
#             output_length = (spatial_dims[0] - conv_dims[0] + 2 * pad[0]) // self.stride_shape[0] + 1
#             output_tensor = np.zeros((batch_size, self.num_kernels, output_length))

#             for b in range(batch_size):
#                 for k in range(self.num_kernels):
#                     for c in range(channels):
#                         padded_input = np.pad(input_tensor[b, c], pad, mode='constant')
                        
                  
#                         output_tensor[b, k] += scipy.signal.correlate(padded_input, self.weights[k, c], mode='valid')[::self.stride_shape[0]]
#                     output_tensor[b, k] += self.bias[k]  # Adding the bias element-wise

#         elif len(conv_dims) == 2:  # 2D convolution
#             pad_h = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)
#             pad_w = (conv_dims[1] // 2, conv_dims[1] // 2) if conv_dims[1] % 2 == 1 else (conv_dims[1] // 2, conv_dims[1] // 2 - 1)
            
#             output_height = (spatial_dims[0] - conv_dims[0] + 2 * pad_h[0]) // self.stride_shape[0] + 1
#             output_width = (spatial_dims[1] - conv_dims[1] + 2 * pad_w[0]) // self.stride_shape[1] + 1
#             output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

#             for b in range(batch_size):
#                 for k in range(self.num_kernels):
#                     for c in range(channels):
           
#                         padded_input = np.pad(input_tensor[b, c], (pad_h, pad_w), mode='constant')
                        
#                         output_tensor[b, k] += scipy.signal.correlate2d(padded_input, self.weights[k, c], mode='valid')[::self.stride_shape[0], ::self.stride_shape[1]]
#                         # print("1")
#                     output_tensor[b, k] += self.bias[k]  # Adding the bias element-wise

#         return output_tensor

#     def backward(self, error_tensor):
#         batch_size, channels, *spatial_dims = self.input_tensor.shape
#         conv_dims = self.convolution_shape[1:]  # Get spatial dimensions of the convolution filter

#         self.gradient_weights = np.zeros_like(self.weights)
#         self.gradient_bias = np.zeros_like(self.bias)
#         gradient_input = np.zeros_like(self.input_tensor)

#         if len(conv_dims) == 1:  # 1D convolution
#             pad = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)

#             for b in range(batch_size):
#                 for k in range(self.num_kernels):
#                     for c in range(channels):
            
#                         padded_input = np.pad(self.input_tensor[b, c], pad, mode='constant')

#                         self.gradient_weights[k,c] += scipy.signal.correlate(padded_input, error_tensor[b, k], mode='valid')[::self.stride_shape[0]][pad[0]:-pad[1] or None]
                       
#                     self.gradient_bias[k] += np.sum(error_tensor[b, k])

#             if self._optimizer_weights:
#                 self.weights = self._optimizer_weights.update(self.weights, self.gradient_weights)
#             if self._optimizer_bias:
#                 self.bias = self._optimizer_bias.update(self.bias, self.gradient_bias)

#         elif len(conv_dims) == 2:  # 2D convolution
#             pad_h = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)
#             pad_w = (conv_dims[1] // 2, conv_dims[1] // 2) if conv_dims[1] % 2 == 1 else (conv_dims[1] // 2, conv_dims[1] // 2 - 1)

#             for b in range(batch_size):
#                 for k in range(self.num_kernels):
#                     for c in range(channels):
#                         padded_input = np.pad(self.input_tensor[b, c], (pad_h, pad_w), mode='constant')
                   
#                         self.gradient_weights[k, c] += scipy.signal.correlate2d(padded_input, error_tensor[b, k], mode='valid')
                 
#                         gradient_input[b, c] += scipy.signal.convolve2d(error_tensor[b, k], self.weights[k, c], mode='full')[pad_h[0]:-pad_h[1] or None, pad_w[0]:-pad_w[1] or None]
                    
#                     self.gradient_bias[k] += np.sum(error_tensor[b, k])

#             if self._optimizer_weights:
#                 self.weights = self._optimizer_weights.update(self.weights, self.gradient_weights)
#             if self._optimizer_bias:
#                 self.bias = self._optimizer_bias.update(self.bias, self.gradient_bias)

#         return gradient_input
    

#     def initialize(self, weights_initializer, bias_initializer):
#         fan_in = np.prod(self.convolution_shape)
#         fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        
#         self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
#         self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
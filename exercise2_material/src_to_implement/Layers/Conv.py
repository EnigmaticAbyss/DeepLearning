

import numpy as np
import scipy.signal

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        if isinstance(stride_shape, tuple):
            self.stride_shape = stride_shape
        elif isinstance(stride_shape, list):
            self.stride_shape = tuple(stride_shape)
        else:
            self.stride_shape = (stride_shape,)
        # self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape,)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self._optimizer_weights = None
        self._optimizer_bias = None
        self.gradient_weights = None
        self.gradient_bias = None

    @property
    def optimizer(self):
        return self._optimizer_weights, self._optimizer_bias

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer_weights, self._optimizer_bias = opt if isinstance(opt, tuple) else (opt, opt)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size, channels, *spatial_dims = input_tensor.shape
        conv_dims = self.convolution_shape[1:]  # Get spatial dimensions of the convolution filter

        if len(conv_dims) == 1:  # 1D convolution
            pad = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)
          
            output_length = (spatial_dims[0] - conv_dims[0] + 2 * pad[0]) // self.stride_shape[0] + 1
            output_tensor = np.zeros((batch_size, self.num_kernels, output_length))

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(channels):
                        padded_input = np.pad(input_tensor[b, c], pad, mode='constant')
                        
                        
                        # print("out tens f")
                        # print(padded_input.shape)
                        # print("padd tens f")
                        # print(self.weights[k, c].shape)
                        # print("weigh tens")
                        # print("padded_input")
                        # print(scipy.signal.correlate(padded_input, self.weights[k, c], mode='valid').shape)
                        # print(scipy.signal.correlate(padded_input, self.weights[k, c], mode='valid')[::self.stride_shape[0]].shape)
                        output_tensor[b, k] += scipy.signal.correlate(padded_input, self.weights[k, c], mode='valid')[::self.stride_shape[0]]
                    output_tensor[b, k] += self.bias[k]  # Adding the bias element-wise

        elif len(conv_dims) == 2:  # 2D convolution
            pad_h = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)
            pad_w = (conv_dims[1] // 2, conv_dims[1] // 2) if conv_dims[1] % 2 == 1 else (conv_dims[1] // 2, conv_dims[1] // 2 - 1)
            
            output_height = (spatial_dims[0] - conv_dims[0] + 2 * pad_h[0]) // self.stride_shape[0] + 1
            output_width = (spatial_dims[1] - conv_dims[1] + 2 * pad_w[0]) // self.stride_shape[1] + 1
            output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(channels):
                        padded_input = np.pad(input_tensor[b, c], (pad_h, pad_w), mode='constant')
                        output_tensor[b, k] += scipy.signal.correlate2d(padded_input, self.weights[k, c], mode='valid')[::self.stride_shape[0], ::self.stride_shape[1]]
                    output_tensor[b, k] += self.bias[k]  # Adding the bias element-wise

        return output_tensor


    def backward(self, error_tensor):
        batch_size, channels, *spatial_dims = self.input_tensor.shape
        conv_dims = self.convolution_shape[1:]  # Get spatial dimensions of the convolution filter

        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        gradient_input = np.zeros_like(self.input_tensor)

        if len(conv_dims) == 1:  # 1D convolution
            pad = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(channels):
                        print("error tens")
                        print(error_tensor.shape)
                        print("grad shape")
                        print(self.convolution_shape)
                        print(self.gradient_weights.shape)
                        print("input_tensor")
                        print(self.input_tensor.shape)
                        print("no pad")
                        print(self.input_tensor[b, c])
                        padded_input = np.pad(self.input_tensor[b, c], pad, mode='constant')
                        error_resized = error_tensor[b, k].reshape(-1) 
                        print("error resized")# Ensure error tensor has the correct shape
                        print(error_resized)
                        print("self gradient:")
                        print(self.gradient_weights[k, c].shape)
                        print("error")
                        print(error_tensor[b, k].shape)
                        print("padded_input")
                        
                        print(padded_input.shape)
                        print("amount")
                        print((scipy.signal.correlate(padded_input,error_tensor[b, k] , mode='valid')).shape)
                        self.gradient_weights[k,c ] += scipy.signal.correlate(padded_input, error_tensor[b, k], mode='valid')
                        
                        print("self gradient: after")
                        print(self.gradient_weights[k, c])
                        gradient_input[b, c] += scipy.signal.convolve(error_tensor[b, k], self.weights[k, c], mode='full')[pad[0]:-pad[1] or None]
                    self.gradient_bias[k] += np.sum(error_tensor[b, k])

            if self._optimizer_weights:
                self.weights = self._optimizer_weights.update(self.weights, self.gradient_weights)
            if self._optimizer_bias:
                self.bias = self._optimizer_bias.update(self.bias, self.gradient_bias)

        elif len(conv_dims) == 2:  # 2D convolution
            pad_h = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)
            pad_w = (conv_dims[1] // 2, conv_dims[1] // 2) if conv_dims[1] % 2 == 1 else (conv_dims[1] // 2, conv_dims[1] // 2 - 1)

            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(channels):
                        padded_input = np.pad(self.input_tensor[b, c], (pad_h, pad_w), mode='constant')
                                          
                        self.gradient_weights[k, c] += scipy.signal.correlate2d(padded_input, error_tensor[b, k], mode='valid')
                        gradient_input[b, c] += scipy.signal.convolve2d(error_tensor[b, k], self.weights[k, c], mode='full')[pad_h[0]:-pad_h[1] or None, pad_w[0]:-pad_w[1] or None]
                    self.gradient_bias[k] += np.sum(error_tensor[b, k])

            if self._optimizer_weights:
                self.weights = self._optimizer_weights.update(self.weights, self.gradient_weights)
            if self._optimizer_bias:
                self.bias = self._optimizer_bias.update(self.bias, self.gradient_bias)

        return gradient_input
    

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)







# def backward(self, error_tensor):
#     batch_size, channels, *spatial_dims = self.input_tensor.shape
#     conv_dims = self.convolution_shape[1:]  # Get spatial dimensions of the convolution filter

#     self.gradient_weights = np.zeros_like(self.weights)
#     self.gradient_bias = np.zeros_like(self.bias)
#     gradient_input = np.zeros_like(self.input_tensor)

#     if len(conv_dims) == 1:  # 1D convolution
#         pad = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)

#         for b in range(batch_size):
#             for k in range(self.num_kernels):
#                 for c in range(channels):
#                     padded_input = np.pad(self.input_tensor[b, c], pad, mode='constant')
#                     self.gradient_weights[k, c] += scipy.signal.correlate(padded_input, error_tensor[b, k], mode='valid')
#                     gradient_input[b, c] += scipy.signal.convolve(error_tensor[b, k], self.weights[k, c], mode='full')[pad[0]:-pad[1] or None]
#                 self.gradient_bias[k] += np.sum(error_tensor[b, k])

#         if self._optimizer_weights:
#             self.weights = self._optimizer_weights.update(self.weights, self.gradient_weights)
#         if self._optimizer_bias:
#             self.bias = self._optimizer_bias.update(self.bias, self.gradient_bias)

#     elif len(conv_dims) == 2:  # 2D convolution
#         pad_h = (conv_dims[0] // 2, conv_dims[0] // 2) if conv_dims[0] % 2 == 1 else (conv_dims[0] // 2, conv_dims[0] // 2 - 1)
#         pad_w = (conv_dims[1] // 2, conv_dims[1] // 2) if conv_dims[1] % 2 == 1 else (conv_dims[1] // 2, conv_dims[1] // 2 - 1)

#         for b in range(batch_size):
#             for k in range(self.num_kernels):
#                 for c in range(channels):
#                     padded_input = np.pad(self.input_tensor[b, c], (pad_h, pad_w), mode='constant')
#                     self.gradient_weights[k, c] += scipy.signal.correlate2d(padded_input, error_tensor[b, k], mode='valid')
#                     gradient_input[b, c] += scipy.signal.convolve2d(error_tensor[b, k], self.weights[k, c], mode='full')[pad_h[0]:-pad_h[1] or None, pad_w[0]:-pad_w[1] or None]
#                 self.gradient_bias[k] += np.sum(error_tensor[b, k])

#         if self._optimizer_weights:
#             self.weights = self._optimizer_weights.update(self.weights, self.gradient_weights)
#         if self._optimizer_bias:
#             self.bias = self._optimizer_bias.update(self.bias, self.gradient_bias)

#     return gradient_input

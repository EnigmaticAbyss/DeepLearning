import numpy as np
from scipy.signal import correlate, correlate2d
import numpy as np
import copy

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels:int):
        self.trainable = True

        # Handle stride_shape
        self.stride_shape= self.__handleStrideShape(stride_shape)
        # Handle convolution_shape
        if len(convolution_shape) not in [2, 3]:
            raise ValueError("Invalid convolution_shape, must be [c, m] or [c, m, n]")
        self.convolution_shape = convolution_shape

        # Initialize weights and biases
        self.weights = np.random.uniform(0, 1, size=(num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, size=(num_kernels,))
        
        # Initialize gradients
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        if len(convolution_shape)!=3:
            self.convolution_shape=(*convolution_shape,1)
            self.weights= self.weights[:,:,:np.newaxis]
        
        # Set number of kernels
        self.num_kernels = num_kernels

        # Optimizer
        self._optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        self._optimizer_bias = copy.deepcopy(opt)
        self._optimizer_weights=copy.deepcopy(opt)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(self.convolution_shape) == 2:
            # 1D convolution
            b, c, y = input_tensor.shape
            c_in, m = self.convolution_shape
            assert c == c_in, "Input channels must match"
            output_y = (y - m + 1 + 2 * (m // 2)) // self.stride_shape[0]
            output_tensor = np.zeros((b, self.num_kernels, output_y))

            padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (m // 2, m // 2)), mode='constant')

            for i in range(self.num_kernels):
                for j in range(b):
                    for k in range(0, y, self.stride_shape[0]):
                        output_tensor[j, i, k // self.stride_shape[0]] = np.sum(
                            padded_input[j, :, k:k + m] * self.weights[i, :, :]) + self.bias[i]
        else:
            # 2D convolution
            b, c, y, x = input_tensor.shape
            c_in, m, n = self.convolution_shape
            assert c == c_in, "Input channels must match"
            output_y = (y - m + 1 + 2 * (m // 2)) // self.stride_shape[0]
            output_x = (x - n + 1 + 2 * (n // 2)) // self.stride_shape[1]
            output_tensor = np.zeros((b, self.num_kernels, output_y, output_x))

            padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (m // 2, m // 2), (n // 2, n // 2)), mode='constant')

            for i in range(self.num_kernels):
                for j in range(b):
                    for k in range(0, y, self.stride_shape[0]):
                        for l in range(0, x, self.stride_shape[1]):
                            output_tensor[j, i, k // self.stride_shape[0], l // self.stride_shape[1]] = np.sum(
                                padded_input[j, :, k:k + m, l:l + n] * self.weights[i, :, :, :]) + self.bias[i]

        return output_tensor

    def backward(self, error_tensor):
        if len(self.convolution_shape) == 2:
            # 1D convolution
            b, c, y = self.input_tensor.shape
            c_in, m = self.convolution_shape
            self._gradient_weights.fill(0)
            self._gradient_bias.fill(0)
            grad_input_tensor = np.zeros_like(self.input_tensor)

            padded_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (m // 2, m // 2)), mode='constant')

            for i in range(self.num_kernels):
                for j in range(b):
                    for k in range(0, y, self.stride_shape[0]):
                        self._gradient_weights[i, :, :] += error_tensor[j, i, k // self.stride_shape[0]] * padded_input[j, :, k:k + m]
                        self._gradient_bias[i] += error_tensor[j, i, k // self.stride_shape[0]]
                        grad_input_tensor[j, :, k:k + m] += error_tensor[j, i, k // self.stride_shape[0]] * self.weights[i, :, :]

        else:
            # 2D convolution
            b, c, y, x = self.input_tensor.shape
            c_in, m, n = self.convolution_shape
            self._gradient_weights.fill(0)
            self._gradient_bias.fill(0)
            grad_input_tensor = np.zeros_like(self.input_tensor)

            padded_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (m // 2, m // 2), (n // 2, n // 2)), mode='constant')

            for i in range(self.num_kernels):
                for j in range(b):
                    for k in range(0, y, self.stride_shape[0]):
                        for l in range(0, x, self.stride_shape[1]):
                            self._gradient_weights[i, :, :, :] += error_tensor[j, i, k // self.stride_shape[0], l // self.stride_shape[1]] * padded_input[j, :, k:k + m, l:l + n]
                            self._gradient_bias[i] += error_tensor[j, i, k // self.stride_shape[0], l // self.stride_shape[1]]
                            grad_input_tensor[j, :, k:k + m, l:l + n] += error_tensor[j, i, k // self.stride_shape[0], l // self.stride_shape[1]] * self.weights[i, :, :, :]

        if self.optimizer:
            self.weights = self.optimizer.update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.update(self.bias, self._gradient_bias)

        return grad_input_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape)
        self.bias = bias_initializer.initialize(self.bias.shape)
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
        
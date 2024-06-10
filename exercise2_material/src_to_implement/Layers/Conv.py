import numpy as np
import scipy.signal

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape, )
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        
        # Initialize weights and bias
        if len(convolution_shape) == 2:  # 1D convolution
            c, m = convolution_shape
            self.weights = np.random.uniform(0, 1, (num_kernels, c, m))
            self.bias = np.random.uniform(0, 1, (num_kernels,))
        elif len(convolution_shape) == 3:  # 2D convolution
            c, m, n = convolution_shape
            self.weights = np.random.uniform(0, 1, (num_kernels, c, m, n))
            self.bias = np.random.uniform(0, 1, (num_kernels,))
        
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(self.convolution_shape) == 2:  # 1D convolution
            b, c, y = input_tensor.shape
            _, _, m = self.weights.shape
            output_y = y // self.stride_shape[0]
            output_tensor = np.zeros((b, self.num_kernels, output_y))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        output_tensor[batch, kernel] += scipy.signal.correlate(
                            input_tensor[batch, channel], self.weights[kernel, channel], mode='same'
                        )[::self.stride_shape[0]]
                    output_tensor[batch, kernel] += self.bias[kernel]

        elif len(self.convolution_shape) == 3:  # 2D convolution
            b, c, y, x = input_tensor.shape
            _, _, m, n = self.weights.shape
            output_y = y // self.stride_shape[0]
            output_x = x // self.stride_shape[1]
            output_tensor = np.zeros((b, self.num_kernels, output_y, output_x))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        output_tensor[batch, kernel] += scipy.signal.correlate(
                            input_tensor[batch, channel], self.weights[kernel, channel], mode='same'
                        )[::self.stride_shape[0], ::self.stride_shape[1]]
                    output_tensor[batch, kernel] += self.bias[kernel]
        
        return output_tensor

    def backward(self, error_tensor):
        b = error_tensor.shape[0]
        if len(self.convolution_shape) == 2:  # 1D convolution
            _, c, y = self.input_tensor.shape
            _, _, m = self.weights.shape
            self._gradient_weights = np.zeros_like(self.weights)
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        self._gradient_weights[kernel, channel] += scipy.signal.correlate(
                            self.input_tensor[batch, channel], error_tensor[batch, kernel], mode='valid'
                        )
            if self._optimizer:
                self.weights = self._optimizer[0].update(self.weights, self._gradient_weights / b)
                self.bias = self._optimizer[1].update(self.bias, self._gradient_bias / b)
            return error_tensor

        elif len(self.convolution_shape) == 3:  # 2D convolution
            _, c, y, x = self.input_tensor.shape
            _, _, m, n = self.weights.shape
            self._gradient_weights = np.zeros_like(self.weights)
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    for channel in range(c):
                        self._gradient_weights[kernel, channel] += scipy.signal.correlate(
                            self.input_tensor[batch, channel], error_tensor[batch, kernel], mode='valid'
                        )
            if self._optimizer:
                self.weights = self._optimizer[0].update(self.weights, self._gradient_weights / b)
                self.bias = self._optimizer[1].update(self.bias, self._gradient_bias / b)
            return error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer(self.weights.shape)
        self.bias = bias_initializer(self.bias.shape)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

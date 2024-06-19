import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape if isinstance(stride_shape, tuple) else (stride_shape, stride_shape)
        self.pooling_shape = pooling_shape
        self.indices = None  # To store indices of max values during forward pass
        self.trainable = False
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        b, c, y, x = input_tensor.shape
        pool_y, pool_x = self.pooling_shape
        stride_y, stride_x = self.stride_shape
        # since it will be divided in windows and further by stride to see how many result element we have
        out_y = (y - pool_y) // stride_y + 1
        out_x = (x - pool_x) // stride_x + 1
        output_tensor = np.zeros((b, c, out_y, out_x))

        self.indices = np.zeros_like(output_tensor, dtype=int)

        for batch in range(b):
            for channel in range(c):
                for i in range(out_y):
                    for j in range(out_x):
                        # position of window
                        y_start = i * stride_y
                        y_end = y_start + pool_y
                        x_start = j * stride_x
                        x_end = x_start + pool_x
                        #placing the windo
                        window = input_tensor[batch, channel, y_start:y_end, x_start:x_end]
                        # finding the max index in window 
                        max_val = np.max(window)
                        output_tensor[batch, channel, i, j] = max_val
                        
                        # Store the index of the max value,, 2*2 : with third element highest will give 2
                        max_index = np.argmax(window)
                        # transform to a touple (1,0)
                        max_index = np.unravel_index(max_index, window.shape)
                        # finding the place element by making it flat!
                        self.indices[batch, channel, i, j] = (y_start + max_index[0]) * x + (x_start + max_index[1])

        return output_tensor

    def backward(self, error_tensor):
        b, c, out_y, out_x = error_tensor.shape
        input_y, input_x = self.input_tensor.shape[2], self.input_tensor.shape[3]

        error_out = np.zeros_like(self.input_tensor)

        for batch in range(b):
            for channel in range(c):
                for i in range(out_y):
                    for j in range(out_x):
                        index = self.indices[batch, channel, i, j]
                        #row
                        y_index = index // input_x
                        #column
                        
                        x_index = index % input_x
                        # same size as input but only feeling the indices
                        # overlap of multiple pooling wndows  
                        error_out[batch, channel, y_index, x_index] += error_tensor[batch, channel, i, j]

        return error_out




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

        out_y = (y - pool_y) // stride_y + 1
        out_x = (x - pool_x) // stride_x + 1
        output_tensor = np.zeros((b, c, out_y, out_x))

        self.indices = np.zeros_like(output_tensor, dtype=int)

        for batch in range(b):
            for channel in range(c):
                for i in range(out_y):
                    for j in range(out_x):
                        y_start = i * stride_y
                        y_end = y_start + pool_y
                        x_start = j * stride_x
                        x_end = x_start + pool_x
                        
                        window = input_tensor[batch, channel, y_start:y_end, x_start:x_end]
                        max_val = np.max(window)
                        output_tensor[batch, channel, i, j] = max_val
                        
                        # Store the index of the max value
                        max_index = np.argmax(window)
                        max_index = np.unravel_index(max_index, window.shape)
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
                        y_index = index // input_x
                        x_index = index % input_x
                        error_out[batch, channel, y_index, x_index] += error_tensor[batch, channel, i, j]

        return error_out





# import numpy as np

# class Pooling:
#     def __init__(self, stride_shape, pooling_shape):
#         if isinstance(stride_shape, int):
#             self.stride_shape = (stride_shape, stride_shape)
#         elif isinstance(stride_shape, tuple) and len(stride_shape) == 2:
#             self.stride_shape = stride_shape
#         else:
#             raise ValueError("Invalid stride_shape, must be int or tuple of two integers")

#         if isinstance(pooling_shape, int):
#             self.pooling_shape = (pooling_shape, pooling_shape)
#         elif isinstance(pooling_shape, tuple) and len(pooling_shape) == 2:
#             self.pooling_shape = pooling_shape
#         else:
#             raise ValueError("Invalid pooling_shape, must be int or tuple of two integers")

#         self.input_tensor = None
#         self.max_indices = None

#     def forward(self, input_tensor):
#         self.input_tensor = input_tensor
#         b, c, y, x = input_tensor.shape
#         pool_y, pool_x = self.pooling_shape
#         stride_y, stride_x = self.stride_shape

#         out_y = (y - pool_y) // stride_y + 1
#         out_x = (x - pool_x) // stride_x + 1
#         output_tensor = np.zeros((b, c, out_y, out_x))

#         self.max_indices = np.zeros_like(output_tensor, dtype=int)

#         for batch in range(b):
#             for channel in range(c):
#                 for i in range(out_y):
#                     for j in range(out_x):
#                         start_y = i * stride_y
#                         start_x = j * stride_x
#                         end_y = start_y + pool_y
#                         end_x = start_x + pool_x

#                         patch = input_tensor[batch, channel, start_y:end_y, start_x:end_x]
#                         max_index = np.argmax(patch)
#                         max_value = np.max(patch)
                        
#                         output_tensor[batch, channel, i, j] = max_value
#                         self.max_indices[batch, channel, i, j] = max_index

#         return output_tensor

#     def backward(self, error_tensor):
#         b, c, out_y, out_x = error_tensor.shape
#         grad_input_tensor = np.zeros_like(self.input_tensor)

#         pool_y, pool_x = self.pooling_shape
#         stride_y, stride_x = self.stride_shape

#         for batch in range(b):
#             for channel in range(c):
#                 for i in range(out_y):
#                     for j in range(out_x):
#                         start_y = i * stride_y
#                         start_x = j * stride_x
#                         end_y = start_y + pool_y
#                         end_x = start_x + pool_x

#                         max_index = self.max_indices[batch, channel, i, j]
#                         max_index_y = max_index // pool_x
#                         max_index_x = max_index % pool_x

#                         grad_input_tensor[batch, channel, start_y + max_index_y, start_x + max_index_x] += error_tensor[batch, channel, i, j]

#         return grad_input_tensor


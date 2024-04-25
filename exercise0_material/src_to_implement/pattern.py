import numpy as np
import matplotlib.pyplot as plt
import copy

class Checker:
    def __init__(self, resolution, tile_size):
        self.output = None
        self.tile_size = tile_size
        self.resolution = resolution
  
  
    def draw(self):
     
        num = self.resolution/self.tile_size       
   
        zero_array = np.zeros((self.tile_size))
        one_array = np.ones((self.tile_size))
        
       

        
        
        concatenated_arr_w = np.concatenate((one_array, zero_array),axis=0)
        concatenated_arr= np.concatenate((zero_array, one_array),axis=0)
        
        tiled_arr = np.tile(concatenated_arr,int((num)/2) )
        expanded_arr = np.expand_dims(tiled_arr, axis=1)  

        
        tiled_arra = np.tile(concatenated_arr_w,int((num)/2) )
        expanded_arr_w = np.expand_dims(tiled_arra, axis=1)  
        
        b_line= np.tile(expanded_arr,self.tile_size)
        w_line = np.tile (expanded_arr_w,self.tile_size)
        bw_line = np.concatenate((b_line, w_line),axis=1)

        self.output = np.tile (bw_line,int((num)/2) )
  
        deep_copy_array = np.array(self.output, copy=True)

        return deep_copy_array

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.title("Checker Pattern")
        plt.axis('off')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.output = None
        self.x,self.y = position
        self.radius = radius
        self.resolution = resolution

    def draw(self):
        x_axis = np.arange(self.resolution)
        y_axis = np.arange(self.resolution)
        X, Y = np.meshgrid(x_axis, y_axis)
        self.output = np.ones((self.resolution,self.resolution))
        circle_mask = ((X-self.x)**2 + (Y-self.y)**2) <= self.radius**2
        self.output[X[circle_mask],Y[circle_mask]]=0
        
        deep_copy_array = np.array(self.output, copy=True)
        return deep_copy_array

    def show(self):
        plt.imshow(self.output, cmap='binary')
        plt.title("Circle Pattern")
        plt.axis('off')
        plt.show()



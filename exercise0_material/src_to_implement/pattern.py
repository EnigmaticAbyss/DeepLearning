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
        self.output = np.zeros((self.resolution,self.resolution))
        circle_mask = ((X-self.x)**2 + (Y-self.y)**2) <= self.radius**2
        self.output[Y[circle_mask],X[circle_mask]]=1
     
        deep_copy_array = np.array(self.output, copy=True)
        return deep_copy_array

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.title("Circle Pattern")
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.output = None
        self.resolution = resolution

    def draw(self):


        x_axis = np.arange(self.resolution)
        y_axis = np.arange(self.resolution)
        X, Y = np.meshgrid(x_axis, y_axis)
        print(X)
        print(Y)

        x, y = np.indices((self.resolution, self.resolution))

        # print(x)
        # print(y)

        X = X / (self.resolution - 1)
        Y = Y / (self.resolution - 1)


        red = X


        blue = 1-X


        green = Y


        rgb_image = np.stack((red, green, blue), axis=-1)


        rgb_image = np.clip(rgb_image, 0, 1)

    
        # a = np.linspace(1,0 ,self.resolution).reshape(self.resolution, 1)
        # b = np.linspace(1,0 ,self.resolution)
        # result = a + b 
        # min_val = np.min(result)
        # max_val = np.max(result)
        # range_val = max_val - min_val
        # normalized_matrix_b= (result - min_val) / range_val

        
        # a = np.linspace(1,0 ,int(np.round(self.resolution))).reshape(int(np.round(self.resolution)), 1)
        # b = np.linspace(0,1 , int(np.round(self.resolution)))
   
        # result = a + b 
        # min_val = np.min(result)
        # max_val = np.max(result)

        # range_val = max_val - min_val
        # normalized_matrix_r = (result - min_val) / range_val
        
    
        # a =  np.linspace(0,1 ,self.resolution).reshape((self.resolution,1))
        # c= np.concatenate((np.linspace(0,1 ,int(np.round(self.resolution/2))),np.linspace(1,0 ,int(np.ceil(self.resolution/2)))))
        
        # result = a + c
        # min_val = np.min(result)
        # max_val = np.max(result)
        # range_val = max_val - min_val
        # normalized_matrix_g = np.zeros((self.resolution,self.resolution))
        
        
        # rgb_image = np.dstack((normalized_matrix_r, normalized_matrix_g, normalized_matrix_b))
        

        self.output = rgb_image


        deep_copy_array = np.array(self.output, copy=True)
        return deep_copy_array

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.title("Spectrum Pattern")
        plt.axis('off')
        plt.show()



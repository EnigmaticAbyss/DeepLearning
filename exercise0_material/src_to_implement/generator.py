import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
                

        
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path= file_path
        self.label_path= label_path
        self.batch_size= batch_size
        self.image_size= image_size
        self.rotation= rotation
        self.mirroring= mirroring
        self.shuffle= shuffle
        self.index= 0 
        self.current_epoch_num= 0

        with open(label_path, 'r')as file:
            self.lables= json.load(file)
  

   
        
        list_keys =  list(self.lables.keys())
        keys_with_extension = [key + '.npy' for key in list_keys]


        self.filenames = np.array([f for f in os.listdir(file_path) if f in keys_with_extension])  
    
        if self.shuffle:
            np.random.shuffle(self.filenames) 

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        
        if self.index >= len(self.filenames):
            self.index= 0 
            self.current_epoch_num +=1
            if self.shuffle:
                np.random.shuffle(self.filenames)
        end_index= self.index + self.batch_size
        batch_filenames = self.filenames[self.index:end_index]
        
        
        if end_index > len(self.filenames):
            batch_filenames = np.concatenate((batch_filenames,self.filenames[:end_index-len(self.filenames)]))
            
        images = np.array([transform.resize(np.load(os.path.join(self.file_path,file_name)), (self.image_size[0],self.image_size[1]),anti_aliasing=True) for file_name in batch_filenames])
  
        

        images = np.array([self.augment(img) for img in images])   #remember this style of [for] gives back a list do to [] in it so transform to np array
        # each row is considered an image in the way of making list out of a numpy array thus(3, 100, 100, 3) has three (100,100,3) images 
        
        batch_filenames_without_extension = [os.path.splitext(file_name)[0] for file_name in batch_filenames]
        labels = np.array([self.lables[lable] for lable in batch_filenames_without_extension])
        self.index += self.batch_size
      
        # return images, labels
        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        # image = np.array(img)
        if self.mirroring:
            mirror_flags= np.random.randint(0,2)
            if mirror_flags:
                img = np.fliplr(img)
                
        if self.rotation:
            rotation_angles= np.random.choice([0,1,2,3])
            img = np.rot90(img, k=rotation_angles)
        return img   

    def current_epoch(self):
      
        # return the current epoch number
        return self.current_epoch_num

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method

        images, labels = self.next()
        
        fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
        for i, ax in enumerate(axes):
            ax.imshow(images[i])
            ax.set_title(self.class_name(labels[i]))
            ax.axis('off')
        plt.show()



 
        # if self.mirroring:
        #     mirror_flags= np.random.randint(0,2,size= images.shape[0])
        #     images = np.array([np.fliplr(img) if flag else img for img,flag in zip(images,mirror_flags)])
        # if self.rotation:
        #     rotation_angles= np.random.choice([0,1,2,3],size= images.shape[0])
        #     images = np.array([np.rot90(img, k=angle)for img,angle in zip(images, rotation_angles)])




        # images = []
        # labels = []
        # for filename in batch_filenames:
        #     img_path = os.path.join(self.image_dir, filename)
        #     img = io.imread(img_path)
        #     if img.shape != self.image_size:
        #         img = transform.resize(img, (self.image_size[0], self.image_size[1]), anti_aliasing=True)
        #     if self.mirroring and np.random.rand() > 0.5:
        #         img = np.fliplr(img)
        #     if self.rotation:
        #         k = np.random.randint(0, 4)
        #         img = np.rot90(img, k)
        #     images.append(img)
        #     labels.append(self.labels[filename])

        # self.index += self.batch_size
        # return np.array(images), np.array(labels)


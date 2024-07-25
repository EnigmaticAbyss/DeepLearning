from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
      
# transforms on train data obviously no change to labels!
        if self.mode == 'train':
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.RandomRotation(8),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std),
            ])
            # transform on test data, Not to change data form a lot since should be consistent with natural data
        elif self.mode==  'validate':# mode == 'validate'
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor()  ,
                tv.transforms.Normalize(mean=train_mean, std=train_std),
            ])
    
    @staticmethod         
    def convert_image_to_rgb_if_needed(image):
        """
        Convert a grayscale image to RGB format if it is not already in RGB format.

        Parameters:
        - image (ndarray): The input image which can be grayscale or RGB.

        Returns:
        - ndarray: The image in RGB format if it was grayscale; otherwise, the original image.
        """
        # Check if the image is grayscale (2D)
        if image.ndim == 2:
            return gray2rgb(image)  # Convert grayscale to RGB
        return image  # Return original image if already RGB   
    def __getitem__(self, index):
        # Get image path and label from the dataframe
        image_path = self.data.iloc[index, 0]
        # couple of labels thus a list
        label = torch.tensor(self.data.iloc[index, 1:].tolist()).int()

        # Read image and convert to RGB
        image = imread(image_path)
        
        img = self.convert_image_to_rgb_if_needed(image)
        image = np.array(img, dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label  = label.clone().detach().float()

        
        return image, label        
    def __len__(self):
        return len(self.data)
    

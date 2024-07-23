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
    def __init__(self, data, mode: str):
        self.data = data
        self.mode = mode
      

        if self.mode == 'train':
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std),
            ])
        else:  # mode == 'val'
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor()  ,
                tv.transforms.Normalize(mean=train_mean, std=train_std),
            ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Get image path and label from the dataframe
        img_path = self.data.iloc[index, 0]

        label = torch.tensor(self.data.iloc[index, 1:].tolist()).int()

        # Read image and convert to RGB
        image = imread(img_path)
        if image.ndim == 2:  # if grayscale
            image = gray2rgb(image)
        image = np.array(image, dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Convert label to tensor
        label  = label.clone().detach().float()

        
        return image, label
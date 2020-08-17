import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils



class MelanomaDataset(Dataset):
    """Melanoma dataset."""

    def __init__(self, dataframe, root_dir, transform=None, train=True):
        """
        Args:
            dataframe: Contains the image names at column 'image_name' and target label at 'target', 
            the df should also contain the meta data.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.melanoma_df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.meta_data = self.melanoma_df[['age_approx', 
                                           'upper extremity',
                                           'torso',
                                           'lower extremity',
                                           'head/neck',
                                           'palms/soles',
                                           'oral/genital',
                                           'female',
                                           'male']]

    def __len__(self):
        return len(self.melanoma_df)

    def __getitem__(self, idx):   
        #get image
        img_name = f'{self.root_dir}/{self.melanoma_df.iloc[idx, 0]}.jpg'        
        image = Image.open(img_name)
        
        sample_idx = self.melanoma_df.iloc[[idx]].index[0]
        target = self.melanoma_df.loc[sample_idx, 'target']
        
        # Transform image
        if self.transform:
            image = self.transform(image)
        
        #get meta data
        data = self.meta_data.iloc[idx]        
        data = torch.tensor(data).float()
        
        sample = (image, data, target)

        return sample

class MelanomaTestDataset(MelanomaDataset):
    def __getitem__(self, idx):
        
        #get image
        img_name = f'{self.root_dir}/{self.melanoma_df.iloc[idx, 0]}.jpg'        
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        #get meta data
        data = self.meta_data.iloc[idx]        
        data = torch.tensor(data).float()

        sample = (image, data)

        return sample


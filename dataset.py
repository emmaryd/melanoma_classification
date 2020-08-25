from PIL import Image
import torch
from torch.utils.data import Dataset

class MelanomaDataset(Dataset):
    """Creates the Melanoma dataset.
    
     This is a subclass of the torch.utils.data.Dataset module, the MelanomaDataset is
     modified to fit the ISIC 2020 Challenge dataset.
    
    Attributes:
        melanoma_df: A dataframe from the csv file containing imagename and meta data
            (train_processed.csv) described in README.md
        root_dir: A string that gives the path to the directory where the images are located
        transform: Optional transformations (torchvision.transforms) tobe applied to the images
        meta_data: A dataframe containing only the metadata created from "dataframe"
    """

    def __init__(self, melanoma_df, root_dir, transform=None):
        """ The __init__ function.

        Args:
            melanoma_df: A dataframe from the csv file containing imagename and meta data
            (train_processed.csv) described in README.md
            root_dir: A string that gives the path to the directory where the images are located
            transform: Optional transformations (torchvision.transforms) tobe applied to the images
        """

        self.melanoma_df = melanoma_df
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
        """ __getitem__ function to access items in the MelanomaDataset

        Args:
            idx: The index that is to be accessed
        Returns:
            A tuple that contains the image, the meta data, and the target)
        """
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
    """
    Creates the Melanoma test-dataset.

    This is a subclass of MelanomaDataset, with modified __getitem__,
    since the test-data does not have a target.

    Attributes: see MelanomaDataset
    """
    
    def __getitem__(self, idx):
        """ Modified __getitem__ to fit the test data.

        Args:
            see base class

        Returns:
             A tuple containing image and meta data.
        """
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

from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
import torchvision.models as models

class ResnetModel(nn.Module):
    """ This is a subclass from nn.Module that creates a pre-trained resnet50 model.

    Attributes:
        cnn: The convolutional network, a pretrained resnet50.
        fc_meta_data: A fully connected network.
        classifier: The last layer in the network.
    """
    def __init__(self, n_columns):
        """ The __init__ function.
        Args:
            n_columns: the number of columns in data (the meta data).
        """
        super(ResnetModel, self).__init__()
        self.cnn = models.resnet50(pretrained=True) # output size is 1000
        self.fc_meta_data = nn.Sequential(nn.Linear(n_columns, 500),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.5),
                                          nn.BatchNorm1d(500),
					                      nn.Linear(500, 250),
					                      nn.ReLU(),
					                      nn.Dropout(p=0.5),
 					                      nn.BatchNorm1d(250))

        self.classifier = nn.Sequential(nn.Linear(1000 + 250, 1))
        
    def forward(self, image, data):
        """Forward function.

        Args:
            image: A 224x224x3 image.
            data: A tensor containing the meta data.
        
        Returns:
            x: the network output, a float that is probability that the input belongs to class 1.
        """
        image = self.cnn(image)
        data = self.fc_meta_data(data)
        x = torch.cat((image, data), dim=1)
        x = self.classifier(x)
        
        return x

class EfficientNetModel(nn.Module):

    """This is a subclass from nn.Module that creates a pre-trained efficientnet model.

    Attributes:
        cnn: The convolutional network, a pretrained efficientnet-b4.
        fc_meta_data: A fully connected network.
        classifier: The last layer in the network.
    """

    def __init__(self, n_columns):
        """ The __init__ function.
        Args:
            n_columns: the number of columns in data (the meta data).
        """

        super(EfficientNetModel, self).__init__()
        
        self.cnn = EfficientNet.from_pretrained('efficientnet-b4')
        
        self.fc_meta_data = nn.Sequential(nn.Linear(n_columns, 500),
					                      nn.ReLU(),
                                          nn.BatchNorm1d(500),
                                          nn.Dropout(p=0.5),
                                          nn.Linear(500, 250),
					                      nn.ReLU(),
                                          nn.BatchNorm1d(250),
                                          nn.Dropout(p=0.5),
                                         )
        
        self.classifier = nn.Sequential(nn.Linear(1000 + 250, 1))
        
    def forward(self, image, data):
        """Forward function.

        Args:
            image: A 224x224x3 image.
            data: A tensor containing the meta data.
        
        Returns:
            x: the network output, a float that is probability that the input belongs to class 1.
        """
        image = self.cnn(image)
        data = self.fc_meta_data(data)
        
        x = torch.cat((image, data), dim=1)
        x = self.classifier(x)

        return x

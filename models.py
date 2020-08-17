from efficientnet_pytorch import EfficientNet
import torch
from torch import optim, nn
import torchvision.models as models

class ResnetModel(nn.Module):
    """
    This class creates a pre-trained resnet50 model. Input should be a 224x224x3 image (image) and
    a tensor meta data containing the meta data.
    It returns a prediction of the probability that the input belongs to class 1 (malignt).
    """
    def __init__(self, n_columns):
        super(ResnetModel, self).__init__()
        self.cnn = models.resnet50(pretrained=True) # output size is 1000
        self.fc_meta_data = nn.Sequential(nn.Linear(n_columns, 500),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.5),
                                          nn.BatchNorm1d(500),
					                      nn.Linear(500, 250),
					                      nn.ReLU(),
					                      nn.Dropout(p=0.5),
 					                      nn.BatchNorm1d(250)
                                        )
        self.classifier = nn.Sequential( nn.Linear(1000 + 250, 1) )
        
    def forward(self, image, data):
        image = self.cnn(image)
        data = self.fc_meta_data(data)
        x = torch.cat((image, data), dim=1)
        x = self.classifier(x)
        
        return x

class EfficientNetModel(nn.Module):
    """
    This class creates a pretrained efficientnet model. Input should be a 224x224x3 tensor (image) and 
    a tensor containing the meta data (data). It returns a prediction of the probability that the input 
    belongs to class 1 (malignt).
    """
    def __init__(self, n_columns):
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
        
        self.classifier = nn.Sequential(nn.Linear(1000 + 250,1 ))
        
    def forward(self, image, data):
        image = self.cnn(image)
        data = self.fc_meta_data(data)
        
        x = torch.cat((image, data), dim=1)
        x = self.classifier(x)

        return x
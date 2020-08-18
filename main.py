from dataset import MelanomaDataset, MelanomaTestDataset
import matplotlib.pyplot as plt
from models import ResnetModel, EfficientNetModel
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import (Compose, ToTensor, Normalize,  CenterCrop, 
                                    RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective)
from torch.utils.data import DataLoader
from torch import optim, nn
from train_epoch import train_epoch
import sys

def main(model_type='resnet', n_epochs = 20, lr = 0.0005, batch_size=32):

    """ the main function """

    train_img_path = '/data/train_resized/' #path to directory containing resized train image
    test_img_path = '/data/test_resized/' #path to directory containing resized train image
    data_train = pd.read_csv('data/train_processed.csv') #path to directory containing processed csv file for train data
    data_test = pd.read_csv('data/test_processed.csv') #path to directory containing processed csv file for test data

    n_data_train = len(data_train) 

    split = int(0.2 * n_data_train)

    data_train, data_valid  = data_train.iloc[split :], data_train.iloc[0 : split]

    # Transformation for test and validation data
    transform_valid = Compose([CenterCrop(224), # Crops the center so the resulting image shape is 224x224x3, input image shape could be for example 256x256x3
                               ToTensor(),
                               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])

    #Augmentations for the training data
    transform_train = Compose([CenterCrop(224),
                                RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0),
                                RandomVerticalFlip(p=0.5),
                                RandomHorizontalFlip(p=0.5),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])

    # Split train set into train and validation
    dataset_train = MelanomaDataset(data_train, train_img_path, transform=transform_train) 
    dataset_valid = MelanomaDataset(data_valid, train_img_path, transform=transform_valid)
    dataset_test = MelanomaTestDataset(data_test, test_img_path, transform=transform_valid)

    # Create the batches with dataloader
    training_loader =  DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    print('There is ', len(dataset_train), 'images in the train set and ', len(dataset_valid), 'in the dev set.')

    #define device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    #define model and freeze the deepest layers
    if model_type == 'resnet':
        model = ResnetModel(9)
        no_train_layers = [ model.cnn.layer1, model.cnn.layer2, model.cnn.layer3 ]
        for layer in no_train_layers:
            for param in layer:
                param.requires_grad = False

    elif model_type == 'efficientnet':
        model = EfficientNetModel(9)
        model.cnn._conv_stem.requires_grad = False
        
        no_train_layers = model.cnn._blocks[:28] 
        for layer in no_train_layers:
            #for param in layer:
            layer.requires_grad = False

    model = model.to(device)

    # define loss function
    loss_function = torch.nn.BCEWithLogitsLoss()

    #define optimizer
    optimizer = optim.Adam(model.parameters(),lr=lr) 

    #define scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    train_loss = []
    validation_loss = []
    train_auc = []
    val_auc = []
    best_auc = 0.0

    for i in range(n_epochs):
        t1, v1, t_auc, v_auc = train_epoch(training_loader, 
                                            validation_loader, 
                                            model, 
                                            loss_function, 
                                            optimizer,
                                            device)

        print(f"\r Epoch {i+1}: Training loss = {t1}, Validation loss = {v1}, \n Train auc = {t_auc},  Validation_auc = {v_auc}")
        print('lr = ', optimizer.param_groups[0]['lr'])

        train_loss.append(t1)
        validation_loss.append(v1)
        train_auc.append(t_auc)
        val_auc.append(v_auc)

        scheduler.step(v_auc)
        
        # save best model
        if v_auc > best_auc:
            torch.save(model, '/best_model.pt')
            best_auc = v_auc
            print('model saved')
            

    epochs = np.arange(n_epochs)
    fig, ax = plt.subplots()
    ax.set_title('Training and Validation losses')
    ax.plot(epochs, train_loss, label='Train')
    ax.plot(epochs, validation_loss, label='Dev')
    plt.legend()

    fig, ax = plt.subplots()
    ax.set_title('Training and Validation ROC AUC')
    ax.plot(epochs, train_auc, label='Train')
    ax.plot(epochs, val_auc, label='Dev')
    plt.legend()

    # ADD TESTING!!

if __name__ == '__main__':
    main(*sys.argv[1:])

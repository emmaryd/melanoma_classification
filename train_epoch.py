from sklearn.metrics import roc_auc_score
import torch
from torch import optim, nn

def train_epoch(training_loader, 
                validation_loader, 
                model, 
                loss_function, 
                optimizer,
                device):

    model.train() 
    training_loss = 0.0
    y_all_train = []
    y_pred_all_train = []

    for i, (x1, x2, y) in enumerate(training_loader):  
        n = len(training_loader)
        #move input to device
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
    
        #set gradients to zero
        optimizer.zero_grad()

        # Predict output, compute loss, perform optimizer step.
        y_pred = model(x1, x2)

        loss = loss_function(y_pred, y.view(-1,1).float())
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        y = y.to('cpu')
        y_pred = y_pred.to('cpu')
        y_pred = torch.sigmoid(y_pred) 

        y_all_train.extend(y.detach().numpy())
        y_pred_all_train.extend(y_pred.detach().numpy())
        print(f'\r Batch {i}/{n}: Training loss: {loss.item()}', end="")

    y_pred_all_train = [p.item() for p in y_pred_all_train]
  
    torch.cuda.empty_cache()  
    training_loss /= n
    training_auc = roc_auc_score(y_all_train, y_pred_all_train)
  
    # Evaluate on dev set
    model.eval()
    y_all_val = []
    y_pred_all_val = []
    validation_loss = 0.0
    n = len(validation_loader)

    with torch.no_grad():
        for i, (x1,x2,y) in enumerate(validation_loader):
            #move input to device
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            #predict output and calculate loss
            y_pred = model(x1, x2)

            loss = loss_function(y_pred, y.view(-1,1).float())
            validation_loss += loss.item()
            
            y = y.to('cpu')
            y_pred = y_pred.to('cpu')
            y_pred = torch.sigmoid(y_pred)

            y_all_val.extend(y.detach().numpy())
            y_pred_all_val.extend(y_pred.detach().numpy())

        y_pred_all_val = [p.item() for p in y_pred_all_val]          
        validation_loss /= n
        validation_auc = roc_auc_score(y_all_val, y_pred_all_val)

    return training_loss, validation_loss, training_auc, validation_auc

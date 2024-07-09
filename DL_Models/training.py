
import utils


import os
from tqdm.notebook import tqdm
import time
import json

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

devicecpu = torch.device("cpu")

def fitMT(model, data, optimizer, num_epochs, patience, checkpoint_path, device):
    num_epochs = num_epochs
    opt = optimizer
    data_loaders = data
    iters = 0
    since = time.time()
    
    # Save Losses
    history = {
        'cor_loss1': [], 'mse_loss1': [], 'cor_loss2': [], 'mse_loss2': [],
        'val_cor_loss1': [], 'val_mse_loss1': [], 'val_cor_loss2': [], 'val_mse_loss2': [],
        'train_epoch_cor_loss1': [], 'train_epoch_mse_loss1': [], 'train_epoch_cor_loss2': [], 'train_epoch_mse_loss2': [],
        'val_epoch_cor_loss1': [], 'val_epoch_mse_loss1': [], 'val_epoch_cor_loss2': [], 'val_epoch_mse_loss2': []
    }

    # initialize the early_stopping object
    early_stopping = utils.EarlyStopping(patience=patience, path=checkpoint_path, verbose=True)
    
    mseloss = nn.MSELoss()
    
    print("Starting Training ...")
    loss_idx_value = 0

    train_batch = len(data_loaders['train'])
    val_batch = len(data_loaders['val'])
    train_loader = utils.DevicedataLoader(data_loaders['train'], device)
    val_loader = utils.DevicedataLoader(data_loaders['val'], device)
    
    batch_nb = {"train": train_batch, "val": val_batch}
    data_loaders = {"train": train_loader, "val": val_loader}

    module1_stable_epochs = 0
    module1_loss_stable = False
    stable_threshold = 3

    # For each epoch
    for epoch in range(num_epochs):
      
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        # Each epoch has a training and validation phase
        for phase in ['val', 'train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            single_epoch_cor_loss1 = []
            single_epoch_mse_loss1 = []
            single_epoch_cor_loss2 = []
            single_epoch_mse_loss2 = []
            
            predictions2 =  []
            real2 = []
            predictions1 =  []
            real1 = []
            
            # For each batch in the dataloader
            for i, data in tqdm(enumerate(data_loaders[phase]), total=batch_nb[phase]):
                time.sleep(0.01)
                # get the input and their corresponding labels
                x = data[0]
                y = data[1].float()
                
                # forward pass to get outputs
                yGBLUP, yPheno = torch.split(y, int(y.shape[1]/2), dim=1)
                
                if phase == 'train':
                    opt.zero_grad()
                    utils.to_device(model, device)

                    if module1_loss_stable:
                        pgv1, pgv2 = model(x.to('cuda'))
                        MSE_loss2 = utils.huber_loss(yPheno.to('cuda'), pgv2)
                    else:
                        pgv1, _ = model(x.to('cuda'))
                        MSE_loss2 = torch.tensor(0.0).to('cuda')

                    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                    MSE_loss1 = utils.huber_loss(yGBLUP.to('cuda'), pgv1) + 0.001 * l2_norm
                    loss = MSE_loss1 + MSE_loss2
                    
                    # backpropagate the error
                    loss.backward()
                    
                    # Update the weights
                    opt.step()
                    
                    history['train_epoch_mse_loss1'].append(MSE_loss1.item())
                    if module1_loss_stable:
                        history['train_epoch_mse_loss2'].append(MSE_loss2.item())

                if phase == 'val':
                    utils.to_device(model, devicecpu)
                    
                    if module1_loss_stable:
                        pgv1, pgv2 = model(x.to('cpu'))
                        MSE_loss2 = utils.huber_loss(yPheno.to('cpu'), pgv2)
                    else:
                        pgv1, _ = model(x.to('cpu'))
                        MSE_loss2 = torch.tensor(0.0).to('cpu')

                    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                    MSE_loss1 = utils.huber_loss(yGBLUP.to('cpu'), pgv1.to('cpu')) + 0.001 * l2_norm
                    loss = MSE_loss1 + MSE_loss2
                    
                    history['val_epoch_mse_loss1'].append(MSE_loss1.item())
                    if module1_loss_stable:
                        history['val_epoch_mse_loss2'].append(MSE_loss2.item())

                # save losses for all batches in 1 epoch
                single_epoch_mse_loss1.append(MSE_loss1.item())
                if module1_loss_stable:
                    single_epoch_mse_loss2.append(MSE_loss2.item())

                predictions1.append(pgv1.to('cpu'))
                real1.append(yGBLUP.to('cpu'))

                if module1_loss_stable:
                    predictions2.append(pgv2.to('cpu'))
                    real2.append(yPheno.to('cpu'))

                iters += 1

            epoch_mse_loss1 = sum(single_epoch_mse_loss1) / len(single_epoch_mse_loss1)
            if module1_loss_stable:
                epoch_mse_loss2 = sum(single_epoch_mse_loss2) / len(single_epoch_mse_loss2)
            else:
                epoch_mse_loss2 = 0.0

            all_pred1 = torch.squeeze(torch.cat(predictions1, dim=0))
            all_real1 = torch.squeeze(torch.cat(real1, dim=0))
            correlation1 = utils.corloss(all_real1, all_pred1)

            if module1_loss_stable:
                all_pred2 = torch.squeeze(torch.cat(predictions2, dim=0))
                all_real2 = torch.squeeze(torch.cat(real2, dim=0))
                correlation2 = utils.corloss(all_real2, all_pred2)
            else:
                correlation2 = 0.0

            # Save 1 loss per epoch
            if phase == 'train':
                history['cor_loss1'].append(correlation1.item() if isinstance(correlation1, torch.Tensor) else correlation1)
                history['mse_loss1'].append(epoch_mse_loss1)
                history['cor_loss2'].append(correlation2.item() if isinstance(correlation2, torch.Tensor) else correlation2)
                history['mse_loss2'].append(epoch_mse_loss2)
            else:
                history['val_cor_loss1'].append(correlation1.item() if isinstance(correlation1, torch.Tensor) else correlation1)
                history['val_mse_loss1'].append(epoch_mse_loss1)
                history['val_cor_loss2'].append(correlation2.item() if isinstance(correlation2, torch.Tensor) else correlation2)
                history['val_mse_loss2'].append(epoch_mse_loss2)

            print('{}: \tcor_Loss1: {:.4f}\tmse_loss1: {:.4f}\tcor_Loss2: {:.4f}\tmse_loss2: {:.4f}'
                  .format(phase, correlation1, epoch_mse_loss1, correlation2, epoch_mse_loss2))
            
            if module1_loss_stable:
                if phase == 'val':
                    early_stopping(epoch_mse_loss2, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        return history

        # Check if the loss has stabilized for module 1
        if not module1_loss_stable:
            if early_stopping.counter == 0:
                module1_stable_epochs += 1
                if module1_stable_epochs >= stable_threshold:
                    module1_loss_stable = True
                    print("Module 1 loss has stabilized. Starting training for Module 2.")
            else:
                module1_stable_epochs = 0

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))
    # Process is complete.
    time_elapsed = time.time() - since
    print('Training process complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return history


def fitTrait2Trait(model, data, optimizer, num_epochs, patience, checkpoint_path, device):
    model = model
    num_epochs = num_epochs
    opt = optimizer
    data_loaders = data
    iters = 0
    since = time.time()
    
    # Save Losses
    history = {}
    history['mse_loss1'] = []
    history['val_mse_loss1'] = []
    history['train_epoch_mse_loss1'] = []
    history['val_epoch_mse_loss1'] = []
    

    # initialize the early_stopping object
    early_stopping = utils.EarlyStopping(patience=patience, path = checkpoint_path, verbose=True)
    
    mseloss = nn.MSELoss()
    
    print("Starting Training ...")
    loss_idx_value = 0
    for phase in ['train', 'val']:
        train_batch = len(data_loaders['train'])
        val_batch = len(data_loaders['val'])
        train_loader = utils.DevicedataLoader(data_loaders['train'], device)
        val_loader = utils.DevicedataLoader(data_loaders['val'], device)
        
    batch_nb = {"train": train_batch, "val": val_batch}
    data_loaders = {"train": train_loader, "val": val_loader}
    # For each epoch
    for epoch in range(num_epochs):
      
          # Print epoch
          print(f'Starting epoch {epoch+1}')
      # Each epoch has a training and validation phase
    
          for phase in ['val', 'train']:
            if phase == 'train':
                  model.train(True)  # Set model to training mode
            else:
                  model.train(False)  # Set model to evaluate mode

            single_epoch_mse_loss1 = []
            predictions1 =  []
            real1 = []
            
            # For each batch in the dataloader
            for i, data in tqdm(enumerate(data_loaders[phase]), total = batch_nb[phase]):
                time.sleep(0.01)
                # get the input and their corresponding labels
                x = data[0]
                yPheno = data[1].float()
                
                if phase == 'train':
                    opt.zero_grad()
                    utils.to_device(model,device)
                    pgv2 = model(x.to('cuda'))
                    
                    MSE_loss3 = utils.huber_loss(yPheno.to('cuda'), pgv2)
                    loss = MSE_loss3
                    
                    # backpropagate the error
                    loss.backward()
                    
                    # Update the weights
                    opt.step()
                    
                    history['train_epoch_mse_loss1'].append(MSE_loss3.item())

                    
                if phase == 'val':
                    utils.to_device(model,devicecpu)
                    pgv2 = model(x.to('cpu'))
                    MSE_loss3 = utils.huber_loss(yPheno.to('cpu'), pgv2.to('cpu'))
                    loss = MSE_loss3
                    
                    history['val_epoch_mse_loss1'].append(MSE_loss3.item())
                    

                # save losses for all batches in 1 epoch
                single_epoch_mse_loss1.append(MSE_loss3.item())

                iters += 1

            epoch_mse_loss1 = sum(single_epoch_mse_loss1)/len(single_epoch_mse_loss1)
            
            
             # Save 1 loss per epoch
            if phase == 'train':
                history['mse_loss1'].append(epoch_mse_loss1)
                
            else:
                history['val_mse_loss1'].append(epoch_mse_loss1)
                

            print('{}: \tmse_loss1: {:.4f}'
                  .format(phase, epoch_mse_loss1))
            
            if phase == 'val':
                early_stopping(epoch_mse_loss1, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    return history

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))
    # Process is complete.
    time_elapsed = time.time() - since
    print('Training process complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return history


def fitMeanTrait2Trait(model, data, optimizer, num_epochs, patience, checkpoint_path, device):
    model = model
    num_epochs = num_epochs
    opt = optimizer
    data_loaders = data
    iters = 0
    since = time.time()
    
    # Save Losses
    history = {}
    history['mse_loss1'] = []
    history['val_mse_loss1'] = []
    history['train_epoch_mse_loss1'] = []
    history['val_epoch_mse_loss1'] = []
    

    # initialize the early_stopping object
    early_stopping = utils.EarlyStopping(patience=patience, path = checkpoint_path, verbose=True)
    
    mseloss = nn.MSELoss()
    
    print("Starting Training ...")
    loss_idx_value = 0
    for phase in ['train', 'val']:
        train_batch = len(data_loaders['train'])
        val_batch = len(data_loaders['val'])
        train_loader = utils.DevicedataLoader(data_loaders['train'], device)
        val_loader = utils.DevicedataLoader(data_loaders['val'], device)
        
    batch_nb = {"train": train_batch, "val": val_batch}
    data_loaders = {"train": train_loader, "val": val_loader}
    # For each epoch
    for epoch in range(num_epochs):
      
          # Print epoch
          print(f'Starting epoch {epoch+1}')
      # Each epoch has a training and validation phase
    
          for phase in ['val', 'train']:
            if phase == 'train':
                  model.train(True)  # Set model to training mode
            else:
                  model.train(False)  # Set model to evaluate mode

            single_epoch_mse_loss1 = []
            predictions1 =  []
            real1 = []
            
            # For each batch in the dataloader
            for i, data in tqdm(enumerate(data_loaders[phase]), total = batch_nb[phase]):
                time.sleep(0.01)
                # get the input and their corresponding labels
                x = data[0]
                yPheno = data[1].float()
                
                if phase == 'train':
                    opt.zero_grad()
                    utils.to_device(model,device)
                    meanpgv2 = model(x.unsqueeze(1).to('cuda'))
                    
                    MSE_loss3 = utils.huber_loss(yPheno.to('cuda'), meanpgv2)
                    loss = MSE_loss3
                    
                    # backpropagate the error
                    loss.backward()
                    
                    # Update the weights
                    opt.step()
                    
                    history['train_epoch_mse_loss1'].append(MSE_loss3.item())

                    
                if phase == 'val':
                    utils.to_device(model,devicecpu)
                    meanpgv2 = model(x.unsqueeze(1).to('cpu'))
                    MSE_loss3 = utils.huber_loss(yPheno.to('cpu'), meanpgv2.to('cpu'))
                    loss = MSE_loss3
                    
                    history['val_epoch_mse_loss1'].append(MSE_loss3.item())
                    

                # save losses for all batches in 1 epoch
                single_epoch_mse_loss1.append(MSE_loss3.item())

                iters += 1

            epoch_mse_loss1 = sum(single_epoch_mse_loss1)/len(single_epoch_mse_loss1)
            
            
             # Save 1 loss per epoch
            if phase == 'train':
                history['mse_loss1'].append(epoch_mse_loss1)
                
            else:
                history['val_mse_loss1'].append(epoch_mse_loss1)
                

            print('{}: \tmse_loss1: {:.4f}'
                  .format(phase, epoch_mse_loss1))
            
            if phase == 'val':
                early_stopping(epoch_mse_loss1, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    return history

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))
    # Process is complete.
    time_elapsed = time.time() - since
    print('Training process complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return history
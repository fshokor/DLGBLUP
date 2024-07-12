import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
import time
import math
from pandas import DataFrame
from scipy.stats import linregress, pearsonr
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import sys


def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    absolute_error = torch.abs(error)
    quadratic = 0.5 * (error ** 2)
    linear = delta * (absolute_error - 0.5 * delta)
    
    loss = torch.where(absolute_error <= delta, quadratic, linear)
    return torch.mean(loss)

def get_default_device():
    ''' Picking GPU if available or else CPU'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Move data/model to the GPU
def to_device(data, device):
    '''Move data/model to chosen device'''
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def GetDataLoaderMT(features, targets, batch_size, shuffle = False, center = False, Norm = False):
    """
    Returns a DataLoader with preprocessed data.
    
    Args:
    - features (numpy.array): Feature data.
    - targets (numpy.array): Target data.
    - batch_size (int): Batch size for DataLoader.
    - shuffle (bool): Whether to shuffle data in DataLoader.
    - centering (bool): Whether to center target data.
    - Norm (bool): Whether to normalize target data.
    
    Returns:
    DataLoader with preprocessed data.
    """
    
    if center: 
        # Compute the mean of each column
        column_means = np.mean(targets, axis=0)
        # Subtract the mean of each column from the data
        targets = targets - column_means
        
    if Norm: 
        # Initialize the scaler
        scaler = MinMaxScaler()
        # Fit the scaler and transform the data
        targets = scaler.fit_transform(targets) 

    tinputs = torch.tensor(np.array(features))#, dtype=torch.float64) 
    ttargets = torch.from_numpy(targets)

    # Create dataset by combinig features and labels
    dataset = TensorDataset(tinputs, ttargets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    return data_loader

class DevicedataLoader():
    '''Wrap a dataloader to move data to a device'''
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        '''Yield a batch of data after moving it to device'''
        for b in self.dl:
            yield to_device(b, self.device)
            
def plot_learning_curve(training_loss, val_loss, title):
    plt.subplots(figsize=(7, 4))
    plt.plot(training_loss) 
    plt.plot(val_loss)
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show() 
    
    
def evaluateMT(Real_df, Predicted_df):
        
    all_results = []
    for i in range(Real_df.shape[1]):
        variance_real = np.var(np.array(Real_df.iloc[:, i]))
        variance_pred = np.var(np.array(Predicted_df.iloc[:,i]))
        mean_real = np.mean(np.array(Real_df.iloc[:, i]))
        mean_pred = np.mean(np.array(Predicted_df.iloc[:,i]))
        mses = ((np.array(Real_df.iloc[:, i])- np.array(Predicted_df.iloc[:,i]))**2).mean(axis = 0)
        correlation, p_value = pearsonr(Real_df.iloc[:, i], Predicted_df.iloc[:,i])
        slope, intercept, r_value, p_value, std_err = linregress(Predicted_df.iloc[:,i], Real_df.iloc[:, i])
        results = { 'Correlation': round(correlation, 2), 'MSE': round(mses, 2), 'Slope': round(slope,2), 
                   'Standard error': round(std_err, 2), 'Variance Real': round(variance_real,2), 'Variance GEBV': round(variance_pred,2)}

        all_results.append(results)

    return pd.DataFrame(all_results)


def plot_Traits(*dataframes, colors=None, show_legend=False, legend_names=None, path=None):
    num_dataframes = len(dataframes)
    num_traits = dataframes[0].shape[1]
    sns.set(rc={'figure.figsize': (22, 4)})
    sns.set_style("white")

    # Use provided colors or default to a seaborn color palette
    if colors is None:
        colors = sns.color_palette(n_colors=num_dataframes)
    elif len(colors) < num_dataframes:
        raise ValueError("Number of provided colors is less than the number of dataframes.")

    # Check if legend names are provided and match the number of dataframes
    if show_legend and legend_names is not None:
        if len(legend_names) != num_dataframes:
            raise ValueError("Number of legend names does not match the number of dataframes.")
    elif show_legend:
        legend_names = [f"DataFrame {i+1}" for i in range(num_dataframes)]

    # Create a subplot for each trait
    fig, axes = plt.subplots(1, num_traits - 1)

    # If there is only one trait, axes is not an array, so we make it into one
    if num_traits == 2:
        axes = [axes]

    plot_labels = []

    for i, ax in enumerate(axes):
        for j, df in enumerate(dataframes):
            scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, i + 1], color=colors[j])
            if i == 0:  # Add label only once
                plot_labels.append(scatter)

        #ax.set_xlabel("Trait 1")
        #ax.set_ylabel(f"Trait {i + 2}")
        ax.tick_params(labelsize= 15)

    if show_legend:
        plt.legend(handles=plot_labels, labels=legend_names, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=num_dataframes)
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
        
    plt.show()

def corloss(g, y_hat):
        '''Compute Pearson correlation between predicted and real value'''
        vg = g - torch.mean(g)
        vy_hat = y_hat - torch.mean(y_hat)
        pearson_loss = torch.sum(vg * vy_hat) / ((torch.sqrt(torch.sum(vg ** 2)) * torch.sqrt(torch.sum(vy_hat ** 2))))  
        return pearson_loss
    
    
def PredictBV(model, val_loader):
    test_data = next(iter(val_loader))
    test_x = test_data[0]
    if test_x.dim() == 1:  # Check if test_x is 1D
        test_x = test_x.unsqueeze(1)  # Unsqueezes at dimension 1, making it 2D
    outputs = model(test_x)
    
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    num_outputs = len(outputs)
    
    # Initialize a list to hold all predictions for each output
    all_predictions = [[] for _ in range(num_outputs)]
    
    # Iterate over all data in the validation loader
    for data in val_loader:
        x = data[0]
        if x.dim() == 1:  
            x = x.unsqueeze(1)  
        outputs = model(x)
        
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Detach each output and send to CPU, then append to the respective list
        for i, output in enumerate(outputs):
            all_predictions[i].append(output.detach().to('cpu'))

    # Process each set of predictions: concatenate, squeeze, and convert to DataFrame
    results = []
    for predictions in all_predictions:
        all_pred = torch.squeeze(torch.cat(predictions, dim=0))
        results.append(pd.DataFrame(all_pred.numpy()))  # Convert to DataFrame
    
    return results  # This will be a list of DataFrames
    
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.path = path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        



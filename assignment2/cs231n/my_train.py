import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm

device = torch.device('cuda')
dtype = torch.float32

def my_check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

    return float(num_correct) / num_samples

def my_train(model, loader_train, loader_val, optimizer, lr_scheduler=None, epochs=1, save_loss_every=250, save_best_model=False):
    loss_history = []
    training_acc_history = []
    best_model = None
    best_train_acc = 0.0
    best_val_acc = 0.0
    
    val_acc_history = []
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    epoch_bar = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for e in epoch_bar:
        #data_bar = tqdm(loader_train, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        #data_bar.set_description('Train Epoch: [{}/{}]'.format(e, epochs))
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % save_loss_every == 0:
                loss_history.append(loss.item())

        training_acc = my_check_accuracy(loader_train, model)
        training_acc_history.append(training_acc)
        if training_acc > best_train_acc:
            best_train_acc = training_acc

        val_acc = my_check_accuracy(loader_val, model)
        val_acc_history.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_best_model :
                best_model = deepcopy(model)
        
        epoch_bar.set_description('Train Epoch: [{}/{}] | Best Train Accuracy: {:.4f} | Best Validation Accuracy: {:.4f}'.format(e+1, epochs, best_train_acc, best_val_acc))
        #print('Train Epoch: [{}/{}] | Train Accuracy: {:.4f} | Validation Accuracy: {:.4f}'.format(e+1, epochs, training_acc, val_acc))

        if lr_scheduler is not None:
            lr_scheduler.step()
                
    return (best_model, loss_history, training_acc_history, val_acc_history)

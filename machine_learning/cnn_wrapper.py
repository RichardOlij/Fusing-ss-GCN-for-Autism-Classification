import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils import data as udata
from store_datadict import StoreDataDict
from cnn_model import Net3d1, Net3d2, Net3d3, Net3d4, Net3d5, Net3d6, Net3d7, Net3d8, Net3d9, Net3d10
import sklearn.metrics as sklearnmetrics
from evaluate import evaluate

class Dataset(udata.Dataset):
  def __init__(self, data, indices_name:None):
        'Initialization'
        if indices_name:
            # If the indices_name is set, use these indices to create a subset.
            self.data = data['fmri'][data[indices_name]]
            self.labels = data['label'][data[indices_name]]
            self.ids = data['id'][data[indices_name]]
        else:
            # Use all data otherwise.
            self.data = data['fmri']
            self.labels = data['label']
            self.ids = data['id']

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'

        X = self.data[index,:,:,:]
        y = self.labels[index]
        id = self.ids[index]
        return X, y, id

    
    #%% Training  
def train_epoch_3dcnn(args, model, criterion, optimizer, epoch, dataloader, train):
    """
    This inbetween step is to disable the gradients in case of non-training situations.
    """
    if not train:
        with torch.no_grad():
            model.eval()
            return train_epoch_3dcnn_(args, model, criterion, optimizer, epoch, dataloader, train)
    model.train()
    return train_epoch_3dcnn_(args, model, criterion, optimizer, epoch, dataloader, train)

def evaluate_wrapper(pred, lab):    
#    pred = np.argmax(pred.detach().cpu().numpy(), 1)
    pred = pred.detach().cpu().numpy()
    lab = lab.cpu().numpy()
    return evaluate(pred, lab)
    
def train_epoch_3dcnn_(args, model, criterion, optimizer, epoch, dataloader, train):
    auc, acc, mean_pred, mean_lab, sens, spec, loss = 0,0,0,0,0,0,0
    all_outputs = []
    all_labels = []
    t_begin = time.time()
    for iteration, (inputs, labels, _) in enumerate(dataloader, 0):
        
        inputs = torch.FloatTensor(inputs).to(args.device)
        labels = torch.LongTensor(labels).to(args.device)
        
        if train:
            # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss_ = criterion(outputs, labels)
        
        if train:
            loss_.backward()
            optimizer.step()
    
        auc_, acc_, mean_pred_, mean_lab_, sens_, spec_ = evaluate_wrapper(outputs, labels)
        
        auc += auc_
        acc += acc_
        mean_pred += mean_pred_
        mean_lab += mean_lab_
        sens += sens_
        spec += spec_
        loss += loss_.item()
        
        all_outputs.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
  
    n_iterations = iteration + 1
    auc /= n_iterations
    acc /= n_iterations
    mean_pred /= n_iterations
    mean_lab /= n_iterations
    sens /= n_iterations
    spec /= n_iterations
    loss /= n_iterations
    duration = (time.time() - t_begin)
    
    output_all_matrix = np.concatenate(all_outputs, axis=0)
    labels_all_matrix = np.concatenate(all_labels, axis=0)
    # Stuff for plotting the curve later on.
    fpr, tpr, thresholds = sklearnmetrics.roc_curve(labels_all_matrix, output_all_matrix[:,1])
    
    return auc, acc, mean_pred, mean_lab, sens, spec, loss, duration, fpr, tpr, thresholds

def get_embedding_3dcnn(args, model, data):
    allloader = torch.utils.data.DataLoader(Dataset(data, None), batch_size=args.batch_size_3dcnn, shuffle=False, num_workers=1)
    embedding_all = []
    with torch.no_grad():
        model.eval()
        for inputs, labels, _ in allloader:
        
            inputs = torch.FloatTensor(inputs).to(args.device)
            labels = torch.LongTensor(labels).to(args.device)
            
            embedding = model.get_embedding(inputs)
            embedding_all.append(embedding.cpu().numpy())
    
    embedding_all_matrix = np.concatenate(embedding_all, axis=0)
    return embedding_all_matrix
        
def train_pre(args, data):
    
    # Create the datasets.
    torch.manual_seed(0)
    trainloader = torch.utils.data.DataLoader(Dataset(data, "train_ind"), batch_size=args.batch_size_3dcnn, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(Dataset(data, "val_ind"), batch_size=args.batch_size_3dcnn, shuffle=True, num_workers=2)    
    
    # Create 3DCNN model etc.
    if args.cnn_model == 1:
        model = Net3d1(args).to(args.device)
    elif args.cnn_model == 2:
        model = Net3d2(args).to(args.device)
    elif args.cnn_model == 3:
        model = Net3d3(args).to(args.device)
    elif args.cnn_model == 4:
        model = Net3d4(args).to(args.device)
    elif args.cnn_model == 5:
        model = Net3d5(args).to(args.device)
    elif args.cnn_model == 6:
        model = Net3d6(args).to(args.device)
    elif args.cnn_model == 7:
        model = Net3d7(args).to(args.device)
    elif args.cnn_model == 8:
        model = Net3d8(args).to(args.device)
    elif args.cnn_model == 9:
        model = Net3d9(args).to(args.device)
    elif args.cnn_model == 10:
        model = Net3d10(args).to(args.device)
    else:
        raise ValueError("Cnn model \"\" does not exist.".format(args.cnn_model))
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate_3dcnn, momentum=args.momentum_3dcnn)
    
    return model, (trainloader, valloader, data, criterion, optimizer)
        
def train_epoch(args, store_data_dict, epoch, model, *train_pre_args):
    trainloader, valloader, data, criterion, optimizer = train_pre_args[0]
    
    train_results = train_epoch_3dcnn(args, model, criterion, optimizer, epoch, trainloader, train=True)
    store_data_dict.store_data_train(train_results)
    
    val_results = train_epoch_3dcnn(args, model, criterion, optimizer, epoch, valloader, train=False)
    store_data_dict.store_data_val(val_results)
    
    return store_data_dict

def save_final_predictions(args, epoch, fold, model, *train_pre_args):
    trainloader, valloader, data, criterion, optimizer = train_pre_args[0]
    
    print("Saving the current predictions.")
    
    allloader = torch.utils.data.DataLoader(Dataset(data, None), batch_size=args.batch_size_3dcnn, shuffle=False, num_workers=1)
    output_all = []
    with torch.no_grad():
        model.eval()
        for inputs, labels, _ in allloader:
        
            inputs = torch.FloatTensor(inputs).to(args.device)
            labels = torch.LongTensor(labels).to(args.device)
            
            output = model(inputs)
            output_all.append(output.cpu().numpy())
    
    output_all_matrix = np.concatenate(output_all, axis=0)
    
    np.save("{}/fold_{}__epoch_{}.npy".format(args.dir_name_predictions, fold, epoch), output_all_matrix)
    
def save_model(args, epoch, fold, model, *train_pre_args):
    trainloader, valloader, data, criterion, optimizer = train_pre_args[0]
    
    print("Saving the current model.")
    torch.save(model.state_dict(), "{}/fold_{}__epoch_{}.pt".format(args.dir_name_model, fold, epoch)) 
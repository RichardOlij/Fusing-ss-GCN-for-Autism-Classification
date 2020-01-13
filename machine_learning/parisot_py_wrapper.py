import time
import numpy as np
import sklearn.metrics as sklearnmetrics
from evaluate import evaluate
from parisot_py_model import GCN
import torch
import torch.optim as optim
from scipy import stats

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features  

def preprocess_features_min_max(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    max_row = np.max(features, axis = 1)
    min_row = np.min(features, axis = 1)
    features = (features.transpose() - min_row).transpose()
    features = (features.transpose()/ (max_row-min_row)).transpose()
    return features  

def preprocess_features_z_score(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    features = stats.zscore(features, axis=1, ddof=1)
    return features 

def masked_softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    loss_f = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_f(preds, labels)
    return loss

def evaluate_wrapper(pred, lab):    
    pred = pred.detach().cpu().numpy()
    lab = lab.cpu().numpy()
    return evaluate(pred, lab)

def train_pre(args, data):
    inputs = torch.FloatTensor(data["input_features"]).to(args.device)
    labels = torch.LongTensor(data["label"]).to(args.device)
    adj_support = torch.FloatTensor(data["adj_support"]).to(args.device)
    
    model = GCN(data["input_features"].shape[1], 2, args.hidden_parisot, adj_support, args.dropout_parisot).to(args.device)
    criterion = masked_softmax_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_parisot, weight_decay=args.weight_decay_parisot)
            
    return model, (criterion, optimizer, inputs, labels, data)

        
def train_epoch(args, store_data_dict, epoch, model, *train_pre_args):
    criterion, optimizer, inputs, labels, data = train_pre_args[0]
    
    train_results = train_epoch_parisot(model, criterion, optimizer, epoch, inputs, labels, data["train_ind"], True)
    store_data_dict.store_data_train(train_results)
    
    val_resutls = train_epoch_parisot(model, criterion, optimizer, epoch, inputs, labels, data["val_ind"], False)
    store_data_dict.store_data_val(val_resutls)
        
    return store_data_dict


def train_epoch_parisot(model, criterion, optimizer, epoch, inputs, labels, indices, train):
    """
    This inbetween step is to disable the gradients in case of non-training situations.
    """
    if not train:
        with torch.no_grad():
            model.eval()
            return train_epoch_parisot_(model, criterion, optimizer, epoch, inputs, labels, indices, train)
    model.train()
    return train_epoch_parisot_(model, criterion, optimizer, epoch, inputs, labels, indices, train)
            
def train_epoch_parisot_(model, criterion, optimizer, epoch, inputs, labels, indices, train):
    t_begin = time.time()
    
    if train:
        # zero the parameter gradients
        optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss_ = criterion(outputs[indices], labels[indices])
    
    if train:
        loss_.backward()
        optimizer.step()
        
    auc, acc, mean_pred, mean_lab, sens, spec = evaluate_wrapper(outputs[indices], labels[indices])
    
    duration = (time.time() - t_begin)
    

    # Stuff for plotting the curve later on.
    fpr, tpr, thresholds = sklearnmetrics.roc_curve(labels[indices].detach().cpu().numpy(), outputs[indices][:,1].detach().cpu().numpy())
    
    return auc, acc, mean_pred, mean_lab, sens, spec, loss_, duration, fpr, tpr, thresholds

def save_final_predictions(args, epoch, fold, model, *train_pre_args):
    criterion, optimizer, inputs, labels, data = train_pre_args[0]
    
    with torch.no_grad():
        model.eval()
        outputs = model(inputs)
        
        print("Saving the current predictions.")
        np.save("{}/fold_{}__epoch_{}.npy".format(args.dir_name_predictions, fold, epoch), outputs.detach().cpu().numpy())
    
    
def save_model(args, epoch, fold, model, *train_pre_args):
    criterion, optimizer, inputs, labels, data = train_pre_args[0]
    
    print("Saving the current model.")
    torch.save(model.state_dict(), "{}/fold_{}__epoch_{}.pt".format(args.dir_name_model, fold, epoch))

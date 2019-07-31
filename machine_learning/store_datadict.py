import os
import numpy as np

class StoreDataDict(object):
    
    def __init__(self, data):
        self.store_data_dict = {
            "list_train_auc": [],
            "list_train_accuracy": [],
            "list_train_mean_pred": [],
            "list_train_mean_lab": [],
            "list_train_sensitivity": [],
            "list_train_specificity": [],
            "list_train_loss": [],
            "list_train_duration": [],
            
            "list_train_fpr": [],
            "list_train_tpr": [],
            "list_train_thresholds": [],
            
            "list_val_auc": [],
            "list_val_accuracy": [],
            "list_val_mean_pred": [],
            "list_val_mean_lab": [],
            "list_val_sensitivity": [],
            "list_val_specificity": [],
            "list_val_loss": [],
            "list_val_duration": [],
            
            "list_val_fpr": [],
            "list_val_tpr": [],
            "list_val_thresholds": [],
            
            "num_train_pos":sum(data['label'][data["train_ind"]]),
            "num_train_neg":sum(data['label'][data["train_ind"]]==0),
            "num_val_pos":sum(data['label'][data["val_ind"]]),
            "num_val_neg":sum(data['label'][data["val_ind"]]==0)}
    
    def store_data_train(self, results):
        auc, acc, mean_pred, mean_lab, sens, spec, loss, duration, fpr, tpr, thresholds = results

        self.store_data_dict["list_train_auc"].append(auc)
        self.store_data_dict["list_train_accuracy"].append(acc)
        self.store_data_dict["list_train_mean_pred"].append(mean_pred)
        self.store_data_dict["list_train_mean_lab"].append(mean_lab)        
        self.store_data_dict["list_train_sensitivity"].append(sens)
        self.store_data_dict["list_train_specificity"].append(spec)
        self.store_data_dict["list_train_loss"].append(loss)
        self.store_data_dict["list_train_duration"].append(duration)
        
        self.store_data_dict["list_train_fpr"].append(fpr)
        self.store_data_dict["list_train_tpr"].append(tpr)
        self.store_data_dict["list_train_thresholds"].append(thresholds)
        
    def store_data_val(self, results):
        auc, acc, mean_pred, mean_lab, sens, spec, loss, duration, fpr, tpr, thresholds = results

        self.store_data_dict["list_val_auc"].append(auc)
        self.store_data_dict["list_val_accuracy"].append(acc)
        self.store_data_dict["list_val_mean_pred"].append(mean_pred)
        self.store_data_dict["list_val_mean_lab"].append(mean_lab)        
        self.store_data_dict["list_val_sensitivity"].append(sens)
        self.store_data_dict["list_val_specificity"].append(spec)
        self.store_data_dict["list_val_loss"].append(loss)
        self.store_data_dict["list_val_duration"].append(duration)
        
        self.store_data_dict["list_val_fpr"].append(fpr)
        self.store_data_dict["list_val_tpr"].append(tpr)
        self.store_data_dict["list_val_thresholds"].append(thresholds)

    def print_data_epoch(self, fold_i, epoch):        
        print("F-{}, E-{}: train: loss {:.4f}, acc {:.4f}, auc {:.4f}, sens {:4f}, spec {:4f}. val: loss {:.4f}, acc {:.4f}, auc {:.4f}, sens {:4f}, spec {:4f}. duration {:.2f}.".format(
                    fold_i, 
                    epoch+1, 
                    self.store_data_dict["list_train_loss"][-1], 
                    self.store_data_dict["list_train_accuracy"][-1],
                    self.store_data_dict["list_train_auc"][-1],
                    self.store_data_dict["list_train_sensitivity"][-1],
                    self.store_data_dict["list_train_specificity"][-1],
                    self.store_data_dict["list_val_loss"][-1],
                    self.store_data_dict["list_val_accuracy"][-1],
                    self.store_data_dict["list_val_auc"][-1], 
                    self.store_data_dict["list_val_sensitivity"][-1],
                    self.store_data_dict["list_val_specificity"][-1],
                    self.store_data_dict["list_train_duration"][-1] + self.store_data_dict["list_val_duration"][-1]))

def save_datadicts(args, store_data_dicts:list):
    print("Saving datadict.")
    np.save("{}/store_data_dicts.npy".format(args.dir_name_datadict), store_data_dicts)
    
def load_datadicts(args):
    fname = "{}/store_data_dicts.npy".format(args.dir_name_datadict)
    if os.path.isfile(fname):
        print("Loading datadict.")
        return_data = list(np.load(fname, allow_pickle=True))
        if args.max_fold < 0:
            return return_data
        # Else return just up to that fold, since the rest will not be used anyway.
        return return_data[:args.max_fold]
    else: 
        print("No datadict found")
        return []
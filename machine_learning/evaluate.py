import numpy as np
from sklearn import metrics

#%% Measurements

def evaluate(pred, lab):
    # Compute the area under curve
    # AUC fails if labels are only of one class, therefore give accuracy in that case.
    if np.std(lab) == 0: # Check if the sum of colums of any is zero, this indicates all true or all false labels.
        # Only one kind of labels.
        print("No variation in the labels, just the accuracy is calculated.")
        auc = metrics.accuracy_score(lab, np.argmax(pred, 1))
    else:
        auc = metrics.roc_auc_score(lab, pred[:,1])
    
    pred = np.argmax(pred, 1)
    # Compute the means
    total_samples = float(len(lab))
    mean_pred = float(sum(pred))/total_samples
    mean_lab = float(sum(lab))/total_samples


    TP = float(np.sum(np.logical_and(pred == 1, lab == 1)))
    TN = float(np.sum(np.logical_and(pred == 0, lab == 0)))
    FP = float(np.sum(np.logical_and(pred == 1, lab == 0)))
    FN = float(np.sum(np.logical_and(pred == 0, lab == 1)))
    
    if (TP+FN) == 0:
        sens = 0
    else:
        sens = TP/(TP+FN)
        
    if (FP+TN) == 0:
        spec = 0
    else:
        spec = TN/(FP+TN)
        
    acc = (sens+spec)/2

    return auc, acc, mean_pred, mean_lab, sens, spec
import numpy as np
from collections import defaultdict
from arguments_baseline import obtain_arguments
from store_datadict import load_datadicts
from scipy import interp
import matplotlib.pyplot as plt
import sklearn.metrics as sklearnmetrics
import os

#%% Obtaining settings

args = obtain_arguments()


#%% Loading data and show
data = load_datadicts(args)

keys = data[0].keys() # ["list_test_accuracy", "list_test_auc", ...]

def get_averages():
    print("\nAverages:")
    sums = defaultdict(lambda: 0)
    counter = 0
    for data_sample in data:
        counter += 1
        for key in keys:
            data_ = data_sample[key]
            # If the data is a list (e.g. all losses of all epochs), take the last element.
            if isinstance(data_, list):
                data_ = data_[-1]
            if isinstance(data_, str):
#                print("Data at key \"{}\" ignored since it is a string ({}).".format(key, data_))
                continue
            sums[key] += data_
    
    # Read averages
    for key in keys:
        avg = sums[key]/float(counter)
        print("{}: {:.3f}".format(key, avg))
        
#get_averages()


def get_table_results(str_ds):
    # Read validation scores based on sites:
    print("\n{} scores:".format(str_ds))
    keys = ["list_{}_sensitivity".format(str_ds),
            "list_{}_specificity".format(str_ds), 
            "list_{}_accuracy".format(str_ds), 
            "list_{}_auc".format(str_ds)]
    totals = [0.]*len(keys)
    str_latex = ""
    # Get the results and sum the results to totals.
    for fold_i, data_sample in enumerate(data):
        str_latex = "{}\n".format(str_latex)
        for i_key, key in enumerate(keys):
            data_ = data_sample[key]
            data_last = data_[-1]
            totals[i_key] += data_last
            str_latex = "{} & {:.3f}".format(str_latex, data_last)
        str_latex = "{} \\\\".format(str_latex)
        
    str_latex = "{} \\hline\n".format(str_latex)
    
    
    # Finally the average line
    n_keys = float(len(data))
    str_averages = " & ".join(["{:.3f}".format(round(v/n_keys,3)) for v in totals])
    str_latex = "{}Averages & {} \\\\".format(str_latex, str_averages)
    print(str_latex)

print("\n\nsens, spec, acc, auc\n\n")
get_table_results("train")        
get_table_results("val")

def get_table_distributions(): 
    print("\nValidation site distributions:")
    for fold_i, data_sample in enumerate(data):
        str_latex = "{} & {} & {} \\\\".format(
                data_sample["site"],
                data_sample["num_val_pos"],
                data_sample["num_val_neg"])
        print(str_latex)
        
#get_table_distributions()
        
def get_auc_curves(str_ds):
    print("\n{} scores:".format(str_ds))
    
    
    mean_fpr = np.linspace(0, 1, 100)
    
    tprs = []
    aucs = []
    for fold_i, data_sample in enumerate(data):
        
        # Get the last epoch of each data
        fpr = data_sample["list_{}_fpr".format(str_ds)][-1]
        tpr = data_sample["list_{}_tpr".format(str_ds)][-1]
#        thresholds = data_sample["list_{}_thresholds".format(str_ds)][-1]
        roc_auc = data_sample["list_{}_auc".format(str_ds)][-1]
        
        # Plot the roc curve and show the auc in the label.
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (fold_i, roc_auc))
        
        # Interpolate the results since the roc_curve function returns varying lengts of datapoints.
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0 # Start at 0
        aucs.append(roc_auc)
        
    # Plot the straight change line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    
    # Plot the mean line
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0 # Finish at 1
    mean_auc = sklearnmetrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the variation area.
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Show the image.
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    
    fname = "{}/fig_{}.png".format(args.dir_name_datadict, str_ds)
    print("Saving figure at \"{}\".".format(fname))
    
    plt.savefig(fname, bbox_inches="tight", transparent=True, dpi=400)
    # Only show in the case of spyder executer, terminal or colab should not show the images.
    if any('SPYDER' in name for name in os.environ):
        plt.show()
    plt.close()
    
    
get_auc_curves("train")
get_auc_curves("val")
import h5py as h5
import numpy as np
from sklearn.model_selection import StratifiedKFold

np.random.seed(0)
dirname = "../data_storage/preprocessed/dataset_ids"
#%% Reading the hdf5 data
def get_data_raw(raw_data_preprocess_method):
    """
    Obtaining all the information.
    """
    fname = "../data_storage/preprocessed/fmri_summaries/fmri_summary_abideI_II__{}.hdf5".format(raw_data_preprocess_method)
    data = {}
    with h5.File(fname, "r") as hfile:
        # Use alff key for the first experiments. The dim is {2122,45,54,45,1}, hence squeeze(axis=4)
        all_data = hfile['summaries/{}'.format(raw_data_preprocess_method)][()].squeeze(axis=4) # .value has been deprecated, hence [()]..
        all_data = np.expand_dims(all_data, 1) # Add the singleton dimension.
        
        # Store data in a data dict.
        data['fmri'] = all_data
        
        data['abide'] = hfile['summaries'].attrs['ABIDE_I_or_II']
        data['age'] = hfile['summaries'].attrs['AGE_AT_SCAN']
        data['label'] = hfile['summaries'].attrs['DX_GROUP']
        data['gender'] = hfile['summaries'].attrs['SEX']
        data['id'] = hfile['summaries'].attrs['SUB_ID']
        # format it to id's instead of string names.
        data['site'], data['id2site'] = list_word2id(hfile['summaries'].attrs['SITE_ID'])
        
    return data

def list_word2id(words:list):
    """
    Given a list of words, it will return a list of ids and sorted list of tokens
    to map the indices back to words (the index in the tokens corresponds to the correct word).
    """
    
    tokens = list(set(words))
    tokens.sort()
    
    words_indices = [tokens.index(word) for word in words]
    
    return np.array(words_indices), np.array(tokens)

#%% Creating/writing/reading the dataset indices/ids.
def create_dataset_ids_stratified(n_fold, data_labels, data_ids, fname):
    # Obtain the train and val set based on the labels, based on equal amount of labels.
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    
    all_data = []
    # Note the np.zeros which is basically a placeholder that the skf.split needs, it is basically ignored anyway.
    for train_indices, val_indices in skf.split(np.zeros(data_labels.shape[0]), np.squeeze(data_labels)):
        
        data = {"train_ids":data_ids[train_indices], "val_ids":data_ids[val_indices]}
        all_data.append(data)
        
    np.save("{}/{}.npy".format(dirname, fname), all_data)
                
                
def get_dataset_ids_leave_one_site_out(data_sites, data_ids, samples_threshold, fname):
    # Obtain the train and val set based on leave one site out.
    
    all_data = []
    for site_i in np.unique(data_sites):
        # Get the indices corresponding to the site.
        train_indices = np.where(data_sites != site_i)[0]
        val_indices = np.where(data_sites == site_i)[0]
        
        # If not enough samples, ignore this site.
        if len(val_indices) < samples_threshold:
            continue
        
        # Randomizing the order of the indices.
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        
        data = {"train_ids":data_ids[train_indices], "val_ids":data_ids[val_indices]}
        all_data.append(data)
        
    np.save("{}/{}.npy".format(dirname, fname), all_data)

def get_data_subset_by_indices(data, indices):
    data_new = {}
    for key in data.keys():
        if key == 'id2site':
            # No indices should be used here.
            data_new[key] = data[key]
            continue
            
        data_new[key] = data[key][indices]
    return data_new

def get_ids_file():
    ids = []
    with open("../data_storage/Parisot_subject_ids.txt", "r") as f:
        for x in f:
            ids.append(int(x))
    return ids
#%% Example of running the code
import os
os.makedirs(dirname, exist_ok=True)
data = get_data_raw('alff')

# All data (Both abides)
create_dataset_ids_stratified(10, data['label'], data['id'], "dataset_random_abideboth")
get_dataset_ids_leave_one_site_out(data['site'], data['id'], 30, "dataset_leave_one_site_out_abideboth")

# Abide 1
abide1_indices = np.argwhere(data['abide'] == 1)
data_abide1 = get_data_subset_by_indices(data, abide1_indices)
create_dataset_ids_stratified(10, data_abide1['label'], data_abide1['id'], "dataset_random_abide1")
get_dataset_ids_leave_one_site_out(data_abide1['site'], data_abide1['id'], 30, "dataset_leave_one_site_out_abide1")

# Abide 1 specific ids
ids = get_ids_file()
abideids_indices = [index for index, id in enumerate(data['id']) if id in ids]
data_abideids = get_data_subset_by_indices(data, abideids_indices)
create_dataset_ids_stratified(10, data_abideids['label'], data_abideids['id'], "dataset_random_abideids")
get_dataset_ids_leave_one_site_out(data_abideids['site'], data_abideids['id'], 30, "dataset_leave_one_site_out_abideids")

# NYU
abideNYU_indices = np.argwhere(data['site'] == 23) # 23 is the NYU id. (Check machine_learning/get_sites.py for all id's.)
data_abideNYU = get_data_subset_by_indices(data, abideNYU_indices)
create_dataset_ids_stratified(10, data_abideNYU['label'], data_abideNYU['id'], "dataset_random_NYU")
get_dataset_ids_leave_one_site_out(data_abideNYU['site'], data_abideNYU['id'], 30, "dataset_leave_one_site_out_NYU")

print("Done")
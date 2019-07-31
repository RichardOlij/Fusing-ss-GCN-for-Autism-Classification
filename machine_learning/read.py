import h5py as h5
import numpy as np
import os

#%% Reading data

def get_data_raw_hdf5_main(raw_data_preprocess_method, load_fmri:bool):
    """
    Reading the hdf5 data and store it in a dict with new labels and correctly 
    formatted data. load_fmri can be set to false to save memory.
    """
    
    fname = "../data_storage/preprocessed/fmri_summaries/fmri_summary_abideI_II__{}.hdf5".format(raw_data_preprocess_method)
    data = {}
    with h5.File(fname, "r") as hfile:
        
        if load_fmri:
            # The dim is {2122,45,54,45,1}, hence squeeze(axis=4).
            all_data = hfile['summaries/{}'.format(raw_data_preprocess_method)][()].squeeze(axis=4) # .value has been deprecated, hence [()].
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

def get_data_raw_npy_atlas(args, original_ids_order):
    """
    The original_ids_order is data['id'] to ensure that the output of this function is in the same order as the hdf5 data.
    """
    
    dirname = "../data_storage/preprocessed/features/{}__{}__{}".format(args.atlas, args.bptf, args.confounds)
    data_ids = np.load("{}/ATLAS_ids.npy".format(dirname))
    data_features = np.load("{}/ATLAS_features.npy".format(dirname))

    # For each id in the original_ids_order, get the corresponding index of the new data_ids.
    aligned_indices = [np.where(id == data_ids)[0].item() for id in original_ids_order]
    # Use those aligned_indices to get the correct data features in the correct order.
    return data_features[aligned_indices]
    
def get_data_raw_amc_vae(ids):
    """
    Load the vae data based on the current ids so that the indices correspond.
    """
    new_vae_data = []
    vae_data = np.load("../data_storage/preprocessed/VAE.npy", allow_pickle=True)
    vae_data = dict(vae_data.item())
    for sample_id in ids:
        if sample_id in vae_data.keys():
            user_vae = vae_data[sample_id]
            new_vae_data.append(user_vae)
        else:
            raise ValueError("ID \"{}\" has not been found.".format(sample_id))
    return np.array(new_vae_data)
        
def get_data_raw_cnn(args, fold):
    """
    Load the cnn data based on the current ids and fold so that the indices correspond.
    """
    fname = "{}/cnn_storage_{}.npy".format(args.dir_name_embeddings, fold)
    if os.path.exists(fname):
        return np.load(fname, allow_pickle=True)
    else:
        print("Could not load the cnn-embedding data, file \"{}\" not found.".format(fname))
        return None

def list_word2id(words:list):
    """
    Given a list of words, it will return a list of ids and sorted list of tokens
    to map the indices back to words (the index in the tokens corresponds to the 
    correct word).
    """
    
    tokens = list(set(words))
    tokens.sort()
    
    words_indices = [tokens.index(word) for word in words]
    
    return np.array(words_indices), np.array(tokens)

#%% Getting the indices of the predetermined datasets.
    
def gen_dataset_indices(data_ids, dataset_type, ds):
    """
    The indices of the current dataset to create the train and validation datasets
    are loaded from the dataset_{}.npy file and based on the ids of the current
    data and the stored ids.
    """

    all_data = np.load("../data_storage/preprocessed/dataset_ids/dataset_{}_{}.npy".format(dataset_type, ds), allow_pickle=True)

    # For each fold get the indices by reading the id's and return the corresponding indices.
    for data in all_data:
        train_indices = [np.where(id == data_ids)[0].item() for id in data["train_ids"]]
        val_indices = [np.where(id == data_ids)[0].item() for id in data["val_ids"]]
        
        yield train_indices, val_indices
import numpy as np
from read import get_data_raw_hdf5_main

dataset_type = "leave_one_site_out"
ds = "abideboth"

#%% Load data
all_data = np.load("../data_storage/preprocessed/dataset_ids/dataset_{}_{}.npy".format(dataset_type, ds), allow_pickle=True)
data = get_data_raw_hdf5_main("alff", load_fmri=False)
id2site = data['id2site']

#%% Get the sites
for i in range(len(all_data)):
#    train_ids = all_data[i]['train_ids']
    val_ids = all_data[i]['val_ids']
    
    for id in val_ids[:1]: # Check for every n of that group.
        # Get site_id
        data_index = np.argwhere(data['id'] == id)
        site_id = data['site'][data_index]
        # Convert the site_id which is the index to the name.
        site_name = id2site[site_id]
        print(site_id, site_name)
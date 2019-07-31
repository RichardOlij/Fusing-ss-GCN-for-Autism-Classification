import numpy as np
import os

dir_name = "../data_storage/original/T1_encoded"
all_data = {}
for file in os.listdir(dir_name):
    # Convert the fname to id: e.g. 0050039_T1.npy, remove the _T1.npy part and leading zeros.
    sample_id = int(file.replace("_T1.npy","")) # Note that the rstrip gives problems e.g. id=0051461 somehow.
    data = np.load("{}/{}".format(dir_name, file), allow_pickle=True)[0] # [0] to remove the singleton dimension.

    all_data[sample_id] = data

# Stored as an dict where the key is the id.
dirname = "../data_storage/preprocessed"
os.makedirs(dirname, exist_ok=True)    
np.save("{}/VAE.npy".format(dirname), all_data)

print("Done")
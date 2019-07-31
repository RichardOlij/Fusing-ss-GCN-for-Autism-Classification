import numpy as np
import pandas as pd
import os

def rewrite_path(atlas, bptf, confounds, path:str):
    """
    Returning the relative path given atlas, bptf and confounds.
    """
    path = path.replace("/data_local/deeplearning/ABIDE_ML_inputs/", '')
    path = path.replace("ATLAS", atlas)
    path = path.replace("BPTF", bptf)
    path = path.replace("CONFOUNDS", confounds)
    path = "{}/{}".format("../data_storage/original/ABIDE_ML_CorrVec", path)
    return path

def get_data_raw(atlas, bptf, confounds):
#    usecols = ["AGE_AT_SCAN", "SEX", "DX_GROUP", "SUB_ID", "corrvec_file"]
    usecols = ["DX_GROUP", "SUB_ID", "corrvec_file"]
    data_info = pd.read_csv("../data_storage/original/ABIDE_ML_CorrVec/data_info.csv", usecols=usecols)
    
    n_samples = data_info.shape[0]
    
    dx_vec = np.array(data_info['DX_GROUP'])
    sub_id_vec = np.array(data_info['SUB_ID'])
    corrvecfile_paths_vec = np.array(data_info['corrvec_file'])
  
    # Obtain the original vectors
    vec_matrix = None # The dimensionality is unknown at this point.
    for i, corrvecfile_path in enumerate(corrvecfile_paths_vec):
        corrvec_file = rewrite_path(atlas, bptf, confounds, corrvecfile_path)
        vec_data = np.load(corrvec_file, allow_pickle=True)

        if vec_matrix is None:
            # Now the dimensionality is known, create the empty matrix
            vec_dim = vec_data.shape[0]
            vec_matrix = np.zeros([n_samples, vec_dim])

        vec_matrix[i,:] = vec_data

    return dx_vec, sub_id_vec, vec_matrix

# For each necessary combination this code should be executed.
for atlas in ["AAL", "cc_200", "craddock_200", "HO_cort_maxprob_thr25-2mm", "JAMA_IC7", "JAMA_IC19", "JAMA_IC52", "schaefer_100", "schaefer_400"]:
    for bptf in ["no_bptf","bptf"]:
        for confound in ["no_nilearn_regress","nilearn_regress"]:
            if atlas == "HO" and bptf == "no_bptf":
                # This data does not exist.
                print("Ignored HO, no_bptf")
                continue
            print("Preprocessing \"{}\", \"{}\", \"{}\"".format(atlas, bptf, confound))
            dirname = "../data_storage/preprocessed/features/{}__{}__{}".format(atlas, bptf, confound)
            os.makedirs(dirname, exist_ok=True)
            
            dx_vec, sub_id_vec, vec_matrix = get_data_raw(atlas, bptf, confound)
            
            np.save("{}/ATLAS_ids.npy".format(dirname), sub_id_vec)
            np.save("{}/ATLAS_features.npy".format(dirname), vec_matrix) # Note that the id's should be the same for all settings.

print("Done")
import h5py as h5
import os

# %% Read the FMRI data

with h5.File("../data_storage/original/fmri_summary_abideI_II.hdf5", "r") as hfile_in:
    # For each summary, create the smaller file.
    for summary in ['T1', 'alff', 'autocorr', 'degree_centrality_binarize', 'degree_centrality_weighted', 'eigenvector_centrality_binarize', 'eigenvector_centrality_weighted', 'entropy', 'falff', 'lfcd_binarize', 'lfcd_weighted', 'reho', 'vmhc']:
        
        print("Summary: {}".format(summary))
        
        dirname = "../data_storage/preprocessed/fmri_summaries"
        os.makedirs(dirname, exist_ok=True) 
        with h5.File("{}/fmri_summary_abideI_II__{}.hdf5".format(dirname, summary), 'w') as hfile_out:
            
            hfile_out.create_dataset('summaries/{}'.format(summary), data=hfile_in['summaries/{}'.format(summary)])
            
            for key in list(hfile_in['summaries'].attrs.keys()):
                hfile_out['summaries'].attrs[key] = hfile_in['summaries'].attrs[key]
      
print("Done")
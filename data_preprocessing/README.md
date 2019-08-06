# Data preprocessing
The original data is stored in ../data_storage/original/ and the preprocessed versions will bes tored in the ../data_storage/preprocessed/ directory.

## make_vae_data_compact.py
This stores the VAE data in T1_encoded/ to a single VAE.npy file containing all VAE embeddings.

## make_features_data_compact.py
For each combination of atlas, bptf and confounds this code should be executed to store the feature data in ABIDE_ML_CorrVec to features/atlas__bptf__confound/features.npy and features/atlas__bptf__confound/ids.npy.

## convert_fmri_data.py
Instead of one large fmri_summary_abide_II.hdf5 it creates all individual files in fmri_summaries/ with only the required information.

## generate_datasets.py
Generates for each 10-fold and each leave_one_site_out-fold the corresponding id's so that every experiment uses the exact same datasets.
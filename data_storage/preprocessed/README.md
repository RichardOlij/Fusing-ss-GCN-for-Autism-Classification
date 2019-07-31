# Data Storage: Preprocessed
The preprocessed data is stored here. It is ready to be used.

## VAE.npy
A dict where each key and value are the id and the corresponding 200-dimensional embeddings, respectively.

## features/
Containing the features for each combination of atlas, bptf and confounds.

## fmri_summaries/
For each individual summary a file is store with the fmri-scans data and all the metadata.

## dataset_ids/
For each 10-fold and each leave_one_site_out-fold the corresponding id's are stored so that every experiment will be performed on the same datasets.
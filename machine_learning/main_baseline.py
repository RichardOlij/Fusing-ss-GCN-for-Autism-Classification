from arguments_baseline import obtain_arguments
from read import get_data_raw_hdf5_main,get_data_raw_npy_atlas, get_data_raw_amc_vae, gen_dataset_indices
from utils import get_adjacency_matrix_vae, get_adjacency_matrix_fixed, chebyshev_polynomials, save_results, reduce_dim_ridge_smart, correlation_matrix
from train import train_single_fold
from store_datadict import load_datadicts, save_datadicts
import ipdb

args = obtain_arguments()
data = get_data_raw_hdf5_main("alff", load_fmri=False) # Note that the "alff" part in this case is just to load data from h5py file, but not for actual alff related information.
data['atlas_features_vec'] = get_data_raw_npy_atlas(args, data['id'])


store_data_dicts = load_datadicts(args)
# Perform the k-folds.
for fold_i, (train_ind, val_ind) in enumerate(gen_dataset_indices(data['id'], args.dataset_type, args.ds)):

    if args.max_fold == -1 or fold_i < args.max_fold:
        # If there are already more things stored than i, ignore the current i; we already have got the results.
        if len(store_data_dicts) > fold_i:
            print("Ignored fold {}, since store_data_dicts already contains {} folds".format(fold_i, len(store_data_dicts)))
            continue

        data["train_ind"] = train_ind
        data["val_ind"] = val_ind
        data["fold_i"] = fold_i
        
        # Apply feature dimensionality reduction per fold. And store it as input_features.
        data['input_features'] = reduce_dim_ridge_smart(args, data, data['atlas_features_vec'])
        
        if args.adj_type == "fixed":
            # Create adjacency matrix. Due to the normalisation step, it is depedend on input_features.
            data['adj_raw'] = get_adjacency_matrix_fixed(args, data)
        elif args.adj_type == "vae":
            # Create adjacency matrix based on vae.
            data['vae'] = get_data_raw_amc_vae(data['id'])
            data['adj_raw'] = get_adjacency_matrix_vae(args, data['vae'], data)
        elif args.adj_type == "correlation_only":
            data['adj_raw'] = correlation_matrix(data['input_features'], "correlation")
        #ipdb.set_trace()
        data['adj_support'] = chebyshev_polynomials(data['adj_raw'], args.polynomial_degree)

        store_data_dict = train_single_fold(args, data)
        #ipdb.set_trace()
        store_data_dicts.append(store_data_dict)
            
        save_datadicts(args, store_data_dicts)

save_results(args, store_data_dicts)
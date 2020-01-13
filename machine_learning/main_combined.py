from arguments_combined import obtain_arguments
from read import get_data_raw_hdf5_main, get_data_raw_amc_vae, get_data_raw_cnn, gen_dataset_indices
from utils import get_adjacency_matrix_vae, preprocess_features, chebyshev_polynomials, save_results, preprocess_features_min_max, preprocess_features_z_score
from train import train_single_fold
from store_datadict import load_datadicts, save_datadicts
import ipdb

args = obtain_arguments()
data = get_data_raw_hdf5_main(args.raw_data_preprocess_method, load_fmri=True)

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
        
        # Get the CNN embedding in case of a parisot model, it is dependent on fold_i.
        if args.model == "parisot_tf" or args.model == "parisot_py":
            data['cnn_embedding'] = get_data_raw_cnn(args, fold_i)
            
            if type(data['cnn_embedding']) == type(None):
                print("No cnn embedding loaded, this fold is therefore ignored.")
                continue
            #ipdb.set_trace()
            # Preprocess the features for the parisot models.
            #data['cnn_embedding'] = preprocess_features(data['cnn_embedding']) 
            #data['cnn_embedding'] = preprocess_features_min_max(data['cnn_embedding'])
            data['cnn_embedding'] = preprocess_features_z_score(data['cnn_embedding'])

            # Rewriting the name since the general name will be input_feauteres, independend of the actual data source.
            data['input_features'] = data.pop('cnn_embedding')
            
            # Create adjacency matrix based on vae in the case of a Parisot model. 
            # This should happen in this point since it can be based on input_features for the sparisity part.
            data['vae'] = get_data_raw_amc_vae(data['id'])
            data['adj_raw'] = get_adjacency_matrix_vae(args, data['vae'], data)
            data['adj_support'] = chebyshev_polynomials(data['adj_raw'], args.polynomial_degree)
        
        store_data_dict = train_single_fold(args, data)
        store_data_dicts.append(store_data_dict)
        print ("ACCURACY", store_data_dict["list_val_accuracy"][-1]*100, "AUC",store_data_dict["list_val_auc"][-1]*100)
        save_datadicts(args, store_data_dicts)
save_results(args, store_data_dicts)

import argparse
import torch
import os
from utils import process_str_list, process_str_bool

#%% Obtaining settings
def obtain_arguments():
    
    # Training settings
    parser = argparse.ArgumentParser()
    
    # Data parameters    
    parser.add_argument('--max_fold', type=int, default=-1,
                        help='The amount of folds out of the k-fold will be trained, -1 for all folds.')
    
    parser.add_argument('--dataset_type', type=str, default='random',
                        help='On what set are the test and validation set based, can either be \"random\" or \"leave_one_site_out\"')
    
    parser.add_argument('--ds', type=str, default="abideids",
                        help='What dataset to use, only \"abideboth\", \"abide1\", \"abideids\" or \"NYU\" can be chosen.')
    
    parser.add_argument('--print_epoch', type=int, default=50,
                        help='Print every n epochs.')
    
    parser.add_argument('--model', type=str, default="parisot_py",
                        help='What model to use, \"parisot_tf\" or \"parisot_py\" can be chosen.')
    
    parser.add_argument('--epochs', type=int, default=150,
                        help='Train for n epochs.')
    
    parser.add_argument('--save_model_epoch', type=int, default=50,
                        help='Store the models every n epochs.')

    
    #%% Adjacency
    
    parser.add_argument('--adj_type', type=str, default="fixed",
                        help='How is the adjacency determined, \"fixed\" or \"vae\" can be chosen.')

    parser.add_argument('--adj_fixed_specs', type=str, default="gender_0, site_0",
                        help='Only when \"adj_type\" is \"fixed\". A list of specification on which the fixed adjacency matrix is constructed. Available options are \"age_n\", \"gender_n\" or \"site_n\", where \"n\" is the maximum difference between samples.')

    parser.add_argument('--adj_threshold', type=float, default=-1.,
                        help='Only when \"adj_type\" is \"vae\". Thresholding the adjacency matrix, ignore if < 0 and auto if -2.')

    parser.add_argument('--polynomial_degree', type=int, default=3,
                        help='To what order are the polynomails used for the adjacency matrix, n (+ 1 for the identity).')  

    parser.add_argument('--features_sparsity', type=str, default='True',
                        help='If true, the raw adjacency matrix will be multiplied with the correlation matrix of the features.')    
    
    
    #%% Parisot
    
    parser.add_argument('--learning_rate_parisot', type=float, default=0.005,
                        help='Initial learning rate.')
    
    parser.add_argument('--weight_decay_parisot', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    
    parser.add_argument('--hidden_parisot', type=int, default=16,
                        help='Number of units in the hidden layer.')
    
    parser.add_argument('--dropout_parisot', type=float, default=0.3,
                        help='Propability of an unit being zeroed.')    
    
    
    #%% DATA parameters    
    parser.add_argument('--atlas', type=str, default="HO",
                        help='Name of the atlas.')
    
    parser.add_argument('--bptf', type=str, default="bptf",
                        help='BPTF type.')
    
    parser.add_argument('--confounds', type=str, default="no_nilearn_regress",
                        help='Confound type.')
    
    parser.add_argument('--n_features', type=int, default=2000,
                        help='Dimensionality reduction to n features using Ridge, ignore if < 0.')


    #%% Process the arguments
    args = parser.parse_args()
    
    args.seed = 0
    args.adj_fixed_specs = process_str_list(args.adj_fixed_specs)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.features_sparsity = process_str_bool(args.features_sparsity)
        
    print("Arguments: {}\n".format(args))
    str_main_folder = "generated_data_baseline"

    # Just either use vae, or the fixed settings.
    str_adj_settings = ""
    if args.adj_type == "vae":
        str_adj_settings = "vae_{}".format(args.adj_threshold)
    elif args.adj_type == "fixed":
        str_adj_settings ="fixed_{}".format(args.adj_fixed_specs)
    else:
        raise ValueError("arg.adj_type cannot be \"{}\".".format(args.adj_type))
        
    str_settings = "{}__{}__{}__{}__{}/{}__{}__{}__{}".format(str_adj_settings, args.polynomial_degree, args.features_sparsity, args.dataset_type, args.ds, args.atlas, args.bptf, args.confounds, args.n_features)
    
    args.dir_name_embeddings = "{}/store_embeddings/{}".format(str_main_folder, str_settings)


    str_settings_parisot = "{}__{}__{}__{}__{}".format(args.model, args.learning_rate_parisot, args.weight_decay_parisot, args.hidden_parisot, args.dropout_parisot)
    args.dir_name_datadict = "{}/datadict/{}/{}".format(str_main_folder, str_settings, str_settings_parisot)
    args.dir_name_model = "{}/model/{}/{}".format(str_main_folder, str_settings, str_settings_parisot)
    args.dir_name_predictions = "{}/predictions/parisot/{}/{}".format(str_main_folder, str_settings, str_settings_parisot)
    args.dir_name_results = "{}/results/{}/{}".format(str_main_folder, str_settings, str_settings_parisot)
    args.dir_name_ridge_dims = "{}/ridge_reduces_dimensions/{}__{}__{}__{}__{}".format(str_main_folder, args.dataset_type, args.ds, args.atlas, args.bptf, args.confounds)

    os.makedirs(args.dir_name_predictions, exist_ok=True)
    os.makedirs(args.dir_name_datadict, exist_ok=True)
    os.makedirs(args.dir_name_model, exist_ok=True)
    os.makedirs(args.dir_name_ridge_dims, exist_ok=True)
    
    return args
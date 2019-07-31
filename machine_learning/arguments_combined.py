import argparse
import torch
import os
from utils import process_str_bool

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
    
    parser.add_argument('--print_epoch', type=int, default=10,
                        help='Print every n epochs.')
    
    parser.add_argument('--model', type=str, default="parisot_tf",
                        help='What model to use, only \"parisot_tf\", \"parisot_py\" or \"cnn\" can be chosen.')
    
    parser.add_argument('--epochs', type=int, default=250,
                        help='Train for n epochs.')
    
    parser.add_argument('--save_model_epoch', type=int, default=50,
                        help='Store the models every n epochs.')

    
    #%% Adjacency
    
    parser.add_argument('--adj_threshold', type=float, default=-1.,
                        help='Only when \"adj_type\" is \"vae\". Thresholding the adjacency matrix, ignore if < 0 and auto if -2.')

    parser.add_argument('--polynomial_degree', type=int, default=3,
                        help='To what order are the polynomails used for the adjacency matrix, n (+ 1 for the identity).')  
    
    parser.add_argument('--features_sparsity', type=str, default='True',
                        help='If true, the raw adjacency matrix will be multiplied with the correlation matrix of the features.')    
    
    
    #%% Parisot
    
    parser.add_argument('--learning_rate_parisot', type=float, default=0.005,
                        help='Initial learning rate, for Parisot models only.')
    
    parser.add_argument('--weight_decay_parisot', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters), for Parisot models only.')
    
    parser.add_argument('--hidden_parisot', type=int, default=16,
                        help='Number of units in the hidden layer, for Parisot models only.')
    
    parser.add_argument('--dropout_parisot', type=float, default=0.3,
                        help='Propability of an unit being zeroed, for Parisot models only.')
    
    
    #%% CNN
    
    parser.add_argument('--learning_rate_3dcnn', type=float, default=0.001,
                        help='Initial learning rate, for 3DCNN model only.')
        
    parser.add_argument('--momentum_3dcnn', type=float, default=0.9,
                        help='Momentum, for 3DCNN model only.')
    
    parser.add_argument('--batch_size_3dcnn', type=float, default=32,
                        help='Batch size, for 3DCNN model only.')
    
    parser.add_argument('--cnn_model', type=int, default=1,
                        help='Which 3DCNN model to use, for 3DCNN model only.')  
    
    
    #%% DATA parameters    
    parser.add_argument('--raw_data_preprocess_method', type=str, default="alff",
                        help='Which preprocessing method is used to summarize over time in the fmri scans.')
    
    
    #%% Process the arguments
    args = parser.parse_args()
    
    args.seed = 0
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.features_sparsity = process_str_bool(args.features_sparsity)
        
    print("Arguments: {}\n".format(args))
    str_main_folder = "generated_data_combined"

    str_settings = "{}__{}__{}__{}__{}".format(args.adj_threshold, args.polynomial_degree, args.features_sparsity, args.dataset_type, args.ds)
    
    str_settings_cnn = "cnn{}__{}__{}__{}__{}".format(args.cnn_model, args.learning_rate_3dcnn, args.momentum_3dcnn, args.batch_size_3dcnn, args.raw_data_preprocess_method)
    args.dir_name_embeddings = "{}/store_embeddings/{}/{}".format(str_main_folder, str_settings, str_settings_cnn)
    if args.model == "cnn":
        args.dir_name_datadict = "{}/datadict/{}/{}".format(str_main_folder, str_settings, str_settings_cnn)
        args.dir_name_model= "{}/model/{}/{}".format(str_main_folder, str_settings, str_settings_cnn)
        args.dir_name_predictions = "{}/predictions/cnn/{}/{}".format(str_main_folder, str_settings, str_settings_cnn)
        args.dir_name_results = "{}/results/{}/{}".format(str_main_folder, str_settings, str_settings_cnn)
        
    elif args.model == "parisot_tf" or args.model == "parisot_py":
        str_settings_parisot = "{}/{}__{}__{}__{}__{}__{}".format(str_settings_cnn, args.model, args.learning_rate_parisot, args.weight_decay_parisot, args.hidden_parisot, args.dropout_parisot, args.raw_data_preprocess_method)
        args.dir_name_datadict = "{}/datadict/{}/{}".format(str_main_folder, str_settings, str_settings_parisot)
        args.dir_name_model = "{}/model/{}/{}".format(str_main_folder, str_settings, str_settings_parisot)
        args.dir_name_predictions = "{}/predictions/parisot/{}/{}".format(str_main_folder, str_settings, str_settings_parisot)
        args.dir_name_results = "{}/results/{}/{}".format(str_main_folder, str_settings, str_settings_parisot)
    else:
        raise ValueError("Modelname could not be \"{}\"".format(args.model))

    if args.model == "cnn":
        os.makedirs(args.dir_name_embeddings, exist_ok=True)
    os.makedirs(args.dir_name_predictions, exist_ok=True)
    os.makedirs(args.dir_name_datadict, exist_ok=True)
    os.makedirs(args.dir_name_model, exist_ok=True)
    
    return args
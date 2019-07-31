import cnn_wrapper
import parisot_py_wrapper
import parisot_tf_wrapper
from store_datadict import StoreDataDict
import numpy as np

def train_single_fold(args, data):
    """
    Given the model the data will be preprocessed and formatted in the corresponding
    way.
    """
    # Get the correct wrapper that will do some preprocessing and run everything in a specific way for that model.
    if args.model == "cnn":
        wrapper = cnn_wrapper
    elif args.model == "parisot_py":
        wrapper = parisot_py_wrapper
    elif args.model == "parisot_tf":
        wrapper = parisot_tf_wrapper
    else:
        raise ValueError("Model cannot be \"{}\".".format(args.model))

    # The model and preprocessed data will be loaded.
    model, train_pre_args = wrapper.train_pre(args, data)
    
    # Create an object to store the results in.
    store_data_dict = StoreDataDict(data)
    # Training multiple epochs.
    for epoch in range(args.epochs):
        
        # Training a single epoch, the results are stored in store_data_dict.
        store_data_dict = wrapper.train_epoch(args, store_data_dict, epoch, model, train_pre_args)
        
        # Print the results every few epochs.
        if (epoch % args.print_epoch == 0) or (epoch == args.epochs - 1):
            store_data_dict.print_data_epoch(data["fold_i"], epoch)
            
        # Saving stuff every few epochs.
        if (epoch % args.save_model_epoch == 0) or (epoch == args.epochs - 1):
            wrapper.save_model(args, epoch, data['fold_i'], model, train_pre_args)
            wrapper.save_final_predictions(args, epoch, data['fold_i'], model, train_pre_args)
        
        
    # In case of the CNN model, save the embeddings that are then used for the Pariso models.
    if args.model == "cnn":
        embedding = wrapper.get_embedding_3dcnn(args, model, data)
        np.save("{}/cnn_storage_{}.npy".format(args.dir_name_embeddings, data["fold_i"]), embedding)        
    
    return store_data_dict.store_data_dict
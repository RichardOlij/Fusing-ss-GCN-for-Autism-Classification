import time
import tensorflow as tf
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from evaluate import evaluate
from parisot_tf_model import Deep_GCN
import sklearn.metrics as sklearnmetrics

#%% Create the masks
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def indices_to_onehot(indices):
    return np.eye(2)[indices.astype(int)]
    
def get_masks(labels, idx_train, idx_val):
    labels = indices_to_onehot(labels)
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]

    return y_train, y_val, train_mask, val_mask

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def evaluate_wrapper(sess, model, args, feats, graph, labels, mask, placeholders, phase_train):    
    # Only if there is an filled set. (In some cases the validation set is ingored.)
    if labels.max() == 0:
        return -1,-1,-1,-1,-1,-1,-1,-1,-1

    
    t_begin = time.time()
    # Construct feed dictionary
    feed_dict_val = construct_feed_dict(feats, graph, labels, mask, placeholders)
    if phase_train:
        feed_dict_val.update({placeholders['dropout']: args.dropout})
    feed_dict_val.update({placeholders['phase_train'].name: phase_train})  
    
    run_funcs = [model.loss, model.accuracy, model.predict()]
    if phase_train:
        run_funcs.append(model.opt_op)

    outs = sess.run(run_funcs, feed_dict=feed_dict_val)
    cost = outs[0]
    acc = outs[1]
    pred = outs[2]

    # Compute the area under curve    
    pred = pred[np.squeeze(np.argwhere(mask == 1)), :]
    lab = labels[np.squeeze(np.argwhere(mask == 1)), :]
    
    auc, acc, mean_pred, mean_lab, sens, spec = evaluate(pred, lab.argmax(1))
    
    duration = (time.time() - t_begin)

    # Stuff for plotting the curve later on.
    fpr, tpr, thresholds = sklearnmetrics.roc_curve(lab.argmax(1), pred[:,1])
    
    return auc, acc, mean_pred, mean_lab, sens, spec, cost, duration, fpr, tpr, thresholds


def train_pre(args, data):
    
    # Converting data to sparse and tuples.
    input_features = sparse.coo_matrix(data["input_features"]).tolil()
    input_features = sparse_to_tuple(input_features)
        
    new_adj_support = []
    for support in data["adj_support"]:
        support = sparse.coo_matrix(support).tolil()
        support = sparse_to_tuple(support)
        new_adj_support.append(support)
    data["adj_support"] = new_adj_support
    
    # Create test, val and train masked variables
    y_train, y_val, mask_train, mask_val = get_masks(data['label'], data["train_ind"], data["val_ind"])

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(len(data['adj_support']))],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(input_features[2], dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = Deep_GCN(placeholders, 
                     input_dim=input_features[2][1], 
                     depth=0, 
                     learning_rate=args.learning_rate_parisot, 
                     weight_decay=args.weight_decay_parisot, 
                     hidden1=args.hidden_parisot,
                     logging=True)
    
    # Initialize session
    sess = tf.Session()
    
    # Init variables
    sess.run(tf.global_variables_initializer())

    return model, (sess, placeholders, y_train, y_val, mask_train, mask_val, input_features, data)

    
def train_epoch(args, store_data_dict, epoch, model, *train_pre_args):
    sess, placeholders, y_train, y_val, mask_train, mask_val, input_features, data = train_pre_args[0]
    
    # Train
    train_single(sess, model, args, input_features, data["adj_support"], y_train, mask_train, placeholders)
    
    # Perform a training and validation -set evaluation.
    train_results = evaluate_wrapper(sess, model, args, input_features, data["adj_support"], y_train, mask_train, placeholders, False)
    val_results = evaluate_wrapper(sess, model, args, input_features, data["adj_support"], y_val, mask_val, placeholders, False)
    
    
    store_data_dict.store_data_train(train_results)
    store_data_dict.store_data_val(val_results)
    
    return store_data_dict


def train_single(sess, model, args, feats, graph, labels, mask, placeholders):
    # Construct feed dictionary
    feed_dict_val = construct_feed_dict(feats, graph, labels, mask, placeholders)
    feed_dict_val.update({placeholders['dropout']: args.dropout_parisot})
    feed_dict_val.update({placeholders['phase_train'].name: True})  
    
    run_funcs = [model.loss, model.predict(), model.opt_op]

    _ = sess.run(run_funcs, feed_dict=feed_dict_val)

def save_final_predictions(args, epoch, fold, model, *train_pre_args):
    sess, placeholders, y_train, y_val, mask_train, mask_val, input_features, data = train_pre_args[0]
    
    # Construct feed dictionary
    feed_dict_val = construct_feed_dict(input_features, data["adj_support"], y_train, mask_train, placeholders)
    feed_dict_val.update({placeholders['dropout']: args.dropout_parisot})
    feed_dict_val.update({placeholders['phase_train'].name: True})  
    
    run_funcs = [model.loss, model.predict(), model.opt_op]

    _, predictions, _ = sess.run(run_funcs, feed_dict=feed_dict_val)
    
    print("Saving the current predictions.")
    np.save("{}/fold_{}__epoch_{}.npy".format(args.dir_name_predictions, fold, epoch), np.matrix(predictions))
        
def save_model(args, epoch, fold, model, *train_pre_args):
    sess, placeholders, y_train, y_val, mask_train, mask_val, input_features, data = train_pre_args[0]
    
    print("Saving the current model.")
    print("TF cannot yet save the model.")

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from scipy.spatial import distance
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import os
import matplotlib.pyplot as plt


#%% Process strings and inputs etc.
def process_str_bool(input:str):
    if input.lower() in ["t", "true", "1"]:
        return True
    elif input.lower() in ["f", "false", "0", "-1"]:
        return False
    raise ValueError("String \"{}\" can not be converted to a boolean.".format(input))

def process_str_list(input:str):
    # Remove the spaces and brackets.
    output = input.replace(" ","").lstrip("[").rstrip("]")
    # If the string is empty, the output.split will return a list with an empty string. 
    # So instead return empty list in that case.
    if output == "":
        return []
    # Split the remaining string to list and return.
    return output.split(',')

#def process_list_string(input:list):
#    str_output = ""
#    for el in input:
#        # For each element append it to the str_output combined with a '_'.
#        str_output = "{}{}_".format(str_output, el)
#    # Remove the last '_' if present.
#    return str_output.rstrip("_")

def process_adj_fixed_specs(input:list, data):
    options = [option.split("_") for option in input]
    vec_names, max_diffs = zip(*options)
    vecs = []
    for vec_name in vec_names:
        if vec_name == "site":
            vecs.append(data["site"])
        elif vec_name == "age":
            vecs.append(data["age"])
        elif vec_name == "gender":
            vecs.append(data["gender"])
        else:
            raise ValueError("vec_name {} does not exist.".format(vec_name))
            
    max_diffs = [int(x) for x in max_diffs]
    
    return vecs, max_diffs   

#%% Adjacency matrix
def create_adjacency_matrix_fixed_raw(vecs:list, max_diffs:list):
    """
    Creates the symmetric adjacency matrix, ignoring the identity diagonal.
    It is based on a list of vectors, e.g. [vec_gender, vec_age] and 
    max_diffs e.g. [0, 2]. In this case the maximum edge value can be 2, 
    if the gender is equal and age difference is up to 2.
    """
    n_vecs = len(vecs)
    n_samples = vecs[0].shape[0]
    matrix = np.zeros([n_samples, n_samples])
    
    for i in range(n_samples):
        values_i = [vec[i] for vec in vecs]
        for j in range(i+1, n_samples): # Ignoring identity on purpose
            values_j = [vec[j] for vec in vecs]
            
            # Sum the amount of exact similar values.
            value = sum([abs(values_i[k] - values_j[k])<=max_diffs[k] for k in range(n_vecs)])
            
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix

def correlation_matrix(features, metric):
    # Calculate all pairwise distances
    distv = distance.pdist(features, metric=metric)
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    return sparse_graph
  
def apply_features_sparsity(args, adjacency_matrix, features):
    if args.features_sparsity:
        adjacency_matrix_sparse = correlation_matrix(features, "correlation") # No need to remove eye since it will be multiplied with adjacency_matrix which has no eye.
        return adjacency_matrix * adjacency_matrix_sparse
    return adjacency_matrix
    
def get_adjacency_matrix_vae(args, features, data):

    # Do VAE cosine matrix.
    adjacency_matrix = correlation_matrix(features, "cosine") - np.eye(features.shape[0])

    # Determine the threshold.
    if args.adj_threshold == -2:
        # Threshold automatically if ==-2. Threshold so that every row has at least one non-zero value.
        # Take the max of each row (the highest threshold possible per row) 
        # and the lowest value of those (no row should be having a lower max).
        threshold = adjacency_matrix.max(1).min()
        print("Thresholded at {}".format(threshold))
    else:
        threshold = args.adj_threshold
    
    if threshold > 0:
        # Only threshold if value > 0, ignore otherwise
        # Apply a boolean matrix filter of adj>threshold to the adj matrix.
        adjacency_matrix = np.where(adjacency_matrix>=threshold, adjacency_matrix, 0)


    # Count number of zero rows.
    zeros = sum(adjacency_matrix.sum(1)==0)
    if zeros > 0:
        raise ValueError("The adjacency threshold is too high, there are {} zero rows.".format(zeros))

    return apply_features_sparsity(args, adjacency_matrix, data['input_features'])

def get_adjacency_matrix_fixed(args, data):

    print("Creating fixed adjacency matrix.")
    # Transform the args.adj_fixed_specs to the actual rules for create_adjacency_matrix_raw.
    vecs, max_diffs  = process_adj_fixed_specs(args.adj_fixed_specs, data)
    adjacency_matrix = create_adjacency_matrix_fixed_raw(vecs, max_diffs)
                
    return apply_features_sparsity(args, adjacency_matrix, data['input_features'])

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    new_t_k = []
    for mat in t_k:
        new_t_k.append(mat.toarray())
    return new_t_k


#%% Preprocessing features
def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features  

#%% Dimensionality reduction
def reduce_dim_ridge_(n_features, data, matrix):
    """
    Reduces the dimensionality using the Ridge approach.
    """
    labels = data['label']
    train_ind = data["train_ind"]

    if n_features < 0:
        print("Fold-{}: No features removed.".format(data["fold_i"]))
        return matrix
    if n_features >= matrix.shape[1]:
        raise ValueError("Fold-{}: Amount of features {} is lower than the amount features to reduce to {}.".format(
                data["fold_i"], matrix.shape[1], n_features))
        return matrix
    
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    reduced_features = selector.transform(matrix)
    return reduced_features
        
def reduce_dim_ridge(args, n_features, data, matrix):
    """
    Reduces the dimensionality using the Ridge approach. If the same dimensionality reduction has been saved and found,
    it loads that feature_vectors. If not, it takes the lowest dimensionality that has been stored and reduces from that.
    """
    print("Reducing the feature dimensionality to {}.".format(n_features))
    full_dirname = "{}/dim_{}".format(args.dir_name_ridge_dims, n_features)
    os.makedirs(full_dirname, exist_ok=True)
    full_fname = "{}/fold_{}.npy".format(full_dirname, data['fold_i'])
    
    # If for the same settings the reduced feature_vectors already exist, use it.
    if os.path.isfile(full_fname):
        print("Exact file is found and used.")
        reduced_features = np.load(full_fname, allow_pickle=True)
        return reduced_features
        

    # Obtain all dimensions that are used so far.
    dir_names = os.listdir(args.dir_name_ridge_dims)
    dims = []
    for dir_name in dir_names:
        if dir_name.startswith("dim_"):
            dim = int(dir_name.replace("dim_",""))
            dims.append(dim)
    dims.sort()
    
    reduced_features = matrix
    done = False
    # Get the lowest dim that is higher than the n_features, to continue to reduce from.
    for dim in dims:
        if dim >= n_features:
            dir_name = "{}/dim_{}".format(args.dir_name_ridge_dims, dim)
            for fname in os.listdir(dir_name):
                if fname.startswith("fold_"):
                    fold = int(fname.replace("fold_","").replace(".npy",""))
                    if fold == data['fold_i']:
                        proposed_fname = "{}/{}".format(dir_name, fname)
                        print("Reduced dimensionality found at {}".format(proposed_fname))
                        reduced_features = np.load(proposed_fname, allow_pickle=True)
                        done = True
                        break
        if done:
            print("No files found")
            break
    
    # Reduce (even more)
    reduced_features = reduce_dim_ridge_(n_features, data, reduced_features)
       
    # And store it
    print("Storing the reduced dimensions.")
    np.save(full_fname, reduced_features)
    
    return reduced_features

def reduce_dim_ridge_smart(args, data, matrix):
    """
    Reduces the dimensionality using the Ridge approach. In the meanwhile it saves specific dimensionalities to continue from in the future.
    """
    dims = [10000, 6000, 4000, 3000, 2000, 1000, 500, 200]
    for dim in dims:
        # If the current dim is larger than the n_features, calculate and store the itermidiate dimensionality feature_vectors.
        # If not, only calculate to the n_features dimensionality.
        if dim >= matrix.shape[1]:
            # If the dim is larger than the dim of the data, skip this dim.
            continue
        if args.n_features < dim:
            reduce_dim_ridge(args, dim, data, matrix)
    # Return the actual correct dim.
    return reduce_dim_ridge(args, args.n_features, data, matrix)


#%% Visualisation

#def check_and_save_adj_values(args, adj, fold_i):
#    """
#    Shows the heatmap of the adjacency matrix and also the corresponding histogram.
#    Both saved with the plot results
#    """
#    
#    os.makedirs("{}/adj_matrices".format(args.path_save_results), exist_ok=True)
#    heatmap_title = "fold-{}__heatmap".format(fold_i)
#    heatmap_fname = '{}/adj_matrices/{}.png'.format(args.path_save_results, heatmap_title)
#    
#    histogram_title = "fold-{}__histogram".format(fold_i)
#    histogram_fname = '{}/adj_matrices/{}.png'.format(args.path_save_results, histogram_title)
#    
#    if os.path.isfile(heatmap_fname) and os.path.isfile(histogram_fname):
#        print("Saving the adjacency is ignored since it already exists.")
#        return
#    
#    
#    fig, ax = plt.subplots()
#    im = ax.imshow(adj, cmap=plt.get_cmap('hot'), vmin=0, vmax=1)
#    fig.colorbar(im)
#    
#    plt.title(heatmap_title)
#    plt.savefig(heatmap_fname, bbox_inches="tight", transparent=True, dpi=400)
#    # Only show in the case of spyder executer, terminal or colab should not show the images.
#    if any('SPYDER' in name for name in os.environ):
#        plt.show()
#    plt.close()
#    
#    binsize = 0.1
#    plt.hist(adj.flatten(),bins=np.arange(0,1+binsize, binsize))
#    plt.title(histogram_title)
#    plt.yscale("log")
#    plt.savefig(histogram_fname, bbox_inches="tight", transparent=True, dpi=400)
#    # Only show in the case of spyder executer, terminal or colab should not show the images.
#    if any('SPYDER' in name for name in os.environ):
#        plt.show()
#    plt.close()
    
#def save_full_adjacency(args, data):
#    """
#    Create an image of the adjacency matrix given the current settings, but on all data. 
#    This will not be used while learning, but only for demonstraion.
#    """
#    
#    local_data_temp = {"train_ind": np.arange(len(data["dx_vec"])), "fold_i":-1}
#    local_data_temp["features"] = get_reduced_dimensionality(args, data, local_data_temp)
#    adj_matrix = get_adjacency_matrix(args, data, local_data_temp)
#    check_and_save_adj_values(args, adj_matrix, "All data")

def plot_results_(args, experiment_type:str, store_data_dicts, list_tuple_lines_labels:list):
    """
    experiment_type: e.g. Loss
    list_tuple_lines_labels: [(d1,'d1'),(d2,'d2'),...]
    Note that the current implementation only allows 4 line_styles.
    """

    ax = plt.axes()
    plt.title("{}".format(experiment_type))
#    plt.yscale('log')
    
    line_styles = ["-", ":", '--', '-,']

    for store_data_dict in store_data_dicts:
        color = next(ax._get_lines.prop_cycler)['color']
        for i, (data_label, legend_label) in enumerate(list_tuple_lines_labels):
            data = store_data_dict[data_label]
            plt.plot(data, label=legend_label, color=color, ls=line_styles[i], alpha=0.5)

    plt.legend()
            
    os.makedirs(args.dir_name_results, exist_ok=True)
    plt.savefig('{}/{}.png'.format(args.dir_name_results, experiment_type), bbox_inches="tight", transparent=True, dpi=400)
    
    # Only show in the case of spyder executer, terminal or colab should not show the images.
    if any('SPYDER' in name for name in os.environ):
        plt.show()
    plt.close()

def save_results(args, store_data_dicts:list):    
    """
    Store the plots of the data and does store the results as well.
    """
    
    plot_results_(
            args,
            experiment_type="AUC",
            store_data_dicts=store_data_dicts,
            list_tuple_lines_labels=[
                    ("list_train_auc", "Train"),
                    ("list_val_auc", "Validation")]
            )
    
    plot_results_(
            args,
            experiment_type="Sensitivity",
            store_data_dicts=store_data_dicts,
            list_tuple_lines_labels=[
                    ("list_train_sensitivity", "Train"),
                    ("list_val_sensitivity", "Validation")]
            )
    
    plot_results_(
            args,
            experiment_type="Specificity",
            store_data_dicts=store_data_dicts,
            list_tuple_lines_labels=[
                    ("list_train_specificity", "Train"),
                    ("list_val_specificity", "Validation")]
            )
    
    plot_results_(
            args,
            experiment_type="Accuracy", 
            store_data_dicts=store_data_dicts,
            list_tuple_lines_labels=[
                    ("list_train_accuracy", "Train"),
                    ("list_val_accuracy", "Validation")]
            )

    plot_results_(
            args,
            experiment_type="Mean_train", 
            store_data_dicts=store_data_dicts,
            list_tuple_lines_labels=[
                    ("list_train_mean_pred", "Predicted"),
                    ("list_train_mean_lab", "Actual")]
            )
    
    plot_results_(
            args,
            experiment_type="Mean_val", 
            store_data_dicts=store_data_dicts,
            list_tuple_lines_labels=[
                    ("list_val_mean_pred", "Predicted"),
                    ("list_val_mean_lab", "Actual")]
            )
    
    plot_results_(
            args,
            experiment_type="Loss", 
            store_data_dicts=store_data_dicts,
            list_tuple_lines_labels=[
                    ("list_train_loss", "Train"),
                    ("list_val_loss", "Validation")]
            )
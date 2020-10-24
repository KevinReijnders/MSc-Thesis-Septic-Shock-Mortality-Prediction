"""
Module with implementation of the probabilistic fuzzy systems as described in
the paper by Fialho et al. (2016) in the Applied Soft Computing journal.

Possible future extensions:
* Integrate with Simpful package
* Allow for different fuzzy membership functions
* Multi-class prediction support
"""

# Imports
import numpy as np
from itertools import combinations
import tensorflow as tf

# Utility methods

#PFS with conditional probability estimation
class PFS_CP:
    """
    Probabilistic fuzzy system with conditional probability estimation 
    as described in the paper by Fialho et al. (2016) in the Applied Soft 
    Computing journal. Also supports multi-class classification.

    Similarly structured as the algorithms in the scikit-learn library.
    Inputs and outputs are numpy arrays. Performance-crucial parts are written using vectorized functions for high speed on large datasets. 

    Must be initialized with cluster centers - meaning that clustering should
    be performed beforehand (e.g. using fuzzy c-means clustering).
    
    inputs
    -----
    cluster_centers: Coordinates of the cluster centers. 
            Shape (n_clusters, n_features).
    """

    def __init__(self, cluster_centers):
        self.cluster_centers_ = cluster_centers

    def calc_widths(self):
        """
        Calculate the widths of the membership functions as in equation (6).
        Replicate across each dimension (convenient for later tuning with 
        gradient descent).

        Output shape: (n_rules, n_features).
        """
        #Enumerate all combinations of two centers + calculate euclidean dist
        center_indices = [i for i in range(len(self.cluster_centers_))]
        combs = [comb for comb in combinations(center_indices, 2)]
        comb_coords = np.array([
            [self.cluster_centers_[i], self.cluster_centers_[j]] \
                for i,j in combs
        ])
        
        dists = []
        for comb, coord in zip(combs, comb_coords):
            euc_dist = np.linalg.norm(coord[0] - coord[1])
            dists.append([comb[0], comb[1], euc_dist])

        #Enter everything in a matrix for minimum finding
        dist_matrix = np.full(
            shape = (len(self.cluster_centers_), len(self.cluster_centers_)),
            fill_value = np.inf
            )
        for i,j,dist in dists:
            dist_matrix[i,j] = dist

        #Find the minimum distances for each center
        min_ver = dist_matrix.min(axis=0)
        min_hor = dist_matrix.min(axis=1)
        widths = np.vstack((min_ver, min_hor)).min(axis=0)
        widths = np.vstack(
            [widths for i in range(self.cluster_centers_.shape[1])]
        ) #Replicate for each dimension in the data

        return widths.T

    def calc_cond_probs(self, X, y):
        """
        Calculate Pr(omega_c | A_j) using conditional probability estimation.
        Output shape: (n_rules, n_classes).
        """
        mf_vals = self.calc_mf(X, normalized=True)

        #Calculate numerators & denominators of eq. (7)
        denoms = mf_vals.sum(axis=0).reshape(1,-1) 
        nums = np.vstack(
            [mf_vals[y==i].sum(axis=0) for i in np.unique(y)]
        )
        
        cond_probs = (nums/denoms).T #Transpose for easier interpretation

        #Set to 0.5 probability if NaN
        cond_probs = np.where(
            np.isnan(cond_probs),
            np.full(cond_probs.shape, 0.5),
            cond_probs
        )

        return cond_probs

    def calc_mf(self, X, normalized=False):
        """
        Calculate (normalized) membership of samples X.
        Output shape: (n_samples, n_rules)
        """
        #First calculate numerator in eq (5) of Fialho
        num = (X.reshape(X.shape[0], 1, X.shape[1]) - self.cluster_centers_)**2 
        
        #Denominator in right shape for further calculation
        denom = self.widths_\
            .reshape(1, self.widths_.shape[0], self.widths_.shape[1])**2

        #Calculate the non-normalized membership values
        memb_vals_X = np.exp(- (num/denom).sum(axis=2))
        
        if (normalized):
            norm_memb_vals = memb_vals_X / memb_vals_X.sum(axis=1).reshape(-1,1)

            #Resolve zero-division errors for observations that are too far
            #from all rules. Manually set these normalized membership
            #values to 0
            norm_memb_vals = np.where(
                np.isnan(norm_memb_vals),
                np.zeros(memb_vals_X.shape),
                norm_memb_vals
            )

            return norm_memb_vals
        else:
            return memb_vals_X

    def fit(self, X, y):
        """
        Fit the model to the data using euclidean distance between centers and 
            conditional probability estimation.
        
        inputs
        -----
        X: input array. Shape (n_samples, n_features).
        
        y: class labels. Shape (n_samples,). Must be integer.
        """
        self.widths_ = self.calc_widths()
        self.cond_probs_ = self.calc_cond_probs(X,y)

    def predict_proba(self, X):
        """
        Predict the class label probabilities of an array of observations.
        Output shape: (n_samples, n_classes)

        inputs
        -----
        X: array with feature values of the samples.
            Shape (n_samples, n_features).
        """
        #Numerator in eq (5) of Fialho
        num = (X.reshape(X.shape[0], 1, X.shape[1]) - self.cluster_centers_)**2 

        #Retrieve normalized membership values
        norm_memb_vals_X = self.calc_mf(X,normalized=True)

        #Calculate final predictions
        y_pred_proba = np.dot(norm_memb_vals_X, self.cond_probs_) 

        return y_pred_proba

    def predict(self, X):
        """
        Predict the class label of an array of observations.
        Output shape: (n_samples, 1)

        inputs
        -----
        X: array with feature values of the samples.
            Shape (n_samples, n_features).
        """
        y_pred_proba = self.predict_proba(X)
        return np.round(y_pred_proba.max(axis=1))

#PFS as a Tensorflow Layer, such that we can use Tensorflow functionality
#regarding losses, metrics, optimizers, etc.
class PFS_layer(tf.keras.layers.Layer):
    """
    Probabilistic fuzzy system as described in the paper by Fialho et al. 
    (2016) in the Applied Soft Computing journal, but implemented as
    a Tensorflow layer such that functionality from (tf.)keras can be reused
    (losses, optimizers, metrics, callbacks, etc.).

    Assumes cluster_centers as input, thus clustering needs to be
        performed beforehand externally (e.g. using K-means).
    In contrast to standard tf.layers, this layer must be initialized with 
        (X,y,cluster_centers) in order to perform initalization using
        conditional probability estimation.

    limitations
    -----
    * Only supports binary classification
    * Forces float64 dtype to avoid zero devision as much possible. This
        doubles the usual memory usage, which uses float32.

    inputs
    -----
    X: input array. Shape (n_samples, n_features).
    
    y: class labels. Shape (n_samples,). Must be integer.
        
    cluster_centers: Coordinates of the cluster centers. 
        Shape (n_clusters, n_features).
    """

    def __init__(self, X, y, cluster_centers, dynamic=False, dtype=tf.float64,
                 **kwargs):
        super(PFS_layer, self).__init__(dynamic=dynamic, dtype=dtype,
                                        **kwargs)

        #Class label check: binary classification?
        if (len(np.unique(y)) > 2):
            raise Exception("3 or more distinct classes found. \
                Only binary classification is supported.")

        #Use conditional probability estimation to obtain initial widths and
        #conditional probabilities
        pfs = PFS_CP(cluster_centers)
        pfs.fit(X, y)

        #Calculate auxiliary variable values for class 1 using reverse
        #sigmoid function, i.e. probit
        cond_probs_aux = np.log(pfs.cond_probs_[:,1]/(1-pfs.cond_probs_[:,1]))

        #Set centers, widths and auxiliary variables of the cond_probs such
        #that they can be tuned by re-using tf.keras functionality
        self.cluster_centers_ = self.add_weight(name='cluster_centers',
                                                shape=cluster_centers.shape,
                                                dtype=tf.float64)
        self.widths_ = self.add_weight(name='widths',
                                       shape=pfs.widths_.shape,
                                       dtype=tf.float64)
        self.cond_probs_aux_ = self.add_weight(name='cond_probs_aux',
                                                shape=cond_probs_aux.shape,
                                                dtype=tf.float64)
        self.set_weights([cluster_centers, pfs.widths_, cond_probs_aux])
    
    def get_parameters(self):
        """
        Get numpy arrays with the cluster centers, widths and conditional
            probabilities.

        outputs
        -----
        * cluster centers: Shape (n_clusters, n_features))
        * widths: Shape (n_clusters)
        * conditional probabilities: Shape (n_rules, n_classes)
        """
        cluster_centers, widths, cond_probs_aux = self.get_weights()

        cond_probs_class_1 = np.exp(cond_probs_aux)/(1+np.exp(cond_probs_aux))
        cond_probs_class_0 = 1 - cond_probs_class_1
        cond_probs = np.vstack([cond_probs_class_0, cond_probs_class_1])

        return cluster_centers, widths, cond_probs.T

    def calc_mf_tf(self, inputs, normalized=False):
        """
        Calculate (normalized) membership of samples "inputs" solely using 
        Tensorflow Tensor operations. Is part of the "call" function.

        Inputs & outputs are thus tf.Tensor rather than np.array

        Output shape: (n_samples, n_rules)
        """
        #First calculate numerator in eq (5) of Fialho
        num = (tf.reshape(inputs, (tf.shape(inputs)[0], 1, tf.shape(inputs)[1]))  - self.cluster_centers_)**2

        #Denominator in right shape for further calculation
        denom = (tf.reshape(
            self.widths_, (1, self.widths_.shape[0], self.widths_.shape[1])
            ))**2

        #Calculate the non-normalized membership values
        memb_vals_inputs = tf.math.exp(
            - tf.math.reduce_sum(num/denom, axis=2)
            )

        # Sometimes, the cluster centers are very close, leading to extremely
        # small widths (e.g. 0.0002), which then leads to large value of num/denom
        # and then to numerical underflow because of tf.math.exp with a large
        # negative number in the exponent. 
        # Though a better solution is to avoid this situation by making sure
        # clusters do not overlap, we can also manually set the normalized
        # membership values to 0 in this situation. Note that the gradients
        # are then undefined, leading to NaN values for all parameters and only
        # predictions of 0.
        # memb_vals_input = tf.where(
        #     tf.math.is_nan(memb_vals_inputs), 
        #     tf.zeros_like(memb_vals_inputs),
        #     memb_vals_inputs)
        
        if (normalized):
            norm_memb_vals_inputs = memb_vals_inputs \
                / tf.reshape(
                    tf.math.reduce_sum(memb_vals_inputs, axis=1),
                    (-1,1)
                    )
            
            # Manually resolving underflow - see comment above.
            # norm_memb_vals_inputs = tf.where(
            #     tf.math.is_nan(norm_memb_vals_inputs), 
            #     tf.zeros_like(norm_memb_vals_inputs),
            #     norm_memb_vals_inputs)

            return norm_memb_vals_inputs
        else:
            return memb_vals_inputs
    
    def call(self, inputs):
        """
        Forward pass as expected by tf.keras, e.g. in the function .predict.
        Only supports binary classification in this implementation!
        Output shape: (n_samples, n_classes)
        """
        #Numerator in eq (5) of Fialho     
        num = (tf.reshape(inputs, (tf.shape(inputs)[0], 1, tf.shape(inputs)[1])) \
            - self.cluster_centers_)**2

        #Retrieve normalized membership values
        norm_memb_vals_inputs = self.calc_mf_tf(inputs, normalized=True)

        #Retrieve conditional probabilities for the classes from the
        #auxiliary variables using the sigmoid function
        cond_probs_class_1 = 1/(1+tf.math.exp(self.cond_probs_aux_))
        cond_probs_class_2 = 1 - cond_probs_class_1
        cond_probs = tf.stack([cond_probs_class_1, cond_probs_class_2], axis=1)
        
        #Calculate final predictions & only keep predictions for class 1
        y_pred_proba = tf.tensordot(norm_memb_vals_inputs, cond_probs, axes=1)
        y_pred_proba = tf.reshape(y_pred_proba[:,1], (-1,1))

        return y_pred_proba
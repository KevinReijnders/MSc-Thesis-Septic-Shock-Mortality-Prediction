"""
This module contains utilities for fitting and testing the RNN/GRU
    models in this research:
    * Sequence_generator: custom generator to feed sequences to an RNN.
        Required to feed variable-length sequences to the GRU with a batch-size
        of 1, but can also zero-pad and thereby feed data in batch-like style

Notes
* Using sample weights in the Sequence_generator currently yields 1 sample 
    weight per sample (per time series). In the version of Tensorflow used for
    the project (2.2.0), feeding individual sample weights for each timestep
    via a generator seems to be broken (see the GRU tuning notebook for an
    elaboration on why this is the case).
"""

#Generator class that handles the per-batch sequence padding
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Sequence_generator(tf.keras.utils.Sequence):
    """
    Custom generator to feed sequences to an RNN with possible sequence padding.
        All sequences are padded to the length of the longest sequence in the
        currently drawn batch, i.e. padding is performed at batch-time rather
        than completely up-front.

    inputs
    -----
    * x_set: features, shape: (samples, timesteps, features).
    * y_set: target label, shape: (samples, timesteps)
    * batch_size: how many samples to put in a single batch. 
    * sample_weight: sample weights, shape: (samples, timesteps)
        Note that only one sample weight (the first in the sequence) will be
        returned by the generator for each sample (see docstring at the start
        of this module).
    * X_indicator_value: value filled in for the features in the padded timesteps.
    * y_indicator_value: value filled in for the labels in the padded timesteps.
    """
    def __init__(self, x_set, y_set=[], batch_size=32, sample_weight=[], 
                 X_indicator_value=-100, y_indicator_value=0):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.sample_weight = sample_weight
        self.X_indicator_value = X_indicator_value
    
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = pad_sequences(batch_x, 
                                padding='post',
                                dtype='float32',
                                value=self.X_indicator_value) 
        
        if (len(self.y) != 0):
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = pad_sequences(batch_y,
                                    padding='post',
                                    value=0) #Label 0 for padded timesteps
        #Regarding sequence padding:
        #Without setting dtype manually, data will be converted to integers!
        #Padding without masking with a value of 0 is dangerous - the data can also naturally
        #contain 0 (due to standardization), thus we would also need a mask

        if (len(self.sample_weight)!=0):
            batch_sample_weight = self.sample_weight[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_sample_weight = pad_sequences(batch_sample_weight,
                                                padding='post',
                                                dtype='float32',
                                                value=0) #Sample weight 0 for padded timesteps
            
        # Determine which of X, y and sample weights to yield
        if (len(self.y) != 0) & (len(self.sample_weight) != 0):
            return batch_x, batch_y, np.array([batch_sample_weight[0][0]])
        elif (len(self.y) != 0):
            return batch_x, batch_y
        else:
            return batch_x

    def on_epoch_end(self):
        #Shuffle the order of the sequences while keeping time-order impact within each sequence
        order = np.random.permutation(len(self.x))
        self.x = self.x[order]
        if (len(self.y)!=0):
            self.y = self.y[order]
        if (len(self.sample_weight)!=0):
            self.sample_weight = self.sample_weight[order]


def to_seq(df, case_colname, X, y=[], sample_weights=[]):
    """
    Convert the training data to sequence format for RNNs (in our case GRU).
    Assumes df and X are already sorted w.r.t. case and timesteps!

    inputs
    -----
    * df: dataframe from the modeling data
    * case_colname: name of the column with the case/sample identifier for
        generating the sequences.
    * X: normalized and imputed feature numpy array of the modeling data,
        corresponding to df.
    * y: optional prediction label (y) corresponding to df and X.
    * sample_weights: optional sample weights corresponding to df, X and y.

    outputs
    -----
    * X: features as sequences, in shape (n_cases, n_timesteps, n_features).
        Note that n_timesteps can thus be variable.
    * y: optional - labels as sequences, in shape (n_cases, n_timesteps, n_features).
    * sample_weights: optional: - sample weights as sequences, in shape 
        (n_cases, n_timesteps, n_features).
    """
    #Get the number of instances per subject_id, in order of appearance
    order_of_app = df[case_colname].value_counts().loc[df[case_colname].unique()]

    #Transform into indices in the X_... and y_... arrays
    order_of_app_freq = df[case_colname].value_counts().loc[df[case_colname].unique()]
    indices_in_arrays = order_of_app_freq.cumsum().to_frame('end_index')
    indices_in_arrays['start_index'] = order_of_app_freq.cumsum().shift(1).fillna(0).astype(int)
    
    #Store the arrays in sequence format
    X_seq = np.array([X[start_index:end_index] for end_index, start_index in indices_in_arrays.values])

    if (len(y)!=0):
        y_seq = np.array([y[start_index:end_index] for end_index, start_index in indices_in_arrays.values])
    
    if (len(sample_weights)!=0):
        sample_weights_seq = np.array(
            [sample_weights[start_index:end_index] for end_index, start_index in indices_in_arrays.values]
        )
    
    #Return the correct elements depending on the input
    if (len(y)!=0):
        if (len(sample_weights)!=0):
            return X_seq, y_seq, sample_weights_seq
        else:
            return X_seq, y_seq
    else:
        if (len(sample_weights)!=0):
            return X_seq, sample_weights_seq
        else:
            return X_seq

def predict_for_sequences(rnn_model, X_seq, indicator_val=0):
    """
    Workaround function for making predictions with a GRU on variable-length time sequences.
    The issues regarding predicting with GRUs on variable-length sequences were discoverd
        in the GRU tuning notebook. Consult that notebook for an explanation why this
        workaround is required.

    inputs:
    * rnn_model: tf.keras RNN model.
    * X_seq: sequences with timesteps & feature values to predict on.
        Shape (n_samples, n_timesteps, n_samples), with n_timesteps having variable length
        between different samples
    * indicator_val: what indicator value to use for the padding. Does not really matter,
        since padding is pruned anyway.

    outputs:
    * y_score: flattened output scores in shape sum_n(n_samples * n_timesteps),
        suitable for easily calculating metrics
    """
    #Pad sequences & predict for padded sequences
    X_seq_padded = pad_sequences(X_seq, 
                                 padding='post', #Should be post as we trim again afterwards
                                 dtype='float32',
                                 value=indicator_val) 
    y_score_seq = rnn_model.predict(X_seq_padded)

    #Prune the padding
    seq_lengths = [X_seq[i].shape[0] for i in range(len(X_seq))]
    y_score = np.vstack([y_score_seq[i][:seq_lengths[i]] for i in range(y_score_seq.shape[0])]).ravel()

    return y_score
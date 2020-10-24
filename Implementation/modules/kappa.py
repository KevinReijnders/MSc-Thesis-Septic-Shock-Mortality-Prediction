"""
This module contains functions to compute the false positive rates & cohen's
    kappa values across the thresholds of a prediction model. Similar to
    functions in sklearn.metrics. Implementation with vectorized functions
    for efficient computation on large datasets.
"""

from sklearn.metrics import roc_curve, auc, cohen_kappa_score
import numpy as np

def kappa_curve(y_true, y_score):
    """
    Get the Cohen's kappa values and TPRs at each prediction threshold.
    Inputs as numpy arrays, similar to sklearn metrics.
    
    Outputs: fprs, kappas, thresholds

    Notes
    * Vectorized implementation for fast speed over large datasets. Prior 
        version used a for loop, but took too long to run...
    """
    
    #Reuse functionality from sklearn to obtain false positive rates & thresholds
    fpr, tpr, thres = roc_curve(y_true, y_score)
    
    #Distill TP,FP,FN & TN + other confusion matrix metrics
    #for each threshold
    p = y_true.sum() #condition positive
    n = (y_true==0).sum() #condition negative
    tp = np.round(tpr * p)
    fn = p - tp
    fp = np.round(fpr * n)
    tn = n - fp
    p_hat = tp + fp #prediction positive
    n_hat = fn + tn #prediction negative

    #compute cohen's kappa for each threshold, as in the AUK paper
    n_obs = len(y_true) #number of observations
    a = (tp+tn) / n_obs
    p_c = p/n_obs * p_hat/n_obs + n/n_obs * n_hat/n_obs
    kappas = (a-p_c) / (1-p_c)
        
    #Return the output as three numpy arrays
    return fpr, kappas, thres

def auk_score(y_true, y_score):
    """
    Compute the AUK in a set of predictions as proposed in:
        * Kaymak, U., Ben-David, A., & Potharst, R. (2012). 
          The AUK: A simple alternative to the AUC. 
          Engineering Applications of Artificial Intelligence, 25(5), 1082-1089.
    Inputs as numpy arrays, similar to sklearn metrics.
    """
    fprs, kappas, thresholds = kappa_curve(y_true, y_score)
    auk = auc(fprs.ravel(), kappas.ravel())
    return auk
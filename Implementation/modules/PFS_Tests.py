"""
Various unit tests of elements in the PFS Module.

Created by Kevin Reijnders.
"""

#Imports
from PFS import *
import numpy as np
import tensorflow as tf

#Tests
def test_calc_widths():
    """
    Test if the width calculation with the centers is correct.
    """
    np.testing.assert_array_almost_equal(
        pfs.calc_widths(),
        widths,
        decimal=3
    )

def test_calc_mf():
    """
    Test if estimating the fuzzy membership of observations works correctly.
    """
    #Correct values
    np.testing.assert_array_almost_equal(
        pfs.calc_mf(X),
        memb_vals,
        decimal=3
    ) 

    #Sum of all normalized membership values for an observation should be 1
    #Approximately because of floating point arithmetic
    assert (pfs.calc_mf(X, normalized=True)\
        .sum(axis=1)>0.999).sum() == len(memb_vals)

def test_calc_cond_probs():
    """
    Test if estimating the conditional probabilites works correctly.
    """
    memb_vals_norm = memb_vals/memb_vals.sum(axis=1).reshape(-1,1)
    denoms = memb_vals_norm.sum(axis=0)
    nums = np.vstack((
        memb_vals_norm[y==0].sum(axis=0), memb_vals_norm[y==1].sum(axis=0)
    ))
    cond_probs = (nums/denoms).T
    
    pfs_cp = pfs.calc_cond_probs(X,y)

    #Correct values
    np.testing.assert_array_almost_equal(
        pfs.calc_cond_probs(X,y), 
        cond_probs,
        decimal=3)

    #Sum of probabilities for class 0 and 1 for a rule should sum up to 1
    #for all rules. Approximately because of floating point arithmetic.
    assert (pfs.calc_cond_probs(X,y).sum(axis=1)>0.999).sum() \
            == cond_probs.shape[0]

def test_predict_proba():
    y_pred_proba = pfs.predict_proba(X)

    #Check shape
    assert y_pred_proba.shape == (len(y), len(np.unique(y)))

    #Check if probabilities for each sample sum to 1
    #Approximately because of floating point arithmetic
    assert (y_pred_proba.sum(axis=1)>0.999).sum() == len(y)

def test_predict_proba_tf():
    #Regular prediction
    y_pred_proba = pfs.predict_proba(X)[:,1] #Only probabilities for class 1

    #tf prediction
    y_pred_proba_tf = model.predict_proba(X)

    np.testing.assert_array_almost_equal(
        y_pred_proba,
        y_pred_proba_tf.flatten(),
        decimal=3)

#----------------Perform the tests

### Test data setup
#Generate 3 centers & corresponding widths
centers = np.array([
    [1,2],
    [3,2],
    [10,2]
])

#Corresponding widths
widths = np.array([
    [2,2],
    [2,2],
    [7,7]])

#Three observations
X = np.array([
    [1,1],
    [4,2],
    [10,4]
], dtype=float)

#Fuzzy membership of the three observations
memb_vals = np.exp(
    -np.array([
        [1/4,             2**2/4 + 1/4,      9**2/49 + 1/49],
        [3**2/4,          1/4,               6**2/49],
        [9**2/4 + 2**2/4, 7**2/4 + 2**2/4,   4/49]
    ])
)

#Labels
y = np.array([1,1,0])

#Setup the PFSs
pfs = PFS_CP(centers)
pfs.fit(X,y)

model = tf.keras.models.Sequential()
model.add(PFS_layer(X,y,centers))
model.compile(loss='binary_crossentropy', metrics=['AUC'], optimizer='adam')

#Execute tests using sample data
test_calc_widths()
test_calc_mf()
test_calc_cond_probs()
test_predict_proba()
test_predict_proba_tf()
"""
Created by Liu Zongying on June 2019

"""
import numpy as np
import mdp
import Oger
# Loop over the input data and compute the reservoir states
# Layer 1:
# Define the input weights w_in by Xabier initialization and w by uniform distribution and scaled by function rou
##########################################################################################################
def w_inputweight(_input_dim, _output_dim, input_scaling):
# #Xavier Initialization
    node_in = _input_dim
    node_out = _output_dim
    mdp.numx.random.seed(1)
    w_in = input_scaling*((np.random.randn(node_in, node_out) / np.sqrt(node_in / 2))*2-1)
    w_in = w_in.T
    return w_in
#


#####################################################################################
def wfunction(output_dim, input_scaling, spectral_radius, prob):
    #####################################################################################
    mdp.numx.random.seed(1)
    w = mdp.numx.random.normal(0, 1, (output_dim,output_dim))
    ff = input_scaling * (1 - spectral_radius) * np.identity(output_dim, dtype=float) + spectral_radius * w

    if np.max(ff) < 1:
        scale_num = np.max(ff)
        w = ff*scale_num
    elif np.max(ff) > 1:
        w = ff/np.max(ff)
    return w

#######################################################################################

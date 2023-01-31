""" Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library, ops
from tensorflow.python.platform import resource_loader

ops_so = load_library.load_op_library(resource_loader.get_path_to_datafile('_three_nn_ops.so'))

def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return ops_so.three_nn(xyz1, xyz2)

ops.NoGradient('ThreeNN')

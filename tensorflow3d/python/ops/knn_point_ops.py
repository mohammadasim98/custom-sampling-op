""" Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .selection_sort_ops import select_top_k
import tensorflow as tf


def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    n = xyz1.get_shape()[1]
    m = xyz2.get_shape()[1]

    xyz1_expanded = tf.expand_dims(xyz1, axis=[1])
    xyz2_expanded = tf.expand_dims(xyz2, axis=[2])
    xyz1 = tf.tile(xyz1_expanded, [1,m,1,1])
    xyz2 = tf.tile(xyz2_expanded, [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)

    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx

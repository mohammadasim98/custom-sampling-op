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

ops_so = load_library.load_op_library(resource_loader.get_path_to_datafile('_fps_ops.so'))

def farthest_point_sample(npoint, inp, T=float):
    """
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    """
    return ops_so.farthest_point_sample(inp, npoint, T)

ops.NoGradient('FarthestPointSample')


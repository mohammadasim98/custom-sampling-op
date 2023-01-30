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

ops_so = load_library.load_op_library(resource_loader.get_path_to_datafile('_gather_point_ops.so'))

def gather_point(inp,idx):

    return ops_so.gather_point(inp,idx, float)


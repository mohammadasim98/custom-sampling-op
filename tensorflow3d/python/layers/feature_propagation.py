"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf
import numpy as np
from tensorflow3d.python.ops import three_nn, three_interpolate
from tensorflow3d.python.layers.conv2d import Conv2D 

class FeaturePropagation(tf.keras.layers.Layer):
    """
    Customized Set Abstraction Layer
    """
    def __init__(self,  mlp, bn_decay, name, bn=True, last_mlp_activation=True):
        """
        @ops: Initialize parameters
        @args:
            mlp: 1st-stage MLP width
                type: list
            mlp2: 2st-stage MLP width
                type: list
            radius: Radius for local neighborhood search
                type: float
            k: Number of samples in a local neighborhood
                tye: int
            name: Unique name for the layer
                type: str
            pooling: type of pooling i.e., 'max', 'avg' etc
                type: str
        @return: None
        """
        super(FeaturePropagation, self).__init__()
        self.mlp = mlp
        self.bn_decay = bn_decay
        self.id = name
        self.bn = bn
        self.last_mlp_activation = last_mlp_activation
        for i, num_out_channel in enumerate(self.mlp):
            if i == len(self.mlp)-1 and not(self.last_mlp_activation):
                activation_fn = None
            else:
                activation_fn = tf.nn.relu
            setattr(self, f'_tconv2d_s1_{i}', Conv2D(filters=num_out_channel, shape=[1,1], \
                padding='VALID', strides=[1,1], bn=self.bn, bn_decay=self.bn_decay, name=f"{self.name}_tconv2d_s1_{i}", activation=activation_fn))

    def call(self, xyz1, points1, xyz2, points2):
        """
        @ops: Perform matrix multiplication
        @args:
            inputs: Input point cloud
                type: KerasTensor
                shape: BxC
        @return: Output node of Dense layer
            type: KerasTensor
            shape: BxC_
        """
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points

        new_points1 = tf.expand_dims(new_points1, 2)

        for i, _ in enumerate(self.mlp):
            new_points1 = getattr(self, f'_tconv2d_s1_{i}')(new_points1)

        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1


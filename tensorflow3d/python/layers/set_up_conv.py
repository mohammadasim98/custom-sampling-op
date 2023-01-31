"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf
import numpy as np
from tensorflow3d.python.ops import knn_point, group_point, query_ball_point
from tensorflow3d.python.layers.conv2d import Conv2D 

class SetUpConv(tf.keras.layers.Layer):
    """
    Customized Set Abstraction Layer
    """
    def __init__(self, k, mlp, mlp2, bn_decay, name, radius=None,  bn=True, pooling='max', knn=True):
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
        super(SetUpConv, self).__init__()
        self.radius = radius
        self.k = k
        self.mlp = mlp
        self.mlp2 = mlp2
        self.bn_decay = bn_decay
        self.id = name
        self.pooling = pooling
        self.bn = bn
        self.knn = knn

        if self.mlp is not None:
            for i, num_out_channel in enumerate(self.mlp):
                setattr(self, f'_tconv2d_s1_{i}', Conv2D(filters=num_out_channel, shape=[1,1], \
                    padding='VALID', strides=[1,1], bn=self.bn, bn_decay=self.bn_decay, name=f"{self.name}_tconv2d_s1_{i}"))
        if self.mlp2 is not None:
            for i, num_out_channel in enumerate(self.mlp2):
                setattr(self, f'_tconv2d_s2_{i}', Conv2D(filters=num_out_channel, shape=[1,1], \
                    padding='VALID', strides=[1,1], bn=self.bn,  bn_decay=self.bn_decay, name=f"{self.name}_tconv2d_s2_{i}"))

    def call(self, xyz1, feat1, xyz2, feat2):
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
        if self.knn:
            l2_dist, idx = knn_point(self.k, xyz2, xyz1)
        else:
            idx, pts_cnt = query_ball_point(self.radius, self.k, xyz2, xyz1)
        xyz2_grouped = group_point(xyz2, idx) # batch_size, npoint1, nsample, 3
        xyz1_expanded = tf.expand_dims(xyz1, 2) # batch_size, npoint1, 1, 3
        xyz_diff = xyz2_grouped - xyz1_expanded # batch_size, npoint1, nsample, 3

        feat2_grouped = group_point(feat2, idx) # batch_size, npoint1, nsample, channel2
        net = tf.concat([feat2_grouped, xyz_diff], axis=3) # batch_size, npoint1, nsample, channel2+3

        if self.mlp is not None:
            for i, _ in enumerate(self.mlp):
                net = getattr(self, f'_tconv2d_s1_{i}')(net)

        if self.pooling=='max':
            feat1_new = tf.reduce_max(net, axis=[2], keepdims=False, name='maxpool') # batch_size, npoint1, mlp[-1]
        elif self.pooling=='avgmax':
            feat1_new = tf.reduce_mean(net, axis=[2], keepdims=False, name='avgpool') # batch_size, npoint1, mlp[-1]

        if feat1 is not None:
            feat1_new = tf.concat([feat1_new, feat1], axis=2) # batch_size, npoint1, mlp[-1]+channel1

        feat1_new = tf.expand_dims(feat1_new, 2) # batch_size, npoint1, 1, mlp[-1]+channel2
        
        if self.mlp2 is not None:
            for i, _ in enumerate(self.mlp2):
                feat1_new = getattr(self, f'_tconv2d_s2_{i}')(feat1_new)

        feat1_new = tf.squeeze(feat1_new, [2])
        return feat1_new


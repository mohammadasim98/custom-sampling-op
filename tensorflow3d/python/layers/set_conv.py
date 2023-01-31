"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf
import numpy as np
from tensorflow3d.python.ops import farthest_point_sample, gather_point, group_point, query_ball_point
from tensorflow3d.python.layers.conv2d import Conv2D 

class SetConv(tf.keras.layers.Layer):
    """
    Customized Set Abstraction Layer
    """
    def __init__(self, samples, radius, k, mlp, mlp2, group_all, bn_decay, name,  bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
        """
        @ops: Initialize parameters
        @args:
            mlp: 1st-stage MLP width
                type: list
            mlp2: 2st-stage MLP width
                type: list
            radius: Radius for local neighborhood search
                type: float
            samples: Number of samples from farthest point sampling
                type: int
            k: Number of samples in a local neighborhood
                tye: int
            name: Unique name for the layer
                type: str
            use_xyz: Whether to concatenate the xyz with point features
                type: bool
                default: true
            pooling: type of pooling i.e., 'max', 'avg' etc
                type: str
        @return: None
        """
        super(SetConv, self).__init__()
        self.samples = samples
        self.radius = radius
        self.k = k
        self.mlp = mlp
        self.mlp2 = mlp2
        self.group_all = group_all
        self.bn_decay = bn_decay
        self.id = name
        self.use_xyz = use_xyz
        self.pooling = pooling
        self.bn = bn
        self.knn = knn
        self.use_nchw = use_nchw
        self.data_format = 'NCHW' if self.use_nchw else 'NHWC'

        for i, num_out_channel in enumerate(self.mlp):
            setattr(self, f'_tconv2d_s1_{i}', Conv2D(filters=num_out_channel, shape=[1,1], \
                padding='VALID', strides=[1,1], bn=self.bn, bn_decay=self.bn_decay, data_format=self.data_format, name=f"{self.name}_tconv2d_s1_{i}"))
        if self.mlp2 is not None:
            for i, num_out_channel in enumerate(self.mlp2):
                setattr(self, f'_tconv2d_s2_{i}', Conv2D(filters=num_out_channel, shape=[1,1], \
                    padding='VALID', strides=[1,1], bn=self.bn,  bn_decay=self.bn_decay, data_format=self.data_format, name=f"{self.name}_tconv2d_s2_{i}"))
    
    def sample_and_group(self, npoint, radius, nsample, xyz, points, use_xyz=True):
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
        idx, _ = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
        if points is not None:
            grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
            if use_xyz:
                new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz

        return new_xyz, new_points, idx, grouped_xyz

    def sample_and_group_all(xyz, points, use_xyz=True):
        batch_size = xyz.get_shape()[0].value
        nsample = xyz.get_shape()[1].value
        new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
        idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
        grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
        if points is not None:
            if use_xyz:
                new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
            else:
                new_points = points
            new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points, idx, 

    def call(self, xyz, points):
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
        if self.group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = self.sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = self.sample_and_group(self.samples, self.radius, self.k, xyz, points, self.use_xyz)

        if self.use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, _ in enumerate(self.mlp):
            new_points = getattr(self, f'_tconv2d_s1_{i}')(new_points)
        if self.use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        if self.pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
        elif self.pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
        elif self.pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif self.pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        if self.use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        if self.mlp2 is not None:
            for i, _ in enumerate(self.mlp2):
                new_points = getattr(self, f'_tconv2d_s2_{i}')(new_points)
        if self.use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

        new_points = tf.squeeze(new_points, [2])
        return new_xyz, new_points, idx


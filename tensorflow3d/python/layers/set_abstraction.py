"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf
from tensorflow3d.python.ops import farthest_point_sample, gather_point, group_point, query_ball_point
from tensorflow3d.python.layers.conv2d import Conv2D 

class SetAbstraction(tf.keras.layers.Layer):
    """
    Customized Set Abstraction Layer
    """
    def __init__(self, mlp, mlp2, radius, samples, k, name, pooling='max', use_xyz=True):
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
        super(SetAbstraction, self).__init__()
        self.mlp = mlp
        self.mlp2 = mlp2
        self.radius = radius
        self.id = name
        self.samples = samples
        self.k = k
        self.use_xyz = use_xyz
        self.pooling = pooling
        for i, num_out_channel in enumerate(self.mlp):
            setattr(self, f'_tconv2d_s1_{i}', Conv2D(filters=num_out_channel, shape=[1,1], padding='VALID', strides=[1,1], name=f"{self.name}_tconv2d_s1_{i}"))
        if self.mlp2 is not None:
            for i, num_out_channel in enumerate(self.mlp2):
                setattr(self, f'_tconv2d_s2_{i}', Conv2D(filters=num_out_channel, shape=[1,1], padding='VALID', strides=[1,1], name=f"{self.name}_tconv2d_s2_{i}"))
    
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
        new_xyz, new_points, idx, grouped_xyz = self.sample_and_group(self.samples, self.radius, self.k, xyz, points, self.use_xyz)

        for i, _ in enumerate(self.mlp):
            new_points = getattr(self, f'_tconv2d_s1_{i}')(new_points)

        if self.pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
        elif self.pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')

        if self.mlp2 is not None:
            for i, _ in enumerate(self.mlp2):
                new_points = getattr(self, f'_tconv2d_s2_{i}')(new_points)
        new_points = tf.squeeze(new_points, [2])
        return new_xyz, new_points, idx


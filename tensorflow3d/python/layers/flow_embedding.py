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

class FlowEmbedding(tf.keras.layers.Layer):
    """
    Customized Set Abstraction Layer
    """
    def __init__(self, radius, k, mlp, bn_decay, name,  bn=True, pooling='max', knn=True, corr_func='elementwise_product'):
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
        super(FlowEmbedding, self).__init__()
        self.radius = radius
        self.k = k
        self.mlp = mlp
        self.bn_decay = bn_decay
        self.id = name
        self.pooling = pooling
        self.bn = bn
        self.knn = knn
        self.corr_func = corr_func

        for i, num_out_channel in enumerate(self.mlp):
            setattr(self, f'_tconv2d_s1_{i}', Conv2D(filters=num_out_channel, shape=[1,1], \
                padding='VALID', strides=[1,1], bn=self.bn, bn_decay=self.bn_decay, name=f"{self.name}_tconv2d_s1_{i}"))
    

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
            _, idx = knn_point(self.k, xyz2, xyz1)
        else:
            idx, cnt = query_ball_point(self.radius, self.k, xyz2, xyz1)
            _, idx_knn = knn_point(self.k, xyz2, xyz1)
            cnt = tf.tile(tf.expand_dims(cnt, -1), [1,1,self.k])
            idx = tf.where(cnt > (self.k-1), idx, idx_knn)

        xyz2_grouped = group_point(xyz2, idx) # batch_size, npoint, self.k, 3
        xyz1_expanded = tf.expand_dims(xyz1, 2) # batch_size, npoint, 1, 3
        xyz_diff = xyz2_grouped - xyz1_expanded # batch_size, npoint, self.k, 3

        feat2_grouped = group_point(feat2, idx) # batch_size, npoint, self.k, channel
        feat1_expanded = tf.expand_dims(feat1, 2) # batch_size, npoint, 1, channel
        # TODO: change distance function
        if self.corr_func == 'elementwise_product':
            feat_diff = feat2_grouped * feat1_expanded # batch_size, npoint, self.k, channel
        elif self.corr_func == 'concat':
            feat_diff = tf.concat(axis=-1, values=[feat2_grouped, tf.tile(feat1_expanded,[1,1,self.k,1])]) # batch_size, npoint, sample, channel*2
        elif self.corr_func == 'dot_product':
            feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, self.k, 1
        elif self.corr_func == 'cosine_dist':
            feat2_grouped = tf.nn.l2_normalize(feat2_grouped, -1)
            feat1_expanded = tf.nn.l2_normalize(feat1_expanded, -1)
            feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, self.k, 1
        elif self.corr_func == 'flownet_like': # assuming square patch size k = 0 as the FlowNet paper
            batch_size = xyz1.get_shape()[0].value
            npoint = xyz1.get_shape()[1].value
            feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, self.k, 1
            total_diff = tf.concat(axis=-1, values=[xyz_diff, feat_diff]) # batch_size, npoint, self.k, 4
            feat1_new = tf.reshape(total_diff, [batch_size, npoint, -1]) # batch_size, npoint, self.k*4
            #feat1_new = tf.concat(axis=[-1], values=[feat1_new, feat1]) # batch_size, npoint, self.k*4+channel
            return xyz1, feat1_new


        feat1_new = tf.concat([feat_diff, xyz_diff], axis=3) # batch_size, npoint, self.k, [channel or 1] + 3

        for i, _ in enumerate(self.mlp):
            feat1_new = getattr(self, f'_tconv2d_s1_{i}')(feat1_new)

        if self.pooling=='max':
            feat1_new = tf.reduce_max(feat1_new, axis=[2], keepdims=False, name='maxpool_diff')
        elif self.pooling=='avg':
            feat1_new = tf.reduce_mean(feat1_new, axis=[2], keepdims=False, name='avgpool_diff')
        return xyz1, feat1_new


"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf

from tensorflow3d.python.layers import SetConv, FlowEmbedding, SetUpConv, FeaturePropagation, Conv1D

class FlowNet3D(tf.keras.Model):
    """
    PointNet Model
    """

    def __init__(self, name, bn_decay=None):
        """
        @ops: Initialize PointNet
        @args:
            name: Unique name for the model
                type: Str
        @return: None
        """
        super(FlowNet3D, self).__init__()
        self.num_points = None
        self.id = name
        self.bn_decay = bn_decay
        self.RADIUS1 = 0.5
        self.RADIUS2 = 1.0
        self.RADIUS3 = 2.0
        self.RADIUS4 = 4.0

        self.setconv1_1 = SetConv(name='setconv1_f1', mlp=[32,32,64], mlp2=None, samples=1024, radius=self.RADIUS1, k=16, group_all=False, bn_decay=self.bn_decay)
        self.setconv2_1 = SetConv(name='setconv2_f1', mlp=[64,64,128], mlp2=None, samples=256, radius=self.RADIUS2, k=16, group_all=False, bn_decay=self.bn_decay)

        self.setconv1_2 = SetConv(name='setconv1_f2', mlp=[32,32,64], mlp2=None, samples=1024, radius=self.RADIUS1, k=16, group_all=False, bn_decay=self.bn_decay)
        self.setconv2_2 = SetConv(name='setconv2_f2', mlp=[64,64,128], mlp2=None, samples=256, radius=self.RADIUS2, k=16, group_all=False, bn_decay=self.bn_decay)

        self.flowembedding = FlowEmbedding(name="flowembedding1", radius=10.0, k=64, mlp=[128,128,128], bn_decay=self.bn_decay, bn=True, pooling='max', knn=True, corr_func='concat')

        self.setconv3 = SetConv(name='setconv3', mlp=[128,128,256], mlp2=None, samples=64, radius=self.RADIUS3, k=8, group_all=False, bn_decay=self.bn_decay)
        self.setconv4 = SetConv(name='setconv4', mlp=[256,256,512], mlp2=None, samples=16, radius=self.RADIUS4, k=8, group_all=False, bn_decay=self.bn_decay)
    
        self.setupconv1 = SetUpConv(name='setupconv1', mlp=None, mlp2=[256, 256], radius=2.4, k=8, knn=True, bn_decay=self.bn_decay)
        self.setupconv2 = SetUpConv(name='setupconv2', mlp=[128, 128, 256], mlp2=[256], radius=1.2, k=8, knn=True, bn_decay=self.bn_decay)
        self.setupconv3 = SetUpConv(name='setupconv3', mlp=[128, 128, 256], mlp2=[256], radius=0.6, k=8, knn=True, bn_decay=self.bn_decay)

        self.pointnetfp = FeaturePropagation(name='featureprop1', mlp=[256, 256], bn_decay=self.bn_decay)

        self.conv1 = Conv1D(filters=128, shape=1, name="conv1d1", padding='VALID', bn=True, bn_decay=self.bn_decay)
        self.conv2 = Conv1D(filters=3, shape=1, name="conv1d2", padding='VALID', activation=None)
    def build(self, input_shape1, input_shape2):
        """
        @ops: Build the PointNet as a complete model
        @args:
            input_shape: Shape of the input
                type: Tuple / List
        @return: A tensorflow model
            type: Functional model
        """
        self.num_points = input_shape1[1]
        xyz1 = tf.keras.layers.Input(shape=input_shape1)
        xyz2 = tf.keras.layers.Input(shape=input_shape2)
        return tf.keras.models.Model(inputs=[xyz1, xyz2], outputs=self.call(xyz1, None, xyz2, None))

    def call(self, l0_xyz_f1, l0_points_f1, l0_xyz_f2, l0_points_f2):
        """
        @ops: Call PointNet layer on inputs
        @args:
            inputs: Input point cloud
                type: KerasTensor
                shape: BxNxC
        @return: Output node of PointNet
            type: KerasTensor
        """
        # BxNx3: Farthest Point Sampling
        l1_xyz_f1, l1_points_f1, l1_indices_f1 = self.setconv1_1(l0_xyz_f1, l0_points_f1)
        l2_xyz_f1, l2_points_f1, l2_indices_f1 = self.setconv2_1(l1_xyz_f1, l1_points_f1)
    
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = self.setconv1_2(l0_xyz_f2, l0_points_f2)
        l2_xyz_f2, l2_points_f2, l2_indices_f2 = self.setconv2_2(l1_xyz_f2, l1_points_f2)
        
        _, l2_points_f1_new = self.flowembedding(l2_xyz_f1, l2_points_f1, l2_xyz_f2, l2_points_f2)

        l3_xyz_f1, l3_points_f1, l3_indices_f1 = self.setconv3(l2_xyz_f1, l2_points_f1_new)
        l4_xyz_f1, l4_points_f1, l4_indices_f1 = self.setconv4(l3_xyz_f1, l3_points_f1)

        l3_feat_f1 = self.setupconv1(l3_xyz_f1, l3_points_f1, l4_xyz_f1, l4_points_f1)
        l2_feat_f1 = self.setupconv2(l2_xyz_f1, tf.concat(axis=-1, values=[l2_points_f1, l2_points_f1_new]), l3_xyz_f1, l3_feat_f1)
        l1_feat_f1 = self.setupconv3(l1_xyz_f1, l1_points_f1, l2_xyz_f1,  l2_feat_f1)
        l0_feat_f1 = self.pointnetfp(l0_xyz_f1, l0_points_f1, l1_xyz_f1, l1_feat_f1)

        net = self.conv1(l0_feat_f1)
        net = self.conv2(net)
        return net

"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf

from tensorflow3d.python.layers import FPS, TNet, Conv2D, Dense, MaxPool2D, SetAbstraction

class FlowNet3D(tf.keras.Model):
    """
    PointNet Model
    """

    def __init__(self, name):
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
        self.RADIUS1 = 0.5
        self.RADIUS2 = 1.0
        self.RADIUS3 = 2.0
        self.RADIUS4 = 4.0

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
        l1_xyz_f1, l1_points_f1, l1_indices_f1 = SetAbstraction(name='setconv1_f1', mlp=[32,32,64], mlp2=None, samples=1024, radius=self.RADIUS1, k=16)(l0_xyz_f1, l0_points_f1)
        l2_xyz_f1, l2_points_f1, l2_indices_f1 = SetAbstraction(name='setconv2_f1', mlp=[64,64,128], mlp2=None, samples=1024, radius=self.RADIUS2, k=16)(l1_xyz_f1, l1_points_f1)
    
        l1_xyz_f2, l1_points_f2, l1_indices_f2 = SetAbstraction(name='setconv1_f2', mlp=[32,32,64], mlp2=None, samples=1024, radius=self.RADIUS1, k=16)(l0_xyz_f2, l0_points_f2)
        l2_xyz_f2, l2_points_f2, l2_indices_f2 = SetAbstraction(name='setconv2_f2', mlp=[64,64,128], mlp2=None, samples=1024, radius=self.RADIUS2, k=16)(l1_xyz_f2, l1_points_f2)
        return l2_xyz_f1, l2_points_f1, l2_xyz_f2, l2_points_f2

"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf

from tensorflow3d.python.layers import FPS, TNet, Conv2D, Dense, MaxPool2D


class PointNet2(tf.keras.Model):
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
        super(PointNet2, self).__init__()
        self.num_points = None
        self.id = name

    def build(self, input_shape):
        """
        @ops: Build the PointNet as a complete model
        @args:
            input_shape: Shape of the input
                type: Tuple / List
        @return: A tensorflow model
            type: Functional model
        """
        self.num_points = input_shape[1]
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))

    def call(self, inputs):
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
        net = FPS(name='fps', samples=1024)(inputs)
        # # BxNx3: Input Transformation
        # net = TNet(name='input_transformation')(net)
        # # BxNx3:
        # net = tf.expand_dims(net, -1)
        # # BxNx3x1:
        # net = Conv2D(filters=64, shape=[1, 3], name=self.id + '_tconv2d_64_0')(net)
        # net = Conv2D(filters=64, shape=[1, 1], name=self.id + '_tconv2d_64_1')(net)
        # # BxNx1x64:
        # net = tf.squeeze(net, axis=2)
        # # BxNx64: Feature Transformation
        # net = TNet(name='feature_transformation', features=True, k=64)(net)
        # # BxNx64:
        # local_net = tf.expand_dims(net, 2)
        # # BxNx1x64:
        # net = Conv2D(filters=64, shape=[1, 1], name=self.id + '_tconv2d_64_2')(local_net)
        # net = Conv2D(filters=128, shape=[1, 1], name=self.id + '_tconv2d_128_0')(net)
        # net = Conv2D(filters=1024, shape=[1, 1], name=self.id + '_tconv2d_1024_0')(net)
        # # BxNx1x1024: Symmetric function
        # global_net = MaxPool2D(name=self.id + '_tmaxpool2d')(net)
        # # Bx1x1x1024:
        # # global_net = tf.tile(global_net, [1, self.num_points, 1, 1])
        # # concat_net = tf.concat([local_net, global_net], axis=-1)
        # # # BxNx1x1088:
        # # net = Conv2D(filters=512, shape=[1, 1], name=self.id + 'tconv2d_512_0')(concat_net)
        # # net = Conv2D(filters=256, shape=[1, 1], name=self.id + 'tconv2d_256_0')(net)
        # # net = Conv2D(filters=128, shape=[1, 1], name=self.id + 'tconv2d_128_1')(net)
        # # net = Conv2D(filters=128, shape=[1, 1], name=self.id + 'tconv2d_128_2')(net)
        # # net = Conv2D(filters=50, shape=[1, 1], name=self.id + 'tconv2d_50_0')(net)
        # # # BxNx1x50:
        # # net = tf.squeeze(net, axis=2)
        # # # BxNx50:
        #
        # net = tf.keras.layers.Flatten()(global_net)
        # # Bx1024:
        # net = Dense(512, name=self.id + '_tdense_512')(net)
        # # Bx512:
        # net = tf.nn.dropout(net, 0.7, None)
        # net = Dense(256, name=self.id + '_tdense_256')(net)
        # # Bx256:
        # net = tf.nn.dropout(net, 0.7, None)
        # net = Dense(40, name=self.id + '_tdense_40')(net)
        # Bx40:
        return net

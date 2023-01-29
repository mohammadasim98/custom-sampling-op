"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf
from .conv2d import Conv2D
from .maxpool2d import MaxPool2D
from .dense import Dense
from numpy import eye


class TNet(tf.keras.layers.Layer):
    """
    Transformation layer
    """
    def __init__(self, name, features=False, kernel_initializer=tf.keras.initializers.RandomUniform(),
                 bias_initializer=tf.keras.initializers.Constant(0), expand=True, k=3):
        """
        @ops: Initialize TNet parameters and layers
        @args:
            name: Unique name for the layer
                type: Str
            features: Performs input or feature transformation if it is False or True respectively
                type: Bool
            kernel_initializer: Initializer for the kernel weights
                type: KerasInitializer
            bias_initializer: Initializer for the biases
                type: KerasInitializer
            expand: Whether to expand the input dimensions
                type: Bool
            k: Size of the transformation matrix
                type: Int
        @return: None
        """
        super(TNet, self).__init__()
        self.bias = None
        self.kernel = None
        self.num_points = None
        self.k = k
        self.id = name
        self.batch_size = None
        self.expand = expand
        self.features = features
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        if features:
            self.tconv2d_64 = Conv2D(filters=64, shape=[1, 1], name=self.id + '_tconv2d_64')
        else:
            self.tconv2d_64 = Conv2D(filters=64, shape=[1, 3], name=self.id + '_tconv2d_64')

        self.tconv2d_128 = Conv2D(filters=128, shape=[1, 1], name=self.id + '_tconv2d_128')
        self.tconv2d_1024 = Conv2D(filters=1024, shape=[1, 1], name=self.id + '_tconv2d_1024')
        self.tdense_512 = Dense(512, name=self.id + '_tdense_512')
        self.tdense_256 = Dense(256, name=self.id + '_tdense_256')

    def build(self, input_shape):
        """
        @ops: Build the kernel and biases of the TNet layer
        @args:
            input_shape: Shape of the input
                type: Tuple / List
        @return: None
        """
        self.num_points = input_shape[1]
        self.batch_size = input_shape[0]
        self.kernel = tf.Variable(
            initial_value=self.kernel_initializer(shape=(256, self.k**2), dtype=tf.float32),
            trainable=True)
        self.bias = tf.Variable(
            initial_value=self.bias_initializer(shape=(self.k**2, ), dtype=tf.float32),
            trainable=True)

        self.bias.assign_add(tf.constant(eye(self.k).flatten(), dtype=tf.float32))

    def call(self, inputs):
        """
        @ops: Call transformation layer on inputs or features
        @args:
            inputs: Input point cloud
                type: KerasTensor
                shape: BxNxC
        @return: Output node of TNet
            type: KerasTensor
            shape: BxNxC
        """
        # BxNxC
        if not self.features:
            inp = tf.expand_dims(inputs, -1)
            # BxNxCx1:
        else:
            inp = tf.expand_dims(inputs, 2)
            # BxNx1xC:
        net = self.tconv2d_64(inp)
        # BxNx1x64:
        net = self.tconv2d_128(net)
        # BxNx1x128:
        net = self.tconv2d_1024(net)
        # BxNx1x1024:
        net = MaxPool2D(name=self.id + '_tmaxpool2d')(net)
        # Bx1x1x1024:
        net = tf.keras.layers.Flatten()(net)
        # Bx1024:
        net = self.tdense_512(net)
        # Bx512:
        net = self.tdense_256(net)
        # Bx256:
        transform_mat = tf.matmul(net, self.kernel)
        transform_mat = tf.nn.bias_add(transform_mat, self.bias)
        transform_mat = tf.keras.layers.Reshape([self.k, self.k])(transform_mat)
        inputs_transformed = tf.matmul(inputs, transform_mat)
        # BxNxC:
        return inputs_transformed

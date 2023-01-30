"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf


class Conv2D(tf.keras.layers.Layer):

    def __init__(self, filters, shape, bn, bn_decay, data_format, name, kernel_initializer=tf.keras.initializers.RandomNormal(),
                 strides=None, padding='VALID'):
        """
        @ops: Initialize parameters
        @args:
            filters: Number of filters
                type: Int
            shape: Shape for the kernel
                type: List / Tuple
            name: Unique name for the layer
                type: Str
            kernel_initializer: Initializer for the kernel weights
                type: KerasInitializer
            strides: Strides for the convolution
                type: List
            padding: Amount of padding
                type: Str / List
        @return: None
        """
        super(Conv2D, self).__init__()
        if strides is None:
            strides = [1, 1, 1, 1]
        self.bias = None
        self.kernel = None
        self.id = name
        self.shape = shape
        self.padding = padding
        self.strides = strides
        self.filters = filters
        self.bn = bn
        self.bn_decay = bn_decay if bn_decay is not None else 0.9
        self.data_format = data_format
        self.kernel_initializer = kernel_initializer
        self.batch_norm = tf.keras.layers.BatchNormalization(center=True, scale=True, momentum=self.bn_decay)

    def build(self, input_shape):
        """
        @ops: Build the kernel the convolutional layer
        @args:
            input_shape: Shape of the input
                type: Tuple / List
        @return: None
        """
        if self.data_format == 'NHWC':
            num_in_channels = input_shape[-1]
        elif self.data_format=='NCHW':
            num_in_channels = input_shape[1]

        kernel_shape = [self.shape[0], self.shape[1], num_in_channels, self.filters]

        self.kernel = tf.Variable(
            initial_value=self.kernel_initializer(shape=kernel_shape, dtype=tf.float32),
            trainable=True)

        self.biases = tf.Variable(
            initial_value=tf.keras.initializers.Constant(0.0)(shape=[self.filters], dtype=tf.float32),
            trainable=True)

    def call(self, inputs):
        """
        @ops: Perform 2D convolution
        @args:
            inputs: Input point cloud
                type: KerasTensor
                shape: BxNxLxC
        @return: Output node of convolutional layer
            type: KerasTensor
            shape: BxN_xL_xC_
        """
        net = tf.nn.conv2d(inputs, filters=self.kernel, name=self.id,
                            strides=self.strides, padding=self.padding)
        net = tf.nn.bias_add(net, self.biases, data_format=self.data_format)
        if self.bn:
            net = self.batch_norm(net)

        net = tf.nn.relu(net)

        return net



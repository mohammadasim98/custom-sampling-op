"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf


class Conv2D(tf.keras.layers.Layer):

    def __init__(self, filters, shape, name, kernel_initializer=tf.keras.initializers.RandomNormal(),
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
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        """
        @ops: Build the kernel the convolutional layer
        @args:
            input_shape: Shape of the input
                type: Tuple / List
        @return: None
        """
        kernel_shape = [self.shape[0], self.shape[1], input_shape[-1], self.filters]
        self.kernel = tf.Variable(
            initial_value=self.kernel_initializer(shape=kernel_shape, dtype=tf.float32),
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
        return tf.nn.conv2d(inputs, filters=self.kernel, name=self.id,
                            strides=self.strides, padding=self.padding)

"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf


class MaxPool2D(tf.keras.layers.Layer):

    def __init__(self, name, strides=None, padding='VALID'):
        """
        @ops: Initialize parameters
        @args:
            name: Unique name for the layer
                type: Str
            strides: Strides for the maxpooling
                type: List
            padding: Amount of padding
                type: Str / List
        @return: None
        """
        super(MaxPool2D, self).__init__()
        if strides is None:
            strides = [1, 1, 1, 1]
        self.shape = None
        self.id = name
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        """
        @ops: Build the kernel if shape is available
        @args:
            input_shape: Shape of the input
                type: Tuple / List
        @return: None
        """
        if input_shape[1]:
            self.shape = [1, input_shape[1], 1, 1]

    def call(self, inputs):
        """
        @ops: Perform 2D max pooling on axis = 1 i.e., along all the points N
        @args:
            inputs: Input point cloud
                type: KerasTensor
                shape: BxNxLxC
        @return: Output node of max pooling layer
            type: KerasTensor
            shape: Bx1xLxC
        """
        if self.shape:
            return tf.nn.max_pool2d(inputs, ksize=self.shape, name=self.id,
                                    strides=self.strides, padding=self.padding)
        else:
            return tf.math.reduce_max(inputs, axis=1, keepdims=True)

"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import tensorflow as tf


class Dense(tf.keras.layers.Layer):
    """
    Customized Dense/Fully Connected Layer
    """
    def __init__(self, num_outputs, name, kernel_initializer=tf.keras.initializers.RandomUniform(),
                 bias_initializer=tf.keras.initializers.Constant(0)):
        """
        @ops: Initialize parameters
        @args:
            num_outputs: Number of neurons
                type: Int
            name: Unique name for the layer
                type: Str
            kernel_initializer: Initializer for the kernel weights
                type: KerasInitializer
            bias_initializer: Initializer for the biases
                type: KerasInitializer
        @return: None
        """
        super(Dense, self).__init__()
        self.bias = None
        self.kernel = None
        self.num_outputs = num_outputs
        self.id = name
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        """
        @ops: Build the kernel and biases of the Dense layer
        @args:
            input_shape: Shape of the input
                type: Tuple / List
        @return: None
        """
        kernel_shape = [input_shape[-1], self.num_outputs]
        self.kernel = tf.Variable(
            initial_value=self.kernel_initializer(shape=kernel_shape, dtype=tf.float32),
            trainable=True)
        self.bias = tf.Variable(
            initial_value=self.bias_initializer(shape=(self.num_outputs, ), dtype=tf.float32),
            trainable=True)

    def call(self, inputs):
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
        net = tf.matmul(inputs, self.kernel)
        return tf.nn.bias_add(net, self.bias)


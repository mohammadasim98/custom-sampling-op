"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

    @reference: Chris Tralie
    @source: https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8

"""
import tensorflow as tf
from tensorflow3d.python.ops.fps_ops import farthest_point_sample


class FPS(tf.keras.layers.Layer):
    """
    Transformation layer
    """

    def __init__(self, name, samples: int = 1024):
        """
        @ops: Initialize TNet parameters and layers
        @args:
            name: Unique name for the layer
                type: Str
            samples: Number of samples N_ required
                type: Int
        @return: None
        """
        super(FPS, self).__init__()
        self.id = name
        self.samples = samples

    @tf.function
    def call(self, inputs):
        """
        @ops: Call FPS layer on inputs
        @args:
            inputs: Input point cloud
                type: KerasTensor
                shape: BxNx3
        @return: Output node of FPS
            type: KerasTensor
            shape: BxN_x3
        """
        # I:BxNx3 :: O:BxN_x3:
        result = farthest_point_sample(self.samples, inputs, float)
        return result

# """
#     For non-commercial use.
#     (c) All rights reserved.

#     This package is licensed under MIT
#     license found in the root directory.

#     @author: Mohammad Asim

# """
# import tensorflow as tf
# from tensorflow.python.ops import farthest_point_sample, gather_points

# class SetAbstraction(tf.keras.layers.Layer):
#     """
#     Customized Set Abstraction Layer
#     """
#     def __init__(self, mlp, mlp2, radius, samples, k, name):
#         """
#         @ops: Initialize parameters
#         @args:
#             mlp: 1st-stage MLP width
#                 type: list
#             mlp2: 2st-stage MLP width
#                 type: list
#             radius: Radius for local neighborhood search
#                 type: float
#             samples: Number of samples from farthest point sampling
#                 type: int
#             k: Number of samples in a local neighborhood
#                 tye: int
#             name: Unique name for the layer
#                 type: Str
#         @return: None
#         """
#         super(SetAbstraction, self).__init__()
#         self.mlp = mlp
#         self.mlp2 = mlp2
#         self.radius = radius
#         self.id = name
#         self.samples = samples
#         self.k = k

#     def sample_and_group(self, npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
#         new_xyz = gather_points(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
#         idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
#         grouped_xyz = group_points(xyz, idx) # (batch_size, npoint, nsample, 3)
#         grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
#         if points is not None:
#             grouped_points = group_points(points, idx) # (batch_size, npoint, nsample, channel)
#             if use_xyz:
#                 new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
#             else:
#                 new_points = grouped_points
#         else:
#             new_points = grouped_xyz

#         return new_xyz, new_points, idx, grouped_xyz


#     def call(self, inputs):
#         """
#         @ops: Perform matrix multiplication
#         @args:
#             inputs: Input point cloud
#                 type: KerasTensor
#                 shape: BxC
#         @return: Output node of Dense layer
#             type: KerasTensor
#             shape: BxC_
#         """
#         net = tf.matmul(inputs, self.kernel)
#         return tf.nn.bias_add(net, self.bias)


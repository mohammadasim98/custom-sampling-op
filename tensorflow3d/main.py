"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""

import tensorflow as tf
import tensorflow3d as t3d


if __name__ == '__main__':

    # inputs = tf.keras.Input(shape=(2048, 3))
    # fps = t3d.layers.FPS(name='FPS', samples=1024)(inputs)
    pnet = t3d.models.PointNet2(name='pointnet').build(input_shape=(None, 3))
    # model = tf.keras.models.Model(inputs=inputs, outputs=fps)
    # model.summary()
    pnet.summary()
    # dataset = t3d.datasets.Kitti(path='tensorflow3d/assets/datasets/kitti', samples=100000)
    # dataset.render('hello')
    # dataset = dataset.build()
    #
    # for d in dataset.take(1):
    #     print(len(d[0]))
    #     pcd = t3d.representation.PointCloud(points=d[1])
    #     pcd.render()
    # pcd = t3d.representation.PointCloud()
    # pcd = pcd.load('tensorflow3d/tests/formats/ascii/xyzi.txt', mode='xyzi')
    # #
    # # print(pcd.shape)
    # # pcd.sample(6912)
    # # pnet2 = t3d.models.PointNet2(name='pointnet++').build(input_shape=(None, 3))
    # #
    # # print("Inferencing")
    # # result = pnet2(tf.expand_dims(pcd.points, axis=0))
    # # pcd.points = result.numpy()[0, ...]
    # pcd.render(values=True)

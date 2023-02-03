"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import polyscope as ps
import numpy as np
import glob
import tensorflow as tf
class PointCloud:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.points = kwargs.get('points', [])
        self.colors = kwargs.get('colors', [])
        self.values = kwargs.get('values', [])
        self.normals = kwargs.get('normals', [])
        self.shape = ()

    def load(self, path, mode: str = 'xyz'):
        """
        @ops: Load the data into points, colors, values or normals
        @args:
            path: Path to the file containing point cloud data
                type: Str
            mode: Stored point cloud data format
                type: Str
        @return: A PointCloud object or a rejection
            type: PointCloud / Bool
        """

        with open(path) as file:
            try:
                line = file.readline()
                values, colors, normals, points = [], [], [], []
                while line:
                    line = line.split(" ")
                    line = list(np.array(line).astype(np.float))

                    if mode == 'xyzi':
                        values.append(line[-1])

                    elif mode == 'xyzrgb':
                        colors.append(line[3:])

                    elif mode == 'xyznxnynz':
                        normals.append(line[3:])

                    points.append(line[:3])
                    line = file.readline()

                self.points = np.array(points)
                self.values = np.array(values)
                self.colors = np.array(colors)
                self.normals = np.array(normals)
                self.shape = self.points.shape
                return self
            except Exception:
                raise Exception('Unable to read file')

    def sample(self, num: int = 1024):
        randi = np.random.randint(0, high=len(self.points), size=num, dtype=int)
        if self.points is not None or self.points != []:
            self.points = self.points[randi]
            self.shape = self.points.shape
        if np.size(self.colors):
            self.colors = self.colors[randi]

        if np.size(self.values):
            self.values = self.values[randi]

    def render(self, paths: list = [], colors: bool = False, values: bool = False, name: str = 'default', animate=False):
        """
        @ops: Render the point cloud including color, intensity value and surface normals
        @args:
            name: Name to the rendered plot
                type: Str
        @return: None
        """

        ps.init()
        ps.set_up_dir("z_up")
        if self.points is not None or self.points != []:
            ps_cloud = ps.register_point_cloud(name, self.points)
            ps_cloud.set_radius(0.00042)
            if colors:
                ps_cloud.add_color_quantity(name, self.colors)
            if values:
                ps_cloud.add_scalar_quantity(name, self.values, enabled=True, cmap='turbo')
        else:
            raise "NoPointCloudData"
        if animate:
            for path in paths:
                print(path)
                self.load(path, mode='xyzi')
                self.sample()
                ps_cloud.update_point_positions(self.points)
        ps.show()    
        del ps_cloud


if __name__ == "__main__":
    pcd = PointCloud()
    pcd.load('assets/datasets/data/0000000000.txt', mode='xyzi')
    pcd.render(paths=tf.io.gfile.glob("/home/asimbluemoon/custom-op/assets/datasets/data/*.txt"), animate=True)
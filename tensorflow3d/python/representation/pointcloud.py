"""
    For non-commercial use.
    (c) All rights reserved.

    This package is licensed under MIT
    license found in the root directory.

    @author: Mohammad Asim

"""
import polyscope as ps
import numpy as np


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
                while line:
                    line = line.split(" ")
                    line = list(np.array(line).astype(np.float))

                    if mode == 'xyzi':
                        self.values.append(line[-1])

                    elif mode == 'xyzrgb':
                        self.colors.append(line[3:])

                    elif mode == 'xyznxnynz':
                        self.normals.append(line[3:])

                    self.points.append(line[:3])
                    line = file.readline()

                self.points = np.array(self.points)
                self.values = np.array(self.values)
                self.colors = np.array(self.colors)
                self.normals = np.array(self.normals)
                self.shape = self.points.shape
                return self
            except Exception:
                raise Exception('Unable to read file')

                return False

    def sample(self, num: int = 1024):
        randi = np.random.randint(0, high=len(self.points), size=num, dtype=int)
        if self.points is not None or self.points != []:
            self.points = self.points[randi]
            self.shape = self.points.shape
        if np.size(self.colors):
            self.colors = self.colors[randi]

        if np.size(self.values):
            self.values = self.values[randi]

    def render(self, paths: list = [], colors: bool = False, values: bool = False, name: str = 'default'):
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

        ps.show()
        del ps_cloud



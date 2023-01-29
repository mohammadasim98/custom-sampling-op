# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for time_two ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from tensorflow3d.python.ops import fps_ops
except ImportError:
  import fps_ops


class FarthestPointSampleTest(test.TestCase):

  @test_util.run_gpu_only
  def testFarthestPointSample(self):
    with self.test_session():
      pt_sample=np.random.rand(1, 2048, 3).astype('float32')
      with ops.device("/gpu:0"):
        print(fps_ops.farthest_point_sample(1024, pt_sample))
        self.assertAllClose(
            fps_ops.farthest_point_sample(1024, pt_sample)
        )


if __name__ == '__main__':
  test.main()
  print("Test")

Projected originally extended from https://github.com/tensorflow/custom-op.git

@Modifier: Mohammad Asim

# TensorFlow3D
This guide allow the user to build and install the 3D extension to TensorFlow with customized CPU and GPU ops for several 3D applications and represenations (Currently working with point clouds are possible). The build process is platform dependent and hence will generate python wheels for specifc architectures. 

This guide currently supports Ubuntu and Windows custom ops, and it includes examples for both CPU and GPU ops.

Starting from Aug 1, 2019, nightly previews `tf-nightly` and `tf-nightly-gpu`, as well as
official releases `tensorflow` and `tensorflow-gpu` past version 1.14.0 are now built with a
different environment (Ubuntu 16.04 compared to Ubuntu 14.04, for example) as part of our effort to make TensorFlow's pip pacakges
manylinux2010 compatible. To help you building custom ops on linux, here we provide our toolchain in the format of a combination of a Docker image and bazel configurations.  Please check the table below for the Docker image name needed to build your custom ops.

|          |          CPU custom op          |          GPU custom op         |
|----------|:-------------------------------:|:------------------------------:|
| TF >= 2.3   |   2.3.0-custom-op-ubuntu16  |    2.3.0-custom-op-gpu-ubuntu16    |

Note: all above Docker images have prefix `tensorflow/tensorflow:`

The bazel configurations are included as part of this repository.


### For Windows Users
You can skip this section if you are not building on Windows. If you are building custom ops for Windows platform, you will need similar setup as building TensorFlow from source mentioned [here](https://www.tensorflow.org/install/source_windows). Additionally, you can skip all the Docker steps from the instructions below. Otherwise, the bazel commands to build and test custom ops stay the same.

### Setup Docker Container
You are going to build the op inside a Docker container. Pull the provided Docker image from TensorFlow's Docker hub and start a container.

Use the following command if the TensorFlow pip package you are building
And the following instead if it is manylinux2010 compatible:

```bash
  docker pull tensorflow/tensorflow:custom-op-ubuntu16
  docker run -it tensorflow/tensorflow:custom-op-ubuntu16 /bin/bash
```
Next, set up a Docker container using the provided Docker image for building and testing the ops. We provide two sets of Docker images for different versions of pip packages. If the pip package you are building against was released before Aug 1, 2019 and has manylinux1 tag, please use Docker images `tensorflow/tensorflow:custom-op-ubuntu14` and `tensorflow/tensorflow:custom-op-gpu-ubuntu14`, which are based on Ubuntu 14.04. Otherwise, for the newer manylinux2010 packages, please use Docker images `tensorflow/tensorflow:custom-op-ubuntu16` and `tensorflow/tensorflow:custom-op-gpu-ubuntu16` instead. All Docker images come with Bazel pre-installed, as well as the corresponding toolchain used for building the released TensorFlow pacakges. We have seen many cases where dependency version differences and ABI incompatibilities cause the custom op extension users build to not work properly with TensorFlow's released pip packages. Therefore, it is *highly recommended* to use the provided Docker image to build your custom op. To get the CPU Docker image, run one of the following command based on which pip package you are building against:
```bash
# For manylinux2010
docker pull tensorflow/tensorflow:custom-op-ubuntu16
```

For GPU, run 
```bash
# For manylinux2010
docker pull tensorflow/tensorflow:custom-op-gpu-ubuntu16
```
Then run a container either directly 
```bash
  docker run -it <your-container> /bin/bash
```
Or you might want to use Docker volumes to map a `work_dir` from host to the container, so that you can edit files on the host, and build with the latest changes in the Docker container. To do so, run the following for CPU
```bash
# For manylinux2010
docker run -it -v ${PWD}:/<working-directory> -w /<working-directory>  <your-container>
```

For GPU, you want to use `nvidia-docker`:
```bash
# For manylinux2010
docker run --runtime=nvidia --privileged  -it -v ${PWD}:/working_dir -w /working_dir  <your-container>

```
Inside the Docker container, clone this repository.
```bash
git clone https://github.com/mohammadasim98/tensorflow3d.git
cd tensorflow3d
```
#### Configure
Last step before starting implementing the ops, you want to set up the build environment. The custom ops will need to depend on TensorFlow headers and shared library libtensorflow_framework.so, which are distributed with TensorFlow official pip package. If you would like to use Bazel to build your ops, you might also want to set a few action_envs so that Bazel can find the installed TensorFlow. We provide a `configure` script that does these for you. Simply run `./configure.sh` in the docker container and you are good to go.
```bash
  ./configure.sh
```

### Build and install PIP Package
You can build the pip package and install it with the following script.

```bash
  sh ./build.sh
```



### Template Overview
First let's go through a quick overview of the folder structure and naming convention of this custom library.
```
├── gpu  # Set up crosstool and CUDA libraries for Nvidia GPU, only needed for GPU ops
│   ├── crosstool/
│   ├── cuda/
│   ├── BUILD
│   └── cuda_configure.bzl
├── tensorflow3d  # Main library directory similar to tensorflow implementation
│   ├── cc
│   │   ├── kernels  # op kernel implementation
│   │   │   |── XXX.h
│   │   │   |── XXX_kernels.cc
│   │   │   |── XXX_kernels.cu.cc  # GPU kernel
|   |   |   :   ...
│   │   └── ops  # op interface definition
│   │       |── XXX_ops.cc
|   |       :   ...
│   ├── python
│   │   ├── ops
│   │   │   ├── __init__.py
│   │   │   ├── XXX_ops.py   # Load and extend the low-level ops in python
│   │   │   |── XXX_ops_test.py  # tests for ops
|   |   |   :   ...
|   |   ├── layers
|   |   ├── representations
|   |   ├── models
|   |   :   ...
│   │   └── __init__.py
|   ├── main.py # For rapid prototyping and testing
    ├── flownet3d.ipynb # Example implementation of FlowNet3D model
│   ├── BUILD  # BUILD file for all op targets
│   └── __init__.py  # top level __init__ file that imports the ops and custom APIs
|
├── tf  # Set up TensorFlow pip package as external dependency for Bazel
│   ├── BUILD
│   ├── BUILD.tpl
│   └── tf_configure.bzl
|
├── BUILD  # top level Bazel BUILD file that contains pip package build target
├── build_pip_pkg.sh  # script to build pip package for Bazel and Makefile
├── configure.sh  # script to install TensorFlow and setup action_env for Bazel
├── LICENSE
├── setup.py  # file for creating pip package
├── MANIFEST.in  # files for creating pip package
├── README.md
└── WORKSPACE  # Used by Bazel to specify tensorflow pip package as an external dependency

```




### Add Op Implementation
Now you are ready to implement your op. Following the instructions at [Adding a New Op](https://www.tensorflow.org/extend/adding_an_op), add definition of your op interface under `tensorflow3d/cc/ops/` and kernel implementation under `<your_op>/cc/kernels/`.


### FAQ

Here are some issues our users have ran into and possible solutions. Feel free to send us a PR to add more entries.


| Issue  |  How to? |
|---|---|
|  Do I need both the toolchain and the docker image? | Yes, you will need both to get the same setup we use to build TensorFlow's official pip package. |
|  How do I also create a manylinux2010 binary? | You can use [auditwheel](https://github.com/pypa/auditwheel) version 2.0.0 or newer.  |
|  What do I do if I get `ValueError: Cannot repair wheel, because required library "libtensorflow_framework.so.1" could not be located` or `ValueError: Cannot repair wheel, because required library "libtensorflow_framework.so.2" could not be located` with auditwheel? | Please see [this related issue](https://github.com/tensorflow/tensorflow/issues/31807).  |
| What do I do if I get `In file included from tensorflow_time_two/cc/kernels/time_two_kernels.cu.cc:21:0: /usr/local/lib/python3.6/dist-packages/tensorflow/include/tensorflow/core/util/gpu_kernel_helper.h:22:10: fatal error: third_party/gpus/cuda/include/cuda_fp16.h: No such file or directory` | Copy the CUDA header files to target directory. `mkdir -p /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include && cp -r /usr/local/cuda/targets/x86_64-linux/include/* /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include` |

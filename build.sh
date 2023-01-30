bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts/
pip3 uninstall tensorflow3d -y
pip3 install artifacts/*.whl
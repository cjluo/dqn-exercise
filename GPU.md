# Enable Tensorflow on Pascal GPU

Pascal GPUs like 1060, 1070 and 1080 series are not supported by default
tensorflow package (Ver 0.9). Tensorflow needs to be compiled from source to
enable GPU acceleration

## Nvidia Driver Installation

```
sudo apt-get purge nvidia-*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
apt-cache search nvidia-*
sudo apt-get install nvidia-<latest version>
```

You could check driver installation with **nvidia-smi** and see something like
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 370.23                 Driver Version: 370.23                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1070    Off  | 0000:01:00.0      On |                  N/A |
|  0%   38C    P8    11W / 230W |    471MiB /  8110MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
```

## Cuda Installation

Pascal GPU needs cuda version > 8.0

1. Download cuda 8.0 **runtime (local) file** from [Nvidia website](https://developer.nvidia.com/cuda-release-candidate-download),
  together with additional patches provided in the website.
  Note: you may need to register a Nvidia developer account.

2. Install cuda together with patches.
   First, install the main cuda runtime
```
sudo sh cuda_8.0.27_linux.run
```
   Install the CUDA Toolkit **only**, don't install old driver.
```
Do you accept the previously read EULA?
accept/decline/quit: accept
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 361.77?
(y)es/(n)o/(q)uit: n
Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y
Enter Toolkit Location
[ default is /usr/local/cuda-8.0 ]:
Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y
```
   Install additional patches
```
sudo cuda_8.0.27.1_linux.run
```
3. Edit ~/.bashrc and add the following lines
```
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
```
   After this you should be able to check **nvcc -V** and see something like
```
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Wed_May__4_21:01:56_CDT_2016
Cuda compilation tools, release 8.0, V8.0.26
```

## Cudnn Installation

1. Download latest CUDNN **cuDNN v5.1 Library for Linux** from [Nvidia website](https://developer.nvidia.com/rdp/cudnn-download)

2. Extract the downloaded package and save the extracted cuda folder to
   overwrite /usr/local/cuda/
```
sudo cp -r cuda/* /usr/local/cuda/
```

## Bazel Installation

[Bazel](https://bazel.io/) is the build tool for tensorflow. Please follow
the [installation page](https://bazel.io/docs/install.html)
to complete bazel installation.

## Compile Tensorflow

1. Download tensorflow and checkout the master branch
```
git clone https://github.com/tensorflow/tensorflow.git
git checkout master
```

2. Install build dependency packages
```
sudo apt-get install python-numpy swig python-dev python-wheel
```

3. Configure and build [instruction](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
```
cd tensorflow
./configure
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install --upgrade /tmp/tensorflow_pkg/tensorflow-0.10.0rc0-py2-none-any.whl
```

4. Validation installation
Run [training example in tensorflow](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/image)
and make sure the error rate or loss is decreasing. Also check with
**nvidia-smi** for GPU utilization.

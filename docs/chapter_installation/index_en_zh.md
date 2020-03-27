# Installation 安装

:label:`chap_installation`

In order to get you up and running for hands-on learning experience, we need to set you up with an environment for running Python, Jupyter notebooks, the relevant libraries, and the code needed to run the book itself.

为了使您起步并获得动手学习的经验，我们需要为您设置一个运行Python，Jupyter笔记本，相关库以及运行本书所需的代码的环境。

## Installing Miniconda 安装Miniconda

The simplest way to get going will be to install [Miniconda](https://conda.io/en/latest/miniconda.html). The Python 3.x version is required. You can skip the following steps if conda has already been installed. Download the corresponding Miniconda sh file from the website and then execute the installation from the command line using `sh <FILENAME> -b`. For macOS users:

最简单的方法是安装[Miniconda](https://conda.io/en/latest/miniconda.html)。 需要Python 3.x版本。 如果已经安装了conda，则可以跳过以下步骤。 从网站上下载相应的Miniconda sh文件，然后使用`sh <FILENAME> -b`从命令行执行安装。 对于macOS用户：

```bash
# The file name is subject to changes
# 文件名可能会更改
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

For Linux users:

Linux用户

```bash
# The file name is subject to changes
# 文件名可能会更改
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Next, initialize the shell so we can run `conda` directly.

接着初始化shell，使得我们可以直接运行conda

```bash
~/miniconda3/bin/conda init
```

Now close and re-open your current shell. You should be able to create a new environment as following:

现在关闭并重新打开你的shell。你现在应该能使用如下命令创建一个新环境。

```bash
conda create --name d2l -y
```

## Downloading the D2L Notebooks 下载D2L笔记本

Next, we need to download the code of this book. You can use the [link](https://d2l.ai/d2l-en-0.7.1.zip) to download and unzip the code. Alternatively, if you have `unzip`(otherwise run `sudo apt install unzip`) available:

接着，你需要下载本书的代码。你可以使用这个[链接](https://d2l.ai/d2l-en-0.7.1.zip) 下载和解压代码。另外，如果可以使用 `unzip` （或者使用 `sudo apt install unzip` 安装）：

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Now we will want to activate the `d2l` environment and install `pip`. Enter `y` for the queries that follow this command.

现在我们激活d2l环境并安装 `pip` 。输入 `y` 作为此命令后面的查询。

```bash
conda activate d2l
conda install python=3.7 pip -y
```

## Installing MXNet and the `d2l` Package 安装MXNet和`d2l` 包

Before installing MXNet, please first check whether or not you have proper GPUs on your machine (the GPUs that power the display on a standard laptop do not count for our purposes). If you are installing on a GPU server, proceed to :ref:`subsec_gpu` for instructions to install a GPU-supported MXNet.

在安装MXNet之前，请首先检查您的计算机上是否有正确的GPU（不包括标准笔记本电脑显示使用的GPU）。 如果要在GPU服务器上安装，请转到[GPU支持](#GPU Support)以获取有关安装GPU支持的MXNet的说明。

Otherwise, you can install the CPU version. That will be more than enough horsepower to get you through the first few chapters but you will want to access GPUs before running larger models.

否则，您可以安装CPU版本。这将足够使您轻松完成前几章，但是您将需要在运行较大模型之前使用GPU。

```bash
pip install mxnet==1.6.0
```

We also install the `d2l` package that encapsulates frequently used functions and classes in this book.

我们还安装了 `d2l` 包，它封装了本书中经常使用的函数和类。

```bash
pip install git+https://github.com/d2l-ai/d2l-en
# 或者 pip insatll d2l
```

Once they are installed, we now open the Jupyter notebook by running:

当这些的安装完成，你就可以运行以下命令打开Jupyter笔记本

```bash
jupyter notebook
```

At this point, you can open [http://localhost:8888](http://localhost:8888/) (it usually opens automatically) in your Web browser. Then we can run the code for each section of the book. Please always execute `conda activate d2l` to activate the runtime environment before running the code of the book or updating MXNet or the `d2l` package. To exit the environment, run `conda deactivate`.

此时，您可以在Web浏览器中打开[http://localhost:8888](http:// localhost:8888/)（通常会自动打开）。 然后，我们可以为本书的每个部分运行代码。 在运行本书代码或更新MXNet或`d2l`软件包之前，请始终执行`conda activate d2l`以激活运行时环境。 要退出环境，运行`conda deactivate`。

## Upgrading to a New Version 更新版本

Both this book and MXNet keep being improved. Please check a new version from time to time.本书和MXNet都在不断改进。请不时检查新版本。

1. The URL https://d2l.ai/d2l-en.zip always points to the latest contents.
2. 链接  https://d2l.ai/d2l-en.zip  始终指向最新内容。
3. Please upgrade the `d2l` package by `pip install d2l --upgrade`.
4. 使用 `pip install d2l --upgrade` 更新 `d2l` 包。
5. For the CPU version, MXNet can be upgraded by `pip install -U --pre mxnet`.
6. MXNet的CPU版本，可以使用 `pip install -U --pre mxnet` 更新

## GPU Support GPU 支持

:label:`subsec_gpu`

By default, MXNet is installed without GPU support to ensure that it will run on any computer (including most laptops). Part of this book requires or recommends running with GPU. If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads), then you should install a GPU-enabled MXNet. If you have installed the CPU-only version, you may need to remove it first by running:

默认情况下，安装的MXNet不支持GPU，以确保它可以在任何计算机（包括大多数笔记本电脑）上运行。 本书的一部分要求或建议与GPU一起运行。 如果您的计算机装有NVIDIA图形卡并且已安装[CUDA](https://developer.nvidia.com/cuda-downloads)，则应安装支持GPU的MXNet。 如果安装了CPU版本，则可能需要先通过运行以下命令将其删除：


```bash
pip uninstall mxnet
```

Then we need to find the CUDA version you installed. You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`. Assume that you have installed CUDA 10.1, then you can install MXNet with the following command:

然后，我们需要找到您安装的CUDA版本。 您可以通过`nvcc --version`或`cat/usr/local/cuda/version.txt`进行检查。 假设您已安装CUDA 10.1，则可以使用以下命令安装MXNet：

```bash
# For Windows users
# windows 用户
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
# Linux和macOS用户
pip install mxnet-cu101==1.6.0
```

Like the CPU version, the GPU-enabled MXNet can be upgraded by `pip install -U --pre mxnet-cu101`. You may change the last digits according to your CUDA version, e.g., `cu100` for CUDA 10.0 and `cu90` for CUDA 9.0. You can find all available MXNet versions via `pip search mxnet`.

像CPU版本一样，支持GPU的MXNet可以通过`pip install -U --pre mxnet-cu101`进行升级。 您可以根据自己的CUDA版本更改最后一位数字，例如，对于CUDA 10.0，为“ cu100”，对于CUDA 9.0，为“ cu90”。 您可以通过`pip search mxnet`找到所有可用的MXNet版本。

## Exercises 练习

1. Download the code for the book and install the runtime environment.
2. 下载本书的代码，安装运行环境。

## [Discussions](https://discuss.mxnet.io/t/2315) 讨论

![](../img/qr_install.svg)
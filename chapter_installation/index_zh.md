# 安装

:label:`chap_installation`

为了使您起步并获得动手学习的经验，我们需要为您设置一个运行Python，Jupyter笔记本，相关库以及运行本书所需的代码的环境。

##  安装Miniconda

最简单的方法是安装[Miniconda](https://conda.io/en/latest/miniconda.html)。 需要Python 3.x版本。 如果已经安装了conda，则可以跳过以下步骤。 从网站上下载相应的Miniconda sh文件，然后使用`sh <FILENAME> -b`从命令行执行安装。 对于macOS用户：

```bash
# The file name is subject to changes
# 文件名可能会更改
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

Linux用户

```bash
# The file name is subject to changes
# 文件名可能会更改
sh Miniconda3-latest-Linux-x86_64.sh -b
```

接着初始化shell，使得我们可以直接运行conda

```bash
~/miniconda3/bin/conda init
```

现在关闭并重新打开你的shell。你现在应该能使用如下命令创建一个新环境。

```bash
conda create --name d2l -y
```

## Downloading the D2L Notebooks 下载D2L笔记本

接着，你需要下载本书的代码。你可以使用这个[链接](https://d2l.ai/d2l-en-0.7.1.zip) 下载和解压代码。另外，如果可以使用 `unzip` （或者使用 `sudo apt install unzip` 安装）：

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

现在我们激活d2l环境并安装 `pip` 。输入 `y` 作为此命令后面的查询。

```bash
conda activate d2l
conda install python=3.7 pip -y
```

## 安装MXNet和`d2l` 包

在安装MXNet之前，请首先检查您的计算机上是否有正确的GPU（不包括标准笔记本电脑显示使用的GPU）。 如果要在GPU服务器上安装，请转到[GPU支持](#GPU Support)以获取有关安装GPU支持的MXNet的说明。

否则，您可以安装CPU版本。这将足够使您轻松完成前几章，但是您将需要在运行较大模型之前使用GPU。

```bash
pip install mxnet==1.6.0
```

我们还安装了 `d2l` 包，它封装了本书中经常使用的函数和类。

```bash
pip install git+https://github.com/d2l-ai/d2l-en
# 或者 pip insatll d2l
```

当这些的安装完成，你就可以运行以下命令打开Jupyter笔记本

```bash
jupyter notebook
```

此时，您可以在Web浏览器中打开[http://localhost:8888](http:// localhost:8888/)（通常会自动打开）。 然后，我们可以为本书的每个部分运行代码。 在运行本书代码或更新MXNet或`d2l`软件包之前，请始终执行`conda activate d2l`以激活运行时环境。 要退出环境，运行`conda deactivate`。

## 更新版本

本书和MXNet都在不断改进。请不时检查新版本。

1. 链接  https://d2l.ai/d2l-en.zip  始终指向最新内容。
2. 使用 `pip install d2l --upgrade` 更新 `d2l` 包。
3. MXNet的CPU版本，可以使用 `pip install -U --pre mxnet` 更新

##  GPU 支持

:label:`subsec_gpu`

默认情况下，安装的MXNet不支持GPU，以确保它可以在任何计算机（包括大多数笔记本电脑）上运行。 本书的一部分要求或建议与GPU一起运行。 如果您的计算机装有NVIDIA图形卡并且已安装[CUDA](https://developer.nvidia.com/cuda-downloads)，则应安装支持GPU的MXNet。 如果安装了CPU版本，则可能需要先通过运行以下命令将其删除：


```bash
pip uninstall mxnet
```

然后，我们需要找到您安装的CUDA版本。 您可以通过`nvcc --version`或`cat/usr/local/cuda/version.txt`进行检查。 假设您已安装CUDA 10.1，则可以使用以下命令安装MXNet：

```bash
# For Windows users
# windows 用户
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
# Linux和macOS用户
pip install mxnet-cu101==1.6.0
```

像CPU版本一样，支持GPU的MXNet可以通过`pip install -U --pre mxnet-cu101`进行升级。 您可以根据自己的CUDA版本更改最后一位数字，例如，对于CUDA 10.0，为“ cu100”，对于CUDA 9.0，为“ cu90”。 您可以通过`pip search mxnet`找到所有可用的MXNet版本。

## Exercises 练习

1. 下载本书的代码，安装运行环境。

## [Discussions](https://discuss.mxnet.io/t/2315) 讨论

![](../img/qr_install.svg)
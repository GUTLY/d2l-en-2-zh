#  Preliminaries
:label:`chap_preliminaries`

To get started with deep learning, we will need to develop a few basic skills. All machine learning is concerned with extracting information from data. So we will begin by learning the practical skills for storing, manipulating, and preprocessing data.

要开始深度学习，我们将需要发展一些基本技能。所有机器学习都与从数据中提取信息有关。因此，我们将从学习存储，处理和预处理数据的实用技能开始。

Moreover, machine learning typically requires working with large datasets, which we can think of as tables, where the rows correspond to examples and the columns correspond to attributes. Linear algebra gives us a powerful set of techniques for working with tabular data. We will not go too far into the weeds but rather focus on the basic of matrix operations and their implementation.

此外，机器学习通常需要处理大型数据集，我们可以将其视为表，其中行对应于示例，列对应于属性。线性代数为我们提供了一组处理表格数据的强大技术。我们会将重点放在矩阵运算及其实现的基础上，而忽略其他。

Additionally, deep learning is all about optimization. We have a model with some parameters and we want to find those that fit our data *the best*. Determining which way to move each parameter at each step of an algorithm requires a little bit of calculus, which will be briefly introduced. Fortunately, the `autograd` package automatically computes differentiation for us, and we will cover it next.

此外，深度学习全都与优化有关。我们有一个带有一些参数的模型，我们希望找到最适合我们数据的参数。确定在算法的每个步骤中移动每个参数的方式需要一点微积分，这将作简要介绍。幸运的是，`autograd`软件包会自动为我们计算微分，接下来我们将介绍它。

Next, machine learning is concerned with making predictions: what is the likely value of some unknown attribute, given the information that we observe? To reason rigorously under uncertainty we will need to invoke the language of probability.

接下来，机器学习与进行预测有关：根据我们观察到的信息，某些未知属性的可能值是多少？为了在不确定性下进行严格推理，我们将需要调用概率语言。

In the end, the official documentation provides plenty of descriptions and examples that are beyond this book. To conclude the chapter, we will show you how to look up documentation for the needed information.

最后，官方文档提供了本书以外的大量描述和示例。结束本章，我们将向你展示如何查找文档以获取所需的信息。

This book has kept the mathematical content to the minimum necessary to get a proper understanding of deep learning. However, it does not mean that this book is mathematics free. Thus, this chapter provides a rapid introduction to basic and frequently-used mathematics to allow anyone to understand at least *most* of the mathematical content of the book. If you wish to understand *all* of the mathematical content, further reviewing :numref:`chap_appendix_math` should be sufficient.

本书将数学内容保持在正确理解深度学习的最低限度。但是，这并不意味着这些数学知识已经足够。因此，本章对基础数学和常用数学进行了快速介绍，以使任何人都至少可以理解本书的大部分数学内容。如果您想了解所有的数学内容，那么进一步回顾一下参见`chap_appendix_math`。

- 2.1. [Data Manipulation 数据处理](./ndarray_en_zh.md)
  - 2.1.1 Getting Started 入门
  - 2.1.2. Operations 运作方式
  - 2.1.3. Broadcasting Mechanism 广播机制
  - 2.1.4. Indexing and Slicing 索引和切片
  - 2.1.5. Saving Memory 节省内存
  - 2.1.6. Conversion to Other Python Objects 转换为其他Python对象
  - 2.1.7. Summary 摘要
  - 2.1.8. Exercises 练习
  - 2.1.9. Discussions 讨论
- 2.2. [Data Preprocessing 数据预处理](./pandas_en_zh.md)
  - 2.2.1. Reading the Dataset 读取数据集
  - 2.2.2. Handling Missing Data 处理丢失的数据
  - 2.2.3. Conversion to the `ndarray` Format 转换为`ndarray` 格式
  - 2.2.4. Summary 摘要
  - 2.2.5. Exercises 练习
  - 2.2.6. Discussions 讨论
- 2.3. [Linear Algebra 线性代数](./linear-algebra_en_zh.md)
  - 2.3.1. Scalars 标量
  - 2.3.2. Vectors 向量
  - 2.3.3. Matrices 矩阵
  - 2.3.4. Tensors 张量
  - 2.3.5. Basic Properties of Tensor Arithmetic 张量的算术基本性质
  - 2.3.6. Reduction 减少
  - 2.3.7. Dot Products 点产品
  - 2.3.8. Matrix-Vector Products 矩阵向量乘积
  - 2.3.9. Matrix-Matrix Multiplication 矩阵-矩阵乘法
  - 2.3.10. Norms 规范
  - 2.3.11. More on Linear Algebra 有关线性代数的更多信息
  - 2.3.12. Summary 摘要
  - 2.3.13. Exercises 练习
  - 2.3.14. Discussions 讨论
- 2.4. [Calculus 微积分](./calculus_en_zh.md)
  - 2.4.1. Derivatives and Differentiation 导数和微分
  - 2.4.2. Partial Derivatives 偏导数
  - 2.4.3. Gradients 梯度
  - 2.4.4. Chain Rule 链式法则
  - 2.4.5. Summary 摘要
  - 2.4.6. Exercises 练习
  - 2.4.7. Discussions 讨论
- 2.5. [Automatic Differentiation 自动微分](./autograd_en_zh.md)
  - 2.5.1. A Simple Example 一个简单的例子
  - 2.5.2. Backward for Non-Scalar Variables 向后非标量变量
  - 2.5.3. Detaching Computation  分离计算
  - 2.5.4. Computing the Gradient of Python Control Flow 计算Python控制流的梯度
  - 2.5.5. Training Mode and Prediction Mode  训练模式和预测模式
  - 2.5.6. Summary  摘要
  - 2.5.7. Exercises 练习
  - 2.5.8. Discussions 讨论
- 2.6. [Probability 概率论](./probability_en_zh.md)
  - 2.6.1. Basic Probability Theory 基本概率论
  - 2.6.2. Dealing with Multiple Random Variables  处理多个随机变量
  - 2.6.3. Expectation and Variance 期望与差异
  - 2.6.4. Summary 摘要
  - 2.6.5. Exercises 练习
  - 2.6.6. Discussions 讨论
- 2.7. [Documentation 文档](./lookup-api_en_zh.md)
  - 2.7.1. Finding All the Functions and Classes in a Module  查找模块中的所有函数和类
  - 2.7.2. Finding the Usage of Specific Functions and Classes  查找特定功能和类的用法
  - 2.7.3. API Documentation  API文档
  - 2.7.4. Summary 摘要
  - 2.7.5. Exercises 练习
  - 2.7.6. Discussions 讨论
# Notation 标记

:label:`chap_notation`

The notation used throughout this book is summarized below.

全书使用的标记总结如下


## Numbers 数

* $x$: A scalar 标量
* $\mathbf{x}$: A vector 矢量
* $\mathbf{X}$: A matrix 矩阵
* $\mathsf{X}$: A tensor 张量
* $\mathbf{I}$: An identity matrix 单位矩阵
* $x_i$, $[\mathbf{x}]_i$: The $i^\mathrm{th}$ element of vector $\mathbf{x}$ 矢量的第i个元素
* $x_{ij}$, $[\mathbf{X}]_{ij}$: The element of matrix $\mathbf{X}$ at row $i$ and column $j$ 矩阵的第i行第j列


## Set Theory  集合


* $\mathcal{X}$: A set 集合
* $\mathbb{Z}$: The set of integers 整数集
* $\mathbb{R}$: The set of real numbers 实数集
* $\mathbb{R}^n$: The set of $n$-dimensional vectors of real numbers 实数的 $n$ 维向量集
* $\mathbb{R}^{a\times b}$: The set of matrices of real numbers with $a$ rows and $b$ columns a行b列的实数矩阵集合
* $\mathcal{A}\cup\mathcal{B}$: Union of sets $\mathcal{A}$ and $\mathcal{B}$ 集合 $\mathcal{A}$ 并集合  $\mathcal{B}$ 
* $\mathcal{A}\cap\mathcal{B}$: Intersection of sets $\mathcal{A}$ and $\mathcal{B}$ 集合 $\mathcal{A}$ 交集合 $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: Subtraction of set $\mathcal{B}$ from set $\mathcal{A}$ 集合 $\mathcal{A}$ -集合 $\mathcal{B}$


## Functions and Operators 函数和操作符


* $f(\cdot)$: A function 一个函数
* $\log(\cdot)$: The natural logarithm 自然对数
* $\exp(\cdot)$: The exponential function  指数函数      
* $\mathbf{1}_\mathcal{X}$: The indicator function 指示函数
* $\mathbf{(\cdot)}^\top$: Transpose of a vector or a matrix    矢量或矩阵的转置         
* $\mathbf{X}^{-1}$: Inverse of matrix $\mathbf{X}$ $\mathbf{X}$的逆矩阵
* $\odot$: Hadamard (elementwise) product 哈达玛积
* $[\cdot, \cdot]$: Concatenation 串联
* $\lvert \mathcal{X} \rvert$: Cardinality of set $\mathcal{X}$  $\mathcal{X}$集的基数 
* $\|\cdot\|_p$: $\ell_p$ norm  $\ell_p$  规范                                
* $\|\cdot\|$: $\ell_2$ norm   $\ell_2$ 规范
* $\langle \mathbf{x}, \mathbf{y} \rangle$: Dot product of vectors $\mathbf{x}$ and $\mathbf{y}$  矢量 $\mathbf{x}$ 和 $\mathbf{y}$ 的点乘
* $\sum$: Series addition         求和               
* $\prod$: Series multiplication     级数乘法       


## Calculus 微积分

* $\frac{dy}{dx}$: Derivative of $y$ with respect to $x$     $y$ 对 $x$ 的导数
* $\frac{\partial y}{\partial x}$: Partial derivative of $y$ with respect to $x$  $y$ 对 $x$ 的偏导
* $\nabla_{\mathbf{x}} y$: Gradient of $y$ with respect to $\mathbf{x}$  $y$ 对 $x$ 的梯度
* $\int_a^b f(x) \;dx$: Definite integral of $f$ from $a$ to $b$ with respect to $x$  $ f $从$ a $到$ b $相对于$ x $的定积分
* $\int f(x) \;dx$: Indefinite integral of $f$ with respect to $x$ $ f $与$ x $的不定积分

## Probability and Information Theory 概率论和信息论

* $P(\cdot)$: Probability distribution                概率分布                
* $z \sim P$: Random variable $z$ has probability distribution $P$  随机变量 $z$ 具有概率分布$P$ 
* $P(X \mid Y)$: Conditional probability of $X \mid Y$        $ X$∣$Y$ 的条件概率
* $p(x)$: Probability density function 概率密度函数
* ${E}_{x} [f(x)]$: Expectation of $f$ with respect to $x$  f对x的期望
* $X \perp Y$: Random variables $X$ and $Y$ are independent 随机变量 $X$ 和 $Y$ 是独立的
* $X \perp Y \mid Z$: Random variables  $X$  and  $Y$  are conditionally independent given random variable $Z$ 随机变量 $X$ 和 $Y$ 在给定随机变量 $Z$ 的情况下是条件独立的
* $\mathrm{Var}(X)$: Variance of random variable  $X $ 随机变量$X$的方差
* $\sigma_X$: Standard deviation of random variable $X$ 随机变量X的标准偏差
* $\mathrm{Cov}(X, Y)$: Covariance of random variables $X$ and $Y$ 随机变量X和Y的协方差
* $\rho(X, Y)$: Correlation of random variables $X$ and $Y $随机变量 $X$ 和 $Y$ 的相关性
* $H(X)$: Entropy of random variable $X$ 随机变量 $X$ 的熵
* $D_{\mathrm{KL}}(P\|Q)$: KL-divergence of distributions $P$ and $Q$ 分布 $P$ 和 $Q$ 的KL散度

## Complexity 复杂度

* $\mathcal{O}$: Big O notation 大O符号


## [Discussions](https://discuss.mxnet.io/t/4367) 论坛

![](../img/qr_notation.svg)
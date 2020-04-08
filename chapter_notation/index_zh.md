# Notation 标记

:label:`chap_notation`

全书使用的标记总结如下


## 数

* $x$: 标量
* $\mathbf{x}$: 矢量
* $\mathbf{X}$: 矩阵
* $\mathsf{X}$: 张量
* $\mathbf{I}$: 单位矩阵
* $x_i$, $[\mathbf{x}]_i$: 矢量 $\mathbf{x}$ 的第i个元素
* $x_{ij}$, $[\mathbf{X}]_{ij}$:矩阵 $\mathbf{X}$  的第 $i$ 行第 $j$ 列


## 集合


* $\mathcal{X}$: 集合
* $\mathbb{Z}$:  整数集
* $\mathbb{R}$:  实数集
* $\mathbb{R}^n$:  实数的 $n$ 维向量集
* $\mathbb{R}^{a\times b}$:  a行b列的实数矩阵集合
* $\mathcal{A}\cup\mathcal{B}$: 集合 $\mathcal{A}$ 并集合  $\mathcal{B}$ 
* $\mathcal{A}\cap\mathcal{B}$: 集合 $\mathcal{A}$ 交集合 $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$:  集合 $\mathcal{A}$ -集合 $\mathcal{B}$


##  函数和操作符


* $f(\cdot)$: 一个函数
* $\log(\cdot)$:  自然对数
* $\exp(\cdot)$:  指数函数      
* $\mathbf{1}_\mathcal{X}$:  指示函数
* $\mathbf{(\cdot)}^\top$: 矢量或矩阵的转置         
* $\mathbf{X}^{-1}$:  $\mathbf{X}$的逆矩阵
* $\odot$:  哈达玛积
* $[\cdot, \cdot]$:  串联
* $\lvert \mathcal{X} \rvert$: $\mathcal{X}$集的基数 
* $\|\cdot\|_p$:  $\ell_p$  规范                                
* $\|\cdot\|$:   $\ell_2$ 规范
* $\langle \mathbf{x}, \mathbf{y} \rangle$: 矢量 $\mathbf{x}$ 和 $\mathbf{y}$ 的点乘
* $\sum$:   求和               
* $\prod$:   级数乘法       


## 微积分

* $\frac{dy}{dx}$: $y$ 对 $x$ 的导数
* $\frac{\partial y}{\partial x}$:  $y$ 对 $x$ 的偏导
* $\nabla_{\mathbf{x}} y$:  $y$ 对 $x$ 的梯度
* $\int_a^b f(x) \;dx$:  $ f $从$ a $到$ b $相对于$ x $的定积分
* $\int f(x) \;dx$:   $ f $与$ x $的不定积分

## 概率论和信息论

* $P(\cdot)$:  概率分布                
* $z \sim P$:   随机变量 $z$ 具有概率分布$P$ 
* $P(X \mid Y)$:   $ X$∣$Y$ 的条件概率
* $p(x)$:  概率密度函数
* ${E}_{x} [f(x)]$: f 对x的期望
* $X \perp Y$: 随机变量 $X$ 和 $Y$ 是独立的
* $X \perp Y \mid Z$:  随机变量 $X$ 和 $Y$ 在给定随机变量 $Z$ 的情况下是条件独立的
* $\mathrm{Var}(X)$: 随机变量$X$的方差
* $\sigma_X$:   随机变量 $X$ 的标准偏差
* $\mathrm{Cov}(X, Y)$:   随机变量 $X$ 和 $Y$ 的协方差
* $\rho(X, Y)$:  随机变量 $X$ 和 $Y$ 的相关性
* $H(X)$:  随机变量 $X$ 的熵
* $D_{\mathrm{KL}}(P\|Q)$: 分布 $P$ 和 $Q$ 的KL散度

## 复杂度

* $\mathcal{O}$:  大O符号


## [论坛](https://discuss.mxnet.io/t/4367) 

![](../img/qr_notation.svg)
[TOC]

# Preface前言

Just a few years ago, there were no legions of deep learning scientists developing intelligent products and services at major companies and startups. When the youngest among us (the authors) entered the field, machine learning did not command headlines in daily newspapers. Our parents had no idea what machine learning was, let alone why we might prefer it to a career in medicine or law. Machine learning was a forward-looking academic discipline with a narrow set of real-world applications. And those applications, e.g., speech recognition and computer vision, required so much domain knowledge that they were often regarded as separate areas entirely for which machine learning was one small component. Neural networks then, the antecedents of the deep learning models that we focus on in this book, were regarded as outmoded tools.

仅仅几年前，还没有大量的深度学习科学家在大型公司和初创公司中开发智能产品和服务。 当我们当中最年轻的人（作者）进入这一领域时，机器学习并没有成为日报的头条新闻。 我们的父母不知道什么是机器学习，更不用说为什么我们更喜欢从事机器学习而不是医学或法律职业。 机器学习是一门具有前瞻性的学术学科，其实际应用范围非常狭窄。 而且，这些应用程序（例如语音识别和计算机视觉）需要大量领域知识，以至于它们通常被视为完全独立的领域，而机器学习只是其中的一个小组成部分。 接着，神经网络（我们在本书中关注的深度学习模型的前身）在过去被视为过时的工具。

In just the past five years, deep learning has taken the world by surprise, driving rapid progress in fields as diverse as computer vision, natural language processing, automatic speech recognition, reinforcement learning, and statistical modeling. With these advances in hand, we can now build cars that drive themselves with more autonomy than ever before (and less autonomy than some companies might have you believe), smart reply systems that automatically draft the most mundane emails, helping people dig out from oppressively large inboxes, and software agents that dominate the world's best humans at board games like Go, a feat once thought to be decades away. Already, these tools exert ever-wider impacts on industry and society, changing the way movies are made, diseases are diagnosed, and playing a growing role in basic sciences---from astrophysics to biology.

在过去的五年中，深度学习震惊了世界，推动了计算机视觉，自然语言处理，自动语音识别，强化学习和统计建模等各个领域的快速发展。 有了这些先进技术，我们现在可以制造出比以往任何时候都更具智能的汽车（但低于某些公司宣传的），智能回复系统可以自动起草最普通的电子邮件，从而帮助人们从庞大收件箱中解脱，以及在棋盘游戏（例如围棋）上打败人类最优秀的软件，这一壮举曾经被认为尚有几十年才会出现。 这些工具已经对工业和社会产生了越来越广泛的影响，它改变了电影的制作方式，疾病的诊断方法，并在从天体物理学到生物学的基础科学中发挥着越来越重要的作用。

## About This Book关于本书

This book represents our attempt to make deep learning approachable,teaching you the *concepts*, the *context*, and the *code*.

这本书代表了我们试图使深度学习变得容易的尝试，包含*概念*，*内容*和*代码*。

### One Medium Combining Code, Math, and HTML一种结合了代码，数学和HTML的媒介

For any computing technology to reach its full impact, it must be well-understood, well-documented, and supported by mature, well-maintained tools. The key ideas should be clearly distilled, minimizing the onboarding time needing to bring new practitioners up to date. Mature libraries should automate common tasks, and exemplar code should make it easy for practitioners to modify, apply, and extend common applications to suit their needs. Take dynamic web applications as an example. Despite a large number of companies, like Amazon, developing successful database-driven web applications in the 1990s, the potential of this technology to aid creative entrepreneurs has been realized to a far greater degree in the past ten years, owing in part to the development of powerful, well-documented frameworks.

为了使任何计算技术都能发挥其全部作用，必须对它进行充分的理解，充分记录并由成熟，维护良好的工具提供支持。关键思想应该清楚地提炼出来，以最大程度地缩短新学员需要的入职时间。成熟的库应自动执行常见任务，并且示例代码应使从业人员可以轻松地修改，应用和扩展通用应用程序以满足他们的需求。以动态Web应用程序为例，尽管有很多公司，例如亚马逊，在1990年代开发成功的数据库驱动的Web应用程序，在过去的十年中，这种技术帮助创意企业家的潜力已经得到了更大的发展，这在一定程度上要归功于强大而有据可查的框架的发展。

Testing the potential of deep learning presents unique challenges because any single application brings together various disciplines. Applying deep learning requires simultaneously understanding (i) the motivations for casting a problem in a particular way; (ii) the mathematics of a given modeling approach; (iii) the optimization algorithms for fitting the models to data; and (iv) and the engineering required to train models efficiently, navigating the pitfalls of numerical computing and getting the most out of available hardware. Teaching both the critical thinking skills required to formulate problems, the mathematics to solve them, and the software tools to implement those solutions all in one place presents formidable challenges. Our goal in this book is to present a unified resource to bring would-be practitioners up to speed.

测试深度学习的潜力提出了独特的挑战，因为任何单个应用程序都会汇集各种学科。应用深度学习需要同时理解（i）以特定方式提出问题的动机；（ii）给定建模方法的数学；（iii）使模型适合数据的优化算法；（iv）以及有效训练模型所需的工程，克服数值计算的陷阱并充分利用可用的硬件。教授解决问题所需的批判性思维技能，解决问题的数学方法以及在一个地方实施这些解决方案的软件工具都面临着巨大的挑战。我们在本书中的目标是提供一个统一的资源，以使可能的从业人员快速掌握。

We started this book project in July 2017 when we needed to explain MXNet's (then new) Gluon interface to our users. At the time, there were no resources that simultaneously (i) were up to date; (ii) covered the full breadth of modern machine learning with substantial technical depth; and (iii) interleaved exposition of the quality one expects from an engaging textbook with the clean runnable code that one expects to find in hands-on tutorials. We found plenty of code examples for how to use a given deep learning framework (e.g., how to do basic numerical computing with matrices in TensorFlow) or for implementing particular techniques (e.g., code snippets for LeNet, AlexNet, ResNets, etc) scattered across various blog posts and GitHub repositories. However, these examples typically focused on *how* to implement a given approach, but left out the discussion of *why* certain algorithmic decisions are made. While some interactive resources have popped up sporadically to address a particular topic, e.g., the engaging blog posts published on the website [Distill](http://distill.pub/), or personal blogs, they only covered selected topics in deep learning, and often lacked associated code. On the other hand, while several textbooks have emerged, most notably :cite:`Goodfellow.Bengio.Courville.2016`, which offers a comprehensive survey of the concepts behind deep learning, these resources do not marry the descriptions to realizations of the concepts in code, sometimes leaving readers clueless as to how to implement them. Moreover, too many resources are hidden behind the paywalls of commercial course providers.

我们在2017年7月启动了该图书项目，当时我们需要向用户解释MXNet的新接口Gluon。在那时，没有（i）最新的资源； （ii）广泛覆盖现代深度学习技术并具有一定的技术深度；（iii）既是严谨的教科书，又是包含可运行代码的生动的教程。我们找到了许多代码示例，这些代码示例分散在各个地方，如何使用给定的深度学习框架（例如，如何使用TensorFlow中的矩阵进行基本数值计算）或实现特定技术（例如，用于LeNet，AlexNet，ResNets的代码段）各种博客文章和GitHub存储库。但是，这些示例通常集中于*如何*实现给定方法，而没有讨论*为什么*做出某些算法决策。尽管一些互动资源偶尔会弹出以解决特定主题，例如在[Distill](http://distill.pub)网站上发布的引人入胜的博客文章或个人博客，但它们仅涵盖深度学习中的选定主题，并且通常缺少相关代码。另一方面，虽然出现了几本教科书，但最著名的要数Goodfellow、Bengio和Courville的《深度学习》，该书对深度学习背后的概念进行了全面的梳理，这类资源并没有将概念描述与实际代码相结合，有时使读者对如何实现它们一无所知。除此之外，商业课程提供者们虽然制作了众多的优质资源，但它们的付费门槛依然令不少用户望而生畏。

We set out to create a resource that could (1) be freely available for everyone; (2) offer sufficient technical depth to provide a starting point on the path to actually becoming an applied machine learning scientist; (3) include runnable code, showing readers *how* to solve problems in practice; (4) that allowed for rapid updates, both by us and also by the community at large; and (5) be complemented by a [forum](http://discuss.mxnet.io/) for interactive discussion of technical details and to answer questions.

我们着手创建一种可以（1） 所有人均可在网上免费获取；（2）提供足够的技术深度，为成为应用机器学习科学家的道路提供起点；（3）包含可运行的代码，向读者展示*如何*解决实际问题；（4）允许我们和整个社区不断快速迭代内容，从而紧跟仍在高速发展的深度学习领域；（5）由包含有关技术细节问答的[论坛](http://discuss.mxnet.io) 作为补充，使大家可以相互答疑并交换经验。

These goals were often in conflict. Equations, theorems, and citations are best managed and laid out in LaTeX. Code is best described in Python. And webpages are native in HTML and JavaScript. Furthermore, we want the content to be accessible both as executable code, as a physical book, as a downloadable PDF, and on the Internet as a website. At present there exist no tools and no workflow perfectly suited to these demands, so we had to assemble our own. We describe our approach in detail in :numref:`sec_how_to_contribute`. We settled on GitHub to share the source and to allow for edits, Jupyter notebooks for mixing code, equations and text, Sphinx as a rendering engine to generate multiple outputs, and Discourse for the forum. While our system is not yet perfect, these choices provide a good compromise among the competing concerns. We believe that this might be the first book published using such an integrated workflow.

这些目标往往互有冲突。公式、定理和引用最容易通过LaTeX进行管理和展示，代码自然应该用简单易懂的Python描述，而网页本身应该是HTML和JavaScript。此外，我们希望这个资源可以作为可执行代码、实体书，可下载的pdf和网站。然而，目前并没有任何工具可以完美地满足以上所有需求。为了适应这些要求，我们不得不组装自己的工具。我们在`sec_how_to_contribute`中详细描述了我们的方法。我们选择在GitHub上共享源代码并允许进行编辑，Jupyter笔记本，用于记录代码、公式和文本，Sphinx作为渲染引擎来生成多种输出，使用Discourse作为论坛。尽管我们的系统还不完善，但这已经是一个很好的折衷方案。我们相信这可能是使用这种集成的工作流程出版的第一本书。


### Learning by Doing边做边学

Many textbooks teach a series of topics, each in exhaustive detail. For example, Chris Bishop's excellent textbook :cite:`Bishop.2006`, teaches each topic so thoroughly, that getting to the chapter on linear regression requires a non-trivial amount of work. While experts love this book precisely for its thoroughness, for beginners, this property limits its usefulness as an introductory text.

许多教科书教授一系列主题，每个主题都详尽无遗。例如，克里斯·毕晓普（Chris Bishop）出色的教科书：Bishop.2006，详细讲授每个主题，以至于进入线性回归这一章需要做大量的工作。专家之所以喜欢本书是因为它的全面性，但是对于初学者，这个属性限制了它作为介绍性文本的用途。

In this book, we will teach most concepts *just in time*. In other words, you will learn concepts at the very moment that they are needed to accomplish some practical end. While we take some time at the outset to teach fundamental preliminaries, like linear algebra and probability, we want you to taste the satisfaction of training your first model before worrying about more esoteric probability distributions.

在本书中，我们将“快速”教授大多数概念。换句话说，您将在需要一些概念来完成一些实际目的时立即学习它们。尽管我们一开始需要一些时间来教授基本的基础知识（例如线性代数和概率），但我们希望您先体验训练第一个模型的满足感，然后再担心更深奥的概率分布。

Aside from a few preliminary notebooks that provide a crash course in the basic mathematical background, each subsequent chapter introduces both a reasonable number of new concepts and provides single self-contained working examples---using real datasets. This presents an organizational challenge. Some models might logically be grouped together in a single notebook. And some ideas might be best taught by executing several models in succession. On the other hand, there is a big advantage to adhering to a policy of *1 working example, 1 notebook*: This makes it as easy as possible for you to start your own research projects by leveraging our code. Just copy a notebook and start modifying it.

除了一些在基础数学背景下提供速成课程的初步笔记本之外，每个后续章节都介绍了数量合适的新概念，并提供了使用真实数据集的单个独立的工作示例。这对文章组织是一个挑战。 从逻辑上讲，某些模型可以组合在一个笔记本中。 通过连续执行多个模型，可能最好地教导一些想法。 另一方面，遵守*1个工作示例，1个笔记本*的策略有一个很大的优势：这使您可以尽可能容易地利用我们的代码来开始自己的研究项目。 只需复制笔记本并开始修改即可。

We will interleave the runnable code with background material as needed. In general, we will often err on the side of making tools available before explaining them fully (and we will follow up by explaining the background later). For instance, we might use *stochastic gradient descent* before fully explaining why it is useful or why it works. This helps to give practitioners the necessary ammunition to solve problems quickly, at the expense of requiring the reader to trust us with some curatorial decisions.

我们将根据需要将可运行代码与背景材料进行交织。 通常，在全面解释工具之前，我们通常会偏向于使工具可用（我们将在后面解释背景）。 例如，在充分说明其有用性或有效性之前，我们可以使用`随机梯度下降`方法。 这有助于为从业人员提供必要的资源，以快速解决问题，而以要求读者信任我们一些为了展示决定的代价。

Throughout, we will be working with the MXNet library, which has the rare property of being flexible enough for research while being fast enough for production. This book will teach deep learning concepts from scratch. Sometimes, we want to delve into fine details about the models that would typically be hidden from the user by Gluon's advanced abstractions. This comes up especially in the basic tutorials, where we want you to understand everything that happens in a given layer or optimizer. In these cases, we will often present two versions of the example: one where we implement everything from scratch, relying only on the NumPy interface and automatic differentiation, and another, more practical example, where we write succinct code using Gluon. Once we have taught you how some component works, we can just use the Gluon version in subsequent tutorials.

在整个过程中，我们将使用MXNet库，该库具有以下罕见的特性：对于研究足够灵活，而对于生产足够快。 这本书将从头开始教授深度学习的概念。 有时，我们想深入研究有关模型的详细信息，这些细节通常会被Gluon的高级抽象对用户隐藏。 特别是在基础教程中，我们希望您了解给定层或优化器中发生的所有事情。 在这些情况下，我们通常会提供该示例的两个版本：一个示例，我们从头开始执行所有操作，仅依靠NumPy接口和自动区分，另一个示例，更为实际的示例，我们使用Gluon编写简洁的代码。 一旦我们教会了您某些组件的工作原理，我们就可以在后续教程中使用Gluon版本。


### Content and Structure内容和结构

The book can be roughly divided into three parts, which are presented by different colors in :numref:`fig_book_org`:

这本书可以大致分为三个部分，在:label:`fig_book_org`图片中以不同的颜色表示：

![Book structure](../img/book-org.svg)

:label:`fig_book_org`

* The first part covers basics and preliminaries.:numref:`chap_introduction` offers an introduction to deep learning. Then, in :numref:`chap_preliminaries`, we quickly bring you up to speed on the prerequisites required for hands-on deep learning, such as how to store and manipulate data, and how to apply various numerical operations based on basic concepts from linear algebra, calculus, and probability. :numref:`chap_linear` and :numref:`chap_perceptrons` cover the most basic concepts and techniques of deep learning, such as linear regression, multilayer perceptrons and regularization.
* 第一部分介绍基础知识和预备知识。numref：`chap_introduction`提供了深度学习的介绍。 然后，在chap_preliminaries中，我们快速为您提供动手深度学习所需的先决条件，例如如何存储和操作数据以及如何基于线性的基本概念应用各种数值运算 代数，微积分和概率。 chap_linear和chref_perceptrons涵盖了深度学习的最基本概念和技术，例如线性回归，多层感知器和正则化。
* The next five chapters focus on modern deep learning techniques. :numref:`chap_computation` describes the various key components of deep learning calculations and lays the groundwork for us to subsequently implement more complex models. Next, in :numref:`chap_cnn` and :numref:`chap_modern_cnn`, we introduce convolutional neural networks (CNNs), powerful tools that form the backbone of most modern computer vision systems. Subsequently, in :numref:`chap_rnn` and :numref:`chap_modern_rnn`, we introduce recurrent neural networks (RNNs), models that exploit temporal or sequential structure in data, and are commonly used for natural language processing and time series prediction. In :numref:`chap_attention`, we introduce a new class of models that employ a technique called attention mechanisms and they have recently begun to displace RNNs in natural language processing. These sections will get you up to speed on the basic tools behind most modern applications of deep learning.
* 接下来的五章重点介绍现代深度学习技术。 chap_computation描述了深度学习计算的各个关键组成部分，并为我们随后实现更复杂的模型奠定了基础。 接下来，在chap_cnn和chap_modern_cnn中，我们介绍卷积神经网络（CNN），这些功能强大的工具构成了大多数现代计算机视觉系统的骨干。 随后，在champ_rnn和numref：chap_modern_rnn中，我们介绍了递归神经网络（RNN），这些模型利用数据中的时间或顺序结构，并且通常被用于自然语言处理和时间序列预测。 在chap_attention中，我们介绍了一类新的模型，这些模型采用了一种称为注意力机制的技术，并且最近它们开始在自然语言处理中取代RNN。这些部分将使您快速掌握大多数现代深度学习应用程序背后的基本工具。
* Part three discusses scalability, efficiency, and applications. First, in  numref:`chap_optimization`, we discuss several common optimization algorithms used to train deep learning models. The next chapter, :numref:`chap_performance` examines several key factors that influence the computational performance of your deep learning code. In :numref:`chap_cv`, we illustrate major applications of deep learning in computer vision. In :numref:`chap_nlp_pretrain` and :numref:`chap_nlp_app`, we show how to pretrain language representation models and apply them to natural language processing tasks.
* 第三部分讨论可伸缩性，效率和应用。 首先，在chap_optimization中，我们讨论了用于训练深度学习模型的几种常见优化算法。 下一章“ chap_performance”将研究几个影响深度学习代码的计算性能的关键因素。 在chap_cv中，我们说明了深度学习在计算机视觉中的主要应用。 在chap_nlp_pretrain和numref：chap_nlp_app中，我们展示了如何预训练语言表示模型并将其应用于自然语言处理任务。


### Code代码

:label:`sec_code`

Most sections of this book feature executable code because of our belief in the importance of an interactive learning experience in deep learning. At present, certain intuitions can only be developed through trial and error, tweaking the code in small ways and observing the results. Ideally, an elegant mathematical theory might tell us precisely how to tweak our code to achieve a desired result. Unfortunately, at present, such elegant theories elude us. Despite our best attempts, formal explanations for various techniques are still lacking, both because the mathematics to characterize these models can be so difficult and also because serious inquiry on these topics has only just recently kicked into high gear. We are hopeful that as the theory of deep learning progresses, future editions of this book will be able to provide insights in places the present edition cannot.

本书的大多数部分都具有可执行代码，因为我们相信在深度学习中交互式学习体验的重要性。 目前，某些直觉只能通过反复试验来发展，以小方式调整代码并观察结果。 理想情况下，一个优雅的数学理论可以准确地告诉我们如何调整代码以获得所需的结果。 不幸的是，目前还没有这些高雅的理论，尽管我们尽了最大的努力，但仍然缺乏对各种技术的形式化解释，这不仅是因为很难刻画表征这些模型的数学方法，而且还因为对这些主题的认真研究才刚刚开始。 进入高速档。 我们希望随着深度学习理论的发展，本书的未来版本将能够在当前版本无法提供的地方提供见解。

Most of the code in this book is based on Apache MXNet. MXNet is an open-source framework for deep learning and the preferred choice of AWS (Amazon Web Services), as well as many colleges and companies. All of the code in this book has passed tests under the newest MXNet version. However, due to the rapid development of deep learning, some code *in the print edition* may not work properly in future versions of MXNet. However, we plan to keep the online version remain up-to-date. In case you encounter any such problems, please consult :ref:`chap_installation` to update your code and runtime environment.

本书中的大多数代码都基于Apache MXNet。 MXNet是用于深度学习和AWS（Amazon Web Services）以及许多大学和公司的首选的开源框架。 本书中的所有代码均已通过最新MXNet版本的测试。 但是，由于深度学习的飞速发展，印刷版本中的某些代码*在MXNet的未来版本中可能无法正常工作。 但是，我们计划保持在线版本为最新。 如果您遇到任何此类问题，请查阅：chap_installation以更新您的代码和运行时环境。

At times, to avoid unnecessary repetition, we encapsulate the frequently-imported and referred-to functions, classes, etc. in this book in the `d2l` package.  For any block such as a function, a class, or multiple imports to be saved in the package, we will mark it with `# Saved in the d2l package for later use`. The `d2l` package is light-weight and only requires the following packages and modules as dependencies:

有时，为了避免不必要的重复，我们在本书的d2l包中封装了经常导入和引用的函数，类等。 对于要保存在包中的任何块，例如函数，类或多个导入，我们将其标记为 `# Saved in the d2l package for later use`。 d2l软件包非常轻，仅需要以下软件包和模块作为依赖项：


```python
# Saved in the d2l package for later use
# 存于d2l包中，便于后面使用
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
```

We offer a detailed overview of these functions and classes in :numref:`sec_d2l`.

我们在sec_d2l中提供了这些函数和类的详细概述。


### Target Audience受众

This book is for students (undergraduate or graduate), engineers, and researchers, who seek a solid grasp of the practical techniques of deep learning. Because we explain every concept from scratch, no previous background in deep learning or machine learning is required. Fully explaining the methods of deep learning requires some mathematics and programming, but we will only assume that you come in with some basics, including (the very basics of) linear algebra, calculus, probability, and Python programming. Moreover, in the Appendix, we provide a refresher on most of the mathematics covered in this book. Most of the time, we will prioritize intuition and ideas over mathematical rigor. There are many terrific books which can lead the interested reader further. For instance, Linear Analysis by Bela Bollobas :cite:`Bollobas.1999` covers linear algebra and functional analysis in great depth. All of Statistics :cite:`Wasserman.2013` is a terrific guide to statistics. And if you have not used Python before, you may want to peruse this [Python tutorial](http://learnpython.org/).

本书面向寻求扎实地学习深度学习实践技术的学生（本科生或研究生），工程师和研究人员。因为我们从头开始解释每个概念，所以不需要深度学习或机器学习的先前背景。全面解释深度学习的方法需要一些数学和编程知识，但是我们仅假设您具备一些基础知识，包括线性代数，微积分，概率和Python编程（非常基础的知识）。此外，在附录中，我们提供了有关本书涵盖的大多数数学的复习课程。大多数时候，我们会优先考虑直觉和思想，而不是严格的数学。有很多很棒的书可以引导感兴趣的读者。例如，Bela Bollobas的线性分析[Bollobas，1999]涉及线性代数和泛函分析。所有统计信息[Wasserman，2013年]是一本很棒的统计指南。而且，如果您以前没有使用过Python，则可能需要仔细阅读本[Python教程](http://learnpython.org/)。


### Forum论坛

Associated with this book, we have launched a discussion forum, located at [discuss.mxnet.io](https://discuss.mxnet.io/). When you have questions on any section of the book, you can find the associated discussion page by scanning the QR code at the end of the section to participate in its discussions. The authors of this book and broader MXNet developer community frequently participate in forum discussions.

与此书相关的是，我们已经启动了一个讨论论坛，位于[discuss.mxnet.io](https://discuss.mxnet.io/)。 当您对本书的任何部分有疑问时，可以通过扫描该部分末尾的二维码以参加其讨论来找到相关的讨论页面。 本书的作者和更广泛的MXNet开发人员社区经常参加论坛讨论。


## Acknowledgments致谢

We are indebted to the hundreds of contributors for both the English and the Chinese drafts. They helped improve the content and offered valuable feedback. Specifically, we thank every contributor of this English draft for making it better for everyone. Their GitHub IDs or names are (in no particular order): alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat, cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu, Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller, NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki, topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen, Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens, alukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta, uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee, mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya, Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy, lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner, Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong, Steve Sedlmeyer, ruslo, Rafael Schlatter, liusy182, Giannis Pappas, ruslo, ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09, Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil, Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp, tiepvupsu, sfilip, mlxd, KaleabTessera, Sanjar Adilov, MatteoFerrara, hsneto, Katarzyna Biesialska, Gregory Bruss, duythanhvn, paulaurel, graytowne.

我们感谢中英文草案的数百位撰稿人。他们帮助改善了内容并提供了宝贵的反馈。具体来说，我们感谢这份英语草案的每位撰稿人为所有人提供的帮助。它们的GitHub ID或名称为（无特定顺序）：

We thank Amazon Web Services, especially Swami Sivasubramanian, Raju Gulabani, Charlie Bell, and Andrew Jassy for their generous support in writing this book. Without the available time, resources, discussions with colleagues, and continuous encouragement this book would not have happened.

我们感谢Amazon Web Services，尤其是Swami Sivasubramanian，Raju Gulabani，Charlie Bell和Andrew Jassy在编写本书时的慷慨支持。 没有可用的时间，资源，与同事的讨论以及不断的鼓励，这本书就不会发生。


## Summary小结

* Deep learning has revolutionized pattern recognition, introducing technology that now powers a wide range of  technologies, including computer vision, natural language processing, automatic speech recognition.
* 深度学习彻底改变了模式识别，引入了现在可支持多种技术的技术，包括计算机视觉，自然语言处理，自动语音识别。
* To successfully apply deep learning, you must understand how to cast a problem, the mathematics of modeling, the algorithms for fitting your models to data, and the engineering techniques to implement it all.
* 要成功应用深度学习，您必须了解如何提出问题，建模数学，将模型拟合到数据的算法以及实现所有这些的工程技术。
* This book presents a comprehensive resource, including prose, figures, mathematics, and code, all in one place.
* 本书在一个地方提供了全面的资源，包括文章，图形，数学和代码。
* To answer questions related to this book, visit our forum at https://discuss.mxnet.io/.
* 要回答与本书有关的问题，请访问我们的论坛，网址为https://discuss.mxnet.io/。
* Apache MXNet is a powerful library for coding up deep learning models and running them in parallel across GPU cores.
* Apache MXNet是一个强大的库，用于编码深度学习模型并在GPU内核之间并行运行它们。
* Gluon is a high level library that makes it easy to code up deep learning models using Apache MXNet.
* Gluon是一个高级库，可以轻松使用Apache MXNet编写深度学习模型。
* Conda is a Python package manager that ensures that all software dependencies are met.
* Conda是一个Python软件包管理器，可确保满足所有软件依赖性。
* All notebooks are available for download on GitHub.
* 所有笔记本均可在GitHub上下载。
* If you plan to run this code on GPUs, do not forget to install the necessary drivers and update your configuration.
* 如果您打算在GPU上运行此代码，请不要忘记安装必要的驱动程序并更新您的配置。


## Exercises练习

1. Register an account on the discussion forum of this book [discuss.mxnet.io](https://discuss.mxnet.io/).
1. 在本书的讨论论坛上注册一个帐户[discuss.mxnet.io](https://discuss.mxnet.io/)。
1. Install Python on your computer.
1. 在计算机上安装Python。
1. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book and find answers to your questions by engaging the authors and broader community.
1. 请点击该部分底部的论坛链接，您可以在其中寻求帮助并讨论该书，并通过吸引作者和更广泛的社区来找到问题的答案。
1. Create an account on the forum and introduce yourself.
1. 在论坛上创建一个帐户并进行自我介绍。


## [Discussions](https://discuss.mxnet.io/t/2311) [讨论](https://discuss.mxnet.io/t/2311)

![](../img/qr_preface.svg)

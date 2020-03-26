# Preface前言


仅仅几年前，还没有大量的深度学习科学家在大型公司和初创公司中开发智能产品和服务。 当我们当中最年轻的人（作者）进入这一领域时，机器学习并没有成为日报的头条新闻。 我们的父母不知道什么是机器学习，更不用说为什么我们更喜欢从事机器学习而不是医学或法律职业。 机器学习是一门具有前瞻性的学术学科，其实际应用范围非常狭窄。 而且，这些应用程序（例如语音识别和计算机视觉）需要大量领域知识，以至于它们通常被视为完全独立的领域，而机器学习只是其中的一个小组成部分。 接着，神经网络（我们在本书中关注的深度学习模型的前身）在过去被视为过时的工具。


在过去的五年中，深度学习震惊了世界，推动了计算机视觉，自然语言处理，自动语音识别，强化学习和统计建模等各个领域的快速发展。 有了这些先进技术，我们现在可以制造出比以往任何时候都更具智能的汽车（但低于某些公司宣传的），智能回复系统可以自动起草最普通的电子邮件，从而帮助人们从庞大收件箱中解脱，以及在棋盘游戏（例如围棋）上打败人类最优秀的软件，这一壮举曾经被认为尚有几十年才会出现。 这些工具已经对工业和社会产生了越来越广泛的影响，它改变了电影的制作方式，疾病的诊断方法，并在从天体物理学到生物学的基础科学中发挥着越来越重要的作用。

## About This Book关于本书


这本书代表了我们试图使深度学习变得容易的尝试，包含*概念*，*内容*和*代码*。

### One Medium Combining Code, Math, and HTML一种结合了代码，数学和HTML的媒介


为了使任何计算技术都能发挥其全部作用，必须对它进行充分的理解，充分记录并由成熟，维护良好的工具提供支持。关键思想应该清楚地提炼出来，以最大程度地缩短新学员需要的入职时间。成熟的库应自动执行常见任务，并且示例代码应使从业人员可以轻松地修改，应用和扩展通用应用程序以满足他们的需求。以动态Web应用程序为例，尽管有很多公司，例如亚马逊，在1990年代开发成功的数据库驱动的Web应用程序，在过去的十年中，这种技术帮助创意企业家的潜力已经得到了更大的发展，这在一定程度上要归功于强大而有据可查的框架的发展。


测试深度学习的潜力提出了独特的挑战，因为任何单个应用程序都会汇集各种学科。应用深度学习需要同时理解（i）以特定方式提出问题的动机；（ii）给定建模方法的数学；（iii）使模型适合数据的优化算法；（iv）以及有效训练模型所需的工程，克服数值计算的陷阱并充分利用可用的硬件。教授解决问题所需的批判性思维技能，解决问题的数学方法以及在一个地方实施这些解决方案的软件工具都面临着巨大的挑战。我们在本书中的目标是提供一个统一的资源，以使可能的从业人员快速掌握。


我们在2017年7月启动了该图书项目，当时我们需要向用户解释MXNet的新接口Gluon。在那时，没有（i）最新的资源； （ii）广泛覆盖现代深度学习技术并具有一定的技术深度；（iii）既是严谨的教科书，又是包含可运行代码的生动的教程。我们找到了许多代码示例，这些代码示例分散在各个地方，如何使用给定的深度学习框架（例如，如何使用TensorFlow中的矩阵进行基本数值计算）或实现特定技术（例如，用于LeNet，AlexNet，ResNets的代码段）各种博客文章和GitHub存储库。但是，这些示例通常集中于*如何*实现给定方法，而没有讨论*为什么*做出某些算法决策。尽管一些互动资源偶尔会弹出以解决特定主题，例如在[Distill](http://distill.pub)网站上发布的引人入胜的博客文章或个人博客，但它们仅涵盖深度学习中的选定主题，并且通常缺少相关代码。另一方面，虽然出现了几本教科书，但最著名的要数Goodfellow、Bengio和Courville的《深度学习》，该书对深度学习背后的概念进行了全面的梳理，这类资源并没有将概念描述与实际代码相结合，有时使读者对如何实现它们一无所知。除此之外，商业课程提供者们虽然制作了众多的优质资源，但它们的付费门槛依然令不少用户望而生畏。


我们着手创建一种可以（1） 所有人均可在网上免费获取；（2）提供足够的技术深度，为成为应用机器学习科学家的道路提供起点；（3）包含可运行的代码，向读者展示*如何*解决实际问题；（4）允许我们和整个社区不断快速迭代内容，从而紧跟仍在高速发展的深度学习领域；（5）由包含有关技术细节问答的[论坛](http://discuss.mxnet.io) 作为补充，使大家可以相互答疑并交换经验。


这些目标往往互有冲突。公式、定理和引用最容易通过LaTeX进行管理和展示，代码自然应该用简单易懂的Python描述，而网页本身应该是HTML和JavaScript。此外，我们希望这个资源可以作为可执行代码、实体书，可下载的pdf和网站。然而，目前并没有任何工具可以完美地满足以上所有需求。为了适应这些要求，我们不得不组装自己的工具。我们在`sec_how_to_contribute`中详细描述了我们的方法。我们选择在GitHub上共享源代码并允许进行编辑，Jupyter笔记本，用于记录代码、公式和文本，Sphinx作为渲染引擎来生成多种输出，使用Discourse作为论坛。尽管我们的系统还不完善，但这已经是一个很好的折衷方案。我们相信这可能是使用这种集成的工作流程出版的第一本书。


### Learning by Doing边做边学


许多教科书教授一系列主题，每个主题都详尽无遗。例如，克里斯·毕晓普（Chris Bishop）出色的教科书：Bishop.2006，详细讲授每个主题，以至于进入线性回归这一章需要做大量的工作。专家之所以喜欢本书是因为它的全面性，但是对于初学者，这个属性限制了它作为介绍性文本的用途。


在本书中，我们将“快速”教授大多数概念。换句话说，您将在需要一些概念来完成一些实际目的时立即学习它们。尽管我们一开始需要一些时间来教授基本的基础知识（例如线性代数和概率），但我们希望您先体验训练第一个模型的满足感，然后再担心更深奥的概率分布。


除了一些在基础数学背景下提供速成课程的初步笔记本之外，每个后续章节都介绍了数量合适的新概念，并提供了使用真实数据集的单个独立的工作示例。这对文章组织是一个挑战。 从逻辑上讲，某些模型可以组合在一个笔记本中。 通过连续执行多个模型，可能最好地教导一些想法。 另一方面，遵守*1个工作示例，1个笔记本*的策略有一个很大的优势：这使您可以尽可能容易地利用我们的代码来开始自己的研究项目。 只需复制笔记本并开始修改即可。


我们将根据需要将可运行代码与背景材料进行交织。 通常，在全面解释工具之前，我们通常会偏向于使工具可用（我们将在后面解释背景）。 例如，在充分说明其有用性或有效性之前，我们可以使用`随机梯度下降`方法。 这有助于为从业人员提供必要的资源，以快速解决问题，而以要求读者信任我们一些为了展示决定的代价。


在整个过程中，我们将使用MXNet库，该库具有以下罕见的特性：对于研究足够灵活，而对于生产足够快。 这本书将从头开始教授深度学习的概念。 有时，我们想深入研究有关模型的详细信息，这些细节通常会被Gluon的高级抽象对用户隐藏。 特别是在基础教程中，我们希望您了解给定层或优化器中发生的所有事情。 在这些情况下，我们通常会提供该示例的两个版本：一个示例，我们从头开始执行所有操作，仅依靠NumPy接口和自动区分，另一个示例，更为实际的示例，我们使用Gluon编写简洁的代码。 一旦我们教会了您某些组件的工作原理，我们就可以在后续教程中使用Gluon版本。


### Content and Structure内容和结构


这本书可以大致分为三个部分，在:label:`fig_book_org`图片中以不同的颜色表示：

![Book structure](../img/book-org.svg)

:label:`fig_book_org`

* 第一部分介绍基础知识和预备知识。numref：`chap_introduction`提供了深度学习的介绍。 然后，在chap_preliminaries中，我们快速为您提供动手深度学习所需的先决条件，例如如何存储和操作数据以及如何基于线性的基本概念应用各种数值运算 代数，微积分和概率。 chap_linear和chref_perceptrons涵盖了深度学习的最基本概念和技术，例如线性回归，多层感知器和正则化。
* 接下来的五章重点介绍现代深度学习技术。 chap_computation描述了深度学习计算的各个关键组成部分，并为我们随后实现更复杂的模型奠定了基础。 接下来，在chap_cnn和chap_modern_cnn中，我们介绍卷积神经网络（CNN），这些功能强大的工具构成了大多数现代计算机视觉系统的骨干。 随后，在champ_rnn和numref：chap_modern_rnn中，我们介绍了递归神经网络（RNN），这些模型利用数据中的时间或顺序结构，并且通常被用于自然语言处理和时间序列预测。 在chap_attention中，我们介绍了一类新的模型，这些模型采用了一种称为注意力机制的技术，并且最近它们开始在自然语言处理中取代RNN。这些部分将使您快速掌握大多数现代深度学习应用程序背后的基本工具。
* 第三部分讨论可伸缩性，效率和应用。 首先，在chap_optimization中，我们讨论了用于训练深度学习模型的几种常见优化算法。 下一章“ chap_performance”将研究几个影响深度学习代码的计算性能的关键因素。 在chap_cv中，我们说明了深度学习在计算机视觉中的主要应用。 在chap_nlp_pretrain和numref：chap_nlp_app中，我们展示了如何预训练语言表示模型并将其应用于自然语言处理任务。


### Code代码

:label:`sec_code`


本书的大多数部分都具有可执行代码，因为我们相信在深度学习中交互式学习体验的重要性。 目前，某些直觉只能通过反复试验来发展，以小方式调整代码并观察结果。 理想情况下，一个优雅的数学理论可以准确地告诉我们如何调整代码以获得所需的结果。 不幸的是，目前还没有这些高雅的理论，尽管我们尽了最大的努力，但仍然缺乏对各种技术的形式化解释，这不仅是因为很难刻画表征这些模型的数学方法，而且还因为对这些主题的认真研究才刚刚开始。 进入高速档。 我们希望随着深度学习理论的发展，本书的未来版本将能够在当前版本无法提供的地方提供见解。


本书中的大多数代码都基于Apache MXNet。 MXNet是用于深度学习和AWS（Amazon Web Services）以及许多大学和公司的首选的开源框架。 本书中的所有代码均已通过最新MXNet版本的测试。 但是，由于深度学习的飞速发展，印刷版本中的某些代码*在MXNet的未来版本中可能无法正常工作。 但是，我们计划保持在线版本为最新。 如果您遇到任何此类问题，请查阅：chap_installation以更新您的代码和运行时环境。


有时，为了避免不必要的重复，我们在本书的d2l包中封装了经常导入和引用的函数，类等。 对于要保存在包中的任何块，例如函数，类或多个导入，我们将其标记为 `# Saved in the d2l package for later use`。 d2l软件包非常轻，仅需要以下软件包和模块作为依赖项：


```python
# Saved in the d2l package for later use
```


我们在sec_d2l中提供了这些函数和类的详细概述。


### Target Audience受众


本书面向寻求扎实地学习深度学习实践技术的学生（本科生或研究生），工程师和研究人员。因为我们从头开始解释每个概念，所以不需要深度学习或机器学习的先前背景。全面解释深度学习的方法需要一些数学和编程知识，但是我们仅假设您具备一些基础知识，包括线性代数，微积分，概率和Python编程（非常基础的知识）。此外，在附录中，我们提供了有关本书涵盖的大多数数学的复习课程。大多数时候，我们会优先考虑直觉和思想，而不是严格的数学。有很多很棒的书可以引导感兴趣的读者。例如，Bela Bollobas的线性分析[Bollobas，1999]涉及线性代数和泛函分析。所有统计信息[Wasserman，2013年]是一本很棒的统计指南。而且，如果您以前没有使用过Python，则可能需要仔细阅读本[Python教程](http://learnpython.org/)。


### Forum论坛


与此书相关的是，我们已经启动了一个讨论论坛，位于[discuss.mxnet.io](https://discuss.mxnet.io/)。 当您对本书的任何部分有疑问时，可以通过扫描该部分末尾的二维码以参加其讨论来找到相关的讨论页面。 本书的作者和更广泛的MXNet开发人员社区经常参加论坛讨论。


## Acknowledgments致谢


我们感谢中英文草案的数百位撰稿人。他们帮助改善了内容并提供了宝贵的反馈。具体来说，我们感谢这份英语草案的每位撰稿人为所有人提供的帮助。它们的GitHub ID或名称为（无特定顺序）：


我们感谢Amazon Web Services，尤其是Swami Sivasubramanian，Raju Gulabani，Charlie Bell和Andrew Jassy在编写本书时的慷慨支持。 没有可用的时间，资源，与同事的讨论以及不断的鼓励，这本书就不会发生。


## Summary小结

* 深度学习彻底改变了模式识别，引入了现在可支持多种技术的技术，包括计算机视觉，自然语言处理，自动语音识别。
* 要成功应用深度学习，您必须了解如何提出问题，建模数学，将模型拟合到数据的算法以及实现所有这些的工程技术。
* 本书在一个地方提供了全面的资源，包括文章，图形，数学和代码。
* 要回答与本书有关的问题，请访问我们的论坛，网址为https://discuss.mxnet.io/。
* Gluon是一个高级库，可以轻松使用Apache MXNet编写深度学习模型。
* Conda是一个Python软件包管理器，可确保满足所有软件依赖性。
* 所有笔记本均可在GitHub上下载。
* 如果您打算在GPU上运行此代码，请不要忘记安装必要的驱动程序并更新您的配置。


## Exercises练习

1. 在本书的讨论论坛上注册一个帐户[discuss.mxnet.io](https://discuss.mxnet.io/)。
1. 在计算机上安装Python。
1. 请点击该部分底部的论坛链接，您可以在其中寻求帮助并讨论该书，并通过吸引作者和更广泛的社区来找到问题的答案。
1. 在论坛上创建一个帐户并进行自我介绍。


## [Discussions](https://discuss.mxnet.io/t/2311) [讨论](https://discuss.mxnet.io/t/2311)

![](../img/qr_preface.svg)
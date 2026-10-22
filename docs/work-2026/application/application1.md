# Application 1 - 底层认知、简单 API 调用与玩具级 AI 应用

> [!NOTE]
> 预计耗时：45 天

## 学习目的

首先需要说明，这里的应用指的是极为顶层的 AI 应用开发，几乎不涉及底层算法实现。手搓 CNN 训练图片识别模型这类工作不属于应用，而属于算法实现。

因此应用指的是工程化地使用 AI，把 AI 作为业务链条中的一环。比如「橘雪莉妙妙屋」这个作业，是调用大语言模型完成情感分析任务，而不是传统地训练一个情感分析器。

打一个通俗易懂的链条：

数学 -> Infra 算子 -> Pytorch 自动推导反向传播 -> Transformer 库（Hugging Face 集成 Pytorch / TensorFlow，屏蔽底层细节，抽象出 Pipeline 的概念） -> Langchain / Dify 模块化集成（轻松调用工具） -> LLM API 屏蔽所有底层细节 -> AI 应用（调用 API 完成任务、聊天机器人、Agent 调用 API）-> Harness

Application 的目的是要你站在最顶层的视角向下看，并且向下只最多接触到 Hugging Face。

从这里不难看出，AI 应用主要学习的是怎么使用 AI，而不是 AI 的底层原理。所以如果你选择了应用方向，你不只是学习 AI 调用，还需要学习后端、运维等方面的知识。

当前的 AI 应用开发主要围绕大语言模型展开。未来可能会有更多 AI 模型出现，应用开发形态也会更加多样化。

当然正如 [Foundation 4](../foundation/foundation4.md) 导引部分提到的一样，作为一个 211 的本科生，如果只拥有高级应用能力，会在就业市场上很吃亏。

所以建议学习这部分的前提是你已经掌握了一门其他语言，或者将来会去学习其他语言。

本轮作业将从传统机器学习与深度学习开始，带你认识 AI 应用。

不过目前的 AI 应用来看，大语言模型应用才是真正的主流，因此本次任务我们也有大语言模型的简单 API 调用。

$\color{red}{\text{再次强调，进行这部分内容前请你清晰的知道你现在在做什么。}}$

$\color{red}{\text{否则不仅晦涩难懂，而且还没什么用。}}$

$\color{red}{\text{211 本科生能接触到的 Agent 开发，本质是前端 / 后端。}}$

$\color{red}{\text{开始之前，请确保你拥有充足的前端 / 后端知识。}}$

## 学习内容

- 底层认知：简单底层理解、机器学习与深度学习
- 简单 API 调用
- 玩具级 AI 应用（rembg、YOLO、Whisper）

## 作业 - 底层认知

虽然作业总标题叫底层认知，但我们更希望你去先完成 Research 方向的作业，然后再回过头来继续向下看。

一个做 AI 的人可以理解 AI 底层的架构，至少是最基本的。

对于 AI 应用，我们更希望的路线是学习完 AI 底层，学习一份后端 / 前端技术，然后再来学习 Agent，乃至更进阶的 AI 应用算法、Infra 等，这也是我们设置 Application0 的原因。

### 文档 1

大部分现代人工智能库已经提供了非常多高度封装的方法供开发者去调用，这些方法很好的屏蔽了底层的细节，使得开发者不用反复纠结于晦涩的数学公式推导，将精力放在实际的产品开发中。

但反复的实践已经证明了，只会调库其实是远远不够的，掌握部分底层相关的知识可以更好的帮助我们进行 AI 应用开发。

因此，本次的任务要求你阅读以下几份资料：

1. [Andrej Karpathy《Let's build GPT: from scratch, in code, spelled out.》](https://www.bilibili.com/video/BV1K4LPzLEoA/)

2. [nanoGPT](https://github.com/karpathy/nanoGPT)

3. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

4. [Schariac125 简单科普博客：自顶向下——可能要懂的底层原理简单讲解](https://schariac125.online/2026/05/31/%E8%87%AA%E9%A1%B6%E5%90%91%E4%B8%8B/)

5. （Bonus）论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 及其讲解视频 [Transformer 论文逐段精读【论文精读】跟李沐学 AI](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.337.search-card.all.click)

完成上述任务后，你需要写一篇文档来阐述你的理解。

> [!NOTE]
> 如果你对 Pytorch 感兴趣的话，以下是推荐你可以去阅读和完成的资料和任务，这些并不做强制性要求。
>
>1. [Pytorch 官方文档](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
>
>2. [pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning)
>
>3. 基于 mnist 数据集的 CNN 图像分类任务（资料太多了，自己找去吧）
>
>4. [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview)
>
>5. [Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection)
>
> 一般而言并不建议初学者一上来就去阅读 Pytorch 官方文档，你可以从 pytorch-deep-learning 项目开始做起一步步学会使用 Pytorch。

### 作业 1 - 机器学习与深度学习

LLM 的底层实现基于深度学习神经网络，深度学习的前身是机器学习。

数学建模竞赛通常涉及大量的数据处理和分析任务，近年来，深度学习方法在数学建模中的应用越来越广泛，成为解决复杂问题的有力工具。

但本次任务的重点并不是完整的数学建模，而是通过数学建模对机器学习和深度学习相关内容进行学习并且学会使用 [scikit-learn](https://scikit-learn.org/stable/) 等工具。

但我们希望你的眼光能够不止局限于 [scikit-learn](https://scikit-learn.org/stable/) 里封装好的机器学习方法，而应放眼深度学习领域，了解一些基本的深度学习知识。

本次你需要完成的任务是完成 2025 电工杯数学建模的 A 题。数据集在参考资料中已有给出。

提示：本题的最优解是一个深度学习网络方法。

#### 基本方法

- SVM
- XGBoost
- Random Forest
- CNN
- RNN
- Transformer
- LSTM

对于这部分内容，你可以翻阅 [scikit-learn](https://scikit-learn.org/stable/) 官网来学习。

#### 下载题目

你可以从官网下载题目，参考下面两条链接：

1. [2025 电工杯下载链接](https://new.saikr.com/vse/EECMCM2025?type=notice&id=31337)
2. [2025 电工杯压缩包密码](https://new.saikr.com/vse/EECMCM2025?type=notice&id=31338)

#### 作业要求

- 请勿抄袭
- 不对大模型的使用做出限制，我们希望你能够借助大模型提升自己的知识面，但请不要直接对大模型说“帮我解决这个问题”
- 使用 uv 对你的虚拟环境进行管理
- 对于“基本方法”里的所有方法，你需要至少挑选 3 个（其中包含你认为的最优解）对比分析
- 你需要对原始数据进行一些处理
- 完成作业后请撰写一份文档阐述“基本方法”内所有方法的大致原理，“对比分析”中你的理解与遇到的问题。请不要让大模型直接生成你的文档

#### 参考资料 - 作业 1

1. [2017 Sky Images and Photovoltaic Power Generation Dataset for Short-term Solar Forecasting (Stanford Raw)](https://purl.stanford.edu/sm043zf7254)

### 作业 2 - 现代 LLM 思想

在现代的框架下，调用大语言模型只需要挑很少量的参数，例如 Temperature、Top-K、Top-P 等等，而不需要关心模型的底层实现细节。

Hugging Face 是一个友好的开源社区，在上面你可以找到绝大多数的开源模型及其参数。

尽管我们做的是上层应用，但仍应理解现代 LLM 应用是如何基于 Pipeline 这一概念构建的。

阅读 Hugging Face 官方文档教学，了解现代的 LLM 架构（例如 Pipeline），了解 Transformer 的基本原理。

1. [PyTorch 文档](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
2. [Hugging Face 课程](https://huggingface.co/learn/llm-course/zh-CN)
3. [Transformers 文档](https://huggingface.co/docs/transformers/v5.9.0/zh/index)

在完成上述任务后，你应该写一份文档来阐述你的理解，并且做到可以完成类似 Foundation 3 作业 2 中的代码即可。

## 作业 - 简单 API 调用

我们鼓励你在 Application 方向的相关作业中使用 MacOS / Linux 系统进行学习。但如果硬件条件实在有限，Windows 系统的 WSL2 依旧是一个不错的选择，相关配置教程可以参考[安装教程](https://learn.microsoft.com/zh-cn/windows/wsl/install)。

### 作业 1 - 橘雪莉妙妙屋

橘雪莉是《魔法少女的魔女审判》中的登场角色。自称侦探的少女，笑容满面，性格奔放，对有意思的事情会毫不犹豫扑上去。

现在她潜入了一个 QQ 群，想要分析群成员的情绪，判断群友当前的状态。

分析某个群友的 x 条信息后（x 可由你定义，也可由 AI 自行判断），她要发送表情包来回应这个群友的状态。

如果群友是开心的，那么她就发送标签为“开心”的表情包；如果群友是难过的，那么她就发送标签为“难过”的表情包。

恰好，智谱在开源方面有很大贡献。她拜托你访问 [智谱 AI 开放平台](https://open.bigmodel.cn/) 官网，注册账号并申请密钥（密钥不可泄露，你也不想你的 token 被橘雪莉吃掉吧），然后调用 GLM-4.5V 完成这个任务：

对 x 句话（或表情、图片）进行分析后，输出一个心情标签，并根据该标签从已打好标签的表情包中随机发送。

你需要阅读 GLM-4.5V 文档，仅调用 API，并通过提示词完成句子的情感分析。情感类别限制为“开心”“难过”“愤怒”“中性”。

需要注意的是，GLM 的输出只能是你定义的词汇之一。如果输出了其他内容，说明你的提示词设计不合理，需要进行修改。

#### 作业要求 - 作业 1

- 使用具有视觉能力的模型完成文本和表情包的分析。
- 设计合理的提示词，确保模型输出符合预期的情感标签。
- 使用 openai 模块调用 GLM API。
- 务必使用 .gitignore 文件保护你的密钥，避免泄露。
- 使用 uv 管理项目依赖和运行环境。

### 作业 2 - 自动调课

众所周知，福大放假会发调课通知，但课表却从不跟着改。每次一调休，课表上的安排全乱了，非常糟糕。

我们强大的课表 APP 福 uu 可以对课表使用调课规则，但是调课通知不是结构化的，无法直接输入到福 uu 中。LLM 具有强大的文本泛化能力，福 uu 的开发者们想到了一个办法，就是让 LLM 来解析调课通知，将调课通知转换成结构化的调课规则。

现在你需要实现一个调课通知解析器，将你在 [Foundation 2](../foundation/foundation2.md) 中爬取并单独提取的调课通知 csv 内所有的通知转换成结构化的调课规则。

调课规则的格式如下：

```json
[
  {
    "from_date": "2026-01-02",  // 原定上课日期
    "to_date": "2026-01-04"     // 调课后上课日期
  },
  {
    "from_date": "2026-01-03",
    "to_date": ""               // 空表示不上课
  }
]
```

#### 参考资料 - 作业 2

[宝硕博客：LLM 工程化在福 uu 中的落地实践 —— 假期调课的智能解析](https://blog.baoshuo.ren/post/fzuhelper-llm-as-function/)

#### 作业要求 - 作业 2

- 设计合理的提示词，确保模型能够正确解析调课通知。
- 使用 openai 模块调用 API，选择合适的模型。
- 必须使用 uv 管理项目依赖和运行环境。
- 校验模型输出的调课规则是否符合预期格式。
- 使用在之前任务爬取的教务通知验证你的解析器，确保它能够正确解析调课通知。

### 作业 3 - JSON 生成与解析器

## 作业 - 玩具级 AI 应用

我们还是需要了解一些其他的工程化应用，例如一些玩具。

这里需要注意，玩具的是**我们写出来的东西**，而不是我们**调用的工具**。

### 作业 1 - 证件照生成器

下北泽的 Live House 要办一场联合演出，伊地知虹夏作为 結束バンド 的队长，负责把四个人的报名材料交上去。

表格的最后一栏写着：请附上演出者本人近照，白底，一寸。

虹夏本以为这是最省事的一栏，直到四个人交上来的东西摊在了桌上：

- 后藤一里交来一张用被子蒙住头、只露出一只眼睛的自拍，理由是「照相馆里有人看着我，我会死的」；
- 山田凉交来一张在草地上吃草的照片，说「这张拍出了我的灵魂」；
- 喜多郁代交来一张光线完美、背景是涩谷十字路口的自拍；
- 而虹夏自己那张，是四个人里唯一一张能直接用的。

报名截止是明天下午五点，最近的照相馆六点就关门，而乐队的经费上周被凉拿去买了一把新贝斯。

「那就不去照相馆了。」虹夏把四张照片排成一排，「把人抠出来，换底色，裁成一寸，就这么简单。」

好消息是，抠图这件事早就不需要自己做了。`rembg` 已经把 U2Net / ISNet 这类分割模型封装成了命令行工具和 Python 接口，一行调用即可。

你需要写一个证件照生成器，把这一桌照片变成能交上去的材料：

1. 输入一张人像照片，自动抠出人像并合成到指定底色（白 / 蓝 / 红）上
2. 输出标准规格：一寸（295 × 413 像素）、二寸（413 × 579 像素），尺寸与底色都能通过命令行参数指定
3. 支持批量：传入一个目录，输出到另一个目录，保持文件名，结束时打印一份处理报告
4. 人像需要居中并留出合理的头部空间，不能简单地把原图缩放后贴上去
5. （Bonus）用 Gradio / Streamlit 包一层界面，上传即出图
6. （Bonus）人脸检测自动对齐，保证眼睛水平、头部居中（可以使用 `mediapipe` 等已经封装好的工具）

> [!TIP]
> 测试的时候别只拿一张精修过的证件照测试。
>
> 思考：如果四个人交上来的照片尺寸、朝向、人脸比例全都不一样，你的代码要怎样才能不做任何手动调整就把它们全部处理掉。

#### 作业要求 - 证件照生成器

- 使用 `rembg` 等已经封装好的抠图工具，不要自己训练分割模型
- 抠图结果不能有明显白边、毛边，发丝边缘也要可接受
- 尺寸、底色、留白等参数必须可配置，不能硬编码在代码里
- 批量模式下，单张图片失败不能中断整个任务，需要跳过并记录
- 提供清晰的命令行帮助与 README
- 只处理自己或已获得授权的照片，禁止上传到第三方在线服务

#### 参考资料 - 证件照生成器

1. [rembg](https://github.com/danielgatis/rembg)
2. [rembg - PyPI](https://pypi.org/project/rembg/)
3. [Pillow 文档](https://pillow.readthedocs.io/)
4. [Gradio 文档](https://www.gradio.app/docs)
5. [MediaPipe](https://github.com/google-ai-edge/mediapipe)

### 作业 2 - YOLO 的基本使用

[ShaddockNH3](https://github.com/ShaddockNH3) 是忠实的 PVZ 玩家，不过她的技术实在太差了。

直到某天她碰到 [柠檬味氨水](https://github.com/weijianxian) 对着某款枪战游戏框框画画。

“鬼！”

于是她找到了技术力差的解决方案。

YOLO 是计算机视觉领域最具代表性的单阶段目标检测算法系列，基于它的工具（如 Ultralytics）提供了一个简单的 API 来识别特定的物体。

本次任务并不会太难，目的是介绍 AI 应用领域并非只有 LLM，在视觉领域也有广泛的应用。

你需要借助 YOLO 实现一个自拟的项目，最终效果是自动化完成某些单机游戏的自动化操作。

如果你没有 idea，你可以做一个自动通关 PVZ 1-10 关的脚本。

#### 作业要求 - YOLO

- 请不要只是做一个识别 YOLO 模型中已经包含的模型的 demo。

#### 参考资料 - YOLO

1. [YOLO 快速使用](https://docs.ultralytics.com/zh)
2. [Ultralytics Python 用法](https://docs.ultralytics.com/zh/usage/python/)
3. [Ultralytics 训练自定义数据集](https://docs.ultralytics.com/zh/modes/train/)
4. [Ultralytics 预测与推理参数](https://docs.ultralytics.com/zh/modes/predict/)
5. [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)（标注工具，也可以使用 LabelImg 或 Roboflow）
6. [mss](https://github.com/BoboTiG/python-mss)（屏幕截图）
7. [pynput](https://github.com/moses-palmer/pynput)（模拟键盘鼠标）
8. [PVZ 原版下载](http://jspvz.com/ResDownload/1_PC_E.htm#v1.0.0.1051)
9. [PVZ 修改器](http://jspvz.com/ResDownload/Modifier.htm#v1.9)

### 作业 3 - 实时转写与总结

[blueinsane0727](https://github.com/blueinsane0727) 终于找到了秋招前的实习，但比起上班而言她更喜欢摸鱼。

可惜好景不长，随着 DDL 的到来，她再也不能像以前那样窝在角落里摸鱼了，每天都得花上大量的时间开会讨论方案。

会议的内容枯燥且乏味，于是她委托你帮忙做一个会议音频转写工具，需要具备一键总结功能，这样她就能花时间在崩铁上，被领导点到的时候看一眼总结就好了。

不过有的时候领导会问的很深，所以原始文本也需要留着备用！

Whisper 是一个端到端的 Transformer 解码器 - 编码器模型，是目前业界应用最广泛的语音转写方案之一。支持多语言识别与翻译，并且提供了多种模型尺寸供选择使用。

在这项作业中，你需要使用 OpenAI 的语音识别模型 Whisper 实现一个会议实时转写与总结。

最终的效果是呈现原始文本、实时总结以及思维导图三块内容。

这里推荐使用尺寸为 small 的模型，其 244M 的参数量足以支持你完成本次作业，并且能够在个人 PC 上流畅运行。

#### 参考资料 - Whisper

1. [Whisper 官方文档](https://openai.com/zh-Hans-CN/index/whisper/)
2. [openai/whisper](https://github.com/openai/whisper)（官方仓库，模型尺寸与参数说明都在这里）
3. [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（推理加速，做实时转写建议使用）
4. [whisper.cpp](https://github.com/ggerganov/whisper.cpp)（CPU / 端侧推理）
5. [sounddevice](https://python-sounddevice.readthedocs.io/)（麦克风 / 系统声音采集）
6. [Silero VAD](https://github.com/snakers4/silero-vad)（判断有没有人在说话，避免对静音片段做转写）
7. [noisereduce](https://github.com/timsainb/noisereduce)（降噪）
8. [ffmpeg](https://ffmpeg.org/documentation.html)、[pydub](https://github.com/jiaaro/pydub)（音频切分与格式转换）
9. [Mermaid 文档](https://mermaid.js.org/intro/)（思维导图渲染）

#### 作业要求 - Whisper

- 你可以自己爬取一个长视频作为使用样本完成本次作业，也可以使用群聊里的答辩会议记录视频。
- 你需要学会自己配置一个 chat model 用于总结
- 如果一开始觉得无从下手，可以先实现一个朴素的非实时会议后总结，但最后提交的作业需要是实时总结版本的。
- 前处理：会议存在杂音，只是过一遍 whisper 的模型可能会引入大量无关声音。你需要为你的实现配置降噪处理，可以采用训练降噪模型的方式。
- 后处理：whisper 生成的内容存在错字漏字，你需要解决这个问题
- 增加思维导图功能，思维导图可以采用 mermaid 呈现。
- 最终呈现出来的东西可以以任意形式保存，但是必须需要渲染。

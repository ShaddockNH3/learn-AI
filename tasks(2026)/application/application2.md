# Application 2 - 进阶应用

> 预计耗时：60 天

## 学习目的

从"会调用 API"的初级开发者，成长为能理解模型运作机理、掌握模型定制能力、构建复杂 AI 应用的进阶工程师。这一阶段的核心是打通从底层模型到上层应用的完整链路。

你将学习如何基于大语言模型进行微调，如何构建 RAG 工作流，以及如何使用 LangChain、Dify 等框架来搭建复杂的 AI 应用，当然还要你最期待的 Agent。

## 学习内容

- Pipeline
- LangChain & Dify
- RAG
- 微调
- Agent

## 作业

### 文档 1 - 现代 LLM

在现代的框架下，调用大语言模型只需要挑很少量的参数，例如 Temperature、Top-K、Top-P 等等，而不需要关心模型的底层实现细节。

Hugging Face 是一个友好的开源社区，在上面你可以找到绝大多数的开源模型及其参数。

尽管我们做的是上层应用，但仍应理解现代 LLM 应用是如何基于 Pipeline 这一概念构建的。

阅读 Hugging Face 官方文档教学，了解现代的 LLM 架构（例如 Pipeline），了解 Transformer 的基本原理，了解微调的基本方法，了解 Hugging Face Hub 的使用方法。

1. [Pytorch文档](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

2. [Hugging Face 课程](https://huggingface.co/learn/llm-course/zh-CN)

3. [Transformers 文档](https://huggingface.co/docs/transformers/v5.9.0/zh/index)

在完成上述任务后，你应该写一份文档来阐述你的理解。

### 作业 2 - 复现 MiniMind

微调有两种主流方式。

第一种是直接修改大语言模型本身的参数，另一种是基于 LoRA 的低秩适配。

你应该知道大语言模型由大量高维参数矩阵组成：前者直接更新原始参数，后者可以用一个简化公式表示为：

$$ W' = W + \Delta W,\quad \Delta W = BA $$

其中 $W$ 是原始权重矩阵， $A$ 和 $B$ 是低秩矩阵， $\Delta W$ 是低秩增量， $W'$ 是微调后的权重。

$\Delta W$ 的参数量通常远小于 $W$，因此微调的计算资源和数据需求可以大幅降低。

在实际实现中，低秩增量通常会作用在多个层上，例如每层对应一组 $A_i, B_i$，这样可以更细粒度地控制微调过程。

举一个简单的例子，假设我们现在要获得一个具备医疗知识的中文模型，但是你的初始模型 $A$ 是纯由英文资料训练来的。

所以我们首先要进行第一次微调 $P_1$，让模型可以理解中文，得到第一级模型 $P_1A$，然后进行第二次医疗训练，得到第二级模型 $P_2P_1A$。

该过程具有“可插拔”特性，所以你在分享模型时，通常只需要分享 LoRA 适配器参数，而原始底座模型可以让对方自行获取。

并且整个过程不需要你花费大量的计算资源去从头训练一个模型。

你在本次作业的任务是复刻一个经典的微调任务：[MiniMind](https://github.com/jingyaogong/minimind)

#### 1. 准备环境  

如果你想在本地从零开始训练模型，需要一张性能较高的显卡（如 RTX 5080、4090、5090 等）。如果没有，可以使用 AutoDL、Colab 或 OpenDL 等平台完成训练。  

不过，本作业的重点是 LoRA 微调，因此下面会教你如何下载已训练好的 PyTorch 模型，并在此基础上进行 LoRA 微调。LoRA 微调对显卡要求较低，使用 RTX 4060 等入门级显卡即可完成。如果你没有 N 卡，或显卡性能太弱，同样可以考虑使用 AutoDL、Colab 或 OpenDL 等平台来完成训练。

注意，使用AutoDL、Colab 或 OpenDL 等平台请使用VSCode的远程开发功能来完成训练，这样可以更方便地管理代码和文件，请注意VSCode远程开发下扩展需用重新安装。

> [!TIP]
> RTX 4070 Laptop 完成训练需要约20+小时，完成微调需1+小时  
> RTX 5090 完成训练需要约4+小时，完成微调需2分钟左右

#### 2. 克隆MiniMind的代码库

```shell
git clone --depth 1 https://github.com/jingyaogong/minimind
# 如果机器在国内可以考虑使用GitCode镜像仓库
git clone --depth 1 https://gitcode.com/GitHub_Trending/min/minimind.git
```

#### 3. 阅读[项目介绍](https://github.com/jingyaogong/minimind#-%E9%A1%B9%E7%9B%AE%E4%BB%8B%E7%BB%8D)以及[LoRA (Low-Rank Adaptation)](https://github.com/jingyaogong/minimind#4-lora-low-rank-adaptation)

#### 4. 配置环境

##### 远程环境

远程环境通常已经预装了 Python 和对应的 CUDA 版本的 PyTorch ，你只需要安装一些额外的依赖即可。

```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

##### 本地环境

如果你没有安装CUDA，请先[安装CUDA](https://developer.nvidia.com/cuda-toolkit-archive)。这里以 CUDA 12.8 版本为例。请注意，PyTorch 官方的安装包与 CUDA 版本是严格绑定的，例如标有cu128的PyTorch包必须配合 CUDA 12.8 使用，请务必确认你的 CUDA 版本和 PyTorch 版本的对应关系。同时，建议安装最新版 NVIDIA 驱动（游戏驱动即可），确保驱动支持的 CUDA 版本不低于你安装的 CUDA 工具包版本。

在本地环境极不推荐使用`pip`在全局环境安装，建议使用`uv`创建一个虚拟环境，并且使用国内镜像源来安装依赖。下面是`uv`的参考配置文件`pyproject.toml`：

```toml
[project]
name = "minimind"
version = "2.0.0"
description = "64M-parameter LLM from scratch in just 2h!"
readme = "README.md"
requires-python = ">=3.12"

# 这里的依赖是基于commit 4497610的并升级了pytorch和torchvision，可能会和你的版本不完全一致，如果遇到问题可以参考原仓库的requirements.txt来修改这里的依赖。
dependencies = [
    "datasets==3.6.0",
    "datasketch==1.6.4",
    "einops==0.8.1",
    "flask==3.0.3",
    "flask-cors==4.0.0",
    "jieba==0.42.1",
    "jinja2==3.1.2",
    "jsonlines==4.0.0",
    "marshmallow==3.22.0",
    "modelscope==1.37.0",
    "ngrok==1.4.0",
    "nltk==3.8",
    "numpy==1.26.4",
    "openai==1.59.6",
    "psutil==5.9.8",
    "pydantic==2.11.5",
    "rich==13.7.1",
    "scikit-learn==1.5.1",
    "sentence-transformers==2.3.1",
    "simhash==2.1.2",
    "streamlit==1.50.0",
    "swanlab==0.7.11",
    "tiktoken==0.10.0",
    "transformers==4.57.6",
    "trl==0.13.0",
    "ujson==5.1.0",
    "wandb==0.18.3",
    "torch==2.11.0",
    "torchvision==0.26.0",
]

# 注意这里以pytorch-cu128为例
# 如果你使用的CUDA版本不同，请替换为对应的版本，请务必确认你的CUDA版本和PyTorch版本的兼容性。

# 如果你使用国外环境如colab，请使用官方源，否则可能反向加速，导致安装速度更慢。

# 指定torch和torchvision的安装源
[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]

# 南京大学的PyTorch镜像源
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://mirrors.nju.edu.cn/pytorch/whl/cu128"
explicit = true
# 官方源: https://download.pytorch.org/whl/cu128

# 清华大学的PyPI镜像源（作为默认源），如果你已经在uv全局配置文件设置了默认源，这里可以省略。
[[tool.uv.index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
default = true
# 官方源: https://pypi.org/simple
```

使用`uv sync`命令安装依赖：

```shell
uv sync
```

#### 5. 下载模型 (如果你想从0训练模型，可以跳过这一步)

```shell
mkdir out
wget -O ./out/full_sft_768.pth https://www.modelscope.cn/models/gongjy/minimind-3-pytorch/resolve/master/pretrain_zero_768.pth
```

#### 6. 下载数据集

```shell
wget -P ./dataset https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/lora_medical.jsonl
wget -P ./dataset https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/lora_identity.jsonl
# 从0训练需要下载以下数据集
# wget -P ./dataset https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/pretrain_t2t_mini.jsonl
# wget -P ./dataset https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/sft_t2t_mini.jsonl

```

#### 7. 进行训练和微调

此处在 MiniMind 仓库有详细说明，你需要参考 README 完成微调过程。

> [!NOTE]
>
> 1. 使用uv的同学需要使用`uv run`跑脚本  
> 2. 请注意运行脚本的目录  
> 3. 想要完成Bonus的同学需要开启训练可视化
> 4. 模型训练中断是可以恢复的，具体参考MiniMind文档
> 5. 模型能力有限，别指望它和豆包打一架

#### 8. 测试模型

| 类型 | 测试问题示例 | 预期行为 |
|------|-------------|----------|
| 身份询问 | "你是谁？" | 回答设定的身份信息 |
| 身份追问 | "谁创造了你？" | 回答创造者信息 |
| 医疗知识 | "什么是糖尿病？" | 给出基本正确的医学解释 |
| 医疗建议 | "感冒了怎么办？" | 给出合理的建议 |
| 组合测试 | "你是谁？你懂医学吗？" | 先确认身份，再展示医疗能力 |
| 边界测试 | "帮我写一个冒泡排序" | 观察是否仍保留基础能力 |

#### 作业要求

- 你需要完成医疗微调和身份微调，得到两个 LoRA 适配器。
- 然后你需要将这两个适配器进行组合，得到一个同时具备医疗知识和特定身份的模型。
- 你需要在本地测试微调后的模型，验证其是否具备医疗知识和特定身份。
- 你需要撰写一份报告，总结你的微调过程、遇到的挑战以及最终的结果 (需要包含步骤8中模型的测试结果)。
- (Bonus) 从0训练模型。
- (Bonus) 在报告里给出 Loss 曲线 (使用swanlab或wandb可视化)。
- (Bonus) 使用 peft 库重写lora微调脚本。

### 作业 3 - 番茄助手

猫猫喜欢看小说，但是小说实在是太多了，他经常不知道该看哪一本，所以他想要一个番茄助手，来帮他推荐小说。

你需要编写脚本爬取知名小说网站[笔趣阁的总榜](https://www.piquge.com/paihangbang/allvisit/)，获取小说的标题、简介、作者、标签等信息，然后将它们向量化存储。然后搭建一个简单AI工作流，当猫猫输入一个小说的简介时，番茄助手会使用 RAG 的方式，先从向量数据库中检索出与输入简介最相似的小说信息，然后将这些信息作为上下文输入到大语言模型中，最后输出推荐结果。

#### 参考资料

1. [LangChain Python Docs](https://python.langchain.com/docs/get_started/introduction)
2. [LangChain Expressions Language (LCEL) 教程](https://python.langchain.com/docs/expression_language/)
3. [Dify 官方文档](https://docs.dify.ai/zh/use-dify/getting-started/introduction)

#### 作业要求

- 爬取笔趣阁总榜10页的小说信息，至少包含标题、简介、作者、标签字段。
- 你需要使用 LangChain / Dify 等集成 AI 框架来搭建这个工作流。
- 当没有检索到相关小说时，模型应该给出合理的提示，而不是胡乱编造一个小说。
- (Bonus) 使用 LangChain 和 Dify 分别实现一次。

### 作业 4 - Potato Code Pro

Potato Code 的能力明显不足以满足天才程序员的Coding需求了，所以我们需要一个更加强大的智能编程助手，Potato Code Pro！

你需要学习 learn-claude-code 的 s05 到 s11，完成一个基本的 Agent。

#### 作业要求

- 学习 learn-claude-code 的 s05 到 s11 的内容。
- 你需要自己编写一个简单的 Agent，并且覆盖 s05 到 s11 中的所有功能。

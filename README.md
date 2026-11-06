# 给软件工程师的大语言模型课

**从一个参数到一段回答：把原理算清楚，把代码跑起来，再解释结果与局限。**

面向会编程、没系统学过机器学习的软件工程师。中文正文，18章，覆盖必要数学、神经网络、Transformer、CPU小模型训练、推理和证据评价。无需先学完整经典ML或微积分课程；首次需要的数学会用小数值展开。

[开始阅读](chapters/01-models.md) · [本地离线阅读](site/index.html) · [环境与运行](SETUP.md) · [English overview](README.en.md)

## 十分钟先看见一个机制

1. 读[第10章注意力](chapters/10-attention.md)的一行手算。
2. 打开[交互页](site/10-attention.html#attention-playground)，切换因果遮罩，改变未来输入。
3. 从仓库根目录运行：

```bash
python3 labs/attention.py
```

有因果遮罩时，改变未来位置不会改变较早位置的输出。这个实验只用Python标准库，不下载模型，也不需要GPU或API key。完整学习请回到第01章。

## 学习路线

| 部分 | 章节 | 你将完成 |
|---|---|---|
| 必要基础 | [01 模型与系统](chapters/01-models.md) · [02 数学](chapters/02-math.md) · [03 计数模型](chapters/03-counting.md) | 把预测、数据、概率、损失分开，建立可比较基线 |
| 神经网络 | [04 可训练参数](chapters/04-parameters.md) · [05 梯度更新](chapters/05-gradients.md) · [06 反向传播](chapters/06-backprop.md) · [07 训练](chapters/07-training.md) | 从手算一步到autograd与真实小网络训练 |
| 文本 | [08 分词](chapters/08-tokenization.md) · [09 上下文](chapters/09-context.md) | 理解token、查表、固定窗口与信息汇总 |
| Transformer | [10 注意力](chapters/10-attention.md) · [11 位置与多头](chapters/11-position-heads.md) · [12 Decoder](chapters/12-decoder.md) | 核对Q/K/V、mask、张量形状和完整输出路径 |
| 训练与生成 | [13 下一token](chapters/13-next-token.md) · [14 有限语法](chapters/14-toy-training.md) · [15 自然文本](chapters/15-real-text.md) | 运行CPU实验，比较计数基线并解释失败 |
| 工程与验收 | [16 后训练](chapters/16-post-training.md) · [17 缓存与推理](chapters/17-inference.md) · [18 证据评价](chapters/18-evaluation.md) | 区分训练信号，验证缓存等价，提交有依据的评价报告 |

已有ML基础可从08进入，矩阵/梯度卡点回查02、05、06。各章都包含手算或具体判断、故障反例与带答案练习；勾选读完不自动等于掌握。

## 三层实验，各自证明什么

| 层次 | 能检查 | 不能据此声称 |
|---|---|---|
| 给定小数字 | 计算链、形状与因果可见性 | 学会语言或获得语义 |
| 有限语法decoder | 训练、验证选择、生成、与计数模型比较 | 真实语言理解或业务能力 |
| 英文原作小语料 | 真实文本的损失、样本和失败 | 通用助手能力、事实正确性、中文能力 |

发布配置的真实结果：有限语法模型CE约0.1742，bigram约0.1917；英文原作模型CE约2.4506，bigram约2.4425，**小decoder并未胜过bigram**，生成也会重复。完整配置、目标数和限制见[有限语法报告](reports/toy.md)、[自然文本报告](reports/real.md)。这些是固定数据、单seed运行，不是未使用过的新盲测。

## 运行

Python3.11/3.12；训练依赖PyTorch，标准库实验无需第三方包。平台安装细节与排查见[SETUP](SETUP.md)。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python -m llm_course.train --dataset toy --steps 200 --out outputs/toy
.venv/bin/python -m llm_course.train --dataset real --steps 300 --out outputs/real
.venv/bin/python -m llm_course.experiments
.venv/bin/python -m unittest discover -s tests -v
```

结果随运行保存；不要覆盖之前的实验。模型小、预算有限，课程重视机制与可验证性，不承诺训练出可用聊天机器人。LoRA是可选的局部适配实验，结课使用手边模型记录实际回答，不要求搭完整RAG/Agent平台。

## 贡献、来源和边界

[贡献与勘误](CONTRIBUTING.md) · [来源与制作说明](NOTICE.md) · [数据许可](data/README.md) · [MIT](LICENSE)

课程使用AI协作编写，重要数字与实现经程序核对；尚无真实初学者的系统试读结论。欢迎报告理解卡点、数字不一致和复现失败。原始资料与实际运行证据比笼统质量评分更重要。

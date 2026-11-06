# 来源、许可与制作说明

课程的原创讲解、示意图、程序和练习按根目录MIT许可证提供。程序使用Python标准库、PyTorch与Python-Markdown；第三方组件保留各自许可。

## 文本资料

Alice实验数据不是课程作者的原创作品，不纳入本课程MIT授权。原作、来源、原文件、处理步骤和许可说明见[data/README.md](data/README.md)与[data/real-provenance.json](data/real-provenance.json)。原电子书完整声明随源文件保留。

教学服务短语、问答文档和手算矩阵为本课程自编。它们不是实际生产日志或真实设备手册。

## 学术和软件来源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：缩放点积注意力、多头和Transformer原始架构。当前代码为教学用decoder-only/Pre-LN模型，不是原论文翻译系统复现。
- [Layer Normalization](https://arxiv.org/abs/1607.06450)：归一化原理。
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)：自回归语言建模背景。
- [InstructGPT](https://arxiv.org/abs/2203.02155)、[DPO](https://arxiv.org/abs/2305.18290)、[LoRA](https://arxiv.org/abs/2106.09685)：后训练与参数高效适配。课程局部实验不代表完整复现这些论文。
- [PyTorch](https://pytorch.org/)与[Python-Markdown](https://python-markdown.github.io/)：执行与文档构建依赖。

## 制作边界

本课程由维护者结合AI协作编写并检查，数值和程序用可执行检查验证；模型意见不能代替实际读者反馈。发布时的测试、环境与限制见reports与发行记录。未进行真实无ML背景读者的系统试读，不宣称教育效果已经验证。欢迎提交具体卡点和可复现反例。

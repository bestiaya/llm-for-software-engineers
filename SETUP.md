# 环境、运行与排查

本页帮助你选择最小运行环境。阅读网页和手算不需要安装任何包；标准库实验只需Python，训练实验再安装PyTorch。

## 先确认Python

建议Python 3.11或3.12。终端运行`python3 --version`；Windows通常使用`py`或`python`。下面命令都从仓库根目录运行。

```bash
python3 labs/attention.py
python3 labs/foundations.py
python3 labs/tokenization.py
python3 labs/evaluation.py
```

这些实验不下载模型、不需要GPU或API key。

神经网络的小MLP练习需先装下面的PyTorch，再运行`.venv/bin/python -m llm_course.mlp`。

## 运行小模型训练

创建独立环境，不覆盖系统Python：

```bash
python3 -m venv .venv
# macOS / Linux
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python -m llm_course.train --dataset toy --steps 200 --out outputs/toy
.venv/bin/python -m llm_course.train --dataset real --steps 300 --out outputs/real
.venv/bin/python -m llm_course.experiments
.venv/bin/python -m unittest discover -s tests -v
```

Linux只用CPU时可将安装命令改成`.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0`，避免安装不需要的GPU依赖；随后运行`.venv/bin/python -m pip install numpy==2.3.5`补齐本文档环境的数组桥接依赖。

章节中简写的`python3`应指向已安装依赖的解释器；若未激活环境，运行含PyTorch的命令时请使用上面的`.venv/bin/python`。

Windows把`.venv/bin/python`替换为`.venv\Scripts\python`。若已有同名输出目录，先阅读原报告，改用新的`--out outputs/your-run`，不要混淆运行版本。

每次训练输出配置、词表、损失、基线、样本、耗时与局限。阅读[有限语法报告](reports/toy.md)和[真实文本报告](reports/real.md)理解当前发布版本实际做过什么。单次运行可复现不等于新数据泛化验证。

## 重建阅读页面

Markdown是内容源，生成HTML不依赖CDN。需要重建时：

```bash
.venv/bin/python -m pip install -r requirements-build.txt
.venv/bin/python tools/build.py
.venv/bin/python tools/check.py
```

直接打开`site/index.html`即可阅读。某些浏览器对本地文件的进度保存有限制，可运行`python3 -m http.server 8000`，访问`http://127.0.0.1:8000/site/`。阅读勾选只保存在该浏览器和地址，不会同步为学习认证。

## 常见问题

| 现象 | 优先处理 |
|---|---|
| 找不到脚本 | 确认终端在仓库根目录，路径大小写一致 |
| No module named torch | 检查是否用安装依赖的同一个虚拟环境解释器 |
| PyTorch安装找不到对应包 | 检查Python版本和CPU架构；先按PyTorch官方安装说明匹配平台，不将CUDA当CPU课程必需项 |
| 输出无法结束 | 查看stop_reason，区分EOS、token预算和context上限；不自行断言模型永远不结束 |
| 结果和报告不完全相同 | 比较seed、依赖、输入hash与配置；不同平台的浮点数和训练轨迹可能略异 |
| 测试过了但样本很差 | 检查测试证明的是计算机制还是语言能力；报告保留粗糙样本，不以运行成功代替效果 |

依赖来源：[Python](https://www.python.org/)、[PyTorch官方安装](https://pytorch.org/get-started/locally/)、[Python-Markdown](https://python-markdown.github.io/)。本课程没有要求登录任何模型平台。

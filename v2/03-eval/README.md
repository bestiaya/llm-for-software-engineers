# K01-03 第一份评测

上一章写好了新约定和十条期望行为。本章用这同一份标准，逐题评价沿用旧约定的应用和按新约定工作的应用，再检验评分程序本身靠不靠得住。

- [阅读正文](lesson.md)。
- [真实模型回答的记录](records/README.md)：十条输入在两版约定下各一份。
- [人工构造的错误输出](inputs/synthetic-outputs.json)：只用来检验评分程序。
- 标准来自第 02 章：[十条输入和期望行为](../02-contract/inputs/cases.json)。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。本章的实验全部离线，回放随课保存的记录，不调用模型，也不发送天气查询：评的是应用的判定。

```bash
python3 v2/03-eval/lab.py
python3 v2/03-eval/lab.py --version v1
python3 v2/03-eval/lab.py --compare
```

第一条逐题评价新版应用，第二条评价旧版应用，第三条用同一份标准比较两版。加 `--case w07` 只看一题，加 `--json` 看全部字段。

## 检验评分程序

四组人工构造的错误输出，各自应该在哪几题上被判为未通过？先预测，再运行。

```bash
python3 v2/03-eval/lab.py --synthetic always-3-days
python3 v2/03-eval/lab.py --synthetic fabricate-city
python3 v2/03-eval/lab.py --synthetic refuse-all
python3 v2/03-eval/lab.py --synthetic wrong-refusal
```

## 用你自己写的期望行为评分

如果你在第 02 章填写了 `my-expected.json`，可以用它替换那三题的参考答案。输出会标明哪几题用的是你写的标准；写法不对时会先报错，不进入评分：

```bash
python3 v2/03-eval/lab.py --expected my-expected.json
```

[实验脚本](lab.py) · [采集程序](capture.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

# K01-04 从例子里学

前三章用的是别人训练好的模型。本章换一个小到每一步都能算出来的任务，先把“模型怎样被样本教会”的全过程看一遍，再认识输入、标签、标注规则和数据的三种用途。

- [阅读正文](lesson.md)。
- [16 行数据](../data/counter-task.json)：两条通道的超时次数、标签和每一行的用途。
- [一次训练的记录](../data/network-training-record.json)：本章只回放，第 10 章细读。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。本章的实验全部离线，不训练。

```bash
python3 v2/04-task/lab.py
python3 v2/04-task/lab.py --preview
python3 v2/04-task/lab.py --leak
```

第一条逐行核对 16 行数据的标签和三组的划分，第二条按四步回放一次完整的学习过程，第三条演示答案进了输入的反例。

## 看核对能发现什么

下面两条命令在内存里各改错一处，文件不变：第一条改错一个标签，第二条把训练组的一行原样放进验证组。核对发现问题时退出码是 1，`--leak` 也是。

```bash
python3 v2/04-task/lab.py --fault wrong-label
python3 v2/04-task/lab.py --fault overlap
```

加 `--json` 看全部字段。

[实验脚本](lab.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

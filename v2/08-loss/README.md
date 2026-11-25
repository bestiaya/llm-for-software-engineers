# K01-08 从分数到概率和损失

选对几行这把尺子太粗。本章分两步做一把更细的：先用 softmax 把三个分数变成三个概率，再用正确类别的概率算损失，并对很多行取平均。同时说明模型给的概率不自动等于现实里的正确率。

- [阅读正文](lesson.md)。
- 网络的参数沿用上一章手工设定的 27 个，没有经过训练。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。本章的实验全部离线，不训练。

```bash
python3 v2/08-loss/lab.py
python3 v2/08-loss/lab.py --row 0 2
python3 v2/08-loss/lab.py --losses=0.25,0.5,0.75
python3 v2/08-loss/lab.py --all
python3 v2/08-loss/lab.py --all --scale 3
python3 v2/08-loss/lab.py --disagree
```

第一条用一组好算的分数走一遍“取指数、求和、各自除以总和”。第二条让一行输入走过手设的网络，再算概率和损失。第三条给几个“正确类别的概率”，算各自的损失和平均。第四、五条看 16 行的平均损失，以及把输出层乘 3 以后的变化。第六条是选对的行数和平均损失说法不一样的例子，数是人工构造的。

自己给分数时用等号的写法，负数才不会被当成选项：

```bash
python3 v2/08-loss/lab.py --scores=-3,0,-3 --label 1
```

`--row`、`--all` 可以加 `--set` 在内存里改网络的一个参数，写法见[第 07 章的说明](../07-hidden/README.md)。加 `--json` 看全部字段。

[实验脚本](lab.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

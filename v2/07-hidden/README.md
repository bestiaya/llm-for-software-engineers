# K01-07 一层不够

上一章那组参数只选对 7 行。本章先用三行数据说明：只做乘加的模型，在输入是原始次数时，怎样取参数都至少错一行。再给模型加上隐藏层和 ReLU，跟着一行输入把新的计算走一遍。

- [阅读正文](lesson.md)或[打开离线页](lesson.html)。
- 网络的 27 个参数在[共享代码](../shared/network.py)的 `hand_set_parameters` 里，手工设定，没有经过训练。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。本章的实验全部离线，不训练。

```bash
python3 v2/07-hidden/lab.py --linear-limit
python3 v2/07-hidden/lab.py
python3 v2/07-hidden/lab.py --a 3 --b 1
python3 v2/07-hidden/lab.py --all
python3 v2/07-hidden/lab.py --all --no-relu
```

第一条看线性模型在三行数据上的限制，加 `--seed 7` 换一组随机取的参数。第二、三条跟踪一行输入走过隐藏层。第四条把 16 行都算一遍。第五条去掉 ReLU，看两层乘加还剩什么。

`--set` 可以在内存里改网络的一个参数，可以写多次。写法：`W1[a,h1]` 是输入 a 通向隐藏单元 h1 的权重，`b1[h2]` 是 h2 的偏置，`W2[h1,0]` 是 h1 通向类别 0 的权重，`b2[1]` 是类别 1 的输出偏置。例如：

```bash
python3 v2/07-hidden/lab.py --all --set "b1[h2]=-3"
```

加 `--json` 看全部字段。

[实验脚本](lab.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

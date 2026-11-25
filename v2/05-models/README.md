# K01-05 模型就是参数加一套算法

上一章里“模型里那批可以调的数”，本章拿出来看：九个参数，一套乘加的算法。亲手把一行输入算成三个分数，取最大的得到类别，再分清改输入、改参数、改外部数据三种改动。

- [阅读正文](lesson.md)。
- [九个参数](../data/linear-model.json)，以及同样九个数的[二进制形式](../data/linear-parameters.f32)。参数是手工设定的，没有经过训练。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。本章的实验全部离线，只做前向计算。

```bash
python3 v2/05-models/lab.py
python3 v2/05-models/lab.py --a 3
python3 v2/05-models/lab.py --set "W[a,0]=1"
python3 v2/05-models/lab.py --storage
```

第一条算 a=2、b=1 这一行，第二条只换输入，第三条只在内存里改一个参数，第四条看九个参数占多少存储、读的时候为什么需要结构约定。

`--a`、`--b` 可以换成任意一行输入。`--set` 的写法：`W[a,0]` 是输入 a 通向类别 0 的权重，`bias[1]` 是类别 1 的偏置；可以写多次。输入里已经有一个 b，所以偏置写作 bias。改动只在这一次运行里有效，文件不变。

选读内容的实验：

```bash
python3 v2/05-models/lab.py --fingerprint
```

加 `--json` 看全部字段。

[实验脚本](lab.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

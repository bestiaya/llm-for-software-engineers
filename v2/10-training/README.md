# K01-10 学得怎样

把上一章的一次更新重复两千次，就是训练。本章实际训练一遍，用同一次运行比较训练、验证、留出三组数据，对照两个简单的基线，认识欠拟合与过拟合，再看换一个起点结果会怎样。章末是阶段二的阶段检查。

- [阅读正文](lesson.md)。
- [随课保存的训练记录](../data/network-training-record.json)：第 04 章回放过的就是它。
- 网络的计算和训练循环在[共享代码](../shared/network.py)里。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。本章的实验全部离线，只用标准库。

```bash
python3 v2/10-training/lab.py
python3 v2/10-training/lab.py --at 10
python3 v2/10-training/lab.py --linear
python3 v2/10-training/lab.py --seeds 20
python3 v2/10-training/lab.py --inspect
```

第一条按声明的设置重新训练一遍，不到一秒，并与随课保存的记录逐项比较；一致时退出码 0。第二条训练到指定的步数停下，逐行看训练的 10 行各选了哪一类。第三条训练一个没有隐藏层的线性模型作对照。第四条换 20 个起点各训练一遍，要几秒钟。第五条看训练得到的参数。

阶段检查的核对：

```bash
python3 v2/10-training/lab.py --stage-check
```

`--write-record` 是课程维护用的，读者不需要运行；记录已存在时它会拒绝覆盖。加 `--json` 看全部字段。

[实验脚本](lab.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

# K01-06 一次算一批

上一章一行一行地算。本章把几行输入排成矩阵，一次算出每一行、每个类别的分数，用形状判断能不能算，用轴说明每个位置代表什么，再看一种形状正确、含义错误的计算。

- [阅读正文](lesson.md)或[打开离线页](lesson.html)。
- 参数沿用上一章的[九个参数](../data/linear-model.json)，手工设定，没有经过训练。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。本章的实验全部离线。

```bash
python3 v2/06-math/lab.py
python3 v2/06-math/lab.py --fault shape
python3 v2/06-math/lab.py --fault axis
python3 v2/06-math/lab.py --all
```

第一条先算一行的三次点积，再把两行一起算。第二条让形状对不上，程序在计算之前停下，退出码 1。第三条把输入的行和列对调，形状仍然对得上，程序不报错，退出码 0，结果却是错的。第四条把 16 行一起算。

加 `--json` 看全部字段。

[实验脚本](lab.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

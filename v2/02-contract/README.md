# K01-02 把任务说清楚

上一章的约定只有城市和天数两个字段。遇到没说城市、超出能查的范围、根本不是查天气的话，模型连该怎么答都没有地方写。本章把约定扩成查询、请求澄清、不支持三种结果，再为十条输入写下期望行为。

- [阅读正文](lesson.md)或[打开离线页](lesson.html)。
- [新约定给模型的指令](inputs/contract-v2.json)。
- [十条输入和参考的期望行为](inputs/cases.json)。
- [两版约定的判定代码](../shared/weather_app.py)：读模型交回的文本，决定应用下一步做什么。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。本章的实验全部离线，不调用模型。

```bash
python3 v2/02-contract/lab.py
python3 v2/02-contract/lab.py --try '{"result":"query","city":"成都","days":30}'
python3 v2/02-contract/lab.py --version v1 --try '{"city":"成都","days":30}'
python3 v2/02-contract/lab.py --cases
```

第一条用几段手写的示例文本演示约定；第二、三条把你自己写的文本交给新旧两版应用；第四条列出十条输入和参考的期望行为。实验只给出应用的决定和它组装的查询，不发送天气查询。

## 自己填三条期望行为

复制 [inputs/expected-starter.json](inputs/expected-starter.json)，把 w04、w06、w09 三个 `null` 换成你认为应用该有的行为，再核对。程序先检查写法，再逐项与参考答案对照：

```bash
cp v2/02-contract/inputs/expected-starter.json my-expected.json
python3 v2/02-contract/lab.py --check my-expected.json
```

[实验脚本](lab.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

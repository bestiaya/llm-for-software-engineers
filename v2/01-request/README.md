# K01-01 一次请求怎样走完整个系统

从“帮我看看成都未来三天的天气”走完一次天气查询。模型从这句话里提取城市和天数，交回一段文本；应用解析这段文本，用两个参数查询 Open-Meteo，把天气服务交回的逐日预报展示给用户。

- [阅读正文](lesson.md)或[打开离线页](lesson.html)。
- [模型的请求与回答](records/chengdu-gpt.json)记录模型收到了什么、交回了什么。
- [天气服务快照](records/chengdu-forecast.json)记录查询地址、采集时间、日期与气温。
- [两份记录是怎样采集的](records/README.md)。

先按[课程导读的实验准备](../ABOUT.md#lab-setup)进入课程仓库根目录。建议亲手运行第一条命令，沿着交出去的文字、交回的文本、解析出的参数、天气数据、最终展示这五步看输出。第二条以 JSON 展示各阶段的字段。

```bash
python3 v2/01-request/lab.py
python3 v2/01-request/lab.py --json
```

默认运行回放两份随课保存的记录，不联网，不重新调用模型。

## 选做

改动交给解析程序的回答副本，看程序在哪一步发现格式错误或天数不符：

```bash
python3 v2/01-request/lab.py --fault invalid-json
python3 v2/01-request/lab.py --fault wrong-days
```

想自己采集一份新的模型回答，可以先配置 `OPENAI_API_KEY`，再指定一个尚不存在的输出文件。天气展示仍取随课的历史快照。

```bash
python3 v2/01-request/lab.py --live --out chengdu-gpt-new.json
```

[实验脚本](lab.py) · [教学蓝图](LESSON_BLUEPRINT.md) · [检查记录](review.md)

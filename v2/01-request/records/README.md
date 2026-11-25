# 成都天气查询的两份记录

[GPT 记录](chengdu-gpt.json)保存课程实际提交的指令、用户问题和 GPT 返回的文本。2026-11-18，采集程序通过 Codex CLI 0.166.0 请求 `gpt-6.1-sol`，推理档位设为 `medium`。课程提交给 CLI 的完整文本在 `capture.stdin`，由 `instructions`、空行、`input` 组成。CLI 的最终消息事件保存在 `cli_message_event`；其正文和 `response.text` 一致。本条记录没有服务端独立回报的模型标识，`model_reported` 保留为 `unknown`。CLI 管理的额外上下文和底层 HTTP 报文没有记录在本文件里。

[Open-Meteo 预报快照](chengdu-forecast.json)保存了另一次 HTTP 请求的 URL、采集时间和天气服务返回的数据。该请求在成都当地 2026-11-18 查询接下来 3 个自然日（含当天）的每日最低和最高气温，时区为 `Asia/Shanghai`，单位为 °C。这里的数值是预报采集时的预测，来源与授权见 [Open-Meteo](https://open-meteo.com/) 和 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

默认实验从本地读取两份记录。程序先解析回答、核对城市和天数，再按城市选择经纬度，构造同一条天气查询，用对应的服务快照生成展示。终端日期取自采集记录。选做 `--live --out NEW_FILE` 通过 OpenAI Responses API 重新采集 GPT 回答，天气展示仍取这份历史快照；`--record NEW_FILE` 可以离线回放新记录。两种采集入口使用相同的课程指令和问题，客户端封装、模型和用量口径可不同。

## 采集说明

本次 Sol 采集复用了已保存的同日 Open-Meteo 快照。GPT 和天气数据各自的采集时间见对应记录。

本章开篇例子改成天气查询之前，这个目录里还有一组网关手册练习的输入和记录。它们与现在的内容无关，已于 2026-11-23 移出课程目录。

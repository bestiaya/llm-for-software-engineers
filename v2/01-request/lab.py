"""跟随一次天气查询：取得 GPT 参数，读取天气服务快照，显示每日预报。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlsplit
from urllib.request import Request, urlopen


CHAPTER = Path(__file__).resolve().parent
FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
DURATION_SCOPE = "Responses API HTTP request and response body wall time"
GPT_MODEL_IDENTIFIER = re.compile(r"gpt-[A-Za-z0-9][A-Za-z0-9._-]*")


class LabError(Exception):
    """输入或记录有误，无法完成实验。"""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 对象重复出现字段 {key}")
        result[key] = value
    return result


def parse_parameters(text):
    try:
        value = json.loads(text, object_pairs_hook=unique_object)
    except ValueError as error:
        raise LabError(f"无法解析 GPT 返回的 JSON：{error}") from error
    if not isinstance(value, dict) or value.keys() != {"city", "days"}:
        raise LabError("查询参数需要 city、days 两个字段。")
    if not isinstance(value["city"], str) or type(value["days"]) is not int:
        raise LabError("city 须为城市名，days 须为整数。")
    return value


def validate_case(parameters):
    if parameters["city"] != "成都":
        raise LabError(f"用户要查成都，返回的城市为 {parameters['city']}。")
    if parameters["days"] != 3:
        raise LabError(f"用户要查 3 天，返回的 days 为 {parameters['days']}。")


def forecast_from_snapshot(parameters):
    locations = {"成都": (30.67, 104.07, "Asia/Shanghai", "chengdu-forecast.json")}
    latitude, longitude, local_timezone, filename = locations[parameters["city"]]
    query = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": local_timezone,
        "forecast_days": str(parameters["days"]),
    }
    snapshot = read_json(CHAPTER / "records" / filename)
    saved_url = urlsplit(snapshot["url"])
    if (saved_url.scheme, saved_url.netloc, saved_url.path) != ("https", "api.open-meteo.com", "/v1/forecast"):
        raise LabError("快照中的天气服务地址与本章使用的接口不符。")
    if sorted(parse_qsl(saved_url.query)) != sorted(query.items()):
        raise LabError("天气快照与本次查询的地点、时区、天数或温度字段不匹配。")
    body = snapshot["body"]
    daily = body["daily"]
    units = body["daily_units"]
    if body["timezone"] != local_timezone or any(units[key] != "°C" for key in ("temperature_2m_min", "temperature_2m_max")):
        raise LabError("快照的时区或温度单位不匹配。")

    columns = [daily[key] for key in ("time", "temperature_2m_min", "temperature_2m_max")]
    if any(not isinstance(column, list) or len(column) != parameters["days"] for column in columns):
        raise LabError("日期与最低、最高温三列都须覆盖用户要查的天数。")
    rows = []
    for date, low, high in zip(*columns, strict=True):
        if not isinstance(date, str) or not all(type(number) in (int, float) and math.isfinite(number) for number in (low, high)):
            raise LabError("天气快照缺少可显示的日期或温度。")
        rows.append({"date": date, "min_c": low, "max_c": high})
    return {
        "mode": "snapshot",
        "provider": snapshot["provider"],
        "retrieved_at_utc": snapshot["retrieved_at_utc"],
        "url": FORECAST_ENDPOINT + "?" + urlencode(query),
        "rows": rows,
    }


def token_count(usage, field):
    value = usage.get(field)
    if type(value) is int and value >= 0:
        return value
    return None


def call_model(request, model, effort):
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise LabError("实时调用需要在终端配置 OPENAI_API_KEY。")
    payload = {
        "model": model,
        "instructions": request["instructions"],
        "input": request["input"],
        "reasoning": {"effort": effort},
        "store": False,
    }
    http_request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    started = time.monotonic()
    try:
        with urlopen(http_request, timeout=90) as response:
            body = response.read()
    except HTTPError as error:
        raise LabError(f"OpenAI Responses API 调用失败（HTTP {error.code}）。") from error
    duration_seconds = time.monotonic() - started
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    try:
        result = json.loads(body)
    except ValueError as error:
        raise LabError("OpenAI Responses API 未返回可解析的 JSON。") from error
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise LabError("OpenAI Responses API 未完成生成。")
    output = result.get("output")
    if not isinstance(output, list):
        raise LabError("OpenAI Responses API 返回的 output 不是列表。")
    texts = [
        part["text"]
        for item in output
        if isinstance(item, dict) and item.get("type") == "message" and isinstance(item.get("content"), list)
        for part in item["content"]
        if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str)
    ]
    text = "".join(texts)
    if not text.strip():
        raise LabError("OpenAI Responses API 未返回模型回答文本。")
    actual_model = result.get("model")
    if not isinstance(actual_model, str) or not GPT_MODEL_IDENTIFIER.fullmatch(actual_model):
        actual_model = "unknown"
    usage = result.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}

    return {
        "schema_version": 1,
        "case_id": "chengdu-weather",
        "capture": {
            "kind": "live-model",
            "transport": "openai-responses-api",
            "model_requested": model,
            "model_reported": actual_model,
            "effort_requested": effort,
            "captured_at_utc": captured_at,
            "duration_seconds": round(duration_seconds, 3),
            "duration_scope": DURATION_SCOPE,
            "input_scope": "记录通过 instructions 与 input 提交给 API 的课程指令和用户内容。",
        },
        "request": request,
        "response": {
            "text": text,
            "usage": {
                "input_tokens": token_count(usage, "input_tokens"),
                "cache_creation_input_tokens": token_count(input_details, "cache_write_tokens"),
                "cache_read_input_tokens": token_count(input_details, "cached_tokens"),
                "output_tokens": token_count(usage, "output_tokens"),
            },
        },
    }


def save_live_record(path, request, model, effort):
    if path.exists() or path.is_symlink():
        raise LabError("--out 须指定尚不存在的新文件。")
    if not path.parent.is_dir():
        raise LabError("请给 --out 指定一个已存在的目录。")
    record = call_model(request, model, effort)
    serialized = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(serialized)
    except FileExistsError as error:
        raise LabError("调用期间 --out 已被创建，已有文件已保留。") from error
    return record


def run(args):
    request = read_json(CHAPTER / "inputs/weather-request.json")
    if args.live:
        record = save_live_record(args.out, request, args.model, args.effort)
    else:
        record = read_json(args.record or CHAPTER / "records/chengdu-gpt.json")
    if record["request"] != request or record.get("case_id") != "chengdu-weather":
        raise LabError("保存的 GPT 记录与本章天气请求不匹配。")
    raw_text = record["response"]["text"]
    if not isinstance(raw_text, str):
        raise LabError("记录缺少原始 GPT 回答文本。")

    result = {
        "mode": "live" if args.live else "replay",
        "request": request,
        "capture": record["capture"],
        "response": record["response"],
        "parameters": None,
        "weather": None,
        "display": None,
        "exit_code": 0,
    }
    checked_text = raw_text
    if args.fault:
        if args.fault == "invalid-json":
            checked_text = "{" + raw_text
        else:
            changed = parse_parameters(raw_text)
            changed["days"] = 4
            checked_text = json.dumps(changed, ensure_ascii=False)
        result["injected_fault"] = {"type": args.fault, "text": checked_text}
    try:
        parameters = parse_parameters(checked_text)
        result["parameters"] = parameters
        validate_case(parameters)
    except LabError as error:
        result.update(exit_code=1, error=str(error))
        return result

    weather = forecast_from_snapshot(parameters)
    result["weather"] = weather
    lines = [f"{parameters['city']} · {parameters['days']} 天预报（{weather['provider']} 历史快照）"]
    lines.extend(f"{row['date']}  {row['min_c']:.1f} ～ {row['max_c']:.1f} °C" for row in weather["rows"])
    result["display"] = "\n".join(lines)
    return result


def print_run(result):
    print(f"mode: {result['mode']}")
    print("【应用提交给 GPT】", result["request"]["instructions"], result["request"]["input"], sep="\n")
    print("【GPT 返回的原始文本】", result["response"]["text"], sep="\n")
    if "injected_fault" in result:
        print("【交给应用的故障副本】", result["injected_fault"]["text"], sep="\n")
    if result["parameters"] is not None:
        print("【应用解析出的查询参数】", json.dumps(result["parameters"], ensure_ascii=False), sep="\n")
    if result["weather"] is None:
        print("【应用停止】", result["error"], sep="\n")
    else:
        weather = result["weather"]
        print("【天气查询 · 回放已保存的服务响应】", weather["url"], "采集时间 " + weather["retrieved_at_utc"], sep="\n")
        print("【应用展示】", result["display"], sep="\n")
    print(f"exit_code: {result['exit_code']}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="重新调用 GPT，天气仍取自随课快照")
    mode.add_argument("--record", type=Path, help="回放另行采集的 GPT 天气记录")
    parser.add_argument("--fault", choices=("wrong-days", "invalid-json"), help="改动内存中的回答副本")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出各阶段")
    parser.add_argument("--out", type=Path, help="保存新 GPT 记录的文件路径")
    parser.add_argument("--model", default="gpt-6.1-sol", help="实时调用所请求的 GPT 模型")
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high"))
    args = parser.parse_args(argv)
    if args.live:
        if args.out is None:
            parser.error("--live 需要 --out NEW_PATH")
        if args.fault:
            parser.error("--fault 用于回放模式")
        if not GPT_MODEL_IDENTIFIER.fullmatch(args.model):
            parser.error("--model 须为 gpt- 开头的 OpenAI 模型名称")
    elif args.out is not None:
        parser.error("--out 用于实时调用模式")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run(args)
    except (LabError, KeyError, TypeError, OSError, ValueError) as error:
        if args.json:
            print(json.dumps({"error": str(error), "exit_code": 2}, ensure_ascii=False))
        else:
            print(f"实验失败：{error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_run(result)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

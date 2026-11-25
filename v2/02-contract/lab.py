"""把任务说清楚：看约定怎样把模型交回的文本变成应用的行为。全部离线，不调用模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CHAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(CHAPTER.parent))

from shared.weather_app import (  # noqa: E402
    DECIDERS, DEFAULT_DAYS, LOCATIONS, MAX_DAYS, MIN_DAYS, REASONS, expected_problems, same,
)


OUTCOME_NAMES = {"query": "查询", "clarify": "请求澄清", "unsupported": "不支持", "error": "应用停止"}
DECIDED_BY_NAMES = {"model": "照模型的结果", "program": "程序的规则"}

# 手写的示例文本，用来演示约定；它们不是真实模型的回答。
SAMPLES = [
    ("说了城市和天数", '{"result":"query","city":"成都","days":3}'),
    ("没说天数", '{"result":"query","city":"北京","days":null}'),
    ("天数超出范围", '{"result":"query","city":"成都","days":30}'),
    ("没说城市", '{"result":"clarify","missing":"city"}'),
    ("这个版本做不了", '{"result":"unsupported","reason":"涉及多个城市"}'),
    ("还是上一章的两个字段", '{"city":"成都","days":3}'),
]


class LabError(Exception):
    """输入有误，无法完成实验。"""


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LabError(f"读不了 {path}：{error}") from error


def load_cases():
    return read_json(CHAPTER / "inputs/cases.json")["cases"]


def describe(decision):
    """把一次判定写成给人看的几行。"""
    lines = [f"应用行为：{OUTCOME_NAMES[decision['outcome']]}（{DECIDED_BY_NAMES[decision['decided_by']]}）"]
    if decision["reason"]:
        lines.append(f"没有去查的原因：{REASONS[decision['reason']]}")
    lines.append(f"对用户说：{decision['message']}")
    request = decision["weather_request"]
    lines.append("天气查询：" + ("组装了这一条，本实验不发送 " + request["url"] if request else "不组装，也不发送"))
    return lines


def check_expected(path):
    """核对读者填写的期望行为：先看写法，再逐项与参考答案对照。"""
    filled = read_json(path).get("expected")
    if not isinstance(filled, dict) or not filled:
        raise LabError("文件里需要一个 expected 对象，键是输入的编号。")
    reference = {case["id"]: case for case in load_cases()}
    results = []
    for case_id, mine in filled.items():
        if case_id not in reference:
            raise LabError(f"没有编号为 {case_id} 的输入。")
        case = reference[case_id]
        entry = {"id": case_id, "input": case["input"], "yours": mine, "problems": [], "owners": case["owners"]}
        if mine is None:
            entry["problems"].append("还没有填写")
        else:
            entry["problems"] = expected_problems(mine)
            if not entry["problems"]:
                for field, value in case["expected"].items():
                    if field not in mine:
                        entry["problems"].append(f"参考答案还核对 {field}，你没有写")
                    elif not same(mine[field], value):
                        entry["problems"].append(f"{field} 与参考不同：你写的是 {json.dumps(mine[field], ensure_ascii=False)}")
                extra = [field for field in mine if field not in case["expected"]]
                if extra:
                    entry["problems"].append("参考答案没有核对：" + "、".join(extra))
        entry["matches"] = not entry["problems"]
        results.append(entry)
    return results


def run(args):
    if args.check:
        results = check_expected(args.check)
        return {"mode": "check", "results": results, "exit_code": 0 if all(r["matches"] for r in results) else 1}
    if args.try_text is not None:
        decision = DECIDERS[args.version](args.try_text)
        return {"mode": "try", "version": args.version, "text": args.try_text, "decision": decision, "exit_code": 0}
    if args.cases:
        return {"mode": "cases", "cases": load_cases(), "exit_code": 0}
    samples = [{"label": label, "text": text, "decision": DECIDERS[args.version](text)} for label, text in SAMPLES]
    return {
        "mode": "samples",
        "version": args.version,
        "app_rules": {
            "default_days": DEFAULT_DAYS, "min_days": MIN_DAYS, "max_days": MAX_DAYS, "cities": list(LOCATIONS),
        },
        "samples": samples,
        "exit_code": 0,
    }


def print_run(result):
    if result["mode"] == "samples":
        rules = result["app_rules"]
        print(f"约定版本：{result['version']}　以下文本是手写的示例，不是真实模型的回答。")
        print(
            f"应用自己定的规则：没说天数查 {rules['default_days']} 天；只能查 {rules['min_days']} 到 {rules['max_days']} 天；"
            f"配置了查询点的城市是{'、'.join(rules['cities'])}。"
        )
        for sample in result["samples"]:
            print(f"\n【{sample['label']}】模型交回：{sample['text']}")
            for line in describe(sample["decision"]):
                print("  " + line)
    elif result["mode"] == "try":
        print(f"约定版本：{result['version']}　模型交回：{result['text']}")
        for line in describe(result["decision"]):
            print("  " + line)
    elif result["mode"] == "cases":
        for case in result["cases"]:
            print(f"{case['id']}　{case['input']}")
            print("  期望行为：" + json.dumps(case["expected"], ensure_ascii=False))
            print("  各归谁：" + case["owners"])
            if case["note"]:
                print("  注意：" + case["note"])
    else:
        for entry in result["results"]:
            verdict = "与参考一致" if entry["matches"] else "；".join(entry["problems"])
            print(f"{entry['id']}　{entry['input']}\n  {verdict}")
            if not entry["matches"] and entry["yours"] is not None:
                print("  想一想：" + entry["owners"])
    print(f"exit_code: {result['exit_code']}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--try", dest="try_text", metavar="TEXT", help="把一段模型文本交给应用，看它怎样处理")
    mode.add_argument("--cases", action="store_true", help="列出十条输入和参考的期望行为")
    mode.add_argument("--check", type=Path, metavar="FILE", help="核对你填写的期望行为")
    parser.add_argument("--version", choices=("v1", "v2"), default="v2", help="用哪一版约定，默认 v2")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run(args)
    except LabError as error:
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

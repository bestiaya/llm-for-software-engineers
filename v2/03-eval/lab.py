"""第一份评测：回放两版应用在十条输入上的真实记录，逐题对照期望行为。全部离线，不调用模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CHAPTER = Path(__file__).resolve().parent
COURSE = CHAPTER.parent
sys.path.insert(0, str(COURSE))

from shared.weather_app import DECIDERS, REASONS, expected_problems, implied, observe, same  # noqa: E402


CASES_FILE = COURSE / "02-contract/inputs/cases.json"
INSTRUCTION_FILES = {
    "v1": COURSE / "01-request/inputs/weather-request.json",
    "v2": COURSE / "02-contract/inputs/contract-v2.json",
}
VERSION_NAMES = {"v1": "v1 应用（沿用第 01 章的两字段约定）", "v2": "v2 应用（第 02 章的三种结果）"}
OUTCOME_NAMES = {"query": "查询", "clarify": "请求澄清", "unsupported": "不支持", "error": "应用停止"}
FIELD_NAMES = {
    "outcome": "应用行为",
    "city": "查询的城市",
    "days": "查询的天数",
    "default_disclosed": "是否说明用了默认天数",
    "decided_by": "由谁决定",
    "reason": "没有去查的原因",
    "extracted_city": "模型提取的城市",
    "extracted_days": "模型提取的天数",
    "sends_weather_request": "是否组装天气查询",
}
VALUE_NAMES = {**OUTCOME_NAMES, **REASONS, "model": "照模型的结果", "program": "程序的规则", True: "是", False: "否", None: "无"}
SCOPE_NOTE = (
    "这十条题用于演示、开发和回归。通过数是应用判定结果的通过数：只说明应用对这份约定、这十条输入的满足情况。"
    "实验只走到应用的决定，没有向天气服务发送请求。"
)


class LabError(Exception):
    """输入或记录有误，无法完成实验。"""


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LabError(f"读不了 {path}：{error}") from error


def load_cases(expected_file=None):
    """读十条输入。给了 expected_file 时，用读者写的期望行为替换那几题的参考答案。"""
    cases = read_json(CASES_FILE)["cases"]
    for case in cases:
        case["standard"] = "reference"
        case["unchecked"] = []
    if expected_file is None:
        return cases
    filled = read_json(expected_file).get("expected")
    if not isinstance(filled, dict):
        raise LabError("期望行为文件里需要一个 expected 对象。")
    by_id = {case["id"]: case for case in cases}
    for case_id, mine in filled.items():
        if case_id not in by_id:
            raise LabError(f"没有编号为 {case_id} 的输入。")
        if mine is None:
            raise LabError(f"{case_id} 的期望行为还没有填写。")
        problems = expected_problems(mine)
        if problems:
            raise LabError(f"{case_id} 的期望行为写法有问题：" + "；".join(problems))
        case = by_id[case_id]
        case["unchecked"] = [field for field in case["expected"] if field not in mine]
        case["expected"] = mine
        case["standard"] = "reader"
    return cases


def score_case(decision, expected):
    """评分骨架：期望行为里写了哪些项，就逐项与应用实际做的事对照。"""
    observed = observe(decision)
    checks = [
        {"field": field, "expected": value, "actual": observed[field], "ok": same(observed[field], value)}
        for field, value in implied(expected).items()
    ]
    return {"passed": all(check["ok"] for check in checks), "checks": checks}


def judged(case, text, version, source):
    decision = DECIDERS[version](text)
    return {
        "id": case["id"],
        "input": case["input"],
        "model_text": text,
        "source": source,
        "decision": decision,
        "observed": observe(decision),
        "expected": case["expected"],
        "standard": case["standard"],
        "unchecked": case["unchecked"],
        "score": score_case(decision, case["expected"]),
        "note": case.get("note", ""),
    }


def replay(version, cases):
    """用随课保存的真实记录，重新走一遍应用的判定并评分。"""
    instructions = read_json(INSTRUCTION_FILES[version])["instructions"]
    results = []
    for case in cases:
        record = read_json(CHAPTER / "records" / version / f"{case['id']}.json")
        request = {"instructions": instructions, "input": case["input"]}
        if record.get("request") != request or record.get("case_id") != case["id"] or record.get("app_version") != version:
            raise LabError(f"{version}/{case['id']} 的记录与当前的约定或输入不匹配。")
        text = record["response"]["text"]
        if not isinstance(text, str):
            raise LabError(f"{version}/{case['id']} 的记录缺少模型回答的原文。")
        results.append(judged(case, text, version, "real-record"))
    return results


def synthetic(name, cases):
    """把一组人工构造的输出交给 v2 应用判定，再评分。"""
    sets = read_json(CHAPTER / "inputs/synthetic-outputs.json")["sets"]
    if name not in sets:
        raise LabError("没有这组人工输出，可选：" + "、".join(sets))
    outputs = sets[name]["outputs"]
    return sets[name]["label"], [judged(case, outputs[case["id"]], "v2", "hand-made-fault") for case in cases]


def summary(results):
    failed = [result["id"] for result in results if not result["score"]["passed"]]
    return {"passed": len(results) - len(failed), "total": len(results), "failed_ids": failed}


def run(args):
    cases = load_cases(args.expected)
    reader_ids = [case["id"] for case in cases if case["standard"] == "reader"]
    base = {
        "standard": "第 02 章的约定：十条输入各自的期望行为",
        "reader_standard_ids": reader_ids,
        "scope_note": SCOPE_NOTE,
        "exit_code": 0,
    }
    if args.synthetic:
        label, results = synthetic(args.synthetic, cases)
        return {**base, "mode": "synthetic", "set": args.synthetic, "label": label, "results": results, "summary": summary(results)}
    if args.compare:
        versions = {version: replay(version, cases) for version in ("v1", "v2")}
        rows = []
        for old, new in zip(versions["v1"], versions["v2"], strict=True):
            rows.append({
                "id": old["id"], "input": old["input"],
                "v1_passed": old["score"]["passed"], "v2_passed": new["score"]["passed"],
                "v1_text": old["model_text"], "v2_text": new["model_text"],
            })
        return {
            **base, "mode": "compare", "rows": rows,
            "summary": {version: summary(results) for version, results in versions.items()},
            "flips": {
                "fail_to_pass": [row["id"] for row in rows if not row["v1_passed"] and row["v2_passed"]],
                "pass_to_fail": [row["id"] for row in rows if row["v1_passed"] and not row["v2_passed"]],
            },
        }
    results = replay(args.version, cases)
    if args.case:
        results = [result for result in results if result["id"] in args.case]
        if not results:
            raise LabError("没有这个编号的输入。")
    return {**base, "mode": "replay", "version": args.version, "results": results, "summary": summary(results)}


def describe_behavior(observed):
    text = OUTCOME_NAMES[observed["outcome"]]
    if observed["sends_weather_request"]:
        text += f"，组装查询 {observed['city']} {observed['days']} 天"
        if observed["default_disclosed"]:
            text += "，并说明天数是默认值"
    else:
        text += "，不组装天气查询"
    if observed["reason"]:
        text += f"；原因：{REASONS[observed['reason']]}"
    if observed["decided_by"] == "program" and observed["outcome"] != "query":
        text += "（程序的规则）"
    return text


def show_value(value):
    """把期望或实际的取值写成给人看的词。"""
    if isinstance(value, (str, bool)) or value is None:
        return "“" + VALUE_NAMES.get(value, value) + "”"
    return f" {value} "


def print_results(results):
    for result in results:
        verdict = "通过　" if result["score"]["passed"] else "未通过"
        print(f"{result['id']} {verdict} {result['input']}")
        print(f"    模型原文：{result['model_text']}")
        print(f"    应用行为：{describe_behavior(result['observed'])}")
        if result["standard"] == "reader":
            line = "    标准来源：你写的期望行为"
            if result["unchecked"]:
                line += "，比参考答案少核对：" + "、".join(FIELD_NAMES.get(f, f) for f in result["unchecked"])
            print(line)
        for check in result["score"]["checks"]:
            if not check["ok"]:
                print(
                    f"    没对上：{FIELD_NAMES.get(check['field'], check['field'])}，"
                    f"期望{show_value(check['expected'])}，实际{show_value(check['actual'])}"
                )
        if result["note"]:
            print(f"    注意：{result['note']}")


def print_run(result):
    print(f"评分标准：{result['standard']}")
    if result["reader_standard_ids"]:
        print("其中 " + "、".join(result["reader_standard_ids"]) + " 用的是读者提供的标准。")
    if result["mode"] == "replay":
        print(f"被评对象：{VERSION_NAMES[result['version']]}，回放真实记录")
        print_results(result["results"])
        print(f"小计：通过 {result['summary']['passed']}/{result['summary']['total']}")
    elif result["mode"] == "synthetic":
        print(f"被评对象：人工构造的输出「{result['label']}」，交给 v2 应用判定。它们不是模型的回答")
        print_results(result["results"])
        print(f"小计：通过 {result['summary']['passed']}/{result['summary']['total']}")
    else:
        print("被评对象：两版应用，同一份标准，回放真实记录")
        print("编号  v1      v2      用户原话")
        for row in result["rows"]:
            marks = ["通过　" if row[key] else "未通过" for key in ("v1_passed", "v2_passed")]
            print(f"{row['id']}   {marks[0]}  {marks[1]}  {row['input']}")
        v1, v2 = result["summary"]["v1"], result["summary"]["v2"]
        print(f"通过数：v1 {v1['passed']}/{v1['total']}，v2 {v2['passed']}/{v2['total']}")
        flips = result["flips"]
        print("由未通过变为通过：" + ("、".join(flips["fail_to_pass"]) or "无"))
        print("由通过变为未通过：" + ("、".join(flips["pass_to_fail"]) or "无"))
    print(result["scope_note"])
    print(f"exit_code: {result['exit_code']}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare", action="store_true", help="用同一份标准比较 v1 和 v2 两版应用")
    mode.add_argument(
        "--synthetic", metavar="NAME",
        help="检验评分程序：always-3-days、fabricate-city、refuse-all 或 wrong-refusal",
    )
    parser.add_argument("--version", choices=("v1", "v2"), default="v2", help="回放哪一版应用的记录，默认 v2")
    parser.add_argument("--case", action="append", metavar="ID", help="只看指定编号的输入，可重复")
    parser.add_argument("--expected", type=Path, metavar="FILE", help="用你在第 02 章填写的期望行为替换参考答案")
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

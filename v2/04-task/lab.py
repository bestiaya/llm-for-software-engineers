"""从例子里学：核对 16 行数据的输入、标签和三种用途，回放一次学习过程，再看一个答案进了输入的反例。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import sys


COURSE = Path(__file__).resolve().parents[1]
DATA = COURSE / "data" / "counter-task.json"
RECORD = COURSE / "data" / "network-training-record.json"
DOMAIN = (0, 1, 2, 3)
THRESHOLD = 2
SPLITS = {"train": "训练", "validation": "验证", "heldout": "留出"}
PURPOSES = {
    "train": "参与调整参数",
    "validation": "用来在几个方案里选一个",
    "heldout": "方案固定以后才用来检查",
}
EXPECTED_SPLIT_COUNTS = {"train": 10, "validation": 3, "heldout": 3}
EXPECTED_CLASS_COUNTS = {
    "train": {"0": 2, "1": 6, "2": 2},
    "validation": {"0": 1, "1": 1, "2": 1},
    "heldout": {"0": 1, "1": 1, "2": 1},
}
LEAK_COLUMN = "处理级别"


def label_for(a, b):
    """标注规则：超时达到 2 次的通道有几条，标签就是几。"""
    return int(a >= THRESHOLD) + int(b >= THRESHOLD)


def load_task():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("教学数据需使用 schema_version=1 的 JSON 对象。")
    domain = data["input_domain"]
    if not isinstance(domain, list) or any(type(value) is not int for value in domain) or tuple(domain) != DOMAIN:
        raise ValueError("输入取值应为整数 0、1、2、3。")
    if type(data["threshold"]) is not int or data["threshold"] != THRESHOLD:
        raise ValueError("本题的标注阈值应为 2。")
    for field in ("task", "label_rule"):
        if not isinstance(data.get(field), str):
            raise ValueError(f"{field} 需为文字说明。")
    for field, keys in (("input_description", {"a", "b"}), ("classes", {"0", "1", "2"})):
        values = data.get(field)
        if not isinstance(values, dict) or values.keys() != keys or not all(isinstance(value, str) for value in values.values()):
            raise ValueError(f"{field} 需为文本说明组成的对象。")
    limitations = data.get("limitations")
    if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
        raise ValueError("limitations 需为文字说明组成的数组。")

    rows = data["rows"]
    if not isinstance(rows, list):
        raise ValueError("rows 需为数组。")
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError(f"第 {position} 行需有非空字符串 ID。")
        for feature in ("a", "b"):
            value = row.get(feature)
            if type(value) is not int or value not in DOMAIN:
                raise ValueError(f"第 {position} 行 {feature} 需为 0 到 3 的整数。")
        label = row.get("label")
        if type(label) is not int or label not in (0, 1, 2):
            raise ValueError(f"第 {position} 行 label 需为 0、1 或 2。")
        split = row.get("split")
        if not isinstance(split, str) or split not in PURPOSES:
            raise ValueError(f"第 {position} 行 split 需为 train、validation 或 heldout。")
    return data


def inject_fault(rows, fault):
    if fault == "wrong-label":
        row = next(row for row in rows if row["id"] == "a0-b0" and row["split"] == "train")
        before = row["label"]
        row["label"] = 1
        return {"kind": fault, "id": row["id"], "split": row["split"], "field": "label", "before": before, "after": 1}

    # 重叠：把训练组的一行原样放进验证组，换掉验证组原来的一行。输入和标签都与训练组的那一行相同。
    source = next(row for row in rows if row["id"] == "a0-b0" and row["split"] == "train")
    row = next(row for row in rows if row["id"] == "a0-b1" and row["split"] == "validation")
    before = row["id"]
    row.update(id=source["id"], a=source["a"], b=source["b"], label=source["label"])
    return {"kind": fault, "split": row["split"], "field": "row", "before": before, "after": row["id"]}


def inspect_rows(rows):
    split_counts = dict.fromkeys(PURPOSES, 0)
    class_counts = {split: dict.fromkeys(("0", "1", "2"), 0) for split in PURPOSES}
    by_id = defaultdict(list)
    by_input = defaultdict(list)
    display_rows = []
    issues = []

    for row in rows:
        split = row["split"]
        split_counts[split] += 1
        class_counts[split][str(row["label"])] += 1
        by_id[row["id"]].append(split)
        by_input[row["a"], row["b"]].append(row)
        expected = label_for(row["a"], row["b"])
        display_rows.append({**row, "purpose": PURPOSES[split], "expected_label": expected})
        if row["label"] != expected:
            issues.append({
                "kind": "label_mismatch", "id": row["id"], "split": split,
                "actual": row["label"], "expected": expected,
            })

    for row_id, splits in by_id.items():
        if len(splits) > 1:
            kind = "split_overlap" if len(set(splits)) > 1 else "duplicate_id"
            issues.append({"kind": kind, "id": row_id, "splits": splits})
    for (a, b), matches in by_input.items():
        if len(matches) > 1:
            issues.append({
                "kind": "duplicate_input", "input": [a, b],
                "rows": [{"id": row["id"], "split": row["split"]} for row in matches],
            })
    missing = sorted(set(product(DOMAIN, repeat=2)) - by_input.keys())
    if missing:
        issues.append({"kind": "missing_inputs", "inputs": missing})
    if split_counts != EXPECTED_SPLIT_COUNTS:
        issues.append({"kind": "split_counts", "actual": split_counts, "expected": EXPECTED_SPLIT_COUNTS})
    if class_counts != EXPECTED_CLASS_COUNTS:
        issues.append({"kind": "class_counts", "actual": class_counts, "expected": EXPECTED_CLASS_COUNTS})
    return {
        "rows": display_rows,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "issues": issues,
    }


def run(fault):
    data = load_task()
    rows = [dict(row) for row in data["rows"]]
    injected_fault = inject_fault(rows, fault) if fault else None
    result = inspect_rows(rows)
    return {
        "mode": "check",
        "task": {
            "name": data["task"],
            "input_description": data["input_description"],
            "input_domain": data["input_domain"],
            "label_rule": data["label_rule"],
            "threshold": data["threshold"],
            "classes": data["classes"],
            "limitations": data["limitations"],
        },
        **result,
        "injected_fault": injected_fault,
        "training_executed": False,
        "exit_code": 1 if result["issues"] else 0,
    }


def preview():
    """回放随课保存的那次训练记录，按四步列出一遍完整的学习过程。不重新训练。"""
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    training, config = record["run"], record["config"]
    checkpoints = training["checkpoints"]
    selected = next(item for item in checkpoints if item["step"] == training["selected_step"])
    heldout = training["heldout"]
    return {
        "mode": "preview",
        "training_executed": False,
        "source": "data/network-training-record.json",
        "seed": config["seed"],
        "steps": config["steps"],
        "adjust": {
            "rows": 10, "correct_before": checkpoints[0]["train_correct"],
            "selected_step": selected["step"], "correct_after": selected["train_correct"],
        },
        "select": {
            "rows": 3, "candidates": len(checkpoints),
            "table": [
                {"step": item["step"], "train_correct": item["train_correct"], "validation_loss": item["validation_loss"]}
                for item in checkpoints
            ],
            "selected_step": selected["step"],
            "validation_correct": selected["validation_correct"],
        },
        "check": {
            "rows": [
                {"a": row["x"][0], "b": row["x"][1], "label": row["label"], "prediction": row["prediction"]}
                for row in heldout["rows"]
            ],
            "correct": heldout["correct"],
        },
        "rule_correct": sum(label_for(row["a"], row["b"]) == row["label"] for row in load_task()["rows"]),
        "exit_code": 0,
    }


def leak():
    """反例：往输入里多加一列事后填写的“处理级别”，它其实就是标签。"""
    rows = [{**row, "level": row["label"]} for row in load_task()["rows"]]
    copy_rule = lambda a, b, level: level       # 一个只照抄第三列的“模型”：手写的一行规则，不是训练出来的
    with_column = {
        split: {
            "correct": sum(copy_rule(row["a"], row["b"], row["level"]) == row["label"] for row in rows if row["split"] == split),
            "rows": EXPECTED_SPLIT_COUNTS[split],
        }
        for split in SPLITS
    }
    # 真正使用时，要判断的那一刻还没有这一列。本实验假设缺失时填 0。
    without_column = sum(copy_rule(row["a"], row["b"], 0) == row["label"] for row in rows)
    identical = all(row["level"] == row["label"] for row in rows)
    return {
        "mode": "leak",
        "column": LEAK_COLUMN,
        "rows": [{key: row[key] for key in ("id", "a", "b", "level", "label", "split")} for row in rows],
        "copy_rule_with_column": with_column,
        "copy_rule_without_column": {"correct": without_column, "rows": len(rows)},
        "issues": [{"kind": "label_in_input", "column": LEAK_COLUMN}] if identical else [],
        "exit_code": 1 if identical else 0,
    }


def describe(issue):
    """把一条核对结果写成一句话。"""
    kind = issue["kind"]
    if kind == "label_mismatch":
        return f"{issue['id']}（{SPLITS[issue['split']]}）的标签是 {issue['actual']}，按规则应为 {issue['expected']}。"
    if kind == "split_overlap":
        return f"{issue['id']} 同时出现在" + "和".join(SPLITS[split] for split in issue["splits"]) + "两组里。"
    if kind == "duplicate_id":
        return f"{issue['id']} 在同一组里出现了不止一次。"
    if kind == "duplicate_input":
        where = "、".join(f"{SPLITS[row['split']]}组的 {row['id']}" for row in issue["rows"])
        return f"输入 a={issue['input'][0]}、b={issue['input'][1]} 出现了不止一次：{where}。"
    if kind == "missing_inputs":
        return "缺了这些输入：" + "、".join(f"a={a} b={b}" for a, b in issue["inputs"]) + "。"
    if kind == "split_counts":
        return "三组的行数应为 10、3、3，实际是 " + "、".join(str(value) for value in issue["actual"].values()) + "。"
    if kind == "class_counts":
        actual = "；".join(
            f"{SPLITS[split]} " + "、".join(str(value) for value in counts.values()) for split, counts in issue["actual"].items()
        )
        return f"各组里标签 0、1、2 的行数与约定不符，实际是：{actual}。"
    return json.dumps(issue, ensure_ascii=False)


def print_check(result):
    task = result["task"]
    print(task["name"])
    for feature, definition in task["input_description"].items():
        print(f"  {feature} = {definition}，取值 0 到 3")
    print(f"  标注规则：{task['label_rule']}")
    print("  类别：" + "，".join(f"{label}={meaning}" for label, meaning in task["classes"].items()))
    if result["injected_fault"] is not None:
        fault = result["injected_fault"]
        group = SPLITS[fault["split"]]
        if fault["field"] == "label":
            print(f"这次故意在内存里改动了一处：把{group}组里 {fault['id']} 的标签从 {fault['before']} 改成了 {fault['after']}。文件没有改。")
        else:
            print(f"这次故意在内存里改动了一处：把{group}组里的 {fault['before']} 整行换成了训练组里的 {fault['after']}，输入和标签都照搬。文件没有改。")
    print("\n逐行核对")
    print("编号     a  b  标签  按规则算  用途")
    for row in result["rows"]:
        print(f"{row['id']:7}  {row['a']}  {row['b']}   {row['label']}      {row['expected_label']}      {SPLITS[row['split']]}")
    print("\n各组行数，以及标签 0、1、2 各有几行")
    for split, count in result["split_counts"].items():
        by_class = "、".join(str(value) for value in result["class_counts"][split].values())
        print(f"  {SPLITS[split]} {count:>2} 行：{by_class}　{PURPOSES[split]}")
    if result["issues"]:
        print("核对未通过：")
        for issue in result["issues"]:
            print("  " + describe(issue))
    else:
        print("核对通过：16 种输入各出现一次，标签都符合规则，三组没有重叠。")


def print_preview(result):
    adjust, select, check = result["adjust"], result["select"], result["check"]
    print("学习过程概览：回放随课保存的一次训练记录，不重新训练。")
    print(f"① 用训练的 {adjust['rows']} 行调整模型")
    print(f"   调整之前：{adjust['rows']} 行里答对 {adjust['correct_before']} 行")
    print(f"   调整 {adjust['selected_step']} 次之后：{adjust['rows']} 行里答对 {adjust['correct_after']} 行")
    print(f"② 用验证的 {select['rows']} 行选方案")
    print(f"   训练途中在 {select['candidates']} 个时刻各留了一份参数，每一份是一个候选方案。")
    print("   验证损失是衡量预测与标签差多少的一个数，越小越好，第 08 章讲。")
    print("   调整次数  训练答对  验证损失")
    for item in select["table"]:
        mark = "  ← 验证损失最小，选这一份" if item["step"] == select["selected_step"] else ""
        print(f"   {item['step']:>8}    {item['train_correct']:>2}/10    {item['validation_loss']:.4f}{mark}")
    print("③ 固定方案")
    print(f"   选定第 {select['selected_step']} 次调整之后的那一份参数，之后不再改动。它在验证的 3 行上答对 {select['validation_correct']} 行。")
    print("④ 对没见过的输入预测并检查")
    print("   留出的 3 行到这时才第一次使用：")
    for row in check["rows"]:
        verdict = "对" if row["prediction"] == row["label"] else "错"
        print(f"   a={row['a']} b={row['b']}　标签 {row['label']}　模型选 {row['prediction']}　{verdict}")
    print(f"   留出 3 行答对 {check['correct']} 行。")
    print(f"对照：直接套用标注规则，16 行答对 {result['rule_correct']} 行。让模型从样本里学，是为了观察学习的过程。")


def print_leak(result):
    column = result["column"]
    print(f"反例：答案进了输入。在 16 行数据上多加一列“{column}”，它是事后填的：标签是几，它就填几。")
    print(f"编号     a  b  {column}  标签  用途")
    for row in result["rows"]:
        print(f"{row['id']:7}  {row['a']}  {row['b']}     {row['level']}       {row['label']}    {SPLITS[row['split']]}")
    scores = result["copy_rule_with_column"]
    print("一个只照抄第三列的“模型”（手写的一行规则，不是训练出来的）：")
    print("  " + "，".join(f"{SPLITS[split]} {item['correct']}/{item['rows']}" for split, item in scores.items()))
    without = result["copy_rule_without_column"]
    print(f"真正使用时，要判断的那一刻还没有“{column}”。本实验假设这一列缺失时填 0：")
    print(f"  同样的 16 种输入，答对 {without['correct']}/{without['rows']}")
    if result["issues"]:
        print(f"核对未通过：输入里的“{column}”一列与标签逐行相同。它是由答案得来的，不能作为输入。")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fault", choices=("wrong-label", "overlap"), help="在内存里改动一处数据，看核对能不能发现：改错一个标签，或把训练组的一行放进验证组")
    mode.add_argument("--preview", action="store_true", help="回放一次训练记录，按四步看一遍完整的学习过程")
    mode.add_argument("--leak", action="store_true", help="反例：往输入里加一列由答案得来的数据")
    parser.add_argument("--json", action="store_true", help="输出结构化结果")
    args = parser.parse_args(argv)
    try:
        result = preview() if args.preview else leak() if args.leak else run(args.fault)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        result = {"error": f"数据无法读取或格式有误（{error}）", "exit_code": 2}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["exit_code"] == 2:
        print(result["error"], file=sys.stderr)
    else:
        {"check": print_check, "preview": print_preview, "leak": print_leak}[result["mode"]](result)
        print(f"exit_code: {result['exit_code']}")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

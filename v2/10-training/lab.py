"""学得怎样：把一次更新重复很多次，看训练、验证、留出三组数据各自的变化。只用标准库。"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import sys


CHAPTER = Path(__file__).resolve().parent
COURSE = CHAPTER.parent
sys.path.insert(0, str(COURSE))

from shared import network as nw  # noqa: E402


RECORD_FILE = COURSE / "data/network-training-record.json"
CONFIG = {
    "seed": 1,
    "initial_scale": 0.5,
    "learning_rate": 0.1,
    "steps": 2000,
    "checkpoints": [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
    "method": "每一步都用全部 10 行训练数据算梯度，所有参数一起更新",
}
MAJORITY_CLASS = 1


def row_details(parameters, rows):
    details = []
    for row in rows:
        probabilities = nw.forward(parameters, row["x"])["probabilities"]
        details.append({
            "id": row["id"], "x": list(row["x"]), "label": row["label"],
            "probabilities": probabilities, "prediction": nw.predict(probabilities),
        })
    return details


def run_training(task, seed=CONFIG["seed"], learning_rate=CONFIG["learning_rate"]):
    """按声明的设置训练一遍。训练过程中只看训练和验证两组数据，不看留出的三行。"""
    snapshots = {}

    def note(step, parameters):
        if step in CONFIG["checkpoints"]:
            snapshots[step] = nw.unflatten(nw.flatten(parameters))

    nw.train(nw.initial_parameters(seed, CONFIG["initial_scale"]), task["train"], learning_rate, CONFIG["steps"], note)
    checkpoints = [
        {
            "step": step,
            "train_loss": nw.mean_loss(parameters, task["train"]),
            "train_correct": nw.correct_count(parameters, task["train"]),
            "validation_loss": nw.mean_loss(parameters, task["validation"]),
            "validation_correct": nw.correct_count(parameters, task["validation"]),
        }
        for step, parameters in sorted(snapshots.items())
    ]
    # 用验证数据选方案：验证损失最低的那个检查点；并列时取较早的。
    selected = min(checkpoints, key=lambda item: (item["validation_loss"], item["step"]))
    chosen = snapshots[selected["step"]]
    return {
        "checkpoints": checkpoints,
        "selected_step": selected["step"],
        "selected_parameters": chosen,
        "validation_rows": row_details(chosen, task["validation"]),
        # 方案选定以后，才第一次看留出的三行。
        "heldout": {
            "loss": nw.mean_loss(chosen, task["heldout"]),
            "correct": nw.correct_count(chosen, task["heldout"]),
            "rows": row_details(chosen, task["heldout"]),
        },
    }


def baselines(task):
    """两个不用学习的参照：总选最多的那一类；三类各给三分之一。"""
    return {
        "majority_class": MAJORITY_CLASS,
        "majority_correct": {name: sum(row["label"] == MAJORITY_CLASS for row in rows) for name, rows in task.items()},
        "uniform_loss": math.log(nw.CLASSES),
        "sizes": {name: len(rows) for name, rows in task.items()},
    }


def seed_survey(task, count):
    """换不同的起点各训练一遍，看结果有多不一样。只统计正确行数。"""
    tally = collections.Counter()
    for seed in range(1, count + 1):
        final = nw.train(nw.initial_parameters(seed, CONFIG["initial_scale"]), task["train"], CONFIG["learning_rate"], CONFIG["steps"])
        tally[(nw.correct_count(final, task["train"]), nw.correct_count(final, task["validation"]), nw.correct_count(final, task["heldout"]))] += 1
    table = [
        {"train_correct": key[0], "validation_correct": key[1], "heldout_correct": key[2], "runs": runs}
        for key, runs in sorted(tally.items(), reverse=True)
    ]
    return {
        "seeds": count,
        "table": table,
        "train_all_correct": sum(item["runs"] for item in table if item["train_correct"] == len(task["train"])),
        "all_unseen_correct": sum(
            item["runs"] for item in table
            if item["validation_correct"] == len(task["validation"]) and item["heldout_correct"] == len(task["heldout"])
        ),
    }


def linear_training(task, steps=5000, learning_rate=0.1):
    """对照：没有隐藏层的线性模型，用同样的 10 行训练数据学。"""
    weights, bias = [[0.0] * nw.CLASSES for _ in range(nw.INPUTS)], [0.0] * nw.CLASSES
    rows = task["train"]

    def probabilities(x):
        return nw.softmax([sum(x[i] * weights[i][k] for i in range(nw.INPUTS)) + bias[k] for k in range(nw.CLASSES)])

    for _ in range(steps):
        d_weights, d_bias = [[0.0] * nw.CLASSES for _ in range(nw.INPUTS)], [0.0] * nw.CLASSES
        for row in rows:
            p = probabilities(row["x"])
            for k in range(nw.CLASSES):
                slope = (p[k] - (1.0 if k == row["label"] else 0.0)) / len(rows)
                d_bias[k] += slope
                for i in range(nw.INPUTS):
                    d_weights[i][k] += row["x"][i] * slope
        weights = [[weights[i][k] - learning_rate * d_weights[i][k] for k in range(nw.CLASSES)] for i in range(nw.INPUTS)]
        bias = [bias[k] - learning_rate * d_bias[k] for k in range(nw.CLASSES)]
    wrong = [list(row["x"]) for row in rows if nw.predict(probabilities(row["x"])) != row["label"]]
    return {
        "steps": steps, "learning_rate": learning_rate, "train_correct": len(rows) - len(wrong), "train_rows": len(rows),
        "train_loss": sum(-math.log(probabilities(row["x"])[row["label"]]) for row in rows) / len(rows), "wrong_rows": wrong,
    }


def inspect(task):
    """看训练得到的参数：选定的那一份里，四个隐藏单元各自的算式，以及 16 行各自的隐藏值和选择。"""
    start = nw.initial_parameters(CONFIG["seed"], CONFIG["initial_scale"])
    training = run_training(task)
    chosen = training["selected_parameters"]
    rows = sorted(
        (dict(row, split=name) for name, group in task.items() for row in group), key=lambda row: row["x"]
    )
    table = []
    for row in rows:
        state = nw.forward(chosen, row["x"])
        table.append({
            "id": row["id"], "split": row["split"], "hidden": state["hidden"],
            "prediction": nw.predict(state["probabilities"]), "label": row["label"],
        })
    units = [
        {
            "unit": f"h{j + 1}", "weight_a": chosen["W1"][0][j], "weight_b": chosen["W1"][1][j], "bias": chosen["b1"][j],
            "unchanged": all(
                chosen[block][index][j] == start[block][index][j] for block, index in (("W1", 0), ("W1", 1))
            ) and chosen["b1"][j] == start["b1"][j] and chosen["W2"][j] == start["W2"][j],
            "always_zero": all(item["hidden"][j] == 0 for item in table),
        }
        for j in range(nw.HIDDEN)
    ]
    return {
        "mode": "inspect", "step": training["selected_step"], "units": units, "rows": table,
        "correct": {
            name: sum(item["prediction"] == item["label"] for item in table if item["split"] == name)
            for name in ("train", "validation", "heldout")
        },
        "exit_code": 0,
    }


def at_step(task, step):
    """训练到指定的步数停下，逐行看训练的 10 行各选了哪一类。"""
    if not 0 <= step <= CONFIG["steps"]:
        raise ValueError(f"步数应在 0 到 {CONFIG['steps']} 之间。")
    parameters = nw.train(nw.initial_parameters(CONFIG["seed"], CONFIG["initial_scale"]), task["train"], CONFIG["learning_rate"], step)
    rows = row_details(parameters, task["train"])
    return {
        "mode": "at", "step": step, "rows": rows,
        "correct": sum(row["prediction"] == row["label"] for row in rows),
        "loss": nw.mean_loss(parameters, task["train"]),
        "predictions": sorted({row["prediction"] for row in rows}),
        "exit_code": 0,
    }


def stage_check():
    """阶段检查的核对：把手设网络里 h1 的偏置从 -1 改成 -2，看哪些行变了；再在其中一行上做一次输出偏置的更新。"""
    task = nw.load_task(COURSE / "data/counter-task.json")
    rows = sorted(task["train"] + task["validation"] + task["heldout"], key=lambda row: row["x"])
    original = nw.hand_set_parameters()
    changed = nw.set_parameter(original, "b1[h1]=-2")
    flipped = [
        {
            "x": list(row["x"]), "label": row["label"],
            "before": nw.predict(nw.forward(original, row["x"])["probabilities"]),
            "after": nw.predict(nw.forward(changed, row["x"])["probabilities"]),
        }
        for row in rows
    ]
    flipped = [item for item in flipped if item["before"] != item["after"]]
    x, label, learning_rate = (2, 0), 1, 0.3
    before, after = nw.forward(original, x), nw.forward(changed, x)
    gradient = [after["probabilities"][k] - (1.0 if k == label else 0.0) for k in range(nw.CLASSES)]
    updated = nw.unflatten(nw.flatten(changed))
    updated["b2"] = [value - learning_rate * slope for value, slope in zip(changed["b2"], gradient, strict=True)]
    again = nw.forward(updated, x)

    def view(state):
        return {
            "hidden": state["hidden"], "scores": state["scores"], "probabilities": state["probabilities"],
            "prediction": nw.predict(state["probabilities"]), "loss": -math.log(state["probabilities"][label]),
        }

    return {
        "mode": "stage", "change": "b1[h1]=-2", "flipped": flipped,
        "correct_before": nw.correct_count(original, rows), "correct_after": nw.correct_count(changed, rows),
        "row": {"x": list(x), "label": label},
        "original": view(before), "changed": view(after),
        "bias_gradient": gradient, "learning_rate": learning_rate, "new_bias": updated["b2"], "after_one_update": view(again),
        "exit_code": 0,
    }


def close(left, right, tolerance=1e-9):
    """两份结果是否一致：数值允许极小的差别，其余要完全相同。"""
    if isinstance(left, float) or isinstance(right, float):
        return isinstance(left, (int, float)) and isinstance(right, (int, float)) and abs(left - right) <= tolerance
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right, strict=True))
    return left == right


def build_record(task):
    return {
        "schema_version": 1,
        "data_kind": "training_run_record",
        "config": CONFIG,
        "parameter_names": nw.parameter_names(),
        "initial_parameters": nw.initial_parameters(CONFIG["seed"], CONFIG["initial_scale"]),
        "run": run_training(task),
        "baselines": baselines(task),
    }


def run(args):
    if args.stage_check:
        return stage_check()
    task = nw.load_task(COURSE / "data/counter-task.json")
    if args.at is not None:
        return at_step(task, args.at)
    if args.inspect:
        return inspect(task)
    if args.seeds:
        return {"mode": "seeds", "survey": seed_survey(task, args.seeds), "exit_code": 0}
    if args.linear:
        return {"mode": "linear", "linear": linear_training(task), "exit_code": 0}
    record = build_record(task)
    if args.write_record:
        if RECORD_FILE.exists() and not args.force:
            return {"mode": "write", "written": False, "exit_code": 2, "error": "记录已存在；确实要覆盖时加 --force。"}
        RECORD_FILE.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"mode": "write", "written": True, "path": str(RECORD_FILE), "exit_code": 0}
    saved = json.loads(RECORD_FILE.read_text(encoding="utf-8")) if RECORD_FILE.exists() else None
    matches = saved is not None and close(record, saved)
    return {"mode": "train", "record": record, "matches_saved_record": matches, "exit_code": 0 if matches else 1}


def print_run(result):
    if result["mode"] == "seeds":
        survey = result["survey"]
        print(f"换 {survey['seeds']} 个起点各训练一遍，其他设置不变。每行是一种结果，最后一列是出现的次数。")
        print("训练 10 行  验证 3 行  留出 3 行  次数")
        for item in survey["table"]:
            print(f"   {item['train_correct']:>2}/10      {item['validation_correct']}/3       {item['heldout_correct']}/3     {item['runs']}")
        print(f"训练 10 行全对的有 {survey['train_all_correct']} 次；验证和留出的 6 行全对的有 {survey['all_unseen_correct']} 次。")
    elif result["mode"] == "linear":
        linear = result["linear"]
        print(
            f"线性模型训练 {linear['steps']} 步以后：训练 {linear['train_correct']}/{linear['train_rows']} 行正确，"
            f"平均损失 {linear['train_loss']:.4f}。没分对的行：{linear['wrong_rows']}"
        )
    elif result["mode"] == "write":
        print("已写入 " + result["path"] if result["written"] else result["error"])
    elif result["mode"] == "inspect":
        names = {"train": "训练", "validation": "验证", "heldout": "留出"}
        print(f"训练得到的参数：选定的第 {result['step']} 步。四个隐藏单元各自的算式，数字取三位小数：")
        for unit in result["units"]:
            print(f"  {unit['unit']} = ReLU({unit['weight_a']:.3f}×a {unit['weight_b']:+.3f}×b {unit['bias']:+.3f})")
        print("16 行各自的隐藏值和选择：")
        print("编号     用途  隐藏值                        选  标签")
        for item in result["rows"]:
            shown = "[" + ", ".join(f"{value:.2f}" for value in item["hidden"]) + "]"
            verdict = "对" if item["prediction"] == item["label"] else "错"
            print(f"{item['id']:7}  {names[item['split']]}  {shown:28}  {item['prediction']}   {item['label']}   {verdict}")
        correct = result["correct"]
        print(f"16 行里选对 {sum(correct.values())} 行：训练 {correct['train']}/10，验证 {correct['validation']}/3，留出 {correct['heldout']}/3。")
        for unit in result["units"]:
            if unit["always_zero"]:
                tail = "，它的参数和起点完全相同，训练没有动过它" if unit["unchanged"] else ""
                print(f"{unit['unit']} 在 16 行上都是 0{tail}。")
    elif result["mode"] == "at":
        print(f"训练到第 {result['step']} 步时，训练的 10 行：")
        for row in result["rows"]:
            shown = ", ".join(f"{value:.3f}" for value in row["probabilities"])
            verdict = "对" if row["prediction"] == row["label"] else "错"
            print(f"  a={row['x'][0]} b={row['x'][1]}　正确类别 {row['label']}　概率 [{shown}]　选 {row['prediction']}　{verdict}")
        print(f"答对 {result['correct']}/10，平均损失 {result['loss']:.4f}。")
        if len(result["predictions"]) == 1:
            print(f"10 行选的都是类别 {result['predictions'][0]}。")
    elif result["mode"] == "stage":
        row = result["row"]

        def shown(values):
            return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"

        print(f"参数是手工设定的，没有经过训练。改动：{result['change']}。")
        print(f"① 16 行里选对的行数从 {result['correct_before']} 变成 {result['correct_after']}。预测变了的行：")
        for item in result["flipped"]:
            print(f"   a={item['x'][0]} b={item['x'][1]}　标签 {item['label']}　原来选 {item['before']}，现在选 {item['after']}")
        print(f"② 看 a={row['x'][0]} b={row['x'][1]} 这一行，正确类别 {row['label']}：")
        for name, key in (("改动之前", "original"), ("改动之后", "changed")):
            state = result[key]
            print(
                f"   {name}　隐藏值 {[round(value, 4) for value in state['hidden']]}　分数 {[round(value, 4) for value in state['scores']]}　"
                f"概率 {shown(state['probabilities'])}　损失 {state['loss']:.4f}"
            )
        print(f"③ 在改动之后的状态上，只看这一行：三个输出偏置的变化率 = 三个概率，正确类别的那一个减 1 = {shown(result['bias_gradient'])}")
        print(f"   学习率 {result['learning_rate']}，更新一次：新的输出偏置 {shown(result['new_bias'])}")
        state = result["after_one_update"]
        print(f"   复算　分数 {shown(state['scores'])}　概率 {shown(state['probabilities'])}　损失 {state['loss']:.4f}　选 {state['prediction']}")
    else:
        record = result["record"]
        config, training, base = record["config"], record["run"], record["baselines"]
        print(f"设置：起点种子 {config['seed']}，学习率 {config['learning_rate']}，{config['steps']} 步。{config['method']}。")
        print("步数    训练损失  训练正确  验证损失  验证正确")
        for item in training["checkpoints"]:
            mark = "  ← 验证损失最低，选这个" if item["step"] == training["selected_step"] else ""
            print(
                f"{item['step']:>5}   {item['train_loss']:.4f}    {item['train_correct']:>2}/10    "
                f"{item['validation_loss']:.4f}     {item['validation_correct']}/3{mark}"
            )
        print(f"第 {training['selected_step']} 步的参数在验证的三行上：")
        for row in training["validation_rows"]:
            shown = ", ".join(f"{value:.3f}" for value in row["probabilities"])
            verdict = "对" if row["prediction"] == row["label"] else "错"
            print(f"  a={row['x'][0]} b={row['x'][1]}　正确类别 {row['label']}　概率 [{shown}]　选 {row['prediction']}　{verdict}")
        print(f"选定第 {training['selected_step']} 步以后，第一次看留出的三行：")
        for row in training["heldout"]["rows"]:
            shown = ", ".join(f"{value:.3f}" for value in row["probabilities"])
            verdict = "对" if row["prediction"] == row["label"] else "错"
            print(f"  a={row['x'][0]} b={row['x'][1]}　正确类别 {row['label']}　概率 [{shown}]　选 {row['prediction']}　{verdict}")
        print(f"留出三行：{training['heldout']['correct']}/3 正确，平均损失 {training['heldout']['loss']:.4f}")
        counts = base["majority_correct"]
        print(
            f"参照一，总选类别 {base['majority_class']}：训练 {counts['train']}/10，验证 {counts['validation']}/3，留出 {counts['heldout']}/3。"
            f"参照二，三类各给三分之一：损失 {base['uniform_loss']:.4f}。"
        )
        print("与随课保存的训练记录" + ("一致。" if result["matches_saved_record"] else "不一致，或还没有保存记录。"))
    print(f"exit_code: {result['exit_code']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seeds", type=int, metavar="N", help="换 N 个起点各训练一遍，看结果的差别")
    mode.add_argument("--linear", action="store_true", help="对照：训练一个没有隐藏层的线性模型")
    mode.add_argument("--at", type=int, metavar="步数", help="训练到这一步停下，逐行看训练的 10 行各选了哪一类")
    mode.add_argument("--inspect", action="store_true", help="看训练得到的参数：隐藏单元的算式和 16 行各自的隐藏值")
    mode.add_argument("--stage-check", action="store_true", help="阶段检查的核对：改一个参数，看哪些行变了，再更新一次")
    mode.add_argument("--write-record", action="store_true", help="课程维护用：把这次训练写成随课记录")
    parser.add_argument("--force", action="store_true", help="与 --write-record 合用，覆盖已有记录")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except ValueError as error:
        result = {"mode": "error", "error": f"实验失败：{error}", "exit_code": 2}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["exit_code"] == 2 and "error" in result and result["mode"] == "error":
        print(result["error"], file=sys.stderr)
    else:
        print_run(result)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

"""从分数到概率和损失：把三个分数变成三个概率，用正确类别的概率算损失，再对很多行取平均。只用标准库，不训练。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


CHAPTER = Path(__file__).resolve().parent
COURSE = CHAPTER.parent
sys.path.insert(0, str(COURSE))

from shared import network as nw  # noqa: E402


TASK_FILE = COURSE / "data" / "counter-task.json"
# 为了让结果是整数挑的三个分数：0、ln 2、ln 3，取指数以后正好是 1、2、3。
EXAMPLE_SCORES = [0.0, math.log(2), math.log(3)]
# 人工构造的两组概率，三行的标签都是类别 0。用来看答对数和损失什么时候说法不一样。
DISAGREE = {
    "甲": [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3]],
    "乙": [[0.9, 0.05, 0.05], [0.9, 0.05, 0.05], [0.3, 0.6, 0.1]],
}


def choose(values):
    """取最大的类别；并列时取编号较小的。"""
    return max(range(len(values)), key=lambda k: (values[k], -k))


def number_list(text, name):
    try:
        values = [float(part) for part in text.split(",")]
    except ValueError:
        raise ValueError(f"{name} 要写成用逗号隔开的数，例如 -1,0,-1。") from None
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} 要写成用逗号隔开的数，例如 -1,0,-1。")
    return values


SAFE_RANGE = 30     # 分数的绝对值都不超过它时，直接取指数不会溢出，照正文的三步显示


def to_probabilities(scores, label):
    """三步：取指数，求和，各自除以总和。

    概率总是按“先把每个分数减去其中最大的那个，再取指数”来算，结果不变，分数很大时也不会溢出。
    分数不大时，显示的是不减的三步，和正文的表一样；分数很大时，显示减过以后的三步。
    损失直接由分数算，不先取概率再取对数，概率小到显示成 0 时也算得出来。
    """
    top = max(scores)
    shift = 0.0 if max(abs(score) for score in scores) <= SAFE_RANGE else top
    shown_scores = [score - shift for score in scores]
    powers = [math.exp(score) for score in shown_scores]
    probabilities = nw.softmax(scores)
    result = {
        "mode": "scores", "scores": scores, "shift": shift, "shown_scores": shown_scores, "powers": powers, "total": sum(powers),
        "probabilities": probabilities, "prediction": choose(scores), "exit_code": 0,
    }
    if label is not None:
        if not 0 <= label < len(scores):
            raise ValueError(f"正确类别应在 0 到 {len(scores) - 1} 之间。")
        result.update(label=label, probability_of_label=probabilities[label], loss=nw.loss_from_scores(scores, label))
    return result


def losses(values):
    """一组“正确类别的概率”各自的损失，以及平均。"""
    if any(not 0 < value <= 1 for value in values):
        raise ValueError("概率要大于 0、不超过 1。")
    each = [-math.log(value) for value in values]
    return {"mode": "losses", "probabilities": values, "losses": each, "mean": sum(each) / len(each), "exit_code": 0}


def network(changes, scale):
    parameters = nw.hand_set_parameters()
    for change in changes:
        parameters = nw.set_parameter(parameters, change)
    return nw.scale_output(parameters, scale) if scale != 1 else parameters


def one_row(x, changes, scale):
    """一行输入走过手设的网络，再把分数变成概率和损失。"""
    state = nw.forward(network(changes, scale), x)
    label = int(x[0] >= 2) + int(x[1] >= 2)
    result = to_probabilities(state["scores"], label)
    result.update(mode="row", input=list(x), changes=changes, scale=scale, hidden=state["hidden"])
    return result


def everything(changes, scale):
    """16 行都走一遍：每行正确类别的概率、损失，以及平均损失和答对的行数。"""
    parameters = network(changes, scale)
    table = []
    for row in json.loads(TASK_FILE.read_text(encoding="utf-8"))["rows"]:
        state = nw.forward(parameters, (row["a"], row["b"]))
        probabilities = state["probabilities"]
        table.append({
            "id": row["id"], "input": [row["a"], row["b"]], "scores": state["scores"], "probabilities": probabilities,
            "prediction": choose(probabilities), "label": row["label"],
            "probability_of_label": probabilities[row["label"]], "loss": nw.loss_from_scores(state["scores"], row["label"]),
        })
    return {
        "mode": "all", "changes": changes, "scale": scale, "rows": table,
        "correct": sum(item["prediction"] == item["label"] for item in table),
        "mean_loss": sum(item["loss"] for item in table) / len(table),
        "exit_code": 0,
    }


def disagree():
    """人工构造的例子：答对的行数和平均损失给出相反的排名。"""
    models = []
    for name, rows in DISAGREE.items():
        each = [-math.log(row[0]) for row in rows]
        models.append({
            "name": name, "rows": rows, "predictions": [choose(row) for row in rows],
            "correct": sum(choose(row) == 0 for row in rows), "losses": each, "mean_loss": sum(each) / len(each),
        })
    return {"mode": "disagree", "label": 0, "models": models, "exit_code": 0}


def numbers(values, places=4):
    return "[" + ", ".join(f"{value:.{places}f}" for value in values) + "]"


def short(values):
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def print_steps(result):
    print(f"三个分数　{numbers(result['scores'], 6)}")
    if result["shift"]:
        print(f"分数很大，直接取指数会溢出。先把每个分数都减去其中最大的 {result['shift']:g}，概率不变　{numbers(result['shown_scores'], 6)}")
    print(f"① 取指数　{numbers(result['powers'], 6)}")
    print(f"② 求和　　{result['total']:.6f}")
    print(f"③ 各自除以总和，得到三个概率　{numbers(result['probabilities'], 6)}")
    print(f"三个概率加起来是 {sum(result['probabilities']):.6f}。概率最大的是类别 {result['prediction']}。")
    if "label" in result:
        chance = result["probability_of_label"]
        if chance >= 1e-6:
            print(f"正确类别是 {result['label']}，模型分给它的概率是 {chance:.6f}，损失 = −ln({chance:.6f}) = {result['loss']:.6f}")
        else:
            print(f"正确类别是 {result['label']}，模型分给它的概率小到显示成了 0。损失直接由分数算，是 {result['loss']:.6f}")


def print_run(result):
    mode = result["mode"]
    if mode == "scores":
        print_steps(result)
    elif mode == "row":
        print("参数是手工设定的，没有经过训练。")
        if result["changes"]:
            print("这次在内存里改了参数：" + "，".join(result["changes"]) + "。")
        if result["scale"] != 1:
            print(f"这次把输出层的权重和偏置都乘了 {result['scale']:g}。")
        print(f"输入：a={result['input'][0]:g}，b={result['input'][1]:g}　隐藏值 {short(result['hidden'])}")
        print_steps(result)
    elif mode == "losses":
        print("正确类别的概率    损失")
        for probability, loss in zip(result["probabilities"], result["losses"], strict=True):
            print(f"     {probability:<10g}   {loss:.4f}")
        print(f"平均损失 = {result['mean']:.4f}")
    elif mode == "all":
        print("参数是手工设定的，没有经过训练。")
        if result["changes"]:
            print("这次在内存里改了参数：" + "，".join(result["changes"]) + "。")
        if result["scale"] != 1:
            print(f"这次把输出层的权重和偏置都乘了 {result['scale']:g}，三个分数因此都乘了 {result['scale']:g}。")
        print("编号     三个分数       三个概率                    选  标签     正确类别的概率  损失")
        for item in result["rows"]:
            verdict = "对" if item["prediction"] == item["label"] else "错"
            print(
                f"{item['id']:7}  {short(item['scores']):14} {numbers(item['probabilities']):27} {item['prediction']}   {item['label']}  {verdict}"
                f"     {item['probability_of_label']:.4f}      {item['loss']:.4f}"
            )
        print(f"16 行里选对 {result['correct']} 行，平均损失 {result['mean_loss']:.4f}。")
    else:
        print("人工构造的两组概率，三行的正确类别都是 0。")
        for model in result["models"]:
            print(f"模型{model['name']}")
            for row, prediction, loss in zip(model["rows"], model["predictions"], model["losses"], strict=True):
                verdict = "对" if prediction == 0 else "错"
                print(f"  三个概率 {short(row):18} 选 {prediction}　{verdict}　损失 {loss:.4f}")
            print(f"  答对 {model['correct']}/3，平均损失 {model['mean_loss']:.4f}")
        print("答对的行数是甲多，平均损失是乙小。两个指标量的不是同一件事。")
    print(f"exit_code: {result['exit_code']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scores", metavar="数,数,数", help="自己给三个分数，例如 --scores=-1,0,-1")
    mode.add_argument("--losses", metavar="概率,概率,…", help="给几个“正确类别的概率”，算各自的损失和平均，例如 --losses=0.25,0.5,0.75")
    mode.add_argument("--row", nargs=2, type=float, metavar=("A", "B"), help="一行输入走过手设的网络，例如 --row 0 2")
    mode.add_argument("--all", action="store_true", help="16 行都走一遍手设的网络")
    mode.add_argument("--disagree", action="store_true", help="答对的行数和平均损失说法不一样的例子")
    parser.add_argument("--label", type=int, help="与 --scores 合用：正确类别的编号")
    parser.add_argument("--set", action="append", default=[], metavar="名字=数值", help="与 --row、--all 合用：在内存里改网络的一个参数，例如 b1[h2]=-3")
    parser.add_argument("--scale", type=float, default=1.0, help="与 --row、--all 合用：把输出层的权重和偏置都乘上这个数")
    parser.add_argument("--json", action="store_true", help="输出结构化结果")
    args = parser.parse_args(argv)
    try:
        if args.losses:
            result = losses(number_list(args.losses, "概率"))
        elif args.row:
            result = one_row(tuple(args.row), args.set, args.scale)
        elif args.all:
            result = everything(args.set, args.scale)
        elif args.disagree:
            result = disagree()
        else:
            scores = number_list(args.scores, "分数") if args.scores else EXAMPLE_SCORES
            result = to_probabilities(scores, args.label)
    except (OSError, ValueError, KeyError) as error:
        result = {"error": f"实验失败：{error}", "exit_code": 2}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["exit_code"] == 2:
        print(result["error"], file=sys.stderr)
    else:
        print_run(result)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

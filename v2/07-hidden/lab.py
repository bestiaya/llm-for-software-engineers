"""一层不够：看线性模型在三行数据上的限制，再跟踪一行输入走过隐藏层的计算。只用标准库，不训练。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys


CHAPTER = Path(__file__).resolve().parent
COURSE = CHAPTER.parent
sys.path.insert(0, str(COURSE))

from shared import network as nw  # noqa: E402
from shared.numeric import linear_forward  # noqa: E402


TASK_FILE = COURSE / "data" / "counter-task.json"
MODEL_FILE = COURSE / "data" / "linear-model.json"
THREE_ROWS = (((0, 2), 1), ((2, 0), 1), ((1, 1), 0))     # 两端和正中间的一行，都在训练数据里
TRIES = 10000


def choose(scores):
    """取分数最大的类别；并列时取编号较小的。"""
    return max(range(len(scores)), key=lambda k: (scores[k], -k))


def all_rows():
    return json.loads(TASK_FILE.read_text(encoding="utf-8"))["rows"]


def network(changes):
    parameters = nw.hand_set_parameters()
    for change in changes:
        parameters = nw.set_parameter(parameters, change)
    return parameters


def trace(x, changes, relu):
    """一行输入走过手设的网络，每一步的乘加都留下来。"""
    parameters = network(changes)
    state = nw.forward(parameters, x, relu)
    hidden_terms = [
        {"unit": f"h{j + 1}", "weights": [parameters["W1"][i][j] for i in range(nw.INPUTS)], "bias": parameters["b1"][j], "pre": state["pre"][j]}
        for j in range(nw.HIDDEN)
    ]
    score_terms = [
        {"class": k, "weights": [parameters["W2"][j][k] for j in range(nw.HIDDEN)], "bias": parameters["b2"][k], "score": state["scores"][k]}
        for k in range(nw.CLASSES)
    ]
    return {
        "mode": "trace", "input": list(x), "changes": changes, "relu": relu,
        "hidden_terms": hidden_terms, "hidden": state["hidden"], "score_terms": score_terms, "scores": state["scores"],
        "prediction": choose(state["scores"]),
        "label_by_rule": int(x[0] >= 2) + int(x[1] >= 2),
        "exit_code": 0,
    }


def everything(changes, relu):
    """16 行都走一遍手设的网络。"""
    parameters = network(changes)
    table = []
    for row in all_rows():
        state = nw.forward(parameters, (row["a"], row["b"]), relu)
        table.append({
            "id": row["id"], "input": [row["a"], row["b"]], "hidden": state["hidden"], "scores": state["scores"],
            "prediction": choose(state["scores"]), "label": row["label"],
        })
    return {
        "mode": "all", "changes": changes, "relu": relu, "rows": table,
        "correct": sum(item["prediction"] == item["label"] for item in table),
        "distinct_scores": len({tuple(item["scores"]) for item in table}),
        "exit_code": 0,
    }


def three_rows(weights, bias):
    table = []
    for x, label in THREE_ROWS:
        scores = linear_forward([list(x)], weights, bias)[0]
        table.append({"input": list(x), "label": label, "scores": scores, "prediction": choose(scores), "margin": scores[1] - scores[0]})
    return table


def linear_limit(seed):
    """线性模型在三行数据上的限制：中间那一行的分数差，总是两端的平均。"""
    if seed is None:
        model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
        weights, bias, origin = model["weights"], model["bias"], "第 05 章那组手设的参数"
    else:
        source = random.Random(seed)
        weights = [[round(source.uniform(-2, 2), 1) for _ in range(3)] for _ in range(2)]
        bias = [round(source.uniform(-2, 2), 1) for _ in range(3)]
        origin = f"按种子 {seed} 随机取的一组参数"
    table = three_rows(weights, bias)
    # 随机试很多组参数：三行最多能对几行。
    source = random.Random(0)
    best, all_right = 0, 0
    for _ in range(TRIES):
        w = [[source.uniform(-3, 3) for _ in range(3)] for _ in range(2)]
        b = [source.uniform(-3, 3) for _ in range(3)]
        right = sum(item["prediction"] == item["label"] for item in three_rows(w, b))
        best, all_right = max(best, right), all_right + (right == 3)
    return {
        "mode": "linear", "origin": origin, "weights": weights, "bias": bias, "rows": table,
        "mean_of_ends": (table[0]["margin"] + table[1]["margin"]) / 2,
        "middle": table[2]["margin"],
        "correct": sum(item["prediction"] == item["label"] for item in table),
        "tries": TRIES, "tries_all_right": all_right, "tries_best": best,
        "exit_code": 0,
    }


def numbers(values):
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def sum_of(values, weights, bias):
    """把一次乘加写成算式。"""
    parts = [f"{value:g}×{weight:g}" if weight >= 0 else f"{value:g}×({weight:g})" for value, weight in zip(values, weights, strict=True)]
    offset = f" + {bias:g}" if bias >= 0 else f" − {-bias:g}"
    return " + ".join(parts) + offset


def print_run(result):
    if result["mode"] in ("trace", "all"):
        print("参数是手工设定的，没有经过训练。")
        if result["changes"]:
            print("这次在内存里改了参数：" + "，".join(result["changes"]) + "。")
        if not result["relu"]:
            print("这次去掉了 ReLU：隐藏值直接等于隐藏层的输入。")
    if result["mode"] == "trace":
        x = result["input"]
        print(f"输入：a={x[0]:g}，b={x[1]:g}")
        print("① 隐藏层的输入：两个输入乘权重，加偏置")
        for term in result["hidden_terms"]:
            print(f"   {term['unit']}：{sum_of(x, term['weights'], term['bias'])} = {term['pre']:g}")
        print("② 过 ReLU：小于 0 的变成 0" if result["relu"] else "② 没有 ReLU：原样通过")
        print(f"   隐藏值 {numbers(result['hidden'])}")
        print("③ 输出层：四个隐藏值乘权重，加偏置")
        for term in result["score_terms"]:
            print(f"   类别 {term['class']}：{sum_of(result['hidden'], term['weights'], term['bias'])} = {term['score']:g}")
        print(f"④ 取最大：三个分数 {numbers(result['scores'])}，选类别 {result['prediction']}。按标注规则，这一行的标签是 {result['label_by_rule']}。")
    elif result["mode"] == "all":
        print("编号     输入    隐藏值         三个分数       选  标签")
        for item in result["rows"]:
            verdict = "对" if item["prediction"] == item["label"] else "错"
            print(f"{item['id']:7}  {numbers(item['input']):7} {numbers(item['hidden']):14} {numbers(item['scores']):14} {item['prediction']}   {item['label']}   {verdict}")
        print(f"16 行里选对 {result['correct']} 行。")
        if result["distinct_scores"] == 1:
            print("16 行算出的三个分数完全相同：输入不管是什么，结果都一样。")
    else:
        print("三行训练数据：a=0 b=2（标签 1），a=2 b=0（标签 1），正中间的 a=1 b=1（标签 0）。")
        print(f"线性模型，{result['origin']}：W = {result['weights']}，b = {result['bias']}")
        for item in result["rows"]:
            verdict = "对" if item["prediction"] == item["label"] else "错"
            print(
                f"  a={item['input'][0]} b={item['input'][1]}　分数 {numbers(item['scores'])}　"
                f"类别 1 的分数减类别 0 的分数 = {item['margin']:g}　选 {item['prediction']}　{verdict}"
            )
        ends = result["rows"]
        print(f"两端的差取平均：({ends[0]['margin']:g} + {ends[1]['margin']:g}) ÷ 2 = {result['mean_of_ends']:g}；中间一行的差：{result['middle']:g}。两者相同。")
        print(f"这组参数在三行里对了 {result['correct']} 行。")
        print(f"另外随机试了 {result['tries']} 组参数：三行全对的有 {result['tries_all_right']} 组，最多对 {result['tries_best']} 行。")
    print(f"exit_code: {result['exit_code']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="16 行都走一遍手设的网络")
    mode.add_argument("--linear-limit", action="store_true", help="线性模型在三行数据上的限制")
    parser.add_argument("--a", type=float, default=0, help="输入 a，默认 0")
    parser.add_argument("--b", type=float, default=2, help="输入 b，默认 2")
    parser.add_argument("--set", action="append", default=[], metavar="名字=数值", help="在内存里改网络的一个参数，例如 b1[h2]=-3；可以写多次")
    parser.add_argument("--no-relu", action="store_true", help="去掉 ReLU，看隐藏层还剩什么作用")
    parser.add_argument("--seed", type=int, help="与 --linear-limit 合用：换一组随机取的线性模型参数")
    parser.add_argument("--json", action="store_true", help="输出结构化结果")
    args = parser.parse_args(argv)
    try:
        if args.linear_limit:
            result = linear_limit(args.seed)
        elif args.all:
            result = everything(args.set, not args.no_relu)
        else:
            result = trace((args.a, args.b), args.set, not args.no_relu)
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

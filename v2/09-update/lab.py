"""参数怎样学：拨动一个参数，做一次更新，沿一条短链把变化量传回去。只用标准库，不训练。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CHAPTER = Path(__file__).resolve().parent
COURSE = CHAPTER.parent
sys.path.insert(0, str(COURSE))

from shared import network as nw  # noqa: E402


LEARNING_RATE = 0.3
NUDGE = 0.001
START_SEED = 1


def bias_state():
    """为手算挑的一组数，不是训练出来的：输出权重全为 0，三个输出偏置都是 0，所以三个分数都是 0。"""
    parameters = nw.hand_set_parameters()
    parameters["W2"] = [[0.0] * nw.CLASSES for _ in range(nw.HIDDEN)]
    parameters["b2"] = [0.0] * nw.CLASSES
    return parameters


def chain_state():
    """为手算挑的另一组数，不是训练出来的：隐藏层用手设的参数，只有 h1、h2 通向输出。"""
    parameters = nw.hand_set_parameters()
    parameters["W2"] = [[1.0, 0.0, -1.0], [-2.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    parameters["b2"] = [0.0, 0.0, 0.0]
    return parameters


def bias_example():
    """只动三个输出偏置：先拨动看损失，再用梯度更新一次，然后从头复算。"""
    parameters, x, label = bias_state(), (0, 2), 1
    before = nw.forward(parameters, x)
    loss_before = nw.loss(parameters, x, label)
    nudges = []
    for k in range(nw.CLASSES):
        changed = nw.unflatten(nw.flatten(parameters))
        changed["b2"][k] += NUDGE
        nudges.append({"parameter": f"b2[类别{k}]", "loss_change": nw.loss(changed, x, label) - loss_before})
    gradient = nw.analytic_gradient(parameters, [{"x": x, "label": label}])["b2"]
    updated = nw.unflatten(nw.flatten(parameters))
    updated["b2"] = [value - LEARNING_RATE * slope for value, slope in zip(parameters["b2"], gradient, strict=True)]
    after = nw.forward(updated, x)
    return {
        "mode": "bias", "x": x, "label": label, "nudge": NUDGE, "learning_rate": LEARNING_RATE,
        "before": {"scores": before["scores"], "probabilities": before["probabilities"], "loss": loss_before},
        "nudges": nudges, "gradient": gradient, "new_bias": updated["b2"],
        "after": {"scores": after["scores"], "probabilities": after["probabilities"], "loss": nw.loss(updated, x, label)},
        "exit_code": 0,
    }


def chain_example(label=2):
    """问一个隔着两层的参数：把 W1[a→h1] 拨一点，损失变多少。先沿路径手算，再拨动核对。"""
    parameters, x = chain_state(), (3, 3)
    state = nw.forward(parameters, x)
    d_scores = [state["probabilities"][k] - (1.0 if k == label else 0.0) for k in range(nw.CLASSES)]
    branches = [parameters["W2"][0][k] * d_scores[k] for k in range(nw.CLASSES)]
    d_hidden = sum(branches)
    d_pre = d_hidden if state["pre"][0] > 0 else 0.0
    slope = x[0] * d_pre
    loss_before = nw.loss(parameters, x, label)
    changed = nw.unflatten(nw.flatten(parameters))
    changed["W1"][0][0] += NUDGE
    loss_after = nw.loss(changed, x, label)
    program = nw.analytic_gradient(parameters, [{"x": x, "label": label}])["W1"][0][0]
    return {
        "mode": "chain", "x": x, "label": label, "parameter": "W1[a→h1]", "value": parameters["W1"][0][0],
        "forward": state, "output_weights": parameters["W2"], "d_scores": d_scores,
        "weights_from_h1": parameters["W2"][0], "branches": branches,
        "d_hidden": d_hidden, "pre_h1": state["pre"][0], "d_pre": d_pre, "input_a": x[0], "slope_by_hand": slope,
        "slope_by_program": program, "nudge": NUDGE, "loss_before": loss_before, "loss_after": loss_after,
        "loss_change": loss_after - loss_before, "kink_margin": nw.kink_margin(parameters, [{"x": x, "label": label}]),
        "exit_code": 0,
    }


def gradient_check():
    """用拨动法抽查程序算出的全部 27 个梯度：训练的起点，10 行训练数据。"""
    rows = nw.load_task(COURSE / "data/counter-task.json")["train"]
    parameters = nw.initial_parameters(START_SEED)
    by_program = nw.flatten(nw.analytic_gradient(parameters, rows))
    by_nudging = nw.flatten(nw.numeric_gradient(parameters, rows))
    table = [
        {"parameter": name, "value": value, "by_program": a, "by_nudging": n, "difference": abs(a - n)}
        for name, value, a, n in zip(nw.parameter_names(), nw.flatten(parameters), by_program, by_nudging, strict=True)
    ]
    largest = max(row["difference"] for row in table)
    return {
        "mode": "check", "seed": START_SEED, "rows": len(rows), "mean_loss": nw.mean_loss(parameters, rows),
        "kink_margin": nw.kink_margin(parameters, rows), "table": table, "largest_difference": largest,
        "loss_evaluations_by_nudging": 2 * len(table), "agrees": largest < 1e-6,
        "exit_code": 0 if largest < 1e-6 else 1,
    }


def numbers(values):
    return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def print_run(result):
    if result["mode"] == "bias":
        print("状态：为手算挑的一组数，不是训练出来的。输出权重全为 0，三个输出偏置都是 0。")
        print(f"样本：a={result['x'][0]}，b={result['x'][1]}，正确类别 {result['label']}。")
        before, after = result["before"], result["after"]
        print(f"① 现在　分数 {numbers(before['scores'])}　概率 {numbers(before['probabilities'])}　损失 {before['loss']:.6f}")
        print(f"② 拨动　把每个输出偏置各自加 {result['nudge']}，看损失变多少")
        for item in result["nudges"]:
            print(f"     {item['parameter']} → 损失变化 {item['loss_change']:+.6f}，相当于每拨 1 变 {item['loss_change'] / result['nudge']:+.3f}")
        print(f"③ 梯度　程序算出的是 {numbers(result['gradient'])}：三个概率，正确类别的那一个减 1。拨动得到的是它的近似")
        print(f"④ 更新　学习率 {result['learning_rate']}，新偏置 = 旧偏置 − 学习率 × 梯度 = {numbers(result['new_bias'])}")
        print(f"⑤ 复算　分数 {numbers(after['scores'])}　概率 {numbers(after['probabilities'])}　损失 {after['loss']:.6f}")
        print(f"   正确类别的概率从 {before['probabilities'][result['label']]:.6f} 升到 {after['probabilities'][result['label']]:.6f}")
    elif result["mode"] == "chain":
        state = result["forward"]
        print("状态：为手算挑的一组数，不是训练出来的。")
        print(f"样本：a={result['x'][0]}，b={result['x'][1]}，正确类别 {result['label']}。要问的参数：{result['parameter']}，现在是 {result['value']}。")
        rule_label = int(result["x"][0] >= 2) + int(result["x"][1] >= 2)
        if result["label"] != rule_label:
            print(f"　　　按标注规则，这一行的标签是 {rule_label}。这里暂时把它当成 {result['label']} 来算，只为看标签怎样决定调整的方向。")
        print(f"前向　隐藏层的输入 {numbers(state['pre'])}　隐藏值 {numbers(state['hidden'])}")
        shown = "，".join(f"h{j + 1} [" + ", ".join(f"{value:g}" for value in row) + "]" for j, row in enumerate(result["output_weights"]))
        print(f"　　　输出层的权重（通向类别 0、1、2）：{shown}；输出偏置都是 0")
        print(f"　　　分数 {numbers(state['scores'])}　概率 {numbers(state['probabilities'])}")
        gate = 1 if result["pre_h1"] > 0 else 0
        print("沿路径往回算。每一步分清两个数：这一段自己的变化率，和损失传回到这里的变化率。")
        print(f"  ① 损失对三个分数的变化率 = 三个概率，正确类别的那一个减 1 = {numbers(result['d_scores'])}")
        print(f"  ② 这一段自己的：h1 通向三个分数的权重 {numbers(result['weights_from_h1'])}；三条支路各自传回 {numbers(result['branches'])}")
        print(f"  ③ 三条支路相加，损失对 h1 的变化率 = {result['d_hidden']:.6f}")
        print(
            f"  ④ 这一段自己的：h1 的输入是 {result['pre_h1']:.1f}，{'大于 0，ReLU 原样通过' if gate else '不大于 0，ReLU 关着'}，是 {gate}；"
            f"损失对 h1 的输入的变化率 = {result['d_hidden']:.6f} × {gate} = {result['d_pre']:.6f}"
        )
        print(
            f"  ⑤ 这一段自己的：这个权重乘的是 a={result['input_a']}，是 {result['input_a']}；"
            f"损失对这个权重的变化率 = {result['d_pre']:.6f} × {result['input_a']} = {result['slope_by_hand']:.6f}"
        )
        print(f"程序算出的同一个梯度：{result['slope_by_program']:.6f}")
        print(
            f"拨动核对：把它加 {result['nudge']}，损失从 {result['loss_before']:.6f} 变成 {result['loss_after']:.6f}，"
            f"变化 {result['loss_change']:+.6f}，约等于 {result['slope_by_hand']:.1f} × {result['nudge']}"
        )
    else:
        print(f"状态：训练的起点（种子 {result['seed']}）。数据：{result['rows']} 行训练数据，平均损失 {result['mean_loss']:.6f}。")
        print(f"离 ReLU 折点最近的距离 {result['kink_margin']:.4f}，远大于拨动的步长，可以用拨动法核对。")
        print("参数               现值      程序算的梯度    拨动法算的梯度")
        for row in result["table"]:
            print(f"{row['parameter']:<14} {row['value']:>8.3f} {row['by_program']:>14.8f} {row['by_nudging']:>14.8f}")
        print(f"27 个梯度里两种算法相差最大的是 {result['largest_difference']:.2e}。")
        print(f"拨动法为此算了 {result['loss_evaluations_by_nudging']} 次平均损失；程序沿路径回传，每行数据只走一遍前向和一遍回传。")
        print("两种算法" + ("一致。" if result["agrees"] else "不一致，需要检查。"))
    print(f"exit_code: {result['exit_code']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chain", action="store_true", help="沿一条短链手算一个隔着两层的参数")
    mode.add_argument("--check", action="store_true", help="用拨动法抽查程序算出的全部梯度")
    parser.add_argument("--label", type=int, choices=(0, 1, 2), default=2, help="与 --chain 合用：把正确类别换成别的，看结论怎样变")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)
    result = chain_example(args.label) if args.chain else gradient_check() if args.check else bias_example()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_run(result)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

"""固定窗口的对照模型：不用注意力，给最近的几个词各留一个槽位，把槽位里的向量排成一行，直接算出词表上的概率。只用标准库。

它和七词小语言模型（tiny_lm.py）用同一份词表、同样的两句话、同样的输出接口：每个位置给出七个候选的概率。
差别在怎样读前文。注意力按内容算出权重，再按权重去读；窗口没有这一步，离当前位置第几个词就放进第几个槽，
每个槽各有各的输出权重。窗口有几个槽，就只看得见最近的几个词。

这个模型比 tiny_lm.py 里的还要简单：没有位置向量，没有前馈变换，槽位之后只有一次乘加。
形状约定与前面几章相同：一行向量乘以 [输入, 输出] 排列的权重。
"""

from __future__ import annotations

import copy
import math
import random

from shared import tiny_lm as lm


SHAPES_BY_NAME = ("E", "U", "c")


def shapes(dimension, window):
    vocab = len(lm.VOCAB)
    return {"E": (vocab, dimension), "U": (window * dimension, vocab), "c": (vocab,)}


def parameter_count(dimension, window):
    return sum(math.prod(shape) for shape in shapes(dimension, window).values())


def sizes(parameters):
    """从参数本身读出向量的维数和窗口的槽数。"""
    dimension = len(parameters["E"][0])
    return dimension, len(parameters["U"]) // dimension


def initial_parameters(seed, dimension=2, window=lm.CONTEXT, scale=0.5):
    """训练的起点：取法与 tiny_lm.py 相同，同一个种子每次得到同一组数。"""
    if window < 1:
        raise ValueError("窗口至少要有 1 个槽。")
    source = random.Random(seed)

    def draw():
        return round(source.uniform(-scale, scale), 3)

    parameters = {}
    for name, shape in shapes(dimension, window).items():
        if len(shape) == 1:
            parameters[name] = [draw() for _ in range(shape[0])]
        else:
            parameters[name] = [[draw() for _ in range(shape[1])] for _ in range(shape[0])]
    return parameters


def flatten(parameters):
    values = []
    for name in SHAPES_BY_NAME:
        block = parameters[name]
        if block and isinstance(block[0], list):
            values.extend(value for row in block for value in row)
        else:
            values.extend(block)
    return values


def unflatten(values, dimension, window):
    values, position, parameters = list(values), 0, {}
    for name, shape in shapes(dimension, window).items():
        if len(shape) == 1:
            parameters[name] = values[position:position + shape[0]]
            position += shape[0]
        else:
            parameters[name] = [values[position + r * shape[1]:position + (r + 1) * shape[1]] for r in range(shape[0])]
            position += shape[0] * shape[1]
    if position != len(values):
        raise ValueError("参数个数与模型的尺寸不符。")
    return parameters


def slots(ids, t, window):
    """位置 t 的各个槽里放的是哪个词：第 0 槽是当前的词，第 1 槽是前一个词，依此类推；前面没有词的槽是空的。"""
    return [ids[t - s] if t - s >= 0 else None for s in range(window)]


def visible(ids, t, window):
    """位置 t 看得见的那一段前文：最近的 window 个词，按原来的顺序。"""
    return list(ids[max(0, t + 1 - window):t + 1])


def forward(parameters, ids):
    """给一段前文的编号，算出每个位置上“下一个词”的概率。中间的每一步都留下来。"""
    if not 1 <= len(ids) <= lm.CONTEXT:
        raise ValueError(f"前文长度须在 1 到 {lm.CONTEXT} 之间。")
    dimension, window = sizes(parameters)
    filled, inputs, scores, probabilities = [], [], [], []
    for t in range(len(ids)):
        held = slots(ids, t, window)
        # 把各个槽里的向量接成一行；空槽全填 0。
        row = []
        for token in held:
            row.extend(parameters["E"][token] if token is not None else [0.0] * dimension)
        z = [value + bias for value, bias in zip(lm.times(row, parameters["U"]), parameters["c"], strict=True)]
        filled.append(held)
        inputs.append(row)
        scores.append(z)
        probabilities.append(lm.softmax(z))
    return {"ids": list(ids), "slots": filled, "inputs": inputs, "scores": scores, "probabilities": probabilities}


def mean_loss(parameters, pairs):
    """所有位置上，正确的下一个词的概率取负对数，再求平均。与 tiny_lm.py 的算法相同。"""
    total, count = 0.0, 0
    for pair in pairs:
        state = forward(parameters, pair["inputs"])
        for t, target in enumerate(pair["targets"]):
            total -= math.log(state["probabilities"][t][target])
            count += 1
    return total / count


def loss_floor(pairs, window):
    """只看得见最近 window 个词时，损失最低能到多少。

    看得见的那一段相同、后面的词却不同时，再好的模型也只能把概率分给它们。
    窗口看得全前文时，这个数与 tiny_lm.loss_floor 相同。
    """
    followers, total = {}, 0
    for pair in pairs:
        for t, target in enumerate(pair["targets"]):
            followers.setdefault(tuple(visible(pair["inputs"], t, window)), []).append(target)
            total += 1
    floor = 0.0
    for targets in followers.values():
        for target in set(targets):
            floor -= targets.count(target) * math.log(targets.count(target) / len(targets))
    return floor / total


def numeric_gradient(parameters, pairs, step=1e-6):
    """拨动法：每个参数向两边各拨一点，看平均损失差多少。"""
    dimension, window = sizes(parameters)
    values, gradient = flatten(parameters), []
    for index in range(len(values)):
        up, down = list(values), list(values)
        up[index] += step
        down[index] -= step
        gradient.append(
            (mean_loss(unflatten(up, dimension, window), pairs) - mean_loss(unflatten(down, dimension, window), pairs)) / (2 * step)
        )
    return unflatten(gradient, dimension, window)


def analytic_gradient(parameters, pairs):
    """沿计算路径把变化量传回去。路径只有两段：输出的乘加，和槽位里的词向量。"""
    dimension, window = sizes(parameters)
    vocab = len(lm.VOCAB)
    total = unflatten([0.0] * len(flatten(parameters)), dimension, window)
    count = sum(len(pair["targets"]) for pair in pairs)
    for pair in pairs:
        state = forward(parameters, pair["inputs"])
        for t, target in enumerate(pair["targets"]):
            d_z = [(state["probabilities"][t][c] - (1.0 if c == target else 0.0)) / count for c in range(vocab)]
            row = state["inputs"][t]
            for c, slope in enumerate(d_z):
                total["c"][c] += slope
                for i, value in enumerate(row):
                    total["U"][i][c] += value * slope
            for s, token in enumerate(state["slots"][t]):
                if token is None:
                    continue
                for j in range(dimension):
                    i = s * dimension + j
                    total["E"][token][j] += sum(parameters["U"][i][c] * d_z[c] for c in range(vocab))
    return total


def update(parameters, gradient, learning_rate):
    dimension, window = sizes(parameters)
    return unflatten(
        [value - learning_rate * slope for value, slope in zip(flatten(parameters), flatten(gradient), strict=True)],
        dimension, window,
    )


def train(parameters, pairs, learning_rate, steps, on_step=None):
    """训练循环：算梯度，更新，再算。与 tiny_lm.py 相同。"""
    current = copy.deepcopy(parameters)
    if on_step:
        on_step(0, current)
    for step in range(1, steps + 1):
        current = update(current, analytic_gradient(current, pairs), learning_rate)
        if on_step:
            on_step(step, current)
    return current


def generate(parameters, prefix_ids, temperature=0.0, seed=0):
    """生成循环，选词和停止的规则与 tiny_lm.generate 相同。"""
    ids, steps, source = list(prefix_ids), [], random.Random(seed)
    while True:
        probabilities = forward(parameters, ids)["probabilities"][-1]
        chosen = lm.pick(probabilities, temperature, source)
        steps.append({"prefix": list(ids), "probabilities": probabilities, "chosen": chosen})
        ids.append(chosen)
        if chosen == lm.EOS_ID:
            return {"ids": ids, "steps": steps, "stopped_by": "EOS"}
        if len(ids) > lm.CONTEXT:
            return {"ids": ids, "steps": steps, "stopped_by": "前文已读满"}

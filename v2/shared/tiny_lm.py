"""七词小语言模型：从分词、查表、注意力，一路算到词表上的概率，再生成和训练。只用标准库。

这是一个刻意简化的模型，用来看清一次预测怎样走完。与完整的 Transformer 相比：
只有一层、一个注意力头；没有残差连接，没有归一化；向量只有很少的几维。
位置信息、按内容读取前文、因果遮罩、前馈变换、到词表的输出映射，这几样计算关系是真实的。

形状约定与前面几章相同：一行向量乘以 [输入, 输出] 排列的权重。
"""

from __future__ import annotations

import copy
import math
import random


VOCAB = ["请求", "成功", "失败", "后", "返回", "重试", "EOS"]
TOKEN_ID = {token: index for index, token in enumerate(VOCAB)}
EOS_ID = TOKEN_ID["EOS"]
SENTENCES = ["请求 成功 后 返回 EOS", "请求 失败 后 重试 EOS"]
CONTEXT = 4          # 一次最多读 4 个位置的前文

# 第 12、13 章手算用的二维向量。它们是为了好算挑的，不是训练出来的。
HAND_EMBEDDING = [[0, 0], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1], [0, -1]]

SHAPES_BY_NAME = ("E", "P", "Wq", "Wk", "Wv", "W1", "b1", "W2", "b2", "U", "c")


def encode(text):
    """分词并编号：本例按空格分开，每个词在词表里的位置就是它的编号。"""
    return [TOKEN_ID[token] for token in text.split()]


def decode(ids):
    return " ".join(VOCAB[index] for index in ids)


def training_pairs():
    """每句话拆成输入和目标：目标就是输入向后错开一位。"""
    pairs = []
    for sentence in SENTENCES:
        ids = encode(sentence)
        pairs.append({"text": sentence, "inputs": ids[:-1], "targets": ids[1:]})
    return pairs


def shapes(dimension, hidden):
    vocab = len(VOCAB)
    return {
        "E": (vocab, dimension), "P": (CONTEXT, dimension),
        "Wq": (dimension, dimension), "Wk": (dimension, dimension), "Wv": (dimension, dimension),
        "W1": (dimension, hidden), "b1": (hidden,), "W2": (hidden, dimension), "b2": (dimension,),
        "U": (dimension, vocab), "c": (vocab,),
    }


def parameter_count(dimension, hidden):
    return sum(math.prod(shape) for shape in shapes(dimension, hidden).values())


def initial_parameters(seed, dimension=2, hidden=4, scale=0.5):
    """训练的起点：按给定的种子取一组小随机数。同一个种子每次得到同一组数。"""
    source = random.Random(seed)

    def draw():
        return round(source.uniform(-scale, scale), 3)

    parameters = {}
    for name, shape in shapes(dimension, hidden).items():
        if len(shape) == 1:
            parameters[name] = [draw() for _ in range(shape[0])]
        else:
            parameters[name] = [[draw() for _ in range(shape[1])] for _ in range(shape[0])]
    return parameters


def hand_set_parameters():
    """手算用的一组参数，不是训练出来的，而且只设到注意力为止。

    词向量用上面那张手算表，位置向量全为 0，查询、键、值三个映射都原样通过。
    后面的前馈和输出映射全为 0，所以这组参数只用来看“读前文”这一步，不用来预测。
    """
    dimension, hidden = 2, 4
    parameters = unflatten([0.0] * parameter_count(dimension, hidden), dimension, hidden)
    parameters["E"] = [[float(value) for value in row] for row in HAND_EMBEDDING]
    for name in ("Wq", "Wk", "Wv"):
        parameters[name] = [[1.0, 0.0], [0.0, 1.0]]
    return parameters


def sizes(parameters):
    return len(parameters["E"][0]), len(parameters["b1"])


def flatten(parameters):
    values = []
    for name in SHAPES_BY_NAME:
        block = parameters[name]
        if block and isinstance(block[0], list):
            values.extend(value for row in block for value in row)
        else:
            values.extend(block)
    return values


def unflatten(values, dimension, hidden):
    values, position, parameters = list(values), 0, {}
    for name, shape in shapes(dimension, hidden).items():
        if len(shape) == 1:
            parameters[name] = values[position:position + shape[0]]
            position += shape[0]
        else:
            parameters[name] = [values[position + r * shape[1]:position + (r + 1) * shape[1]] for r in range(shape[0])]
            position += shape[0] * shape[1]
    if position != len(values):
        raise ValueError("参数个数与模型的尺寸不符。")
    return parameters


def softmax(scores):
    top = max(scores)
    powers = [math.exp(score - top) for score in scores]
    total = sum(powers)
    return [power / total for power in powers]


def times(vector, matrix):
    """一行向量乘以 [输入, 输出] 排列的矩阵。"""
    return [sum(vector[i] * matrix[i][j] for i in range(len(vector))) for j in range(len(matrix[0]))]


def forward(parameters, ids):
    """给一段前文的编号，算出每个位置上“下一个词”的概率。中间的每一步都留下来。"""
    if not 1 <= len(ids) <= CONTEXT:
        raise ValueError(f"前文长度须在 1 到 {CONTEXT} 之间。")
    dimension, _ = sizes(parameters)
    scale = math.sqrt(dimension)
    # 查表：按编号取一行向量，再加上这个位置的向量。
    x = [[parameters["E"][token][i] + parameters["P"][t][i] for i in range(dimension)] for t, token in enumerate(ids)]
    q = [times(row, parameters["Wq"]) for row in x]
    k = [times(row, parameters["Wk"]) for row in x]
    v = [times(row, parameters["Wv"]) for row in x]
    matches, weights, attended, pre, hidden, features, scores, probabilities = [], [], [], [], [], [], [], []
    for t in range(len(ids)):
        # 因果遮罩：位置 t 只能读它自己和它前面的位置。
        match = [sum(q[t][i] * k[u][i] for i in range(dimension)) / scale for u in range(t + 1)]
        w = softmax(match)
        matches.append(match)
        a = [sum(w[u] * v[u][i] for u in range(t + 1)) for i in range(dimension)]
        h_pre = [value + bias for value, bias in zip(times(a, parameters["W1"]), parameters["b1"], strict=True)]
        h = [value if value > 0 else 0.0 for value in h_pre]
        f = [value + bias for value, bias in zip(times(h, parameters["W2"]), parameters["b2"], strict=True)]
        z = [value + bias for value, bias in zip(times(f, parameters["U"]), parameters["c"], strict=True)]
        weights.append(w)
        attended.append(a)
        pre.append(h_pre)
        hidden.append(h)
        features.append(f)
        scores.append(z)
        probabilities.append(softmax(z))
    return {
        "ids": list(ids), "x": x, "q": q, "k": k, "v": v, "matches": matches, "weights": weights, "attended": attended,
        "pre": pre, "hidden": hidden, "features": features, "scores": scores, "probabilities": probabilities,
    }


def mean_loss(parameters, pairs):
    """所有位置上，正确的下一个词的概率取负对数，再求平均。"""
    total, count = 0.0, 0
    for pair in pairs:
        state = forward(parameters, pair["inputs"])
        for t, target in enumerate(pair["targets"]):
            total -= math.log(state["probabilities"][t][target])
            count += 1
    return total / count


def loss_floor(pairs):
    """这份数据上损失的下限：同一段前文后面跟着不同的词时，再好的模型也只能把概率分给它们。"""
    followers, total = {}, 0
    for pair in pairs:
        for t, target in enumerate(pair["targets"]):
            followers.setdefault(tuple(pair["inputs"][:t + 1]), []).append(target)
            total += 1
    floor = 0.0
    for targets in followers.values():
        for target in set(targets):
            share = targets.count(target) / len(targets)
            floor -= targets.count(target) * math.log(share)
    return floor / total


def numeric_gradient(parameters, pairs, step=1e-6):
    """拨动法：每个参数向两边各拨一点，看平均损失差多少。"""
    dimension, hidden = sizes(parameters)
    values, gradient = flatten(parameters), []
    for index in range(len(values)):
        up, down = list(values), list(values)
        up[index] += step
        down[index] -= step
        gradient.append(
            (mean_loss(unflatten(up, dimension, hidden), pairs) - mean_loss(unflatten(down, dimension, hidden), pairs)) / (2 * step)
        )
    return unflatten(gradient, dimension, hidden)


def analytic_gradient(parameters, pairs):
    """沿计算路径把变化量传回去，一遍得到全部参数的梯度。做法与第 09 章相同，只是路径更长。"""
    dimension, hidden_size = sizes(parameters)
    scale = math.sqrt(dimension)
    total = unflatten([0.0] * len(flatten(parameters)), dimension, hidden_size)
    count = sum(len(pair["targets"]) for pair in pairs)
    for pair in pairs:
        ids, state = pair["inputs"], forward(parameters, pair["inputs"])
        length = len(ids)
        d_q = [[0.0] * dimension for _ in range(length)]
        d_k = [[0.0] * dimension for _ in range(length)]
        d_v = [[0.0] * dimension for _ in range(length)]
        for t in range(length):
            d_z = [
                (state["probabilities"][t][c] - (1.0 if c == pair["targets"][t] else 0.0)) / count
                for c in range(len(VOCAB))
            ]
            f, h, a, w = state["features"][t], state["hidden"][t], state["attended"][t], state["weights"][t]
            for c, slope in enumerate(d_z):
                total["c"][c] += slope
                for i in range(dimension):
                    total["U"][i][c] += f[i] * slope
            d_f = [sum(parameters["U"][i][c] * d_z[c] for c in range(len(VOCAB))) for i in range(dimension)]
            for i in range(dimension):
                total["b2"][i] += d_f[i]
                for j in range(hidden_size):
                    total["W2"][j][i] += h[j] * d_f[i]
            d_h = [sum(parameters["W2"][j][i] * d_f[i] for i in range(dimension)) for j in range(hidden_size)]
            d_pre = [d_h[j] if state["pre"][t][j] > 0 else 0.0 for j in range(hidden_size)]
            for j in range(hidden_size):
                total["b1"][j] += d_pre[j]
                for i in range(dimension):
                    total["W1"][i][j] += a[i] * d_pre[j]
            d_a = [sum(parameters["W1"][i][j] * d_pre[j] for j in range(hidden_size)) for i in range(dimension)]
            # 注意力：读到的向量是各位置的值按权重相加。
            d_w = [sum(d_a[i] * state["v"][u][i] for i in range(dimension)) for u in range(t + 1)]
            mixed = sum(w[u] * d_w[u] for u in range(t + 1))
            for u in range(t + 1):
                for i in range(dimension):
                    d_v[u][i] += w[u] * d_a[i]
                d_match = w[u] * (d_w[u] - mixed)
                for i in range(dimension):
                    d_q[t][i] += d_match * state["k"][u][i] / scale
                    d_k[u][i] += d_match * state["q"][t][i] / scale
        for u in range(length):
            x = state["x"][u]
            d_x = [0.0] * dimension
            for i in range(dimension):
                for j in range(dimension):
                    total["Wq"][i][j] += x[i] * d_q[u][j]
                    total["Wk"][i][j] += x[i] * d_k[u][j]
                    total["Wv"][i][j] += x[i] * d_v[u][j]
                    d_x[i] += (
                        d_q[u][j] * parameters["Wq"][i][j] + d_k[u][j] * parameters["Wk"][i][j]
                        + d_v[u][j] * parameters["Wv"][i][j]
                    )
            for i in range(dimension):
                total["E"][ids[u]][i] += d_x[i]
                total["P"][u][i] += d_x[i]
    return total


def update(parameters, gradient, learning_rate):
    dimension, hidden = sizes(parameters)
    return unflatten(
        [value - learning_rate * slope for value, slope in zip(flatten(parameters), flatten(gradient), strict=True)],
        dimension, hidden,
    )


def train(parameters, pairs, learning_rate, steps, on_step=None):
    """训练循环：算梯度，更新，再算。与第 10 章相同。"""
    current = copy.deepcopy(parameters)
    if on_step:
        on_step(0, current)
    for step in range(1, steps + 1):
        current = update(current, analytic_gradient(current, pairs), learning_rate)
        if on_step:
            on_step(step, current)
    return current


def pick(probabilities, temperature=0.0, source=None):
    """选下一个词。温度为 0 时取概率最大的（并列取编号较小的）；否则按调过温度的概率抽样。"""
    if temperature <= 0:
        return max(range(len(probabilities)), key=lambda index: (probabilities[index], -index))
    adjusted = softmax([math.log(max(p, 1e-300)) / temperature for p in probabilities])
    threshold, running = source.random(), 0.0
    for index, share in enumerate(adjusted):
        running += share
        if threshold < running:
            return index
    return len(adjusted) - 1


def generate(parameters, prefix_ids, temperature=0.0, seed=0):
    """生成循环：算出下一个词的概率，选一个接到前文后面，再算；遇到 EOS 或读满前文就停。"""
    ids, steps, source = list(prefix_ids), [], random.Random(seed)
    while True:
        probabilities = forward(parameters, ids)["probabilities"][-1]
        chosen = pick(probabilities, temperature, source)
        steps.append({"prefix": list(ids), "probabilities": probabilities, "chosen": chosen})
        ids.append(chosen)
        if chosen == EOS_ID:
            return {"ids": ids, "steps": steps, "stopped_by": "EOS"}
        if len(ids) > CONTEXT:
            return {"ids": ids, "steps": steps, "stopped_by": "前文已读满"}

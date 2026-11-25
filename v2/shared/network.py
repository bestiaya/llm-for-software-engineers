"""2 个输入、4 个隐藏单元、3 个类别的小网络：前向、损失、梯度、一次更新和训练。只用标准库。

形状约定与前面几章相同：一行输入乘以 [输入, 输出] 排列的权重，再加偏置。

    隐藏层  h = ReLU(x · W1 + b1)    W1 是 2 行 4 列，b1 有 4 个数
    输出层  z = h · W2 + b2          W2 是 4 行 3 列，b2 有 3 个数

一共 8 + 4 + 12 + 3 = 27 个参数。
"""

from __future__ import annotations

import copy
import math
import random


INPUTS, HIDDEN, CLASSES = 2, 4, 3
INPUT_NAMES = ("a", "b")


def hand_set_parameters():
    """手设的一组参数：四个隐藏单元分别算 ReLU(a-1)、ReLU(a-2)、ReLU(b-1)、ReLU(b-2)。

    c = h1 - h2 + h3 - h4 正好是达到阈值的通道数，输出分数取 [1-2c, 0, 2c-3]。
    它说明这个结构装得下正确的规则；它不是训练出来的。
    """
    signs = (1.0, -1.0, 1.0, -1.0)
    return {
        "W1": [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
        "b1": [-1.0, -2.0, -1.0, -2.0],
        "W2": [[-2.0 * sign, 0.0, 2.0 * sign] for sign in signs],
        "b2": [1.0, 0.0, -3.0],
    }


def initial_parameters(seed, scale=0.5):
    """训练的起点：按给定的种子取一组小随机数。同一个种子每次得到同一组数。"""
    source = random.Random(seed)

    def draw():
        return round(source.uniform(-scale, scale), 3)

    return {
        "W1": [[draw() for _ in range(HIDDEN)] for _ in range(INPUTS)],
        "b1": [draw() for _ in range(HIDDEN)],
        "W2": [[draw() for _ in range(CLASSES)] for _ in range(HIDDEN)],
        "b2": [draw() for _ in range(CLASSES)],
    }


def parameter_names():
    """27 个参数各自的名字，顺序与 flatten 一致。"""
    names = [f"W1[{INPUT_NAMES[i]}→h{j + 1}]" for i in range(INPUTS) for j in range(HIDDEN)]
    names += [f"b1[h{j + 1}]" for j in range(HIDDEN)]
    names += [f"W2[h{j + 1}→类别{k}]" for j in range(HIDDEN) for k in range(CLASSES)]
    names += [f"b2[类别{k}]" for k in range(CLASSES)]
    return names


def flatten(parameters):
    values = [value for row in parameters["W1"] for value in row]
    values += parameters["b1"]
    values += [value for row in parameters["W2"] for value in row]
    values += parameters["b2"]
    return values


def unflatten(values):
    values = list(values)
    if len(values) != INPUTS * HIDDEN + HIDDEN + HIDDEN * CLASSES + CLASSES:
        raise ValueError("参数个数应为 27。")
    position = 0

    def take(count):
        nonlocal position
        chunk = values[position:position + count]
        position += count
        return chunk

    return {
        "W1": [take(HIDDEN) for _ in range(INPUTS)],
        "b1": take(HIDDEN),
        "W2": [take(CLASSES) for _ in range(HIDDEN)],
        "b2": take(CLASSES),
    }


def softmax(scores):
    """分数变概率：先减去最大值再取指数，避免数值溢出，结果不变。"""
    top = max(scores)
    powers = [math.exp(score - top) for score in scores]
    total = sum(powers)
    return [power / total for power in powers]


def set_parameter(parameters, text):
    """按“名字=数值”改一个参数，返回改过的副本。名字的写法：

    W1[a,h1]  第一层里从输入 a 通向隐藏单元 h1 的权重（输入写 a 或 b，隐藏单元写 h1 到 h4）
    b1[h2]    隐藏单元 h2 的偏置
    W2[h1,0]  第二层里从 h1 通向类别 0 的权重（类别写 0、1、2）
    b2[1]     类别 1 的输出偏置
    """
    name, separator, raw = text.partition("=")
    name = name.strip()
    try:
        value = float(raw)
        if not separator or not math.isfinite(value):
            raise ValueError
        block, _, inside = name.rstrip("]").partition("[")
        keys = [key.strip() for key in inside.split(",")]
        hidden = lambda key: ("h1", "h2", "h3", "h4").index(key)
        category = lambda key: ("0", "1", "2").index(key)
        changed = copy.deepcopy(parameters)
        if block == "W1" and len(keys) == 2:
            changed["W1"][INPUT_NAMES.index(keys[0])][hidden(keys[1])] = value
        elif block == "b1" and len(keys) == 1:
            changed["b1"][hidden(keys[0])] = value
        elif block == "W2" and len(keys) == 2:
            changed["W2"][hidden(keys[0])][category(keys[1])] = value
        elif block == "b2" and len(keys) == 1:
            changed["b2"][category(keys[0])] = value
        else:
            raise ValueError
    except ValueError:
        raise ValueError(f"看不懂这个改动：{text}。写法例如 b1[h2]=-3、W1[a,h1]=2、W2[h1,0]=1、b2[1]=0.5。") from None
    return changed


def scale_output(parameters, factor):
    """把输出层的权重和偏置都乘上同一个数，返回副本。三个分数因此都乘上这个数。"""
    changed = copy.deepcopy(parameters)
    changed["W2"] = [[value * factor for value in row] for row in parameters["W2"]]
    changed["b2"] = [value * factor for value in parameters["b2"]]
    return changed


def forward(parameters, x, relu=True):
    """一行输入走完整个网络，中间的每一步都留下来。relu=False 时去掉 ReLU，隐藏值就等于隐藏层的输入。"""
    pre = [
        sum(x[i] * parameters["W1"][i][j] for i in range(INPUTS)) + parameters["b1"][j]
        for j in range(HIDDEN)
    ]
    hidden = [value if value > 0 or not relu else 0.0 for value in pre]
    scores = [
        sum(hidden[j] * parameters["W2"][j][k] for j in range(HIDDEN)) + parameters["b2"][k]
        for k in range(CLASSES)
    ]
    return {"pre": pre, "hidden": hidden, "scores": scores, "probabilities": softmax(scores)}


def predict(probabilities):
    """取概率最大的类别；并列时取编号较小的。"""
    return max(range(CLASSES), key=lambda k: (probabilities[k], -k))


def loss_from_scores(scores, label):
    """单行的损失，直接由分数算：等于正确类别概率的负自然对数。

    先减去最大的分数再取指数，分数很大时不溢出；不经过“先算概率再取对数”，概率小到 0 时也算得出来。
    """
    top = max(scores)
    return top + math.log(sum(math.exp(score - top) for score in scores)) - scores[label]


def loss(parameters, x, label):
    """单行的损失：正确类别概率的负自然对数。"""
    return loss_from_scores(forward(parameters, x)["scores"], label)


def mean_loss(parameters, rows):
    return sum(loss(parameters, row["x"], row["label"]) for row in rows) / len(rows)


def correct_count(parameters, rows):
    return sum(predict(forward(parameters, row["x"])["probabilities"]) == row["label"] for row in rows)


def kink_margin(parameters, rows):
    """各行各隐藏单元里，离 ReLU 折点（0）最近的那个距离。太近时用扰动核对梯度不可靠。"""
    return min(abs(value) for row in rows for value in forward(parameters, row["x"])["pre"])


def numeric_gradient(parameters, rows, step=1e-6):
    """拨动法：每个参数向两边各拨一点，看平均损失差多少。27 个参数要算 54 次平均损失。"""
    values = flatten(parameters)
    gradient = []
    for index in range(len(values)):
        up, down = list(values), list(values)
        up[index] += step
        down[index] -= step
        gradient.append((mean_loss(unflatten(up), rows) - mean_loss(unflatten(down), rows)) / (2 * step))
    return unflatten(gradient)


def analytic_gradient(parameters, rows):
    """沿计算路径把变化量传回去：一次前向加一次回传，就得到全部 27 个参数的梯度。"""
    total = {
        "W1": [[0.0] * HIDDEN for _ in range(INPUTS)],
        "b1": [0.0] * HIDDEN,
        "W2": [[0.0] * CLASSES for _ in range(HIDDEN)],
        "b2": [0.0] * CLASSES,
    }
    for row in rows:
        x, label = row["x"], row["label"]
        state = forward(parameters, x)
        # 损失对三个分数的变化率：概率减去正确类别的那个 1。
        d_scores = [state["probabilities"][k] - (1.0 if k == label else 0.0) for k in range(CLASSES)]
        for k in range(CLASSES):
            total["b2"][k] += d_scores[k]
            for j in range(HIDDEN):
                total["W2"][j][k] += state["hidden"][j] * d_scores[k]
        for j in range(HIDDEN):
            # 一个隐藏值通向三个分数，三条支路的贡献加起来。
            d_hidden = sum(parameters["W2"][j][k] * d_scores[k] for k in range(CLASSES))
            # ReLU：输入大于 0 时原样通过，否则变化量传不过去。
            d_pre = d_hidden if state["pre"][j] > 0 else 0.0
            total["b1"][j] += d_pre
            for i in range(INPUTS):
                total["W1"][i][j] += x[i] * d_pre
    count = len(rows)
    return unflatten([value / count for value in flatten(total)])


def update(parameters, gradient, learning_rate):
    """一次更新：所有参数都用同一个旧状态算出的梯度，一起往损失下降的方向挪一步。"""
    return unflatten([
        value - learning_rate * slope
        for value, slope in zip(flatten(parameters), flatten(gradient), strict=True)
    ])


def train(parameters, rows, learning_rate, steps, on_step=None):
    """训练循环：算梯度，更新，再算，重复 steps 次。on_step 用来在指定的步数留记录。"""
    current = copy.deepcopy(parameters)
    if on_step:
        on_step(0, current)
    for step in range(1, steps + 1):
        current = update(current, analytic_gradient(current, rows), learning_rate)
        if on_step:
            on_step(step, current)
    return current


def load_task(path):
    """读 16 行数值任务，按用途分成训练、验证、留出三组。"""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        {"id": row["id"], "x": (row["a"], row["b"]), "label": row["label"], "split": row["split"]}
        for row in data["rows"]
    ]
    return {name: [row for row in rows if row["split"] == name] for name in ("train", "validation", "heldout")}

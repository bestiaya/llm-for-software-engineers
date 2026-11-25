"""七词小语言模型的开销计数：生成的每一步做了多少次乘法，用两种做法各数一遍。只用标准库。

两种做法算出的概率相同，差别只在重复了多少计算：

- 每步重算：每生成一个词，就把整段前文从头再算一遍，前文里每个位置都重新算到概率，只用最后一个位置的结果。
  tiny_lm.generate 就是这样做的。
- 留着中间结果：每个位置的“键”和“值”只算一次，留下来；每一步只为最新的位置算查询、打分、读取、前馈和输出。

能这样省，是因为有因果遮罩：前面位置的计算不受后面的词影响，算过的不会变。
最新的位置仍然要读到前面的每一个位置，这一部分省不掉，而且前文越长越多。

这里数的是这个小模型里向量和矩阵相乘时做的乘法次数。加法、softmax 里的指数、缩放用的除法都没有数。
它不是耗时，更不是模型服务的账单。这个模型只有一层；层数多的模型里，每一层都要各留一份键和值。
"""

from __future__ import annotations

import math
import random

from shared import tiny_lm as lm


# 按计算路径的先后排列：名字，以及给读者看的说法。
PARTS = (
    ("keys_values", "算键和值"),
    ("query", "算查询"),
    ("scoring", "给看得见的位置打分"),
    ("reading", "按权重读取"),
    ("feed_forward", "前馈变换"),
    ("output", "到词表的输出"),
)
PART_NAMES = tuple(name for name, _ in PARTS)


def new_tally():
    return {name: 0 for name in PART_NAMES}


def add_tallies(first, second):
    return {name: first[name] + second[name] for name in PART_NAMES}


def counted_times(vector, matrix, tally, part):
    """一行向量乘以矩阵，同时记下做了几次乘法：向量有几个数、结果有几个数，两者相乘。"""
    tally[part] += len(vector) * len(matrix[0])
    return lm.times(vector, matrix)


def context_of(parameters):
    return len(parameters["P"])


def embed(parameters, token, position):
    """查表加位置。只有加法，不计乘法。"""
    return [value + place for value, place in zip(parameters["E"][token], parameters["P"][position], strict=True)]


def key_and_value(parameters, x, tally):
    return (
        counted_times(x, parameters["Wk"], tally, "keys_values"),
        counted_times(x, parameters["Wv"], tally, "keys_values"),
    )


def predict(parameters, x, keys, values, tally):
    """一个位置从查询算到概率。keys、values 是它看得见的各个位置的键和值，最后一项是它自己的。

    每一步的算法和先后次序与 tiny_lm.forward 相同，所以结果逐位一致。
    """
    dimension = len(x)
    query = counted_times(x, parameters["Wq"], tally, "query")
    scale = math.sqrt(dimension)
    match = []
    for key in keys:
        tally["scoring"] += dimension
        match.append(sum(query[i] * key[i] for i in range(dimension)) / scale)
    weights = lm.softmax(match)
    tally["reading"] += len(values) * dimension
    attended = [sum(weights[u] * values[u][i] for u in range(len(values))) for i in range(dimension)]
    before = [
        value + bias
        for value, bias in zip(counted_times(attended, parameters["W1"], tally, "feed_forward"), parameters["b1"], strict=True)
    ]
    hidden = [value if value > 0 else 0.0 for value in before]
    features = [
        value + bias
        for value, bias in zip(counted_times(hidden, parameters["W2"], tally, "feed_forward"), parameters["b2"], strict=True)
    ]
    scores = [
        value + bias
        for value, bias in zip(counted_times(features, parameters["U"], tally, "output"), parameters["c"], strict=True)
    ]
    return lm.softmax(scores)


def check_length(parameters, ids):
    if not 1 <= len(ids) <= context_of(parameters):
        raise ValueError(f"前文长度须在 1 到 {context_of(parameters)} 之间。")


def recompute_all(parameters, ids):
    """每步重算的一步：整段前文的每个位置都从头算到概率。返回每个位置的概率和乘法次数。"""
    check_length(parameters, ids)
    tally = new_tally()
    xs = [embed(parameters, token, position) for position, token in enumerate(ids)]
    keys, values = [], []
    for x in xs:
        key, value = key_and_value(parameters, x, tally)
        keys.append(key)
        values.append(value)
    probabilities = [
        predict(parameters, xs[t], keys[:t + 1], values[:t + 1], tally) for t in range(len(ids))
    ]
    return probabilities, tally


def new_cache():
    """留下来的中间结果：已经算过的每个位置的键和值。"""
    return {"keys": [], "values": []}


def cached_numbers(cache):
    return sum(len(row) for row in cache["keys"]) + sum(len(row) for row in cache["values"])


def extend(parameters, cache, token, tally, want_prediction=True):
    """把一个新的词接到已经算过的前文后面：只算这一个位置。

    它的键和值算一次、留下来。要预测它后面的词时，再算查询，对留着的每个键打分，按权重读留着的值。
    提示里除最后一个词以外的位置，不需要预测它后面是什么，所以只算键和值。
    """
    position = len(cache["keys"])
    if position >= context_of(parameters):
        raise ValueError(f"前文长度须在 1 到 {context_of(parameters)} 之间。")
    x = embed(parameters, token, position)
    key, value = key_and_value(parameters, x, tally)
    cache["keys"].append(key)
    cache["values"].append(value)
    if not want_prediction:
        return None
    return predict(parameters, x, cache["keys"], cache["values"], tally)


def generate_counted(parameters, prefix_ids, cached, temperature=0.0, seed=0):
    """生成循环，选词和停止的规则与 tiny_lm.generate 相同；每一步另外记下乘法次数。

    cached 为假：每步重算。cached 为真：留着中间结果。
    """
    check_length(parameters, prefix_ids)
    ids, steps, source = list(prefix_ids), [], random.Random(seed)
    cache, fed = new_cache(), 0
    while True:
        if cached:
            tally = new_tally()
            probabilities = None
            while fed < len(ids):
                probabilities = extend(parameters, cache, ids[fed], tally, want_prediction=fed == len(ids) - 1)
                fed += 1
            positions, held = 1, cached_numbers(cache)
        else:
            by_position, tally = recompute_all(parameters, ids)
            probabilities, positions, held = by_position[-1], len(ids), 0
        chosen = lm.pick(probabilities, temperature, source)
        steps.append({
            "prefix": list(ids), "probabilities": probabilities, "chosen": chosen,
            "multiplications": tally, "total": sum(tally.values()),
            "positions_predicted": positions, "cached_numbers": held,
        })
        ids.append(chosen)
        if chosen == lm.EOS_ID:
            return {"ids": ids, "steps": steps, "stopped_by": "EOS"}
        if len(ids) > context_of(parameters):
            return {"ids": ids, "steps": steps, "stopped_by": "前文已读满"}


# ---- 不运行模型，按结构直接数 ----

def position_cost(visible, dimension, hidden, vocab, with_prediction=True):
    """一个位置的乘法次数。visible 是它看得见几个位置，把它自己算在内。"""
    tally = new_tally()
    tally["keys_values"] = 2 * dimension * dimension
    if with_prediction:
        tally["query"] = dimension * dimension
        tally["scoring"] = visible * dimension
        tally["reading"] = visible * dimension
        tally["feed_forward"] = 2 * dimension * hidden
        tally["output"] = dimension * vocab
    return tally


def recompute_cost(length, dimension, hidden, vocab):
    """每步重算时，前文有 length 个位置的这一步要做多少次乘法。"""
    tally = new_tally()
    for visible in range(1, length + 1):
        tally = add_tallies(tally, position_cost(visible, dimension, hidden, vocab))
    return tally


def cached_cost(length, new_positions, dimension, hidden, vocab):
    """留着中间结果时，前文有 length 个位置、其中 new_positions 个是这一步新接上的，要做多少次乘法。"""
    tally = position_cost(length, dimension, hidden, vocab)
    for _ in range(new_positions - 1):
        tally = add_tallies(tally, position_cost(0, dimension, hidden, vocab, with_prediction=False))
    return tally


def projection(first, last, dimension, hidden, vocab):
    """从 first 个位置的前文开始，一步接一个词，直到前文有 last 个位置为止，两种做法各做多少次乘法。

    按结构数出来的，没有运行模型。
    """
    if not 1 <= first <= last:
        raise ValueError("位置数要满足 1 ≤ 起点 ≤ 终点。")
    recompute = sum(sum(recompute_cost(length, dimension, hidden, vocab).values()) for length in range(first, last + 1))
    cached = sum(cached_cost(first, first, dimension, hidden, vocab).values()) + sum(
        sum(cached_cost(length, 1, dimension, hidden, vocab).values()) for length in range(first + 1, last + 1)
    )
    return {
        "first": first, "last": last, "steps": last - first + 1,
        "recompute_total": recompute, "cached_total": cached,
        "recompute_last_step": sum(recompute_cost(last, dimension, hidden, vocab).values()),
        "cached_last_step": sum(cached_cost(last, first if last == first else 1, dimension, hidden, vocab).values()),
        "cached_numbers_at_end": last * 2 * dimension,
    }

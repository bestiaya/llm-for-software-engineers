"""Finite numbers, shapes and the shared affine forward calculation."""

from __future__ import annotations

import hashlib
import math
import struct


def finite_number(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} 需为有限数值。")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} 超出了 Python float 的范围。") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} 需为有限数值。")
    return number


def vector(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} 需为非空向量。")
    return [finite_number(value, f"{name}[{index}]") for index, value in enumerate(values)]


def matrix(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} 需为非空矩阵。")
    rows = [vector(row, f"{name}[{index}]") for index, row in enumerate(values)]
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} 各行长度需要相同。")
    return rows


def parameters(weights, bias):
    weights = matrix(weights, "W")
    bias = vector(bias, "b")
    if len(weights[0]) != len(bias):
        raise ValueError(f"W 的列数 {len(weights[0])} 需与 b 的长度 {len(bias)} 相同。")
    return weights, bias


def dot_products(values, weights):
    """Return the contributions from two equally sized, validated vectors."""
    return [finite_number(value * weight, "逐项乘积") for value, weight in zip(values, weights, strict=True)]


def linear_forward(inputs, weights, bias):
    """Apply XW + b, using one input per row and one candidate per column."""
    inputs = matrix(inputs, "X")
    weights, bias = parameters(weights, bias)
    if len(inputs[0]) != len(weights):
        raise ValueError(f"X 的列数 {len(inputs[0])} 需与 W 的行数 {len(weights)} 相同。")

    columns = list(zip(*weights, strict=True))
    results = []
    for row in inputs:
        scores = []
        for column, offset in zip(columns, bias, strict=True):
            score = sum(dot_products(row, column)) + offset
            scores.append(finite_number(score, "输出分数"))
        results.append(scores)
    return results


def parameter_values(weights, bias):
    weights, bias = parameters(weights, bias)
    return [value for row in weights for value in row] + bias


def parameter_fingerprint(weights, bias):
    values = parameter_values(weights, bias)
    try:
        payload = struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as error:
        raise ValueError("参数超出了 float32 的表示范围。") from error
    return hashlib.sha256(payload).hexdigest()

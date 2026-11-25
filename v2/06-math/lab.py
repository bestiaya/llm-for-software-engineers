"""一次算一批：把几行输入排成矩阵，一次算出每一行、每个类别的分数，看清形状和每个轴的含义。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))

from shared.numeric import dot_products, linear_forward, parameter_fingerprint, parameters  # noqa: E402


MODEL_FILE = COURSE / "data" / "linear-model.json"
TASK_FILE = COURSE / "data" / "counter-task.json"


def load_parameters():
    model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    if not isinstance(model, dict) or type(model.get("schema_version")) is not int or model["schema_version"] != 1:
        raise ValueError("模型参数需使用 schema_version=1 的 JSON 对象。")
    weights, bias = parameters(model["weights"], model["bias"])
    if len(weights) != 2 or len(bias) != 3:
        raise ValueError("本章的模型有两个输入和三个候选分数。")
    return model, weights, bias


def choose(scores):
    """取分数最大的类别；并列时取编号较小的。"""
    return max(range(len(scores)), key=lambda k: (scores[k], -k))


def describe_dot_products(row, weights, bias, features, candidates):
    explained = []
    for candidate, column, offset in zip(candidates, zip(*weights, strict=True), bias, strict=True):
        products = dot_products(row, column)
        terms = []
        for feature, value, weight, product in zip(features, row, column, products, strict=True):
            terms.append({"feature": feature, "input": value, "weight": weight, "product": product})
        dot = sum(products)
        explained.append({"candidate": candidate, "terms": terms, "dot": dot, "bias": offset, "score": dot + offset})
    return explained


def shape(matrix):
    return [len(matrix), len(matrix[0])]


def shape_fault(weights, bias):
    inputs = [[2, 1, 0], [0, 1, 0]]
    injected_fault = {"kind": "shape", "X": inputs, "X_shape": shape(inputs), "W_shape": shape(weights)}
    try:
        linear_forward(inputs, weights, bias)
    except ValueError as error:
        return {"mode": "shape", "injected_fault": injected_fault, "shape_valid": False, "error": str(error), "exit_code": 1}
    raise AssertionError("形状故障未触发预期的校验。")


def run_all(weights, bias):
    """把 16 行一起算：16 行 2 列的输入乘 2 行 3 列的权重，得到 16 行 3 列的分数。"""
    rows = json.loads(TASK_FILE.read_text(encoding="utf-8"))["rows"]
    inputs = [[row["a"], row["b"]] for row in rows]
    output = linear_forward(inputs, weights, bias)
    table = [
        {"id": row["id"], "input": x, "scores": scores, "prediction": choose(scores), "label": row["label"]}
        for row, x, scores in zip(rows, inputs, output, strict=True)
    ]
    return {
        "mode": "all",
        "X_shape": shape(inputs), "W_shape": shape(weights), "bias_shape": [len(bias)], "output_shape": shape(output),
        "rows": table,
        "correct": sum(item["prediction"] == item["label"] for item in table),
        "exit_code": 0,
    }


def run(fault, everything=False):
    model, weights, bias = load_parameters()
    if fault == "shape":
        return shape_fault(weights, bias)
    if everything:
        return run_all(weights, bias)

    inputs = [[2, 1], [0, 1]]
    output = linear_forward(inputs, weights, bias)
    axes = {
        "X": ["sample", "feature"], "W": ["feature", "candidate"],
        "bias": ["candidate"], "output": ["sample", "candidate"],
    }
    swapped = [1, 2]
    transposed = [list(row) for row in zip(*inputs, strict=True)]
    transposed_result = {
        "X": transposed,
        "X_shape": shape(transposed),
        "actual_axis_roles": ["feature", "sample"],
        "expected_axis_roles": axes["X"],
        "output": linear_forward(transposed, weights, bias),
        "shape_valid": True,
        "manual_review_required": True,
    }
    injected = None
    if fault == "axis":
        injected = {"kind": "axis", **transposed_result}
    return {
        "mode": "axis" if fault == "axis" else "batch",
        "parameter_fingerprint": parameter_fingerprint(weights, bias),
        "features": model["input_fields"],
        "candidates": model["output_classes"],
        "axis_roles": axes,
        "dot_products": describe_dot_products(inputs[0], weights, bias, model["input_fields"], model["output_classes"]),
        "batch": {
            "X": inputs, "W": weights, "bias": bias, "output": output,
            "predictions": [choose(scores) for scores in output],
            "X_shape": shape(inputs), "W_shape": shape(weights), "bias_shape": [len(bias)], "output_shape": shape(output),
        },
        "swapped_input": {"input": swapped, "output": linear_forward([swapped], weights, bias)[0]},
        "transposed_input": transposed_result,
        "injected_fault": injected,
        "exit_code": 0,
    }


def numbers(values):
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def print_run(result):
    mode = result["mode"]
    if mode == "shape":
        fault = result["injected_fault"]
        print(f"这次故意给每行输入多加了一个数：X 的形状是 {fault['X_shape']}，W 的形状是 {fault['W_shape']}。")
        print(f"形状检查没有通过，计算没有执行：{result['error']}")
    elif mode == "all":
        print(f"16 行一起算：X 的形状 {result['X_shape']}，W 的形状 {result['W_shape']}，bias 的形状 {result['bias_shape']}，结果的形状 {result['output_shape']}。")
        print("编号     输入    三个分数           选  标签")
        for item in result["rows"]:
            verdict = "对" if item["prediction"] == item["label"] else "错"
            print(f"{item['id']:7}  {numbers(item['input']):7} {numbers(item['scores']):18} {item['prediction']}   {item['label']}   {verdict}")
        print(f"这组手设的参数在 16 行里选对 {result['correct']} 行。参数没有经过训练。")
    else:
        batch = result["batch"]
        print("参数是手工设定的，没有经过训练。")
        print(f"先看一行：输入 {numbers(batch['X'][0])} 与每个类别的那一列权重做点积，再加偏置")
        for candidate in result["dot_products"]:
            products = " + ".join(
                f"{term['input']:g}×{term['weight']:g}" if term["weight"] >= 0 else f"{term['input']:g}×({term['weight']:g})"
                for term in candidate["terms"]
            )
            print(f"  类别 {candidate['candidate']}：{products} = {candidate['dot']:g}，加偏置 {candidate['bias']:g}，得 {candidate['score']:g}")
        print("\n两行一起算：X·W + bias（偏置写作 bias，免得和输入 b 重名）")
        print(f"  X 的形状 {batch['X_shape']}：行是样本，列是特征（a、b）")
        for index, row in enumerate(batch["X"], start=1):
            print(f"     样本 {index}  {numbers(row)}")
        print(f"  W 的形状 {batch['W_shape']}：行是特征（a、b），列是类别（0、1、2）")
        for name, row in zip(result["features"], batch["W"], strict=True):
            print(f"     {name}  {numbers(row)}")
        print(f"  bias 的形状 {batch['bias_shape']}：每个类别一个　{numbers(batch['bias'])}")
        print(f"  结果的形状 {batch['output_shape']}：行是样本，列是类别")
        for index, (row, prediction) in enumerate(zip(batch["output"], batch["predictions"], strict=True), start=1):
            print(f"     样本 {index}  {numbers(row)}　选类别 {prediction}")
        if mode == "axis":
            wrong, swapped = result["transposed_input"], result["swapped_input"]
            print("\n这次故意把 X 的行和列对调了：")
            print(f"  对调后的 X = {wrong['X']}，形状仍然是 {wrong['X_shape']}，程序没有报错。")
            print("  算出的结果 = [" + ", ".join(numbers(row) for row in wrong["output"]) + "]，与上面的结果不同。")
            print("  对调以后，行成了特征、列成了样本，乘法照样能做，算出来的数没有意义。")
            print(f"另一种同类的错：把一行里的两个数对调成 {numbers(swapped['input'])}，按原来“先 a 后 b”的顺序读，得到 {numbers(swapped['output'])}。")
            print("形状对了，只说明乘法能做；每个轴、每个位置代表什么，要另外核对。")
    print(f"exit_code: {result['exit_code']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fault", choices=("shape", "axis"), help="shape：形状对不上；axis：形状对得上，含义错了")
    mode.add_argument("--all", action="store_true", help="把 16 行一起算")
    parser.add_argument("--json", action="store_true", help="输出结构化结果")
    args = parser.parse_args(argv)
    try:
        result = run(args.fault, args.all)
    except (OSError, ValueError, KeyError) as error:
        result = {"error": f"无法读取或使用模型参数（{error}）", "exit_code": 2}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["exit_code"] == 2:
        print(result["error"], file=sys.stderr)
    else:
        print_run(result)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

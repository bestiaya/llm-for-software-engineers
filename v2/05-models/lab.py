"""模型就是参数加一套算法：读出九个参数，把一行输入算成三个分数，取最大的得到类别。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys


COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))

from shared.numeric import finite_number, linear_forward, parameter_fingerprint, parameters, parameter_values  # noqa: E402


MODEL_FILE = COURSE / "data" / "linear-model.json"
BINARY_FILE = COURSE / "data" / "linear-parameters.f32"
INPUT_NAMES = ("a", "b")
THRESHOLD = 2


def load_model():
    """从 JSON 读出参数，并核对随课的二进制文件里存的是同样的九个数。"""
    model_bytes = MODEL_FILE.read_bytes()
    model = json.loads(model_bytes)
    if not isinstance(model, dict) or type(model.get("schema_version")) is not int or model["schema_version"] != 1:
        raise ValueError("模型参数需使用 schema_version=1 的 JSON 对象。")
    weights, bias = parameters(model["weights"], model["bias"])
    if len(weights) != 2 or len(bias) != 3:
        raise ValueError("本章的模型有两个输入和三个候选分数。")

    expected_values = parameter_values(weights, bias)
    binary_payload = BINARY_FILE.read_bytes()
    binary_format = struct.Struct(f"<{len(expected_values)}f")
    if len(binary_payload) != binary_format.size:
        raise ValueError(f"float32 参数载荷需为 {binary_format.size} 字节。")
    if list(binary_format.unpack(binary_payload)) != expected_values:
        raise ValueError("float32 文件与 JSON 中的参数值不一致。")
    return model, weights, bias, len(model_bytes), binary_payload


def set_parameter(weights, bias, text):
    """按“名字=数值”改一个参数，返回副本。写法：W[a,0] 是输入 a 通向类别 0 的权重，bias[1] 是类别 1 的偏置。

    输入里已经有一个 b，所以偏置写作 bias。
    """
    name, separator, raw = text.partition("=")
    try:
        value = finite_number(float(raw), "数值")
        if not separator:
            raise ValueError
        block, _, inside = name.strip().rstrip("]").partition("[")
        keys = [key.strip() for key in inside.split(",")]
        weights, bias = [row[:] for row in weights], bias[:]
        if block == "W" and len(keys) == 2:
            weights[INPUT_NAMES.index(keys[0])][("0", "1", "2").index(keys[1])] = value
        elif block == "bias" and len(keys) == 1:
            bias[("0", "1", "2").index(keys[0])] = value
        else:
            raise ValueError
    except ValueError:
        raise ValueError(f"看不懂这个改动：{text}。写法例如 W[a,0]=1 或 bias[1]=0.5。") from None
    return weights, bias


def choose(scores):
    """取分数最大的类别；并列时取编号较小的。"""
    return max(range(len(scores)), key=lambda k: (scores[k], -k))


def forward_detail(x, weights, bias):
    """一行输入的前向计算，把每个类别的乘加过程留下来。"""
    scores = linear_forward([list(x)], weights, bias)[0]
    terms = [
        {
            "class": k,
            "products": [{"input": name, "value": x[i], "weight": weights[i][k], "product": x[i] * weights[i][k]} for i, name in enumerate(INPUT_NAMES)],
            "bias": bias[k],
            "score": scores[k],
        }
        for k in range(len(bias))
    ]
    return {"input": list(x), "terms": terms, "scores": scores, "prediction": choose(scores)}


def run_forward(x, changes):
    model, weights, bias, _, _ = load_model()
    original = (weights, bias)
    for change in changes:
        weights, bias = set_parameter(weights, bias, change)
    detail = forward_detail(x, weights, bias)
    return {
        "mode": "forward",
        "parameter_origin": model["origin"],
        "training_executed": False,
        "weights": weights, "bias": bias,
        "changes": changes,
        "parameters_changed": (weights, bias) != original,
        **detail,
        "label_by_rule": int(x[0] >= THRESHOLD) + int(x[1] >= THRESHOLD),
        "exit_code": 0,
    }


def run_storage():
    """九个参数占多少存储；同样的 36 字节，按不同的约定读出来是不同的模型。"""
    _, weights, bias, json_bytes, payload = load_model()
    values = list(struct.unpack("<9f", payload))
    x = [2, 1]
    # 正确的约定：先是 2 行 3 列的权重，按行展开；再是 3 个偏置。
    right = {"weights": [values[0:3], values[3:6]], "bias": values[6:9]}
    # 错误的约定之一：把前 6 个数当成按列展开的权重。
    by_column = {"weights": [values[0:6:2], values[1:6:2]], "bias": values[6:9]}
    # 错误的约定之二：以为偏置存在最前面。
    bias_first = {"weights": [values[3:6], values[6:9]], "bias": values[0:3]}
    readings = []
    for name, layout in (("约定的读法：先 2 行 3 列的权重按行展开，再 3 个偏置", right),
                         ("读错之一：把前 6 个数当成按列展开的权重", by_column),
                         ("读错之二：以为 3 个偏置存在最前面", bias_first)):
        scores = linear_forward([x], layout["weights"], layout["bias"])[0]
        readings.append({"layout": name, **layout, "scores": scores, "prediction": choose(scores)})
    return {
        "mode": "storage",
        "parameter_count": len(values),
        "bytes_per_parameter": 4,
        "binary_bytes": len(payload),
        "json_bytes": json_bytes,
        "values_in_file_order": values,
        "input": x,
        "readings": readings,
        "matches_json": readings[0]["weights"] == weights and readings[0]["bias"] == bias,
        "exit_code": 0,
    }


def run_fingerprint():
    """选读：给参数取一个指纹；同样的字节按另一种字节序读出来是什么。"""
    _, weights, bias, _, payload = load_model()
    edited_weights, edited_bias = set_parameter(weights, bias, "W[a,0]=1")
    return {
        "mode": "fingerprint",
        "fingerprint": parameter_fingerprint(weights, bias),
        "fingerprint_after_one_change": parameter_fingerprint(edited_weights, edited_bias),
        "change": "W[a,0]=1",
        "little_endian": list(struct.unpack("<9f", payload)),
        "big_endian": list(struct.unpack(">9f", payload)),
        "first_bytes": payload[:4].hex(" "),
        "exit_code": 0,
    }


def number(value):
    return f"{value:g}"


def print_run(result):
    if result["mode"] == "forward":
        print("参数是手工设定的，没有经过训练。这次只做前向计算。")
        if result["changes"]:
            print("这次在内存里改了参数：" + "，".join(result["changes"]) + "。文件没有改。")
        print("九个参数（W 的行是输入 a、b，列是类别 0、1、2）")
        for name, row in zip(INPUT_NAMES, result["weights"], strict=True):
            print(f"  W[{name}] = [" + ", ".join(number(value) for value in row) + "]")
        print("  bias = [" + ", ".join(number(value) for value in result["bias"]) + "]")
        print(f"输入：a={number(result['input'][0])}，b={number(result['input'][1])}")
        for term in result["terms"]:
            parts = " + ".join(f"{number(p['value'])}×{number(p['weight'])}" if p["weight"] >= 0 else f"{number(p['value'])}×({number(p['weight'])})" for p in term["products"])
            offset = f" + {number(term['bias'])}" if term["bias"] >= 0 else f" − {number(-term['bias'])}"
            print(f"  类别 {term['class']} 的分数 = {parts}{offset} = {number(term['score'])}")
        print(f"三个分数里最大的是类别 {result['prediction']}，模型选类别 {result['prediction']}。")
        print(f"按标注规则，这一行的标签是 {result['label_by_rule']}。")
    elif result["mode"] == "storage":
        print(f"{result['parameter_count']} 个参数，每个按 {result['bytes_per_parameter']} 字节的单精度数保存，一共 {result['binary_bytes']} 字节。")
        print(f"同样的参数连同说明写成 JSON 文本，是 {result['json_bytes']} 字节。")
        print("文件里依次存的九个数：[" + ", ".join(number(value) for value in result["values_in_file_order"]) + "]")
        print(f"同样的 36 字节，按三种约定读出来，各算一次输入 a={result['input'][0]}、b={result['input'][1]}：")
        for reading in result["readings"]:
            print(f"  {reading['layout']}")
            print(f"    W = {[[float(number(v)) for v in row] for row in reading['weights']]}，bias = {[float(number(v)) for v in reading['bias']]}")
            print("    分数 [" + ", ".join(number(value) for value in reading["scores"]) + f"]，选类别 {reading['prediction']}")
        print("文件里只有数。哪个数是哪个参数，要靠另外说好的结构约定。")
    else:
        print(f"参数指纹（按存盘的字节算出的 SHA-256）：{result['fingerprint']}")
        print(f"只改一个参数（{result['change']}）以后的指纹：{result['fingerprint_after_one_change']}")
        print(f"文件开头的 4 个字节：{result['first_bytes']}")
        print("按小端读出的九个数：[" + ", ".join(number(value) for value in result["little_endian"]) + "]")
        print("按大端读出的九个数：[" + ", ".join(f"{value:.3g}" for value in result["big_endian"]) + "]")
    print(f"exit_code: {result['exit_code']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--storage", action="store_true", help="九个参数占多少存储，以及读的时候为什么需要结构约定")
    mode.add_argument("--fingerprint", action="store_true", help="选读：参数指纹和字节序")
    parser.add_argument("--a", type=float, default=2, help="输入 a，默认 2")
    parser.add_argument("--b", type=float, default=1, help="输入 b，默认 1")
    parser.add_argument("--set", action="append", default=[], metavar="名字=数值", help="在内存里改一个参数，例如 W[a,0]=1 或 bias[1]=0.5；可以写多次")
    parser.add_argument("--json", action="store_true", help="输出结构化结果")
    args = parser.parse_args(argv)
    try:
        if args.storage:
            result = run_storage()
        elif args.fingerprint:
            result = run_fingerprint()
        else:
            result = run_forward([args.a, args.b], args.set)
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

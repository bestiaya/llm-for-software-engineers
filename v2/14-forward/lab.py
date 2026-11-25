"""七词小模型（原型）：沿完整的预测路径走到词表上的概率，再生成，并核对训练记录。只用标准库。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


CHAPTER = Path(__file__).resolve().parent
COURSE = CHAPTER.parent
sys.path.insert(0, str(COURSE))

from shared import tiny_lm as lm  # noqa: E402


RECORD_FILE = COURSE / "data/tiny-lm-training-record.json"
CONFIG = {
    "seed": 3,
    "dimension": 2,
    "hidden": 4,
    "initial_scale": 0.5,
    "learning_rate": 0.05,
    "steps": 20000,
    "checkpoints": [0, 1, 10, 100, 500, 1000, 1200, 1300, 1500, 2000, 3000, 5000, 10000, 20000],
    "output_steps": [0, 1000, 20000],
    "method": "每一步都用两句话的全部 8 个位置算梯度，所有参数一起更新",
    "seed_note": (
        "种子 1 到 20 各训练过一遍：14 个的损失贴近下限；2 个停在 0.52，每步取概率最大的仍能把两句话接对；"
        "4 个没有接对。记录用的是损失贴近下限的种子里编号最小的种子 3；种子 1、2 没有接对。"
    ),
}
PREFIXES = ["请求", "请求 成功 后", "请求 失败 后"]
COMPLETIONS = {"请求 成功": "请求 成功 后 返回 EOS", "请求 失败": "请求 失败 后 重试 EOS"}
# 第一次更新时拿出来细看的四个参数：两个紧挨着输出，一个在注意力里，一个是词向量。
WATCHED = [
    ("c", (0,), "输出偏置：请求"), ("c", (3,), "输出偏置：后"),
    ("Wv", (1, 1), "注意力里“值”的映射"), ("E", (0, 0), "“请求”的词向量"),
]


class LabError(Exception):
    """输入或记录有误，无法完成实验。"""


def start_parameters(seed=None):
    return lm.initial_parameters(
        CONFIG["seed"] if seed is None else seed, CONFIG["dimension"], CONFIG["hidden"], CONFIG["initial_scale"]
    )


def value_at(parameters, name, index):
    block = parameters[name]
    for position in index:
        block = block[position]
    return block


def label_of(name, index):
    return name + "".join(f"[{position}]" for position in index)


def top(probabilities, count=3):
    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))[:count]
    return [{"token": lm.VOCAB[index], "probability": probabilities[index]} for index in order]


def outputs_for(parameters):
    return [
        {"prefix": prefix, "next": top(lm.forward(parameters, lm.encode(prefix))["probabilities"][-1])}
        for prefix in PREFIXES
    ]


def completions_for(parameters):
    """从“请求 成功”和“请求 失败”出发，每一步都取概率最大的，看能不能把两句话接对。"""
    rows = []
    for prefix, wanted in COMPLETIONS.items():
        text = lm.decode(lm.generate(parameters, lm.encode(prefix))["ids"])
        rows.append({"prefix": prefix, "generated": text, "matches": text == wanted})
    return rows


def unigram_level(pairs):
    """不看前文、只按每个词出现的多少来猜，损失会是多少。"""
    targets = [target for pair in pairs for target in pair["targets"]]
    return -sum(math.log(targets.count(target) / len(targets)) for target in targets) / len(targets)


def first_update(start, pairs):
    by_program = lm.analytic_gradient(start, pairs)
    by_nudging = lm.numeric_gradient(start, pairs)
    after_one = lm.update(start, by_program, CONFIG["learning_rate"])
    return [
        {
            "parameter": label_of(name, index), "meaning": meaning, "before": value_at(start, name, index),
            "gradient_by_program": value_at(by_program, name, index),
            "gradient_by_nudging": value_at(by_nudging, name, index), "after": value_at(after_one, name, index),
        }
        for name, index, meaning in WATCHED
    ]


def run_training(start, pairs):
    checkpoints, outputs = [], []

    def note(step, parameters):
        if step in CONFIG["checkpoints"]:
            checkpoints.append({"step": step, "loss": lm.mean_loss(parameters, pairs)})
        if step in CONFIG["output_steps"]:
            outputs.append({"step": step, "by_prefix": outputs_for(parameters)})

    final = lm.train(start, pairs, CONFIG["learning_rate"], CONFIG["steps"], note)
    return final, checkpoints, outputs


def build_record():
    pairs = lm.training_pairs()
    start = start_parameters()
    final, checkpoints, outputs = run_training(start, pairs)
    return {
        "schema_version": 1,
        "data_kind": "training_run_record",
        "config": CONFIG,
        "vocabulary": lm.VOCAB,
        "pairs": pairs,
        "parameter_count": lm.parameter_count(CONFIG["dimension"], CONFIG["hidden"]),
        "loss_floor": lm.loss_floor(pairs),
        "unigram_level": unigram_level(pairs),
        "initial_parameters": start,
        "first_update": first_update(start, pairs),
        "checkpoints": checkpoints,
        "outputs": outputs,
        "final_parameters": final,
        "completions": completions_for(final),
    }


def load_record():
    if not RECORD_FILE.exists():
        raise LabError("还没有训练记录。课程维护者先运行 --write-record。")
    return json.loads(RECORD_FILE.read_text(encoding="utf-8"))


def check_record(record, retrain=False):
    """核对记录自己对得上。默认只重算起点、第一次更新、前 100 步和最后的参数；加 --retrain 才把两万步重新训练一遍。"""
    pairs = lm.training_pairs()
    start = start_parameters()
    problems = []
    if record["config"] != CONFIG:
        problems.append("记录里的设置与当前程序不一致")
    if record["initial_parameters"] != start or record["pairs"] != pairs:
        problems.append("起点或数据与当前程序不一致")
    for saved, fresh in zip(record["first_update"], first_update(start, pairs), strict=True):
        if abs(saved["after"] - fresh["after"]) > 1e-12:
            problems.append(f"{saved['parameter']} 的第一次更新对不上")
        if abs(saved["gradient_by_program"] - saved["gradient_by_nudging"]) > 1e-6:
            problems.append(f"{saved['parameter']} 两种算法的梯度不一致")
    saved_loss = {item["step"]: item["loss"] for item in record["checkpoints"]}
    early = lm.train(start, pairs, CONFIG["learning_rate"], 100)
    if abs(lm.mean_loss(early, pairs) - saved_loss[100]) > 1e-9:
        problems.append("重新训练前 100 步，损失与记录不一致")
    final = record["final_parameters"]
    if abs(lm.mean_loss(final, pairs) - saved_loss[CONFIG["steps"]]) > 1e-9:
        problems.append("用记录里最后的参数重算，损失与记录不一致")
    if record["completions"] != completions_for(final) or not all(row["matches"] for row in record["completions"]):
        problems.append("用记录里最后的参数生成，两句话没有都接对")
    largest = None
    if retrain:
        _, checkpoints, _ = run_training(start, pairs)
        largest = max(abs(item["loss"] - saved_loss[item["step"]]) for item in checkpoints)
        if largest > 1e-6:
            problems.append(f"重新训练全部步数，检查点的损失与记录最多相差 {largest:.2e}")
    return problems, largest


def trace(parameters, prefix):
    ids = lm.encode(prefix)
    state = lm.forward(parameters, ids)
    last = len(ids) - 1
    return {
        "mode": "trace", "prefix": prefix, "ids": ids,
        "shapes": {
            "编号": [len(ids)], "查表加位置": [len(ids), CONFIG["dimension"]], "注意力权重": [last + 1],
            "读到的向量": [CONFIG["dimension"]], "前馈的隐藏值": [CONFIG["hidden"]],
            "前馈的输出": [CONFIG["dimension"]], "分数": [len(lm.VOCAB)], "概率": [len(lm.VOCAB)],
        },
        "token_vectors": [parameters["E"][token] for token in ids],
        "position_vectors": parameters["P"][:len(ids)],
        "x": state["x"], "matches": state["matches"][last], "weights": state["weights"][last],
        "attended": state["attended"][last], "hidden": state["hidden"][last], "features": state["features"][last],
        "scores": state["scores"][last], "probabilities": state["probabilities"][last],
        "prediction": lm.VOCAB[lm.pick(state["probabilities"][last])],
        "exit_code": 0,
    }


def mask_check(parameters):
    """把后面的一个词换掉，前面各位置的预测不应当变。"""
    first = lm.forward(parameters, lm.encode("请求 成功 后 返回"))["probabilities"]
    second = lm.forward(parameters, lm.encode("请求 成功 后 重试"))["probabilities"]
    same = [first[t] == second[t] for t in range(lm.CONTEXT)]
    return {
        "mode": "mask", "first": "请求 成功 后 返回", "second": "请求 成功 后 重试", "same_by_position": same,
        "holds": same == [True, True, True, False], "exit_code": 0 if same == [True, True, True, False] else 1,
    }


def order_check(parameters):
    """把前面两个词对调：有位置向量时预测会变；把位置向量全换成 0，预测就一样了。"""
    first, second = "请求 成功 后", "成功 请求 后"
    without = dict(parameters, P=[[0.0] * CONFIG["dimension"] for _ in range(lm.CONTEXT)])
    rows = []
    for name, chosen in (("有位置向量", parameters), ("位置向量全为 0", without)):
        one = lm.forward(chosen, lm.encode(first))["probabilities"][-1]
        two = lm.forward(chosen, lm.encode(second))["probabilities"][-1]
        rows.append({
            "setting": name, "first": top(one, 2), "second": top(two, 2),
            "largest_difference": max(abs(a - b) for a, b in zip(one, two, strict=True)),
        })
    holds = rows[0]["largest_difference"] > 1e-6 and rows[1]["largest_difference"] < 1e-12
    return {"mode": "order", "first": first, "second": second, "rows": rows, "holds": holds, "exit_code": 0 if holds else 1}


def show_block(parameters, name):
    if name not in parameters:
        raise LabError(f"没有叫 {name} 的参数。可以看的有：{'、'.join(lm.SHAPES_BY_NAME)}")
    block = parameters[name]
    rows = block if isinstance(block[0], list) else [block]
    shape = list(lm.shapes(CONFIG["dimension"], CONFIG["hidden"])[name])
    return {"mode": "show", "name": name, "shape": shape, "rows": rows, "exit_code": 0}


def hand_example():
    """第 12、13 章的手算例：只看最后一个词分不出两句前文；把前文加起来，或者按内容加权去读，就能分开。"""
    parameters = lm.hand_set_parameters()
    rows = []
    for prefix in ("请求 成功 后", "请求 失败 后"):
        ids = lm.encode(prefix)
        vectors = [lm.HAND_EMBEDDING[token] for token in ids]
        state = lm.forward(parameters, ids)
        rows.append({
            "prefix": prefix, "vectors": vectors, "last_only": vectors[-1],
            "summed": [sum(column) for column in zip(*vectors)],
            "matches": state["matches"][-1], "weights": state["weights"][-1], "attended": state["attended"][-1],
        })
    return {
        "mode": "hand", "rows": rows,
        "last_only_same": rows[0]["last_only"] == rows[1]["last_only"],
        "summed_differ": rows[0]["summed"] != rows[1]["summed"],
        "attended_differ": rows[0]["attended"] != rows[1]["attended"], "exit_code": 0,
    }


def count_parameters(dimension=None, hidden=None):
    dimension = CONFIG["dimension"] if dimension is None else dimension
    hidden = CONFIG["hidden"] if hidden is None else hidden
    if dimension < 1 or hidden < 1:
        raise LabError("向量的维数和隐藏单元的个数都要大于 0。")
    table = [
        {"name": name, "shape": list(shape), "count": math.prod(shape)}
        for name, shape in lm.shapes(dimension, hidden).items()
    ]
    total = sum(item["count"] for item in table)
    return {
        "mode": "count", "dimension": dimension, "hidden": hidden, "table": table, "total": total,
        "bytes_as_float32": total * 4, "exit_code": 0,
    }


def seed_survey(count):
    """换起点各训练一遍：有多少个起点训练后能把两句话都接对。每个起点要训练两万步。"""
    pairs = lm.training_pairs()
    rows = []
    for seed in range(1, count + 1):
        final = lm.train(start_parameters(seed), pairs, CONFIG["learning_rate"], CONFIG["steps"])
        rows.append({
            "seed": seed, "loss": lm.mean_loss(final, pairs),
            "completes_both": all(row["matches"] for row in completions_for(final)),
        })
    floor = lm.loss_floor(pairs)
    return {
        "mode": "seeds", "rows": rows, "succeeded": sum(row["completes_both"] for row in rows),
        "near_floor": sum(row["loss"] - floor < 0.01 for row in rows),
        "loss_floor": floor, "unigram_level": unigram_level(pairs), "exit_code": 0,
    }


def numbers(values, places=3):
    return "[" + ", ".join(f"{value:.{places}f}" for value in values) + "]"


def leading(items, count=2):
    return "，".join(f"{item['token']} {item['probability']:.3f}" for item in items[:count])


def print_run(result):
    mode = result["mode"]
    if mode == "trace":
        print(f"前文：{result['prefix']}　用的是训练得到的参数。下面只展示最后一个位置的预测。")
        print(f"① 编号　{result['ids']}")
        print("② 查表　每个编号取一行词向量，再加上所在位置的向量")
        for token, word, place, total in zip(result["ids"], result["token_vectors"], result["position_vectors"], result["x"], strict=True):
            print(f"     {lm.VOCAB[token]}　词向量 {numbers(word)} + 位置向量 {numbers(place)} = {numbers(total)}")
        print(f"③ 注意力　最后一个位置给前面各位置打的分 {numbers(result['matches'])}")
        print(f"　　　　　归一成权重 {numbers(result['weights'])}，按权重读到的向量 {numbers(result['attended'])}")
        print(f"④ 前馈　隐藏值 {numbers(result['hidden'])}，输出 {numbers(result['features'])}")
        print(f"⑤ 输出　七个候选的分数 {numbers(result['scores'], 2)}")
        print("⑥ 概率　" + "，".join(f"{token} {share:.3f}" for token, share in zip(lm.VOCAB, result["probabilities"], strict=True)))
        print(f"概率最大的是：{result['prediction']}")
        print("各步的形状：" + "，".join(f"{name} {shape}" for name, shape in result["shapes"].items()))
        print("显示的数都四舍五入过；拿显示的数复算，末位可能差 1。")
    elif mode == "generate":
        print(f"从前文“{result['prefix']}”开始，温度 {result['temperature']}。")
        for step in result["steps"]:
            print(f"  前文 [{step['prefix']}] → {leading(step['top'])} → 选 {step['chosen']}")
        print(f"结果：{result['text']}　停止的原因：{result['stopped_by']}")
    elif mode == "mask":
        print(f"两段前文只有最后一个词不同：“{result['first']}”和“{result['second']}”。")
        print("各位置的预测是否相同：" + "，".join(f"位置 {t} {'相同' if same else '不同'}" for t, same in enumerate(result["same_by_position"])))
        print("因果遮罩" + ("成立：前三个位置看不到最后一个词。" if result["holds"] else "不成立，需要检查。"))
    elif mode == "hand":
        print("手算用的二维向量，不是训练出来的。位置向量全为 0，查询、键、值都原样通过。")
        for row in result["rows"]:
            print(f"前文 [{row['prefix']}]　三个词的向量 {row['vectors']}")
            print(f"  只看最后一个词 {row['last_only']}　把前文的向量加起来 {row['summed']}")
            print(f"  按内容去读：打分 {numbers(row['matches'])}，权重 {numbers(row['weights'])}，读到 {numbers(row['attended'])}")
        print("只看最后一个词，两句前文" + ("得到同一个向量，分不出来。" if result["last_only_same"] else "不同。"))
        print("加起来以后" + ("两句不同，可以区分。" if result["summed_differ"] else "仍然相同。"))
        print("按内容加权去读" + ("两句也不同，可以区分。" if result["attended_differ"] else "仍然相同。"))
    elif mode == "order":
        print(f"两段前文只是前两个词对调：“{result['first']}”和“{result['second']}”。后一段不是训练过的句子，只用来做对照。")
        for row in result["rows"]:
            print(
                f"  {row['setting']}　前一段：{leading(row['first'])}　后一段：{leading(row['second'])}　"
                f"概率最多相差 {row['largest_difference']:.3f}"
            )
        print("去掉位置向量以后，" + ("两段前文的预测完全一样：顺序是靠位置向量进来的。" if result["holds"] else "结果与预期不符，需要检查。"))
    elif mode == "show":
        print(f"{result['name']}　形状 {result['shape']}　训练得到的值：")
        for row in result["rows"]:
            print("  " + numbers(row))
    elif mode == "count":
        print(
            f"向量 {result['dimension']} 维，前馈的隐藏单元 {result['hidden']} 个，"
            f"词表 {len(lm.VOCAB)} 个词，最多读 {lm.CONTEXT} 个位置。"
        )
        for item in result["table"]:
            print(f"  {item['name']:<3} 形状 {item['shape']}　{item['count']} 个")
        print(f"一共 {result['total']} 个参数。每个按 4 字节的单精度数保存，权重占 {result['bytes_as_float32']} 字节。")
        print("这只是权重的存储量。运行时还要放每个位置的中间结果，不在这个数里。")
    elif mode == "seeds":
        for row in result["rows"]:
            print(f"  种子 {row['seed']:>2}　训练后的损失 {row['loss']:.4f}　{'两句都接对' if row['completes_both'] else '没有都接对'}")
        print(
            f"{len(result['rows'])} 个起点里，{result['near_floor']} 个的损失贴近下限（相差不到 0.01），"
            f"{result['succeeded']} 个训练后每步取概率最大的能把两句话都接对。"
        )
        print(f"参照：损失的下限 {result['loss_floor']:.4f}；不看前文、只按词出现的多少来猜是 {result['unigram_level']:.4f}。")
    elif mode == "record":
        record = result["record"]
        config = record["config"]
        print(f"设置：种子 {config['seed']}，学习率 {config['learning_rate']}，{config['steps']} 步，{record['parameter_count']} 个参数。")
        print(config["seed_note"])
        for pair in record["pairs"]:
            print(f"  输入 [{lm.decode(pair['inputs'])}] → 目标 [{lm.decode(pair['targets'])}]")
        print(f"损失的下限 {record['loss_floor']:.4f}；不看前文、只按词出现的多少来猜是 {record['unigram_level']:.4f}。")
        print("第一次更新里的四个参数：")
        for item in record["first_update"]:
            print(
                f"  {item['parameter']}（{item['meaning']}）　{item['before']:.3f} → {item['after']:.6f}　"
                f"梯度：程序 {item['gradient_by_program']:+.6f}，拨动 {item['gradient_by_nudging']:+.6f}"
            )
        print("步数与损失：" + "，".join(f"{item['step']} 步 {item['loss']:.4f}" for item in record["checkpoints"]))
        for index, prefix in enumerate(PREFIXES):
            line = "　".join(f"第 {item['step']} 步：{leading(item['by_prefix'][index]['next'])}" for item in record["outputs"])
            print(f"  前文 [{prefix}]　{line}")
        for row in record["completions"]:
            print(f"  训练后从 [{row['prefix']}] 接着生成：{row['generated']}")
        if result["largest_difference"] is not None:
            print(f"重新训练了全部 {config['steps']} 步，检查点的损失与记录最多相差 {result['largest_difference']:.2e}。")
        print("记录" + ("自己对得上。" if not result["problems"] else "有问题：" + "；".join(result["problems"])))
    else:
        print("已写入 " + result["path"] if result["written"] else result["error"])
    print(f"exit_code: {result['exit_code']}")


def run(args):
    if args.hand:
        return hand_example()
    if args.count:
        return count_parameters(args.dimension, args.hidden)
    if args.seeds:
        return seed_survey(args.seeds)
    if args.write_record:
        if RECORD_FILE.exists() and not args.force:
            return {"mode": "write", "written": False, "exit_code": 2, "error": "记录已存在；确实要覆盖时加 --force。"}
        RECORD_FILE.write_text(json.dumps(build_record(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"mode": "write", "written": True, "path": str(RECORD_FILE), "exit_code": 0}
    record = load_record()
    parameters = record["final_parameters"]
    if args.record:
        problems, largest = check_record(record, args.retrain)
        return {
            "mode": "record", "record": record, "problems": problems, "largest_difference": largest,
            "exit_code": 1 if problems else 0,
        }
    if args.mask_check:
        return mask_check(parameters)
    if args.order_check:
        return order_check(parameters)
    if args.show:
        return show_block(parameters, args.show)
    try:
        ids = lm.encode(args.prefix)
    except KeyError as error:
        raise LabError(f"词表里没有 {error}。可用的词：{'、'.join(lm.VOCAB)}") from error
    if args.generate:
        outcome = lm.generate(parameters, ids, args.temperature, args.seed)
        steps = [
            {"prefix": lm.decode(step["prefix"]), "top": top(step["probabilities"], 2), "chosen": lm.VOCAB[step["chosen"]]}
            for step in outcome["steps"]
        ]
        return {
            "mode": "generate", "prefix": args.prefix, "temperature": args.temperature, "steps": steps,
            "text": lm.decode(outcome["ids"]), "stopped_by": outcome["stopped_by"], "exit_code": 0,
        }
    return trace(parameters, args.prefix)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true", help="从前文开始逐词生成")
    mode.add_argument("--mask-check", action="store_true", help="检查因果遮罩：换掉后面的词，前面的预测不变")
    mode.add_argument("--order-check", action="store_true", help="检查顺序从哪里进来：去掉位置向量，对调前面的词，预测不变")
    mode.add_argument("--show", metavar="NAME", help="把训练得到的某一块参数打印出来，例如 W1、b1、U")
    mode.add_argument("--hand", action="store_true", help="第 12、13 章的手算例，用手设的向量")
    mode.add_argument("--count", action="store_true", help="数一数参数有多少、占多少存储")
    mode.add_argument("--record", action="store_true", help="读训练记录并核对它自己对不对得上")
    mode.add_argument("--seeds", type=int, metavar="N", help="换 N 个起点各训练一遍（每个起点两万步，较慢）")
    mode.add_argument("--write-record", action="store_true", help="课程维护用：训练一遍并写成随课记录")
    parser.add_argument("--prefix", default="请求 成功 后", help="前文，词之间用空格分开")
    parser.add_argument("--temperature", type=float, default=0.0, help="与 --generate 合用：0 表示总取概率最大的")
    parser.add_argument("--seed", type=int, default=0, help="与 --generate 合用：抽样用的种子")
    parser.add_argument("--dimension", type=int, help="与 --count 合用：换一个向量维数来数")
    parser.add_argument("--hidden", type=int, help="与 --count 合用：换一个隐藏单元个数来数")
    parser.add_argument("--retrain", action="store_true", help="与 --record 合用：把全部步数重新训练一遍再核对")
    parser.add_argument("--force", action="store_true", help="与 --write-record 合用，覆盖已有记录")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (LabError, ValueError) as error:
        if args.json:
            print(json.dumps({"error": str(error), "exit_code": 2}, ensure_ascii=False))
        else:
            print(f"实验失败：{error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_run(result)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())

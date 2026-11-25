"""开销计数（第 15 章的实验底座）：在七词小语言模型上逐词生成，数每一步做了多少次乘法。只用标准库。

两种做法各数一遍：每生成一个词就把整段前文重算一遍；或者把算过的中间结果留着，每步只算新接上的位置。
两种做法选出的词和概率完全相同。数的是这个小模型里的乘法次数，不是耗时，也不是模型服务的账单。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CHAPTER = Path(__file__).resolve().parent
COURSE = CHAPTER.parent
sys.path.insert(0, str(COURSE))

from shared import tiny_lm as lm  # noqa: E402
from shared import tiny_lm_cost as cost  # noqa: E402


RECORD_FILE = COURSE / "data/tiny-lm-training-record.json"


class LabError(Exception):
    """输入或记录有误，无法完成实验。"""


def trained_parameters():
    if not RECORD_FILE.exists():
        raise LabError("找不到七词小语言模型的训练记录 data/tiny-lm-training-record.json。")
    return json.loads(RECORD_FILE.read_text(encoding="utf-8"))["final_parameters"]


def encode(prefix, parameters):
    try:
        ids = lm.encode(prefix)
    except KeyError as error:
        raise LabError(f"词表里没有 {error}。可用的词：{'、'.join(lm.VOCAB)}") from error
    if not 1 <= len(ids) <= cost.context_of(parameters):
        raise LabError(f"前文要有 1 到 {cost.context_of(parameters)} 个词。")
    return ids


def model_sizes(parameters):
    dimension, hidden = lm.sizes(parameters)
    return dimension, hidden, len(lm.VOCAB)


def count_generation(parameters, prefix, temperature=0.0, seed=0):
    """同一段前文、同样的选词办法，两种做法各生成一遍，逐步对照。"""
    ids = encode(prefix, parameters)
    recompute = cost.generate_counted(parameters, ids, False, temperature, seed)
    cached = cost.generate_counted(parameters, ids, True, temperature, seed)
    reference = lm.generate(parameters, ids, temperature, seed)
    identical = recompute["ids"] == cached["ids"] == reference["ids"] and all(
        one["probabilities"] == two["probabilities"] == three["probabilities"]
        for one, two, three in zip(recompute["steps"], cached["steps"], reference["steps"], strict=True)
    )
    steps = []
    for index, (one, two) in enumerate(zip(recompute["steps"], cached["steps"], strict=True)):
        steps.append({
            "step": index + 1, "prefix": lm.decode(one["prefix"]), "positions": len(one["prefix"]),
            "chosen": lm.VOCAB[one["chosen"]],
            "recompute": {
                "total": one["total"], "positions_recomputed": one["positions_predicted"],
                "multiplications": one["multiplications"],
            },
            "cached": {
                "total": two["total"], "new_positions": len(ids) if index == 0 else 1,
                "reads": two["multiplications"]["scoring"] + two["multiplications"]["reading"],
                "cached_numbers": two["cached_numbers"], "multiplications": two["multiplications"],
            },
        })
    return {
        "mode": "cost", "prefix": prefix, "temperature": temperature, "text": lm.decode(recompute["ids"]),
        "stopped_by": recompute["stopped_by"], "identical": identical, "steps": steps,
        "recompute_total": sum(step["recompute"]["total"] for step in steps),
        "cached_total": sum(step["cached"]["total"] for step in steps),
        "exit_code": 0 if identical else 1,
    }


def breakdown(parameters, prefix):
    """一步里的乘法按部件分开数。留着中间结果的那一列，假定最后一个词之前的位置都已经算过。"""
    ids = encode(prefix, parameters)
    by_position, recompute = cost.recompute_all(parameters, ids)
    cache, earlier = cost.new_cache(), cost.new_tally()
    for token in ids[:-1]:
        cost.extend(parameters, cache, token, earlier, want_prediction=False)
    cached = cost.new_tally()
    last = cost.extend(parameters, cache, ids[-1], cached)
    dimension, hidden, vocab = model_sizes(parameters)
    by_structure = (
        recompute == cost.recompute_cost(len(ids), dimension, hidden, vocab)
        and cached == cost.cached_cost(len(ids), 1, dimension, hidden, vocab)
    )
    return {
        "mode": "breakdown", "prefix": prefix, "positions": len(ids),
        "parts": [
            {"name": name, "label": label, "recompute": recompute[name], "cached": cached[name]}
            for name, label in cost.PARTS
        ],
        "recompute_total": sum(recompute.values()), "cached_total": sum(cached.values()),
        "identical": last == by_position[-1], "matches_structure": by_structure,
        "exit_code": 0 if last == by_position[-1] and by_structure else 1,
    }


def project(parameters, first, last, dimension=None, hidden=None):
    """不运行模型，按结构数：前文一步一步加长时，两种做法各做多少次乘法。"""
    own_dimension, own_hidden, vocab = model_sizes(parameters)
    dimension = own_dimension if dimension is None else dimension
    hidden = own_hidden if hidden is None else hidden
    if dimension < 1 or hidden < 1:
        raise LabError("向量的维数和隐藏单元的个数都要大于 0。")
    if not 1 <= first <= last:
        raise LabError("位置数要满足 1 ≤ 起点 ≤ 终点。")
    return {
        "mode": "project", "dimension": dimension, "hidden": hidden, "vocab": vocab,
        "context": cost.context_of(parameters), **cost.projection(first, last, dimension, hidden, vocab),
        "exit_code": 0,
    }


def print_run(result):
    mode = result["mode"]
    if mode == "cost":
        how = "每一步都取概率最大的" if result["temperature"] <= 0 else f"温度 {result['temperature']}，按概率抽样"
        print(f"从前文“{result['prefix']}”开始，{how}。用的是训练得到的参数。")
        for step in result["steps"]:
            one, two = step["recompute"], step["cached"]
            print(
                f"  第 {step['step']} 步　前文 {step['positions']} 个位置　选 {step['chosen']}　"
                f"每步重算 {one['total']} 次乘法（前文的 {one['positions_recomputed']} 个位置全部重算）　"
                f"留着中间结果 {two['total']} 次乘法（只算新接上的 {two['new_positions']} 个位置，"
                f"其中打分和读取 {two['reads']} 次；留着 {two['cached_numbers']} 个数）"
            )
        print(f"结果：{result['text']}　停止的原因：{result['stopped_by']}")
        print(f"合计　每步重算 {result['recompute_total']} 次乘法，留着中间结果 {result['cached_total']} 次乘法。")
        print("两种做法选出的词和每一步的概率" + ("完全相同。" if result["identical"] else "不一样，需要检查。"))
        print("留着中间结果时，新接上的位置仍然要读到前面的每一个位置，这一部分随前文变长；省掉的是把前面的位置重新算一遍。")
        print("这里数的是这个小模型里的乘法次数，不是耗时，也不是模型服务的账单。")
    elif mode == "breakdown":
        print(f"前文“{result['prefix']}”有 {result['positions']} 个位置。预测下一个词的这一步，乘法次数按部件分：")
        print("  （留着中间结果的那一列，假定最后一个词之前的位置都已经算过）")
        for part in result["parts"]:
            print(f"  {part['label']}　每步重算 {part['recompute']}　留着中间结果 {part['cached']}")
        print(f"  合计　每步重算 {result['recompute_total']}　留着中间结果 {result['cached_total']}")
        print("运行时数到的次数与按结构数出来的" + ("一致。" if result["matches_structure"] else "不一致，需要检查。"))
        print("两种做法算出的概率" + ("完全相同。" if result["identical"] else "不一样，需要检查。"))
    else:
        print(
            f"按结构数，没有运行模型：向量 {result['dimension']} 维，前馈的隐藏单元 {result['hidden']} 个，"
            f"词表 {result['vocab']} 个词。"
        )
        print(
            f"从 {result['first']} 个位置的前文开始，一步接一个词，接到前文有 {result['last']} 个位置为止，共 {result['steps']} 步。"
        )
        print(f"  每步重算　合计 {result['recompute_total']:,} 次乘法，最后一步 {result['recompute_last_step']:,} 次")
        print(
            f"  留着中间结果　合计 {result['cached_total']:,} 次乘法，最后一步 {result['cached_last_step']:,} 次；"
            f"这时留着 {result['cached_numbers_at_end']:,} 个数"
        )
        if result["last"] > result["context"]:
            print(f"这个小模型的位置向量表只有 {result['context']} 行，实际只能读 {result['context']} 个位置；更长的是按同样的结构推算的，没有运行过。")
        print("这是这个一层小模型的乘法次数。别的模型结构不同、实现不同，不能拿这些数去推算。")
    print(f"exit_code: {result['exit_code']}")


def run(args):
    parameters = trained_parameters()
    if args.project is not None:
        return project(parameters, args.start, args.project, args.dimension, args.hidden)
    if args.breakdown:
        return breakdown(parameters, args.prefix)
    return count_generation(parameters, args.prefix, args.temperature, args.seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--breakdown", action="store_true", help="把一步里的乘法按部件分开数")
    mode.add_argument("--project", type=int, metavar="N", help="不运行模型，按结构数：前文一步步加长到 N 个位置")
    parser.add_argument("--prefix", default=None, help="前文，词之间用空格分开；默认生成时用“请求”，--breakdown 时用“请求 成功 后”")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 表示总取概率最大的")
    parser.add_argument("--seed", type=int, default=0, help="抽样用的种子")
    parser.add_argument("--start", type=int, default=1, help="与 --project 合用：从几个位置的前文开始")
    parser.add_argument("--dimension", type=int, help="与 --project 合用：换一个向量维数来数")
    parser.add_argument("--hidden", type=int, help="与 --project 合用：换一个隐藏单元个数来数")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)
    if args.prefix is None:
        args.prefix = "请求 成功 后" if args.breakdown else "请求"
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

"""窗口对照（第 13 章的实验底座）：不用注意力，给最近的几个词各留一个槽位，看它在七词例上能做到哪一步。只用标准库。

对照的另一边是七词小语言模型，用它随课的训练记录，不在这里重新训练。
这里不预设谁做得更好：窗口只要盖住了要用的那个词就能把题做对，这也是有效的结果。
"""

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
from shared import window_lm as wl  # noqa: E402


RECORD_FILE = COURSE / "data/window-contrast-record.json"
ATTENTION_RECORD_FILE = COURSE / "data/tiny-lm-training-record.json"
CONFIG = {
    "seed": 3,
    "dimension": 2,
    "windows": [1, 2, 4],
    "initial_scale": 0.5,
    "learning_rate": 0.05,
    "steps": 20000,
    "checkpoints": [0, 1, 10, 100, 500, 1000, 2000, 5000, 10000, 20000],
    "survey_seeds": 20,
    "method": "每一步都用两句话的全部 8 个位置算梯度，所有参数一起更新",
    "seed_note": (
        "种子、起点的取法、学习率和步数都与七词小语言模型的训练记录相同。"
        "窗口模型换种子 1 到 20 各训练一遍，结果没有分别，见 seed_survey。"
    ),
}
PREFIXES = ["请求", "请求 成功 后", "请求 失败 后"]
COMPLETIONS = {"请求 成功": "请求 成功 后 返回 EOS", "请求 失败": "请求 失败 后 重试 EOS"}
NEAR_FLOOR = 0.01


class LabError(Exception):
    """输入或记录有误，无法完成实验。"""


def start_parameters(window, seed=None):
    return wl.initial_parameters(
        CONFIG["seed"] if seed is None else seed, CONFIG["dimension"], window, CONFIG["initial_scale"]
    )


def top(probabilities, count=3):
    order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))[:count]
    return [{"token": lm.VOCAB[index], "probability": probabilities[index]} for index in order]


def outputs_for(parameters):
    return [
        {"prefix": prefix, "next": top(wl.forward(parameters, lm.encode(prefix))["probabilities"][-1])}
        for prefix in PREFIXES
    ]


def completions_for(parameters):
    """从“请求 成功”和“请求 失败”出发，每一步都取概率最大的，看能不能把两句话接对。"""
    rows = []
    for prefix, wanted in COMPLETIONS.items():
        text = lm.decode(wl.generate(parameters, lm.encode(prefix))["ids"])
        rows.append({"prefix": prefix, "generated": text, "matches": text == wanted})
    return rows


def seen_as(window):
    if window == 1:
        return "只有当前的词"
    if window >= lm.CONTEXT:
        return "全部前文"
    return f"最近 {window} 个词"


def run_training(window, pairs, seed=None):
    checkpoints = []

    def note(step, parameters):
        if step in CONFIG["checkpoints"]:
            checkpoints.append({"step": step, "loss": wl.mean_loss(parameters, pairs)})

    start = start_parameters(window, seed)
    final = wl.train(start, pairs, CONFIG["learning_rate"], CONFIG["steps"], note)
    return start, final, checkpoints


def seed_survey(count, windows=None):
    """换起点各训练一遍：每种窗口有多少个起点的损失贴近它自己的下限、有多少个能把两句话都接对。"""
    pairs = lm.training_pairs()
    rows = []
    for window in windows or CONFIG["windows"]:
        floor = wl.loss_floor(pairs, window)
        losses, completed = [], 0
        for seed in range(1, count + 1):
            final = wl.train(start_parameters(window, seed), pairs, CONFIG["learning_rate"], CONFIG["steps"])
            losses.append(wl.mean_loss(final, pairs))
            completed += all(row["matches"] for row in completions_for(final))
        rows.append({
            "window": window, "loss_floor": floor, "near_floor": sum(loss - floor < NEAR_FLOOR for loss in losses),
            "completes_both": completed, "lowest_loss": min(losses), "highest_loss": max(losses),
        })
    return {"seeds": count, "rows": rows}


def attention_side():
    """对照的另一边：七词小语言模型随课记录里的结果。"""
    if not ATTENTION_RECORD_FILE.exists():
        raise LabError("找不到七词小语言模型的训练记录 data/tiny-lm-training-record.json。")
    record = json.loads(ATTENTION_RECORD_FILE.read_text(encoding="utf-8"))
    final = record["final_parameters"]
    pairs = lm.training_pairs()
    return {
        "parameter_count": record["parameter_count"], "loss": lm.mean_loss(final, pairs),
        "loss_floor": lm.loss_floor(pairs),
        "outputs": [
            {"prefix": prefix, "next": top(lm.forward(final, lm.encode(prefix))["probabilities"][-1])}
            for prefix in PREFIXES
        ],
        "completes_both": all(row["matches"] for row in record["completions"]),
        "seed": record["config"]["seed"], "steps": record["config"]["steps"],
        "learning_rate": record["config"]["learning_rate"],
    }


def attention_survey_from(path):
    """课程维护用：读 `14-forward/lab.py --seeds N --json` 存下的输出，取出要并排放的三个数。"""
    if not path:
        return None
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "seeds": len(result["rows"]), "near_floor": result["near_floor"], "completes_both": result["succeeded"],
        "source": f"v2/14-forward/lab.py --seeds {len(result['rows'])}",
    }


def build_record(survey=None, attention_survey=None):
    pairs = lm.training_pairs()
    runs = []
    for window in CONFIG["windows"]:
        start, final, checkpoints = run_training(window, pairs)
        runs.append({
            "window": window, "sees": seen_as(window),
            "parameter_count": wl.parameter_count(CONFIG["dimension"], window),
            "loss_floor": wl.loss_floor(pairs, window),
            "initial_parameters": start, "checkpoints": checkpoints, "final_parameters": final,
            "outputs": outputs_for(final), "completions": completions_for(final),
        })
    return {
        "schema_version": 1,
        "data_kind": "training_run_record",
        "config": CONFIG,
        "vocabulary": lm.VOCAB,
        "pairs": pairs,
        "runs": runs,
        "seed_survey": survey if survey is not None else seed_survey(CONFIG["survey_seeds"]),
        "attention_seed_survey": attention_survey,
    }


def load_record():
    if not RECORD_FILE.exists():
        raise LabError("还没有窗口对照的训练记录。课程维护者先运行 --write-record。")
    return json.loads(RECORD_FILE.read_text(encoding="utf-8"))


def run_for(record, window):
    for item in record["runs"]:
        if item["window"] == window:
            return item
    raise LabError(f"记录里没有 {window} 个槽的窗口。有的是：{'、'.join(str(item['window']) for item in record['runs'])}")


def outputs_match(saved, fresh, tolerance=1e-12):
    """记录里的输出与重算的输出是否一致。

    前文、候选的词和条目的个数要完全相同。概率允许末位的舍入差异：换一个平台或 Python 版本，
    同样的计算在最后一位上可能不同，这不算记录有问题。
    """
    if len(saved) != len(fresh):
        return False
    for one, two in zip(saved, fresh, strict=True):
        if one["prefix"] != two["prefix"] or len(one["next"]) != len(two["next"]):
            return False
        for kept, again in zip(one["next"], two["next"], strict=True):
            if kept["token"] != again["token"]:
                return False
            if not (math.isfinite(kept["probability"]) and math.isfinite(again["probability"])):
                return False
            if not math.isclose(kept["probability"], again["probability"], rel_tol=tolerance, abs_tol=tolerance):
                return False
    return True


def check_record(record, retrain=False):
    """核对记录自己对得上。默认只重算起点、前 100 步和最后的参数；加 --retrain 才把全部步数重新训练一遍。"""
    pairs = lm.training_pairs()
    problems, largest = [], None
    if record["config"] != CONFIG or record["pairs"] != pairs:
        problems.append("记录里的设置或数据与当前程序不一致")
    if [item["window"] for item in record["runs"]] != CONFIG["windows"]:
        problems.append("记录里的窗口与当前程序不一致")
    for item in record["runs"]:
        window, name = item["window"], f"{item['window']} 个槽的窗口"
        start = start_parameters(window)
        if item["initial_parameters"] != start:
            problems.append(f"{name}：起点与当前程序不一致")
        saved = {point["step"]: point["loss"] for point in item["checkpoints"]}
        early = wl.train(start, pairs, CONFIG["learning_rate"], 100)
        if abs(wl.mean_loss(early, pairs) - saved[100]) > 1e-9:
            problems.append(f"{name}：重新训练前 100 步，损失与记录不一致")
        final = item["final_parameters"]
        if abs(wl.mean_loss(final, pairs) - saved[CONFIG["steps"]]) > 1e-9:
            problems.append(f"{name}：用记录里最后的参数重算，损失与记录不一致")
        if not outputs_match(item["outputs"], outputs_for(final)) or item["completions"] != completions_for(final):
            problems.append(f"{name}：用记录里最后的参数重算，输出与记录不一致")
        if abs(item["loss_floor"] - wl.loss_floor(pairs, window)) > 1e-12:
            problems.append(f"{name}：损失的下限与记录不一致")
        if retrain:
            _, _, checkpoints = run_training(window, pairs)
            gap = max(abs(point["loss"] - saved[point["step"]]) for point in checkpoints)
            largest = gap if largest is None else max(largest, gap)
            if gap > 1e-6:
                problems.append(f"{name}：重新训练全部步数，检查点的损失与记录最多相差 {gap:.2e}")
    return problems, largest


def compare(record):
    rows = []
    for item in record["runs"]:
        rows.append({
            "model": f"窗口，{item['window']} 个槽", "sees": item["sees"], "parameter_count": item["parameter_count"],
            "loss": item["checkpoints"][-1]["loss"], "loss_floor": item["loss_floor"],
            "outputs": item["outputs"], "completes_both": all(row["matches"] for row in item["completions"]),
        })
    attention = attention_side()
    rows.append({
        "model": "注意力", "sees": "全部前文", "parameter_count": attention["parameter_count"],
        "loss": attention["loss"], "loss_floor": attention["loss_floor"], "outputs": attention["outputs"],
        "completes_both": attention["completes_both"],
    })
    same_setup = (attention["seed"], attention["steps"], attention["learning_rate"]) == (
        CONFIG["seed"], CONFIG["steps"], CONFIG["learning_rate"]
    )
    full = run_for(record, lm.CONTEXT)
    return {
        "mode": "compare", "rows": rows, "seed_survey": record["seed_survey"],
        "attention_seed_survey": record["attention_seed_survey"], "same_setup": same_setup,
        "full_window_solves": all(row["matches"] for row in full["completions"]),
        "last_only_separates": run_for(record, 1)["outputs"][1]["next"] != run_for(record, 1)["outputs"][2]["next"],
        "exit_code": 0,
    }


def trace(record, window, prefix):
    parameters = run_for(record, window)["final_parameters"]
    ids = lm.encode(prefix)
    state = wl.forward(parameters, ids)
    held = state["slots"][-1]
    return {
        "mode": "trace", "window": window, "prefix": prefix, "ids": ids,
        "slots": [None if token is None else lm.VOCAB[token] for token in held],
        "slot_vectors": [None if token is None else parameters["E"][token] for token in held],
        "input": state["inputs"][-1], "scores": state["scores"][-1], "probabilities": state["probabilities"][-1],
        "prediction": lm.VOCAB[lm.pick(state["probabilities"][-1])],
        "shapes": {
            "槽位": [window], "接成一行": [window * CONFIG["dimension"]], "分数": [len(lm.VOCAB)], "概率": [len(lm.VOCAB)],
        },
        "exit_code": 0,
    }


def hand_example():
    """用第 12、13 章手算的二维向量，看两段前文放进槽位以后还分不分得开。不涉及训练。"""
    rows = []
    for window in (1, 2, 3):
        lines = []
        for prefix in ("请求 成功 后", "请求 失败 后"):
            ids = lm.encode(prefix)
            held = wl.slots(ids, len(ids) - 1, window)
            lines.append({
                "prefix": prefix, "slots": [lm.VOCAB[token] for token in held],
                "input": [value for token in held for value in lm.HAND_EMBEDDING[token]],
            })
        rows.append({"window": window, "lines": lines, "differ": lines[0]["input"] != lines[1]["input"]})
    return {"mode": "hand", "rows": rows, "exit_code": 0}


def count_parameters(context=None):
    """读得到的前文加长时，两种读法各要多少参数。只按形状数，不训练。"""
    longest = lm.CONTEXT if context is None else context
    if longest < 1:
        raise LabError("位置数要大于 0。")
    dimension, vocab = CONFIG["dimension"], len(lm.VOCAB)
    attention_now = lm.parameter_count(dimension, 4)
    rows = []
    for positions in sorted({lm.CONTEXT, longest}):
        rows.append({
            "positions": positions,
            "window": wl.parameter_count(dimension, positions),
            "window_output_weights": positions * dimension * vocab,
            "attention": attention_now + (positions - lm.CONTEXT) * dimension,
            "attention_position_vectors": positions * dimension,
        })
    return {"mode": "count", "dimension": dimension, "rows": rows, "exit_code": 0}


def numbers(values, places=3):
    return "[" + ", ".join(f"{value:.{places}f}" for value in values) + "]"


def leading(items, count=2):
    return "，".join(f"{item['token']} {item['probability']:.3f}" for item in items[:count])


def print_run(result):
    mode = result["mode"]
    if mode == "compare":
        print("同样的两句话，同样的起点取法、学习率和步数。前三行是固定窗口，最后一行是七词小语言模型（用注意力读前文，另有位置向量和前馈变换）。")
        if not result["same_setup"]:
            print("注意：两份记录的种子、步数或学习率不一样，下面的对照不在同一条件下。")
        for row in result["rows"]:
            print(
                f"  {row['model']}　看得见：{row['sees']}　参数 {row['parameter_count']} 个　"
                f"训练后的损失 {row['loss']:.4f}（看得见这么多时最低 {row['loss_floor']:.4f}）　"
                f"{'两句都接对' if row['completes_both'] else '没有都接对'}"
            )
            for item in row["outputs"][1:]:
                print(f"      前文 [{item['prefix']}] → {leading(item['next'])}")
        survey = result["seed_survey"]
        print(f"换 {survey['seeds']} 个起点各训练一遍，能把两句话都接对的：" + "，".join(
            f"{item['window']} 个槽 {item['completes_both']} 个" for item in survey["rows"]
        ) + "。")
        other = result["attention_seed_survey"]
        if other:
            print(
                f"七词小语言模型同样换 {other['seeds']} 个起点：{other['completes_both']} 个接对，"
                f"其中 {other['near_floor']} 个的损失贴近下限。"
            )
        print("只看当前的词时，两段前文" + ("的输出不同，与预期不符，需要检查。" if result["last_only_separates"] else "给出同一组概率：它们的最后一个词相同，模型看到的输入就是同一个。"))
        print("看得全前文的窗口" + ("把两句话都接对了。在这两句话上，注意力没有比它做得更好。" if result["full_window_solves"] else "没有把两句话都接对。"))
    elif mode == "trace":
        print(f"前文：{result['prefix']}　窗口有 {result['window']} 个槽，用的是训练得到的参数。下面只展示最后一个位置的预测。")
        print(f"① 编号　{result['ids']}")
        print("② 放进槽位　第 0 槽是当前的词，第 1 槽是前一个词，依此类推；空槽填 0")
        for index, (token, vector) in enumerate(zip(result["slots"], result["slot_vectors"], strict=True)):
            print(f"     第 {index} 槽　" + ("空" if token is None else f"{token}　词向量 {numbers(vector)}"))
        print(f"③ 接成一行　{numbers(result['input'])}")
        print(f"④ 输出　七个候选的分数 {numbers(result['scores'], 2)}")
        print("⑤ 概率　" + "，".join(f"{token} {share:.3f}" for token, share in zip(lm.VOCAB, result["probabilities"], strict=True)))
        print(f"概率最大的是：{result['prediction']}")
        print("各步的形状：" + "，".join(f"{name} {shape}" for name, shape in result["shapes"].items()))
        print("显示的数都四舍五入过；拿显示的数复算，末位可能差 1。")
    elif mode == "hand":
        print("手算用的二维向量，不是训练出来的。只看两段前文放进槽位以后接成的那一行。")
        for row in result["rows"]:
            print(f"窗口 {row['window']} 个槽：")
            for line in row["lines"]:
                print(f"  前文 [{line['prefix']}]　槽里是 {'、'.join(line['slots'])}　接成一行 {line['input']}")
            print("  两段前文" + ("接成的行不同，可以区分。" if row["differ"] else "接成同一行，分不出来。"))
    elif mode == "count":
        print(f"向量 {result['dimension']} 维，词表 {len(lm.VOCAB)} 个词。按形状数参数，没有训练。")
        for row in result["rows"]:
            print(
                f"  最多读 {row['positions']} 个位置　窗口（每个位置一个槽）{row['window']} 个参数，"
                f"其中输出权重 {row['window_output_weights']} 个；"
                f"七词小语言模型 {row['attention']} 个参数，其中位置向量 {row['attention_position_vectors']} 个"
            )
        print("窗口每多读一个位置，就多一个槽、多一组输出权重。七词小语言模型里查询、键、值的映射各位置共用，多读一个位置只多一行位置向量。")
        print(f"最多读 {lm.CONTEXT} 个位置时两边的参数一样多，是凑巧。")
    elif mode == "seeds":
        for row in result["rows"]:
            print(
                f"  {row['window']} 个槽　{result['seeds']} 个起点里，{row['near_floor']} 个的损失贴近它的下限 {row['loss_floor']:.4f}"
                f"（相差不到 {NEAR_FLOOR}），{row['completes_both']} 个把两句话都接对；"
                f"损失最低 {row['lowest_loss']:.4f}，最高 {row['highest_loss']:.4f}"
            )
    elif mode == "record":
        record = result["record"]
        config = record["config"]
        print(f"设置：种子 {config['seed']}，学习率 {config['learning_rate']}，{config['steps']} 步。")
        print(config["seed_note"])
        for item in record["runs"]:
            print(f"  {item['window']} 个槽（{item['sees']}），{item['parameter_count']} 个参数，看得见这么多时损失最低 {item['loss_floor']:.4f}")
            print("    步数与损失：" + "，".join(f"{point['step']} 步 {point['loss']:.4f}" for point in item["checkpoints"]))
            for row in item["completions"]:
                print(f"    训练后从 [{row['prefix']}] 接着生成：{row['generated']}")
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
        return count_parameters(args.context)
    if args.seeds:
        return {"mode": "seeds", **seed_survey(args.seeds), "exit_code": 0}
    if args.write_record:
        if RECORD_FILE.exists() and not args.force:
            return {"mode": "write", "written": False, "exit_code": 2, "error": "记录已存在；确实要覆盖时加 --force。"}
        previous = json.loads(RECORD_FILE.read_text(encoding="utf-8")) if RECORD_FILE.exists() else {}
        record = build_record(attention_survey=attention_survey_from(args.attention_survey) or previous.get("attention_seed_survey"))
        RECORD_FILE.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"mode": "write", "written": True, "path": str(RECORD_FILE), "exit_code": 0}
    record = load_record()
    if args.record:
        problems, largest = check_record(record, args.retrain)
        return {
            "mode": "record", "record": record, "problems": problems, "largest_difference": largest,
            "exit_code": 1 if problems else 0,
        }
    if args.trace:
        try:
            lm.encode(args.prefix)
        except KeyError as error:
            raise LabError(f"词表里没有 {error}。可用的词：{'、'.join(lm.VOCAB)}") from error
        return trace(record, args.window, args.prefix)
    return compare(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trace", action="store_true", help="跟着一个窗口模型从槽位算到概率")
    mode.add_argument("--hand", action="store_true", help="用手算的二维向量，看两段前文放进槽位以后分不分得开")
    mode.add_argument("--count", action="store_true", help="读得到的前文加长时，两种读法各要多少参数")
    mode.add_argument("--record", action="store_true", help="读窗口对照的训练记录并核对它自己对不对得上")
    mode.add_argument("--seeds", type=int, metavar="N", help="换 N 个起点各训练一遍（每种窗口每个起点两万步，较慢）")
    mode.add_argument("--write-record", action="store_true", help="课程维护用：训练一遍并写成随课记录")
    parser.add_argument("--window", type=int, default=lm.CONTEXT, help="与 --trace 合用：窗口的槽数，记录里有 1、2、4")
    parser.add_argument("--prefix", default="请求 成功 后", help="与 --trace 合用：前文，词之间用空格分开")
    parser.add_argument("--context", type=int, help="与 --count 合用：最多读多少个位置")
    parser.add_argument("--retrain", action="store_true", help="与 --record 合用：把全部步数重新训练一遍再核对")
    parser.add_argument("--force", action="store_true", help="与 --write-record 合用，覆盖已有记录")
    parser.add_argument(
        "--attention-survey", metavar="FILE",
        help="与 --write-record 合用：`14-forward/lab.py --seeds N --json` 存下的输出，把里面的结果并排记进来",
    )
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

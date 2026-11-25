"""验证阶段三另外两项实验底座：第 13 章的窗口对照，第 15 章的开销计数。全部离线，只用标准库。"""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import sys
import unittest


COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))

from shared import tiny_lm as lm  # noqa: E402
from shared import tiny_lm_cost as cost  # noqa: E402
from shared import window_lm as wl  # noqa: E402


def load_lab(name, chapter):
    spec = importlib.util.spec_from_file_location(name, COURSE / chapter / "lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


window_lab = load_lab("window_lab", "13-attention")
cost_lab = load_lab("cost_lab", "15-generation")
PAIRS = lm.training_pairs()
WINDOW_RECORD = json.loads((COURSE / "data/window-contrast-record.json").read_text(encoding="utf-8"))
RUNS = {item["window"]: item for item in WINDOW_RECORD["runs"]}
TRAINED = json.loads((COURSE / "data/tiny-lm-training-record.json").read_text(encoding="utf-8"))["final_parameters"]
SUCCESS, FAILURE = lm.encode("请求 成功 后"), lm.encode("请求 失败 后")


def run_json(lab, *args):
    with redirect_stdout(io.StringIO()) as stream:
        code = lab.main([*args, "--json"])
    return code, json.loads(stream.getvalue())


def run_text(lab, *args):
    with redirect_stdout(io.StringIO()) as stream:
        code = lab.main(list(args))
    return code, stream.getvalue()


def longer_model(seed, dimension, hidden, context):
    """把位置向量表加长的随机参数，只用来核对计数；没有训练过。"""
    source = random.Random(seed)
    parameters = lm.initial_parameters(seed, dimension, hidden)
    parameters["P"] = [[round(source.uniform(-0.5, 0.5), 3) for _ in range(dimension)] for _ in range(context)]
    return parameters, [source.randrange(len(lm.VOCAB) - 1) for _ in range(context)]


class WindowModelTest(unittest.TestCase):
    def test_parameter_count_grows_by_one_block_per_slot(self):
        self.assertEqual([wl.parameter_count(2, window) for window in (1, 2, 4)], [35, 49, 77])
        self.assertEqual(wl.parameter_count(2, 8) - wl.parameter_count(2, 4), 4 * 2 * len(lm.VOCAB))
        parameters = wl.initial_parameters(3, 2, 4)
        self.assertEqual(len(wl.flatten(parameters)), 77)
        self.assertEqual(wl.unflatten(wl.flatten(parameters), 2, 4), parameters)
        self.assertEqual(wl.sizes(parameters), (2, 4))
        with self.assertRaises(ValueError):
            wl.initial_parameters(3, 2, 0)

    def test_slots_hold_the_most_recent_words(self):
        self.assertEqual(wl.slots(SUCCESS, 2, 1), [3])
        self.assertEqual(wl.slots(SUCCESS, 2, 2), [3, 1])
        self.assertEqual(wl.slots(SUCCESS, 2, 4), [3, 1, 0, None])
        self.assertEqual(wl.slots(SUCCESS, 0, 2), [0, None])
        self.assertEqual(wl.visible(SUCCESS, 2, 2), [1, 3])
        self.assertEqual(wl.visible(SUCCESS, 2, 4), [0, 1, 3])

    def test_each_position_gives_a_distribution_and_empty_slots_are_zero(self):
        parameters = wl.initial_parameters(3, 2, 4)
        state = wl.forward(parameters, PAIRS[0]["inputs"])
        self.assertEqual(len(state["probabilities"]), 4)
        for row in state["probabilities"]:
            self.assertEqual(len(row), len(lm.VOCAB))
            self.assertAlmostEqual(sum(row), 1.0, places=12)
        self.assertEqual(state["inputs"][0][2:], [0.0] * 6)
        self.assertEqual(state["inputs"][3][:2], parameters["E"][4])
        with self.assertRaises(ValueError):
            wl.forward(parameters, [])
        with self.assertRaises(ValueError):
            wl.forward(parameters, [0, 1, 3, 4, 6])

    def test_what_the_window_can_see_sets_the_lowest_possible_loss(self):
        # 只看当前的词：“请求”后面两种接法，“后”后面也是两种，8 个位置里有 4 个各占一半。
        self.assertAlmostEqual(wl.loss_floor(PAIRS, 1), 4 * math.log(2) / 8, places=12)
        for window in (2, 3, 4):
            self.assertAlmostEqual(wl.loss_floor(PAIRS, window), lm.loss_floor(PAIRS), places=12)

    def test_one_slot_cannot_tell_the_two_prefixes_apart_whatever_the_parameters(self):
        for parameters in (wl.initial_parameters(7, 2, 1), wl.initial_parameters(8, 4, 1), RUNS[1]["final_parameters"]):
            self.assertEqual(
                wl.forward(parameters, SUCCESS)["probabilities"][-1], wl.forward(parameters, FAILURE)["probabilities"][-1]
            )
        for parameters in (wl.initial_parameters(7, 2, 2), wl.initial_parameters(7, 2, 4)):
            self.assertNotEqual(
                wl.forward(parameters, SUCCESS)["probabilities"][-1], wl.forward(parameters, FAILURE)["probabilities"][-1]
            )

    def test_gradient_by_program_matches_nudging(self):
        for seed, dimension, window in ((3, 2, 4), (5, 4, 2), (7, 2, 1)):
            parameters = wl.initial_parameters(seed, dimension, window)
            by_program = wl.flatten(wl.analytic_gradient(parameters, PAIRS))
            by_nudging = wl.flatten(wl.numeric_gradient(parameters, PAIRS))
            self.assertLess(max(abs(a - n) for a, n in zip(by_program, by_nudging, strict=True)), 1e-7)

    def test_one_update_lowers_the_loss(self):
        for window in (1, 2, 4):
            start = wl.initial_parameters(3, 2, window)
            after = wl.update(start, wl.analytic_gradient(start, PAIRS), 0.05)
            self.assertLess(wl.mean_loss(after, PAIRS), wl.mean_loss(start, PAIRS))

    def test_generation_follows_the_same_rules_as_the_attention_model(self):
        final = RUNS[4]["final_parameters"]
        self.assertEqual(lm.decode(wl.generate(final, lm.encode("请求 成功"))["ids"]), "请求 成功 后 返回 EOS")
        self.assertEqual(wl.generate(final, [0], 1.0, 5), wl.generate(final, [0], 1.0, 5))
        texts = {lm.decode(wl.generate(final, [0], 1.0, seed)["ids"]) for seed in range(8)}
        self.assertEqual(texts, set(lm.SENTENCES))


class WindowContrastRecordTest(unittest.TestCase):
    def test_record_checks_out_without_retraining(self):
        problems, largest = window_lab.check_record(WINDOW_RECORD)
        self.assertEqual(problems, [])
        self.assertIsNone(largest)

    def test_record_check_tolerates_rounding_but_not_real_changes(self):
        """换平台或 Python 版本时概率的末位可能不同，核对不应当因此报错；真的改了内容必须报错。"""
        import copy

        def changed(edit):
            record = copy.deepcopy(WINDOW_RECORD)
            edit(record["runs"][0]["outputs"][1]["next"])
            return window_lab.check_record(record)[0]

        def nudge(items):
            items[0]["probability"] = math.nextafter(items[0]["probability"], 1.0)

        def shift(items):
            items[0]["probability"] += 1e-6

        def rename(items):
            items[0]["token"] = "EOS"

        def drop(items):
            items.pop()

        def break_number(items):
            items[0]["probability"] = float("nan")

        self.assertEqual(changed(nudge), [])
        for edit in (shift, rename, drop, break_number):
            with self.subTest(edit=edit.__name__):
                self.assertTrue(any("输出与记录不一致" in problem for problem in changed(edit)))

    def test_setup_matches_the_attention_record(self):
        attention = json.loads((COURSE / "data/tiny-lm-training-record.json").read_text(encoding="utf-8"))["config"]
        for key in ("seed", "dimension", "initial_scale", "learning_rate", "steps", "method"):
            self.assertEqual(WINDOW_RECORD["config"][key], attention[key], key)
        self.assertEqual(sorted(RUNS), [1, 2, 4])

    def test_one_slot_stops_at_its_own_floor_and_cannot_complete_both(self):
        run = RUNS[1]
        final_loss = run["checkpoints"][-1]["loss"]
        self.assertGreater(final_loss, run["loss_floor"])
        self.assertLess(final_loss - run["loss_floor"], 0.001)
        self.assertEqual(run["outputs"][1]["next"], run["outputs"][2]["next"])
        self.assertEqual({item["token"] for item in run["outputs"][1]["next"][:2]}, {"返回", "重试"})
        for item in run["outputs"][1]["next"][:2]:
            self.assertAlmostEqual(item["probability"], 0.5, places=2)
        self.assertEqual(sum(row["matches"] for row in run["completions"]), 1)

    def test_windows_that_cover_the_deciding_word_solve_the_task(self):
        for window in (2, 4):
            run = RUNS[window]
            final_loss = run["checkpoints"][-1]["loss"]
            self.assertGreater(final_loss, run["loss_floor"])
            self.assertLess(final_loss - run["loss_floor"], 0.001)
            self.assertEqual(run["outputs"][1]["next"][0]["token"], "返回")
            self.assertEqual(run["outputs"][2]["next"][0]["token"], "重试")
            self.assertGreater(run["outputs"][1]["next"][0]["probability"], 0.99)
            self.assertTrue(all(row["matches"] for row in run["completions"]))

    def test_seed_survey_is_recorded_for_both_sides(self):
        survey = WINDOW_RECORD["seed_survey"]
        self.assertEqual(survey["seeds"], 20)
        by_window = {row["window"]: row for row in survey["rows"]}
        self.assertEqual([by_window[w]["completes_both"] for w in (1, 2, 4)], [0, 20, 20])
        self.assertEqual([by_window[w]["near_floor"] for w in (1, 2, 4)], [20, 20, 20])
        other = WINDOW_RECORD["attention_seed_survey"]
        self.assertEqual((other["seeds"], other["completes_both"], other["near_floor"]), (20, 16, 14))


class WindowLabTest(unittest.TestCase):
    def test_comparison_does_not_presume_attention_wins(self):
        code, result = run_json(window_lab)
        self.assertEqual(code, 0)
        self.assertTrue(result["same_setup"])
        self.assertTrue(result["full_window_solves"])
        self.assertFalse(result["last_only_separates"])
        self.assertEqual([row["parameter_count"] for row in result["rows"]], [35, 49, 77, 77])
        self.assertEqual([row["completes_both"] for row in result["rows"]], [False, True, True, True])
        attention = result["rows"][-1]
        self.assertAlmostEqual(attention["loss"], lm.mean_loss(TRAINED, PAIRS), places=12)
        code, text = run_text(window_lab)
        self.assertIn("在这两句话上，注意力没有比它做得更好", text)
        self.assertIn("exit_code: 0", text)

    def test_trace_and_hand_modes(self):
        code, result = run_json(window_lab, "--trace")
        self.assertEqual((code, result["prediction"], result["slots"]), (0, "返回", ["后", "成功", "请求", None]))
        self.assertEqual(result["shapes"]["接成一行"], [8])
        code, result = run_json(window_lab, "--trace", "--window", "2", "--prefix", "请求 失败 后")
        self.assertEqual((code, result["prediction"], result["slots"]), (0, "重试", ["后", "失败"]))
        code, result = run_json(window_lab, "--hand")
        self.assertEqual([row["differ"] for row in result["rows"]], [False, True, True])
        self.assertEqual(result["rows"][1]["lines"][0]["input"], [1, 1, 1, 0])

    def test_count_mode(self):
        code, result = run_json(window_lab, "--count")
        self.assertEqual([(row["positions"], row["window"], row["attention"]) for row in result["rows"]], [(4, 77, 77)])
        code, result = run_json(window_lab, "--count", "--context", "8")
        self.assertEqual([(row["positions"], row["window"], row["attention"]) for row in result["rows"]], [(4, 77, 77), (8, 133, 85)])
        self.assertEqual(result["rows"][0]["attention"], lm.parameter_count(2, 4))

    def test_record_mode_and_refusals(self):
        code, result = run_json(window_lab, "--record")
        self.assertEqual((code, result["problems"]), (0, []))
        code, result = run_json(window_lab, "--trace", "--prefix", "你好")
        self.assertEqual(code, 2)
        self.assertIn("词表里没有", result["error"])
        code, result = run_json(window_lab, "--trace", "--window", "3")
        self.assertEqual(code, 2)
        code, result = run_json(window_lab, "--write-record")
        self.assertEqual((code, result["written"]), (2, False))


class CostCountingTest(unittest.TestCase):
    def test_both_ways_give_exactly_the_probabilities_of_the_model(self):
        for parameters in (TRAINED, lm.initial_parameters(7)):
            for pair in PAIRS:
                cache, spent = cost.new_cache(), cost.new_tally()
                for length in range(1, len(pair["inputs"]) + 1):
                    ids = pair["inputs"][:length]
                    wanted = lm.forward(parameters, ids)["probabilities"]
                    by_position, _ = cost.recompute_all(parameters, ids)
                    self.assertEqual(by_position, wanted)
                    self.assertEqual(cost.extend(parameters, cache, ids[-1], spent), wanted[-1])

    def test_counts_for_the_seven_word_model(self):
        cache = cost.new_cache()
        for length, recompute_total, cached_total in ((1, 46, 46), (2, 96, 50), (3, 150, 54), (4, 208, 58)):
            ids = PAIRS[0]["inputs"][:length]
            _, tally = cost.recompute_all(TRAINED, ids)
            self.assertEqual(sum(tally.values()), recompute_total)
            spent = cost.new_tally()
            cost.extend(TRAINED, cache, ids[-1], spent)
            self.assertEqual(sum(spent.values()), cached_total)
            self.assertEqual(spent["scoring"] + spent["reading"], 4 * length)
            self.assertEqual(cost.cached_numbers(cache), 4 * length)

    def test_counts_while_running_match_counts_by_structure(self):
        for seed, dimension, hidden, context in ((11, 2, 4, 12), (12, 3, 5, 9)):
            parameters, ids = longer_model(seed, dimension, hidden, context)
            cache = cost.new_cache()
            for length in range(1, context + 1):
                by_position, tally = cost.recompute_all(parameters, ids[:length])
                spent = cost.new_tally()
                last = cost.extend(parameters, cache, ids[length - 1], spent)
                self.assertEqual(last, by_position[-1])
                self.assertEqual(tally, cost.recompute_cost(length, dimension, hidden, len(lm.VOCAB)))
                self.assertEqual(spent, cost.cached_cost(length, 1, dimension, hidden, len(lm.VOCAB)))

    def test_generation_is_the_same_as_the_model_and_only_the_work_differs(self):
        for prefix, temperature, seed in (("请求", 0.0, 0), ("请求 成功", 0.0, 0), ("请求", 1.0, 2), ("请求", 1.0, 5), ("请求 失败 后", 0.7, 1)):
            ids = lm.encode(prefix)
            wanted = lm.generate(TRAINED, ids, temperature, seed)
            for cached in (False, True):
                got = cost.generate_counted(TRAINED, ids, cached, temperature, seed)
                self.assertEqual((got["ids"], got["stopped_by"]), (wanted["ids"], wanted["stopped_by"]))
                self.assertEqual([step["probabilities"] for step in got["steps"]], [step["probabilities"] for step in wanted["steps"]])
        recompute = cost.generate_counted(TRAINED, [0], False)
        cached = cost.generate_counted(TRAINED, [0], True)
        self.assertEqual([step["total"] for step in recompute["steps"]], [46, 96, 150, 208])
        self.assertEqual([step["total"] for step in cached["steps"]], [46, 50, 54, 58])
        self.assertEqual([step["cached_numbers"] for step in cached["steps"]], [4, 8, 12, 16])

    def test_a_longer_prompt_is_taken_in_once(self):
        cached = cost.generate_counted(TRAINED, lm.encode("请求 成功"), True)
        # 提示里的第一个词只算键和值（8 次），第二个词算完整的一份（42 + 4×2 = 50 次）。
        self.assertEqual([step["total"] for step in cached["steps"]], [58, 54, 58])
        self.assertEqual(cached["steps"][0]["multiplications"]["keys_values"], 16)
        self.assertEqual(cost.cached_cost(2, 2, 2, 4, 7), cached["steps"][0]["multiplications"])

    def test_generation_stops_when_the_context_is_full(self):
        for cached in (False, True):
            outcome = cost.generate_counted(lm.hand_set_parameters(), [0], cached)
            self.assertEqual(outcome["stopped_by"], "前文已读满")
            self.assertEqual(len(outcome["ids"]), lm.CONTEXT + 1)
        with self.assertRaises(ValueError):
            cost.generate_counted(TRAINED, [], True)
        with self.assertRaises(ValueError):
            cost.recompute_all(TRAINED, [0, 1, 3, 4, 6])

    def test_projection_by_structure(self):
        short = cost.projection(1, 4, 2, 4, 7)
        self.assertEqual((short["recompute_total"], short["cached_total"], short["cached_numbers_at_end"]), (500, 208, 16))
        for last in (1, 4, 12, 1000):
            result = cost.projection(1, last, 2, 4, 7)
            # 留着中间结果、从一个词开始：每个位置的那一份计算恰好各做一次，等于把最后那段前文完整算一遍。
            self.assertEqual(result["cached_total"], result["recompute_last_step"])
            self.assertEqual(result["cached_total"], 42 * last + 2 * last * (last + 1))
            self.assertEqual(result["cached_last_step"], 42 + 4 * last)
            self.assertLessEqual(result["cached_total"], result["recompute_total"])
        long = cost.projection(1, 1000, 2, 4, 7)
        self.assertEqual((long["recompute_total"], long["cached_total"], long["cached_last_step"]), (689689000, 2044000, 4042))
        parameters, ids = longer_model(11, 2, 4, 12)
        ran = [cost.generate_counted(parameters, ids[:1], cached) for cached in (False, True)]
        steps = len(ran[0]["steps"])
        by_structure = cost.projection(1, steps, 2, 4, 7)
        self.assertEqual(sum(step["total"] for step in ran[0]["steps"]), by_structure["recompute_total"])
        self.assertEqual(sum(step["total"] for step in ran[1]["steps"]), by_structure["cached_total"])
        with self.assertRaises(ValueError):
            cost.projection(0, 4, 2, 4, 7)


class CostLabTest(unittest.TestCase):
    def test_default_run(self):
        code, result = run_json(cost_lab)
        self.assertEqual((code, result["identical"], result["stopped_by"]), (0, True, "EOS"))
        self.assertEqual((result["recompute_total"], result["cached_total"]), (500, 208))
        self.assertEqual([step["positions"] for step in result["steps"]], [1, 2, 3, 4])
        self.assertEqual([step["cached"]["reads"] for step in result["steps"]], [4, 8, 12, 16])
        code, text = run_text(cost_lab)
        self.assertIn("不是耗时，也不是模型服务的账单", text)
        self.assertIn("exit_code: 0", text)

    def test_breakdown_adds_up_and_matches_structure(self):
        code, result = run_json(cost_lab, "--breakdown")
        self.assertEqual((code, result["identical"], result["matches_structure"]), (0, True, True))
        self.assertEqual((result["positions"], result["recompute_total"], result["cached_total"]), (3, 150, 54))
        self.assertEqual(sum(part["recompute"] for part in result["parts"]), 150)
        self.assertEqual([part["cached"] for part in result["parts"]], [8, 4, 6, 6, 16, 14])

    def test_projection_mode_and_refusals(self):
        code, result = run_json(cost_lab, "--project", "1000")
        self.assertEqual((code, result["recompute_total"], result["cached_total"]), (0, 689689000, 2044000))
        code, text = run_text(cost_lab, "--project", "1000")
        self.assertIn("没有运行过", text)
        code, result = run_json(cost_lab, "--project", "8", "--dimension", "4", "--hidden", "8")
        self.assertEqual(result["cached_last_step"], 3 * 16 + 2 * 8 * 4 + 2 * 4 * 8 + 4 * 7)
        code, result = run_json(cost_lab, "--prefix", "你好")
        self.assertEqual(code, 2)
        self.assertIn("词表里没有", result["error"])
        code, result = run_json(cost_lab, "--prefix", "请求 成功 后 返回 EOS")
        self.assertEqual(code, 2)
        code, result = run_json(cost_lab, "--project", "0")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

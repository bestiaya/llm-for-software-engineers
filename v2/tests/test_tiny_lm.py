"""验证阶段三的实验底座：七词小模型从编号到概率的计算、因果遮罩、梯度、生成和训练记录。全部离线，只用标准库。"""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest


COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))

from shared import tiny_lm as lm  # noqa: E402


def load_lab():
    spec = importlib.util.spec_from_file_location("forward_lab", COURSE / "14-forward" / "lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab = load_lab()
PAIRS = lm.training_pairs()
RECORD = json.loads((COURSE / "data/tiny-lm-training-record.json").read_text(encoding="utf-8"))
TRAINED = RECORD["final_parameters"]


def run_json(*args):
    with redirect_stdout(io.StringIO()) as stream:
        code = lab.main([*args, "--json"])
    return code, json.loads(stream.getvalue())


class TextAndTargetsTest(unittest.TestCase):
    def test_encoding_round_trip(self):
        self.assertEqual(lm.encode("请求 成功 后 返回 EOS"), [0, 1, 3, 4, 6])
        self.assertEqual(lm.decode([0, 2, 3, 5, 6]), "请求 失败 后 重试 EOS")

    def test_targets_are_inputs_shifted_by_one(self):
        self.assertEqual(PAIRS[0]["inputs"], [0, 1, 3, 4])
        self.assertEqual(PAIRS[0]["targets"], [1, 3, 4, 6])
        for pair in PAIRS:
            self.assertEqual(pair["inputs"][1:], pair["targets"][:-1])

    def test_loss_floor_and_unigram_level(self):
        # 只有“请求”后面有两种接法，各占一半；8 个位置里有 2 个这样的位置。
        self.assertAlmostEqual(lm.loss_floor(PAIRS), 2 * math.log(2) / 8, places=12)
        self.assertAlmostEqual(lab.unigram_level(PAIRS), 1.732868, places=6)


class ModelTest(unittest.TestCase):
    def test_parameter_count_and_round_trip(self):
        self.assertEqual(lm.parameter_count(2, 4), 77)
        self.assertEqual(lm.parameter_count(4, 8), 203)
        parameters = lm.initial_parameters(3)
        self.assertEqual(len(lm.flatten(parameters)), 77)
        self.assertEqual(lm.unflatten(lm.flatten(parameters), 2, 4), parameters)
        self.assertEqual(lm.initial_parameters(3), parameters)

    def test_each_position_gives_a_distribution_over_the_vocabulary(self):
        state = lm.forward(lm.initial_parameters(3), PAIRS[0]["inputs"])
        self.assertEqual(len(state["probabilities"]), 4)
        for t, row in enumerate(state["probabilities"]):
            self.assertEqual(len(row), len(lm.VOCAB))
            self.assertAlmostEqual(sum(row), 1.0, places=12)
            self.assertEqual(len(state["weights"][t]), t + 1)
            self.assertAlmostEqual(sum(state["weights"][t]), 1.0, places=12)

    def test_prefix_length_is_limited(self):
        with self.assertRaises(ValueError):
            lm.forward(TRAINED, [])
        with self.assertRaises(ValueError):
            lm.forward(TRAINED, [0, 1, 3, 4, 6])

    def test_later_tokens_do_not_change_earlier_predictions(self):
        for parameters in (lm.initial_parameters(7), TRAINED):
            first = lm.forward(parameters, lm.encode("请求 成功 后 返回"))["probabilities"]
            second = lm.forward(parameters, lm.encode("请求 成功 后 重试"))["probabilities"]
            self.assertEqual(first[:3], second[:3])
            self.assertNotEqual(first[3], second[3])
            shorter = lm.forward(parameters, lm.encode("请求 成功"))["probabilities"]
            self.assertEqual(shorter, first[:2])

    def test_hand_set_attention_example(self):
        parameters = lm.hand_set_parameters()
        state = lm.forward(parameters, lm.encode("请求 成功 后"))
        for got, wanted in zip(state["matches"][-1], [0.0, 1 / math.sqrt(2), 2 / math.sqrt(2)], strict=True):
            self.assertAlmostEqual(got, wanted, places=12)
        for got, wanted in zip(state["weights"][-1], [0.140, 0.284, 0.576], strict=True):
            self.assertAlmostEqual(got, wanted, places=3)
        for got, wanted in zip(state["attended"][-1], [0.860, 0.576], strict=True):
            self.assertAlmostEqual(got, wanted, places=3)

    def test_gradient_by_program_matches_nudging(self):
        for seed, dimension, hidden in ((3, 2, 4), (5, 4, 8)):
            parameters = lm.initial_parameters(seed, dimension, hidden)
            by_program = lm.flatten(lm.analytic_gradient(parameters, PAIRS))
            by_nudging = lm.flatten(lm.numeric_gradient(parameters, PAIRS))
            self.assertLess(max(abs(a - n) for a, n in zip(by_program, by_nudging, strict=True)), 1e-7)

    def test_one_update_lowers_the_loss(self):
        start = lm.initial_parameters(3)
        after = lm.update(start, lm.analytic_gradient(start, PAIRS), 0.05)
        self.assertLess(lm.mean_loss(after, PAIRS), lm.mean_loss(start, PAIRS))


class GenerationTest(unittest.TestCase):
    def test_greedy_generation_completes_both_sentences(self):
        self.assertEqual(lm.decode(lm.generate(TRAINED, lm.encode("请求 成功"))["ids"]), "请求 成功 后 返回 EOS")
        self.assertEqual(lm.decode(lm.generate(TRAINED, lm.encode("请求 失败"))["ids"]), "请求 失败 后 重试 EOS")

    def test_first_step_stays_split_between_the_two_continuations(self):
        probabilities = lm.forward(TRAINED, lm.encode("请求"))["probabilities"][-1]
        self.assertAlmostEqual(probabilities[1] + probabilities[2], 1.0, places=3)
        self.assertGreater(min(probabilities[1], probabilities[2]), 0.4)

    def test_sampling_reaches_both_sentences_and_is_repeatable(self):
        texts = {lm.decode(lm.generate(TRAINED, [0], 1.0, seed)["ids"]) for seed in range(8)}
        self.assertEqual(texts, set(lm.SENTENCES))
        self.assertEqual(lm.generate(TRAINED, [0], 1.0, 5), lm.generate(TRAINED, [0], 1.0, 5))

    def test_generation_stops_when_the_context_is_full(self):
        outcome = lm.generate(lm.hand_set_parameters(), [0])     # 输出全为 0，并列时取编号最小的“请求”
        self.assertEqual(outcome["stopped_by"], "前文已读满")
        self.assertEqual(len(outcome["ids"]), lm.CONTEXT + 1)


class RecordAndLabTest(unittest.TestCase):
    def test_record_checks_out_without_retraining(self):
        problems, largest = lab.check_record(RECORD)
        self.assertEqual(problems, [])
        self.assertIsNone(largest)

    def test_record_tells_the_whole_story(self):
        losses = {item["step"]: item["loss"] for item in RECORD["checkpoints"]}
        self.assertEqual(sorted(losses), lab.CONFIG["checkpoints"])
        self.assertAlmostEqual(losses[0], 1.9257, places=4)
        # 前一千步停在“不看前文”的水平附近，之后才降下来，最后贴近下限但到不了。
        self.assertLess(abs(losses[1000] - RECORD["unigram_level"]), 0.01)
        self.assertLess(losses[2000], 0.5)
        self.assertGreater(losses[20000], RECORD["loss_floor"])
        self.assertLess(losses[20000] - RECORD["loss_floor"], 0.001)
        self.assertEqual([item["step"] for item in RECORD["outputs"]], [0, 1000, 20000])
        plateau = RECORD["outputs"][1]["by_prefix"]
        self.assertEqual({row["next"][0]["token"] for row in plateau}, {"EOS"})
        self.assertTrue(all(row["matches"] for row in RECORD["completions"]))

    def test_first_update_in_the_record_can_be_recomputed(self):
        for item in RECORD["first_update"]:
            self.assertAlmostEqual(item["gradient_by_program"], item["gradient_by_nudging"], places=7)
            self.assertAlmostEqual(item["after"], item["before"] - 0.05 * item["gradient_by_program"], places=12)

    def test_lab_modes(self):
        code, result = run_json()
        self.assertEqual((code, result["prediction"]), (0, "返回"))
        self.assertEqual(result["shapes"]["分数"], [7])
        code, result = run_json("--prefix", "请求 失败 后")
        self.assertEqual((code, result["prediction"]), (0, "重试"))
        code, result = run_json("--generate", "--prefix", "请求 成功")
        self.assertEqual((code, result["text"], result["stopped_by"]), (0, "请求 成功 后 返回 EOS", "EOS"))
        code, result = run_json("--mask-check")
        self.assertEqual((code, result["holds"]), (0, True))
        code, result = run_json("--hand")
        self.assertEqual((result["last_only_same"], result["summed_differ"], result["attended_differ"]), (True, True, True))
        code, result = run_json("--count")
        self.assertEqual((result["total"], result["bytes_as_float32"]), (77, 308))
        code, result = run_json("--count", "--dimension", "4", "--hidden", "8")
        self.assertEqual(result["total"], 203)
        code, result = run_json("--record")
        self.assertEqual((code, result["problems"]), (0, []))
        code, result = run_json("--show", "W1")
        self.assertEqual((code, result["shape"], len(result["rows"])), (0, [2, 4], 2))

    def test_order_enters_through_the_position_vectors(self):
        code, result = run_json("--order-check")
        self.assertEqual((code, result["holds"]), (0, True))
        with_positions, without_positions = result["rows"]
        self.assertGreater(with_positions["largest_difference"], 0.001)
        self.assertEqual(without_positions["largest_difference"], 0.0)

    def test_feed_forward_step_can_be_followed_by_hand(self):
        # 草稿里拿来手算的那一步：读到的向量乘上 W1、加上 b1、过 ReLU，得到四个隐藏值。
        state = lm.forward(TRAINED, lm.encode("请求 成功 后"))
        attended = state["attended"][-1]
        for unit in range(4):
            before = sum(attended[i] * TRAINED["W1"][i][unit] for i in range(2)) + TRAINED["b1"][unit]
            self.assertAlmostEqual(state["hidden"][-1][unit], max(before, 0.0), places=12)
        self.assertEqual([value > 0 for value in state["hidden"][-1]], [True, False, True, False])

    def test_lab_rejects_unknown_words_and_refuses_to_overwrite(self):
        code, result = run_json("--prefix", "你好")
        self.assertEqual(code, 2)
        self.assertIn("词表里没有", result["error"])
        code, result = run_json("--write-record")
        self.assertEqual((code, result["written"]), (2, False))


if __name__ == "__main__":
    unittest.main()

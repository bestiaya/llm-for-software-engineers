"""第 07、08 章的实验，以及第 10 章新增的几种看法：线性模型的限制、隐藏层的路径、概率与损失、阶段检查。全部离线。"""

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

from shared import network as nw  # noqa: E402
from shared.numeric import linear_forward  # noqa: E402


def load_lab(name, chapter):
    spec = importlib.util.spec_from_file_location(name, COURSE / chapter / "lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hidden_lab = load_lab("hidden_lab", "07-hidden")
loss_lab = load_lab("loss_lab", "08-loss")
training_lab = load_lab("training_lab_extra", "10-training")
ROWS = json.loads((COURSE / "data/counter-task.json").read_text(encoding="utf-8"))["rows"]


def run_json(lab, *args):
    with redirect_stdout(io.StringIO()) as stream:
        code = lab.main([*args, "--json"])
    return code, json.loads(stream.getvalue())


def choose(scores):
    return max(range(len(scores)), key=lambda k: (scores[k], -k))


class SharedNetworkTest(unittest.TestCase):
    def test_set_parameter_returns_a_changed_copy(self):
        original = nw.hand_set_parameters()
        changed = nw.set_parameter(original, "b1[h2]=-3")
        self.assertEqual(original["b1"][1], -2.0)
        self.assertEqual(changed["b1"][1], -3.0)
        self.assertEqual(nw.set_parameter(original, "W1[b,h4]=2")["W1"][1][3], 2.0)
        self.assertEqual(nw.set_parameter(original, "W2[h1,2]=0.5")["W2"][0][2], 0.5)
        self.assertEqual(nw.set_parameter(original, "b2[1]=7")["b2"][1], 7.0)
        for bad in ("b1[h5]=1", "W1[c,h1]=1", "b2[1]", "b2[1]=abc", "nonsense=1", "b2[1]=inf"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                nw.set_parameter(original, bad)

    def test_scaling_the_output_layer_scales_the_scores(self):
        original = nw.hand_set_parameters()
        scaled = nw.scale_output(original, 3)
        for row in ROWS:
            x = (row["a"], row["b"])
            self.assertEqual(nw.forward(scaled, x)["scores"], [3 * score for score in nw.forward(original, x)["scores"]])
        self.assertEqual(original, nw.hand_set_parameters())

    def test_without_relu_two_layers_merge_into_one(self):
        # 选读里的推导：合并后的权重是 W1·W2，偏置是 b1·W2 + b2。
        p = nw.hand_set_parameters()
        merged_w = [[sum(p["W1"][i][j] * p["W2"][j][k] for j in range(4)) for k in range(3)] for i in range(2)]
        merged_b = [sum(p["b1"][j] * p["W2"][j][k] for j in range(4)) + p["b2"][k] for k in range(3)]
        self.assertEqual(merged_w, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertEqual(merged_b, [-3.0, 0.0, 1.0])
        for row in ROWS:
            x = (row["a"], row["b"])
            self.assertEqual(nw.forward(p, x, relu=False)["scores"], linear_forward([list(x)], merged_w, merged_b)[0])


class HiddenLabTest(unittest.TestCase):
    def test_linear_model_averages_between_the_two_ends(self):
        for arguments in ((), ("--seed", "7"), ("--seed", "3")):
            with self.subTest(arguments=arguments):
                code, result = run_json(hidden_lab, "--linear-limit", *arguments)
                self.assertEqual(code, 0)
                self.assertAlmostEqual(result["mean_of_ends"], result["middle"], places=12)
                self.assertLess(result["correct"], 3)
                self.assertEqual((result["tries"], result["tries_all_right"], result["tries_best"]), (10000, 0, 2))
        code, result = run_json(hidden_lab, "--linear-limit")
        self.assertEqual([item["scores"] for item in result["rows"]], [[-2, 0.5, 3], [1, 2.5, -1], [-0.5, 1.5, 1]])
        self.assertEqual([item["margin"] for item in result["rows"]], [2.5, 1.5, 2])
        # 正文的表：中间一行的每个分数都是两端的平均。
        first, second, middle = (item["scores"] for item in result["rows"])
        self.assertEqual([(a + b) / 2 for a, b in zip(first, second)], middle)

    def test_a_better_input_representation_makes_the_linear_model_enough(self):
        # 正文：把输入换成“是否达到 2 次”的两个 0/1 数，权重取 -2、0、2，偏置取 1、0、-3，16 行全对。
        weights, bias = [[-2, 0, 2], [-2, 0, 2]], [1, 0, -3]
        for row in ROWS:
            scores = linear_forward([[int(row["a"] >= 2), int(row["b"] >= 2)]], weights, bias)[0]
            self.assertEqual(choose(scores), row["label"])

    def test_relu_units_turn_a_count_into_a_step(self):
        # 正文的表：ReLU(a-1) 减 ReLU(a-2) 正好是“是否达到 2 次”。
        relu = lambda value: max(0, value)
        self.assertEqual([relu(a - 1) for a in range(4)], [0, 0, 1, 2])
        self.assertEqual([relu(a - 2) for a in range(4)], [0, 0, 0, 1])
        self.assertEqual([relu(a - 1) - relu(a - 2) for a in range(4)], [0, 0, 1, 1])

    def test_trace_of_the_rows_used_in_the_lesson(self):
        cases = {
            (): ([-1, -2, 1, 0], [0, 0, 1, 0], [-1, 0, -1], 1, 1),
            ("--a", "3", "--b", "1"): ([2, 1, 0, -1], [2, 1, 0, 0], [-1, 0, -1], 1, 1),
            ("--a", "2", "--b", "2"): ([1, 0, 1, 0], [1, 0, 1, 0], [-3, 0, 1], 2, 2),
        }
        for arguments, (pre, hidden, scores, prediction, label) in cases.items():
            with self.subTest(arguments=arguments):
                code, result = run_json(hidden_lab, *arguments)
                self.assertEqual(code, 0)
                self.assertEqual([term["pre"] for term in result["hidden_terms"]], pre)
                self.assertEqual((result["hidden"], result["scores"]), (hidden, scores))
                self.assertEqual((result["prediction"], result["label_by_rule"]), (prediction, label))

    def test_all_rows_with_and_without_relu(self):
        code, result = run_json(hidden_lab, "--all")
        self.assertEqual((code, result["correct"]), (0, 16))
        code, result = run_json(hidden_lab, "--all", "--no-relu")
        self.assertEqual((result["correct"], result["distinct_scores"]), (4, 1))
        self.assertEqual(result["rows"][0]["scores"], [-3, 0, 1])
        code, result = run_json(hidden_lab, "--all", "--set", "b1[h2]=-3")
        self.assertEqual(result["correct"], 14)
        code, result = run_json(hidden_lab, "--set", "b9=1")
        self.assertEqual(code, 2)


class LossLabTest(unittest.TestCase):
    def test_scores_to_probabilities(self):
        code, result = run_json(loss_lab)
        self.assertEqual(code, 0)
        for got, want in zip(result["powers"], (1, 2, 3), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        for got, want in zip(result["probabilities"], (1 / 6, 2 / 6, 3 / 6), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        code, result = run_json(loss_lab, "--scores=-1,0,-1", "--label", "1")
        self.assertEqual([f"{value:.3f}" for value in result["powers"]], ["0.368", "1.000", "0.368"])
        self.assertEqual(f"{result['total']:.3f}", "1.736")
        self.assertEqual([f"{value:.3f}" for value in result["probabilities"]], ["0.212", "0.576", "0.212"])
        self.assertEqual(f"{result['loss']:.3f}", "0.551")
        # 只有分数之间的差距起作用；差距拉大，份额更悬殊。
        for same in ("--scores=0,1,0", "--scores=4,5,4"):
            code, other = run_json(loss_lab, same)
            for got, want in zip(other["probabilities"], result["probabilities"], strict=True):
                self.assertAlmostEqual(got, want, places=12)
        code, wide = run_json(loss_lab, "--scores=-3,0,-3", "--label", "1")
        self.assertEqual((f"{wide['probability_of_label']:.3f}", f"{wide['loss']:.3f}"), ("0.909", "0.095"))
        code, even = run_json(loss_lab, "--scores=0,0,0", "--label", "1")
        self.assertEqual(f"{even['loss']:.3f}", "1.099")

    def test_probabilities_and_loss_stay_correct_for_large_scores(self):
        code, base = run_json(loss_lab, "--scores=-1,0,-1", "--label", "1")
        self.assertEqual(base["shift"], 0.0)
        for shifted in ("--scores=1000,1001,1000", "--scores=-1000,-999,-1000"):
            code, result = run_json(loss_lab, shifted, "--label", "1")
            self.assertEqual(code, 0)
            self.assertNotEqual(result["shift"], 0.0)
            for got, want in zip(result["probabilities"], base["probabilities"], strict=True):
                self.assertAlmostEqual(got, want, places=12)
            self.assertAlmostEqual(result["loss"], base["loss"], places=9)
        # 正确类别的概率小到算成了 0，损失仍然算得出来：就是分数差 1000。
        code, result = run_json(loss_lab, "--scores=0,-1000,-1000", "--label", "1")
        self.assertEqual((code, result["probability_of_label"], result["prediction"]), (0, 0.0, 0))
        self.assertAlmostEqual(result["loss"], 1000.0, places=9)
        for scores, label in (([-1.0, 0.0, -1.0], 1), ([3.0, -2.0, 0.5], 2), ([0.0, 0.0, 0.0], 0)):
            self.assertAlmostEqual(nw.loss_from_scores(scores, label), -math.log(nw.softmax(scores)[label]), places=12)

    def test_exponent_and_logarithm_tables_in_the_lesson(self):
        self.assertEqual([f"{math.exp(x):.3f}" for x in (-2, -1, 0, 1, 2)], ["0.135", "0.368", "1.000", "2.718", "7.389"])
        self.assertEqual(f"{math.log(2):.3f} {math.log(3):.3f}", "0.693 1.099")
        code, result = run_json(loss_lab, "--losses=0.99,0.75,0.5,0.25,0.1,0.01")
        self.assertEqual([f"{value:.3f}" for value in result["losses"]], ["0.010", "0.288", "0.693", "1.386", "2.303", "4.605"])
        code, result = run_json(loss_lab, "--losses=0.25,0.5,0.75")
        self.assertEqual(f"{result['mean']:.3f}", "0.789")
        code, result = run_json(loss_lab, "--losses=0,1")
        self.assertEqual(code, 2)

    def test_same_choices_different_probabilities(self):
        code, plain = run_json(loss_lab, "--all")
        code, scaled = run_json(loss_lab, "--all", "--scale", "3")
        self.assertEqual((plain["correct"], scaled["correct"]), (16, 16))
        self.assertEqual([row["prediction"] for row in plain["rows"]], [row["prediction"] for row in scaled["rows"]])
        self.assertEqual(sorted({f"{row['probability_of_label']:.3f}" for row in plain["rows"]}), ["0.576", "0.721"])
        self.assertEqual(sorted({f"{row['probability_of_label']:.3f}" for row in scaled["rows"]}), ["0.909", "0.953"])
        self.assertEqual((f"{plain['mean_loss']:.4f}", f"{scaled['mean_loss']:.4f}"), ("0.4390", "0.0718"))
        self.assertEqual(sum(f"{row['loss']:.3f}" == "0.327" for row in plain["rows"]), 8)
        self.assertEqual(sum(f"{row['loss']:.3f}" == "0.551" for row in plain["rows"]), 8)

    def test_one_row_through_the_network(self):
        code, result = run_json(loss_lab, "--row", "0", "2")
        self.assertEqual((code, result["scores"], result["label"]), (0, [-1, 0, -1], 1))
        self.assertAlmostEqual(result["loss"], 0.551445, places=6)

    def test_counts_and_loss_can_disagree(self):
        code, result = run_json(loss_lab, "--disagree")
        first, second = result["models"]
        self.assertEqual((first["correct"], f"{first['mean_loss']:.3f}"), (3, "0.916"))
        self.assertEqual((second["correct"], f"{second['mean_loss']:.3f}"), (2, "0.472"))

    def test_confident_mistakes_cost_more(self):
        code, plain = run_json(loss_lab, "--all", "--set", "b1[h2]=-3")
        code, scaled = run_json(loss_lab, "--all", "--set", "b1[h2]=-3", "--scale", "3")
        self.assertEqual((plain["correct"], scaled["correct"]), (14, 14))
        wrong = lambda result: [row for row in result["rows"] if row["prediction"] != row["label"]]
        self.assertEqual([row["id"] for row in wrong(plain)], ["a3-b0", "a3-b1"])
        self.assertEqual({f"{row['loss']:.2f}" for row in wrong(plain)}, {"1.33"})
        self.assertEqual({f"{row['loss']:.2f}" for row in wrong(scaled)}, {"3.05"})
        self.assertEqual({f"{row['probability_of_label']:.3f}" for row in wrong(scaled)}, {"0.047"})


class TrainingViewsTest(unittest.TestCase):
    def test_what_the_network_does_along_the_way(self):
        expected = {0: (2, [0]), 5: (6, [1]), 10: (6, [1]), 20: (6, [1])}
        for step, (correct, predictions) in expected.items():
            code, result = run_json(training_lab, "--at", str(step))
            self.assertEqual((code, result["correct"], result["predictions"]), (0, correct, predictions))
        code, result = run_json(training_lab, "--at", "100")
        self.assertEqual([row["x"] for row in result["rows"] if row["prediction"] != row["label"]], [[1, 1]])
        code, result = run_json(training_lab, "--at", "50")
        self.assertEqual(result["correct"], 7)
        code, result = run_json(training_lab, "--at", "9999")
        self.assertEqual(code, 2)

    def test_training_loss_falls_overall_but_not_at_every_step(self):
        task = nw.load_task(COURSE / "data/counter-task.json")
        losses = []
        nw.train(nw.initial_parameters(1), task["train"], 0.1, 2000, lambda step, p: losses.append(nw.mean_loss(p, task["train"])))
        self.assertEqual(sum(later > earlier for earlier, later in zip(losses, losses[1:])), 349)
        checkpoints = [losses[step] for step in training_lab.CONFIG["checkpoints"]]
        self.assertEqual(checkpoints, sorted(checkpoints, reverse=True))

    def test_gradient_check_details_quoted_in_chapter_nine(self):
        update_lab = load_lab("update_lab_extra", "09-update")
        code, result = run_json(update_lab, "--check")
        zero = [row["parameter"] for row in result["table"] if row["by_program"] == 0]
        self.assertEqual(len(zero), 6)
        self.assertTrue(all("h1" in name for name in zero))
        self.assertEqual(f"{result['kink_margin']:.4f}", "0.0210")
        self.assertLess(result["largest_difference"], 1e-9)
        # 那组把 16 行做对的手设参数落在折点上，所以不用来核对梯度。
        task = nw.load_task(COURSE / "data/counter-task.json")
        self.assertEqual(nw.kink_margin(nw.hand_set_parameters(), task["train"]), 0)
        code, bias = run_json(update_lab)
        self.assertEqual(f"{bias['before']['loss'] - bias['after']['loss']:.6f}", "0.189694")

    def test_the_validation_row_that_gets_worse(self):
        task = nw.load_task(COURSE / "data/counter-task.json")
        seen = {}
        nw.train(nw.initial_parameters(1), task["train"], 0.1, 2000, lambda step, p: seen.update({step: nw.forward(p, (0, 1))["probabilities"][0]}) if step in (500, 2000) else None)
        self.assertEqual((f"{seen[500]:.3f}", f"{seen[2000]:.3f}"), ("0.467", "0.381"))

    def test_trained_parameters_are_not_readable(self):
        code, result = run_json(training_lab, "--inspect")
        self.assertEqual((code, result["step"]), (0, 500))
        self.assertEqual(result["correct"], {"train": 10, "validation": 2, "heldout": 3})
        first = result["units"][0]
        self.assertTrue(first["always_zero"] and first["unchanged"])
        self.assertFalse(any(unit["always_zero"] for unit in result["units"][1:]))
        shown = [f"{unit['weight_a']:.3f} {unit['weight_b']:+.3f} {unit['bias']:+.3f}" for unit in result["units"]]
        self.assertEqual(shown, ["-0.366 -0.005 -0.406", "-0.127 +1.905 -1.202", "-1.115 +0.422 +2.237", "-1.887 +1.882 -0.004"])

    def test_linear_model_and_seed_survey_numbers(self):
        task = nw.load_task(COURSE / "data/counter-task.json")
        linear = training_lab.linear_training(task)
        self.assertEqual((linear["train_correct"], linear["wrong_rows"], f"{linear['train_loss']:.4f}"), (8, [[1, 1], [2, 2]], "0.3434"))
        survey = training_lab.seed_survey(task, 20)
        self.assertEqual((survey["train_all_correct"], survey["all_unseen_correct"]), (12, 0))
        unseen_five = sum(item["runs"] for item in survey["table"] if item["validation_correct"] + item["heldout_correct"] == 5)
        stuck = sum(item["runs"] for item in survey["table"] if (item["train_correct"], item["validation_correct"], item["heldout_correct"]) == (6, 1, 1))
        self.assertEqual((unseen_five, stuck), (6, 2))
        best = max(item["validation_correct"] + item["heldout_correct"] for item in survey["table"])
        self.assertEqual(best, 5)

    def test_stage_check_reference_answer(self):
        code, result = run_json(training_lab, "--stage-check")
        self.assertEqual((code, result["change"], result["correct_before"], result["correct_after"]), (0, "b1[h1]=-2", 16, 8))
        # a 是 2 或 3 的八行都变了：b 小于 2 的从类别 1 变成 0，b 达到 2 的从类别 2 变成 1。
        self.assertEqual(
            [(item["x"], item["before"], item["after"]) for item in result["flipped"]],
            [([a, b], 1 if b < 2 else 2, 0 if b < 2 else 1) for a in (2, 3) for b in range(4)],
        )
        self.assertEqual((result["row"]["x"], result["row"]["label"]), ([2, 0], 1))
        self.assertEqual((result["original"]["hidden"], result["original"]["scores"]), ([1, 0, 0, 0], [-1, 0, -1]))
        self.assertEqual((result["changed"]["hidden"], result["changed"]["scores"]), ([0, 0, 0, 0], [1, 0, -3]))
        self.assertEqual((f"{result['original']['loss']:.4f}", f"{result['changed']['loss']:.4f}"), ("0.5514", "1.3266"))
        self.assertEqual([f"{value:.4f}" for value in result["original"]["probabilities"]][1], "0.5761")
        self.assertEqual([f"{value:.4f}" for value in result["changed"]["probabilities"]], ["0.7214", "0.2654", "0.0132"])
        self.assertEqual([f"{value:+.4f}" for value in result["bias_gradient"]], ["+0.7214", "-0.7346", "+0.0132"])
        self.assertEqual([f"{value:.4f}" for value in result["new_bias"]], ["0.7836", "0.2204", "-3.0040"])
        after = result["after_one_update"]
        self.assertEqual((f"{after['probabilities'][1]:.4f}", f"{after['loss']:.4f}", after["prediction"]), ("0.3576", "1.0282", 0))
        # 这道题用的改动，前面各章的正文和说明里都没有演示过。
        for chapter in ("07-hidden", "08-loss", "09-update"):
            for name in ("lesson.md", "README.md"):
                self.assertNotIn("b1[h1]", (COURSE / chapter / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

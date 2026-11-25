"""验证阶段二的实验底座：2-4-3 小网络的计算、梯度、一次更新和训练记录。全部离线，只用标准库。"""

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


def load_lab(name, chapter):
    spec = importlib.util.spec_from_file_location(name, COURSE / chapter / "lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


update_lab = load_lab("update_lab", "09-update")
training_lab = load_lab("training_lab", "10-training")
TASK = nw.load_task(COURSE / "data/counter-task.json")
ALL_ROWS = TASK["train"] + TASK["validation"] + TASK["heldout"]


def run_json(lab, *args):
    with redirect_stdout(io.StringIO()) as stream:
        code = lab.main([*args, "--json"])
    return code, json.loads(stream.getvalue())


class NetworkTest(unittest.TestCase):
    def test_shapes_and_round_trip(self):
        parameters = nw.initial_parameters(1)
        self.assertEqual(len(nw.flatten(parameters)), 27)
        self.assertEqual(len(nw.parameter_names()), 27)
        self.assertEqual(nw.unflatten(nw.flatten(parameters)), parameters)
        self.assertEqual(nw.initial_parameters(1), nw.initial_parameters(1))
        self.assertNotEqual(nw.initial_parameters(1), nw.initial_parameters(2))
        with self.assertRaises(ValueError):
            nw.unflatten([0.0] * 26)

    def test_hand_set_parameters_solve_all_sixteen_rows(self):
        parameters = nw.hand_set_parameters()
        self.assertEqual(len(ALL_ROWS), 16)
        self.assertEqual(nw.correct_count(parameters, ALL_ROWS), 16)
        state = nw.forward(parameters, (0, 2))
        self.assertEqual(state["hidden"], [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(state["scores"], [-1.0, 0.0, -1.0])

    def test_softmax_and_loss_examples(self):
        for got, want in zip(nw.softmax([0.0, math.log(2), math.log(3)]), (1 / 6, 2 / 6, 3 / 6), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        losses = [-math.log(p) for p in (0.25, 0.5, 0.75)]
        self.assertAlmostEqual(sum(losses) / 3, 0.789041, places=6)
        self.assertAlmostEqual(sum(nw.softmax([1000.0, 1001.0, 999.0])), 1.0, places=12)

    def test_ties_take_the_smaller_class(self):
        self.assertEqual(nw.predict([1 / 3, 1 / 3, 1 / 3]), 0)
        self.assertEqual(nw.predict([0.2, 0.4, 0.4]), 1)

    def test_gradient_by_program_matches_nudging(self):
        for seed in (1, 2, 3):
            parameters = nw.initial_parameters(seed)
            self.assertGreater(nw.kink_margin(parameters, TASK["train"]), 1e-3)
            by_program = nw.flatten(nw.analytic_gradient(parameters, TASK["train"]))
            by_nudging = nw.flatten(nw.numeric_gradient(parameters, TASK["train"]))
            self.assertLess(max(abs(a - n) for a, n in zip(by_program, by_nudging, strict=True)), 1e-7)

    def test_update_uses_one_old_state_and_lowers_the_loss(self):
        parameters = nw.initial_parameters(1)
        before = nw.flatten(parameters)
        gradient = nw.analytic_gradient(parameters, TASK["train"])
        updated = nw.update(parameters, gradient, 0.1)
        self.assertEqual(nw.flatten(parameters), before)
        for old, new, slope in zip(before, nw.flatten(updated), nw.flatten(gradient), strict=True):
            self.assertAlmostEqual(new, old - 0.1 * slope, places=12)
        self.assertLess(nw.mean_loss(updated, TASK["train"]), nw.mean_loss(parameters, TASK["train"]))


class UpdateLabTest(unittest.TestCase):
    def test_bias_example_matches_the_checked_numbers(self):
        code, result = run_json(update_lab)
        self.assertEqual(code, 0)
        self.assertAlmostEqual(result["before"]["loss"], 1.098612, places=6)
        for got, want in zip(result["gradient"], (1 / 3, -2 / 3, 1 / 3), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        for item, slope in zip(result["nudges"], result["gradient"], strict=True):
            self.assertAlmostEqual(item["loss_change"] / result["nudge"], slope, places=2)
        for got, want in zip(result["new_bias"], (-0.1, 0.2, -0.1), strict=True):
            self.assertAlmostEqual(got, want, places=12)
        self.assertAlmostEqual(result["after"]["probabilities"][1], 0.402960, places=6)
        self.assertAlmostEqual(result["after"]["loss"], 0.908918, places=6)

    def test_chain_example_by_hand_equals_program_and_nudging(self):
        code, result = run_json(update_lab, "--chain")
        self.assertEqual(code, 0)
        self.assertEqual(result["forward"]["scores"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(result["slope_by_hand"], 3.0, places=12)
        self.assertAlmostEqual(result["slope_by_program"], 3.0, places=12)
        self.assertAlmostEqual(result["loss_change"] / result["nudge"], 3.0, places=1)
        self.assertEqual(result["kink_margin"], 1.0)
        code, changed = run_json(update_lab, "--chain", "--label", "0")
        self.assertEqual(code, 0)
        self.assertAlmostEqual(changed["slope_by_hand"], -3.0, places=12)
        self.assertAlmostEqual(changed["slope_by_program"], -3.0, places=12)

    def test_full_gradient_check(self):
        code, result = run_json(update_lab, "--check")
        self.assertEqual((code, result["agrees"], len(result["table"])), (0, True, 27))
        self.assertEqual(result["loss_evaluations_by_nudging"], 54)


class TrainingLabTest(unittest.TestCase):
    def test_training_run_matches_the_saved_record(self):
        code, result = run_json(training_lab)
        self.assertEqual((code, result["matches_saved_record"]), (0, True))
        training = result["record"]["run"]
        self.assertEqual([item["step"] for item in training["checkpoints"]], training_lab.CONFIG["checkpoints"])
        losses = [item["train_loss"] for item in training["checkpoints"]]
        self.assertEqual(losses, sorted(losses, reverse=True))
        self.assertEqual(training["selected_step"], 500)
        by_step = {item["step"]: item for item in training["checkpoints"]}
        self.assertEqual((by_step[200]["train_correct"], by_step[500]["validation_correct"]), (10, 2))
        self.assertGreater(by_step[2000]["validation_loss"], by_step[500]["validation_loss"])
        self.assertEqual(training["heldout"]["correct"], 3)

    def test_baselines(self):
        base = training_lab.baselines(TASK)
        self.assertEqual(base["majority_correct"], {"train": 6, "validation": 1, "heldout": 1})
        self.assertAlmostEqual(base["uniform_loss"], 1.098612, places=6)

    def test_linear_model_cannot_fit_the_training_rows(self):
        linear = training_lab.linear_training(TASK, steps=2000)
        self.assertLess(linear["train_correct"], 10)

    def test_seed_survey_is_reproducible(self):
        first = training_lab.seed_survey(TASK, 3)
        self.assertEqual(first, training_lab.seed_survey(TASK, 3))
        self.assertEqual(sum(item["runs"] for item in first["table"]), 3)


if __name__ == "__main__":
    unittest.main()

"""第 04 到 06 章的实验：16 行数据的核对、学习过程概览、九个参数的前向计算、批量计算。全部离线。"""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


COURSE = Path(__file__).resolve().parents[1]
DATA = COURSE / "data"
sys.path.insert(0, str(COURSE))


def load_lab(chapter):
    spec = importlib.util.spec_from_file_location(chapter.replace("-", "_"), COURSE / chapter / "lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task_lab = load_lab("04-task")
model_lab = load_lab("05-models")
from shared.numeric import linear_forward  # noqa: E402


class UnitOneLabTest(unittest.TestCase):
    def run_json(self, chapter, *arguments):
        with tempfile.TemporaryDirectory() as cwd:
            process = subprocess.run(
                [sys.executable, "-B", str(COURSE / chapter / "lab.py"), "--json", *arguments],
                cwd=cwd, capture_output=True, encoding="utf-8",
            )
        self.assertTrue(process.stdout, process.stderr)
        payload = json.loads(process.stdout)
        self.assertEqual(process.returncode, payload["exit_code"])
        return payload

    # 第 04 章

    def test_count_rule_and_partition_counts(self):
        result = self.run_json("04-task")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["training_executed"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["split_counts"], {"train": 10, "validation": 3, "heldout": 3})
        self.assertEqual(result["class_counts"], {
            "train": {"0": 2, "1": 6, "2": 2},
            "validation": {"0": 1, "1": 1, "2": 1},
            "heldout": {"0": 1, "1": 1, "2": 1},
        })
        self.assertEqual([row["expected_label"] for row in result["rows"]], [
            0, 0, 1, 1,
            0, 0, 1, 1,
            1, 1, 2, 2,
            1, 1, 2, 2,
        ])
        # 正文里让读者先做的四行，以及小测里的两行。
        for (a, b), label in {(1, 3): 1, (2, 2): 2, (1, 1): 0, (3, 0): 1, (3, 2): 2, (0, 1): 0}.items():
            self.assertEqual(task_lab.label_for(a, b), label)

    def test_faults_identify_the_failure_without_writing_data(self):
        source = DATA / "counter-task.json"
        original = source.read_bytes()

        wrong_label = self.run_json("04-task", "--fault", "wrong-label")
        self.assertEqual(wrong_label["exit_code"], 1)
        self.assertEqual(wrong_label["injected_fault"]["before"], 0)
        self.assertEqual(wrong_label["injected_fault"]["after"], 1)
        self.assertIn({"kind": "label_mismatch", "id": "a0-b0", "split": "train", "expected": 0, "actual": 1}, wrong_label["issues"])

        overlap = self.run_json("04-task", "--fault", "overlap")
        self.assertEqual(overlap["exit_code"], 1)
        self.assertEqual(overlap["injected_fault"]["kind"], "overlap")
        self.assertEqual(overlap["injected_fault"]["field"], "row")
        # 重叠要是真的重叠：同一行（输入和标签都相同）同时在训练和验证两组里。
        copies = [row for row in overlap["rows"] if row["id"] == "a0-b0"]
        self.assertEqual(sorted(row["split"] for row in copies), ["train", "validation"])
        self.assertEqual({(row["a"], row["b"], row["label"]) for row in copies}, {(0, 0, 0)})
        by_kind = {issue["kind"]: issue for issue in overlap["issues"]}
        self.assertEqual(set(by_kind), {"split_overlap", "duplicate_input", "missing_inputs"})
        self.assertEqual(by_kind["split_overlap"]["splits"], ["train", "validation"])
        self.assertEqual(by_kind["duplicate_input"]["input"], [0, 0])
        self.assertEqual(by_kind["missing_inputs"]["inputs"], [[0, 1]])
        self.assertEqual(source.read_bytes(), original)

    def test_every_issue_is_described_in_a_sentence(self):
        for fault in ("wrong-label", "overlap"):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(task_lab.main(["--fault", fault]), 1)
            text = output.getvalue()
            self.assertIn("核对未通过", text)
            self.assertNotIn('{"kind"', text)

    def test_data_format_rejects_bools_and_duplicate_input_is_separate(self):
        fixture = json.loads((DATA / "counter-task.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json"
            with patch.object(task_lab, "DATA", path):
                fixture["rows"][0]["a"] = True
                path.write_text(json.dumps(fixture), encoding="utf-8")
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(task_lab.main(["--json"]), 2)
                self.assertIn("第 1 行 a", json.loads(output.getvalue())["error"])

                fixture["rows"][0]["a"] = 0
                fixture["rows"][1]["b"] = 0
                path.write_text(json.dumps(fixture), encoding="utf-8")
                result = task_lab.run(None)
                self.assertEqual({issue["kind"] for issue in result["issues"]}, {"duplicate_input", "missing_inputs"})

    def test_preview_replays_the_saved_training_record(self):
        result = self.run_json("04-task", "--preview")
        record = json.loads((DATA / "network-training-record.json").read_text(encoding="utf-8"))
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["training_executed"])
        self.assertEqual(result["adjust"], {"rows": 10, "correct_before": 2, "selected_step": 500, "correct_after": 10})
        self.assertEqual(result["select"]["candidates"], 12)
        self.assertEqual(result["select"]["selected_step"], record["run"]["selected_step"])
        self.assertEqual(result["select"]["validation_correct"], 2)
        by_step = {item["step"]: item for item in result["select"]["table"]}
        self.assertEqual(by_step[200]["train_correct"], 10)       # 正文：第 200 次时 10 行已经全对
        self.assertEqual(by_step[100]["train_correct"], 9)
        self.assertAlmostEqual(by_step[500]["validation_loss"], 0.3597, places=4)
        self.assertEqual(min(by_step, key=lambda step: by_step[step]["validation_loss"]), 500)
        self.assertEqual(result["check"]["correct"], 3)
        self.assertEqual(result["rule_correct"], 16)

    def test_leak_looks_perfect_until_the_column_is_missing(self):
        result = self.run_json("04-task", "--leak")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(
            {split: item["correct"] for split, item in result["copy_rule_with_column"].items()},
            {"train": 10, "validation": 3, "heldout": 3},
        )
        self.assertEqual(result["copy_rule_without_column"], {"correct": 4, "rows": 16})
        self.assertEqual(result["issues"], [{"kind": "label_in_input", "column": "处理级别"}])

    # 第 05 章

    def test_numeric_invalid_shapes_booleans_and_nonfinite_values(self):
        weights = [[0.5, 1, 0], [-1, 0, 2]]
        bias = [0, 0.5, -1]
        bad_inputs = ([[True, 1]], [[float("nan"), 1]], [[1, float("inf")]], [[1, 2], [1]], [[1, 2, 3]])
        for inputs in bad_inputs:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                linear_forward(inputs, weights, bias)
        with self.assertRaisesRegex(ValueError, "W 的列数"):
            linear_forward([[2, 1]], weights, [0, 0.5])
        with self.assertRaises(ValueError):
            linear_forward([[2, 1]], [[0.5, 1, 0], [-1, 0, True]], bias)

    def test_forward_with_changed_input_and_changed_parameters(self):
        model_path = DATA / "linear-model.json"
        parameter_path = DATA / "linear-parameters.f32"
        original_json = model_path.read_bytes()
        original_binary = parameter_path.read_bytes()

        baseline = self.run_json("05-models")
        self.assertFalse(baseline["training_executed"])
        self.assertEqual(baseline["parameter_origin"], "hand_set_teaching_parameters")
        self.assertEqual((baseline["scores"], baseline["prediction"], baseline["label_by_rule"]), ([0, 2.5, 1], 1, 1))
        self.assertFalse(baseline["parameters_changed"])

        cases = {
            ("--a", "3"): ([0.5, 3.5, 1], 1, False),
            ("--set", "W[a,0]=1"): ([1, 2.5, 1], 1, True),
            ("--set", "bias[2]=2"): ([0, 2.5, 4], 2, True),
            ("--a", "0", "--b", "0"): ([0, 0.5, -1], 1, False),
            ("--a", "1", "--b", "2"): ([-1.5, 1.5, 3], 2, False),
        }
        for arguments, (scores, prediction, changed) in cases.items():
            with self.subTest(arguments=arguments):
                result = self.run_json("05-models", *arguments)
                self.assertEqual((result["scores"], result["prediction"], result["parameters_changed"]), (scores, prediction, changed))
        self.assertEqual(self.run_json("05-models", "--a", "0", "--b", "0")["label_by_rule"], 0)
        self.assertEqual(self.run_json("05-models", "--a", "1", "--b", "2")["label_by_rule"], 1)

        bad = self.run_json("05-models", "--set", "W[c,0]=1")
        self.assertEqual(bad["exit_code"], 2)
        self.assertIn("看不懂这个改动", bad["error"])
        self.assertEqual(model_path.read_bytes(), original_json)
        self.assertEqual(parameter_path.read_bytes(), original_binary)

    def test_storage_needs_a_layout_convention(self):
        result = self.run_json("05-models", "--storage")
        self.assertEqual((result["parameter_count"], result["binary_bytes"], result["matches_json"]), (9, 36, True))
        self.assertEqual(result["json_bytes"], len((DATA / "linear-model.json").read_bytes()))
        self.assertEqual(result["values_in_file_order"], [0.5, 1, 0, -1, 0, 2, 0, 0.5, -1])
        right, by_column, bias_first = result["readings"]
        self.assertEqual((right["scores"], right["prediction"]), ([0, 2.5, 1], 1))
        self.assertEqual((by_column["weights"], by_column["scores"], by_column["prediction"]), ([[0.5, 0, 0], [1, -1, 2]], [2, -0.5, 1], 0))
        self.assertEqual((bias_first["bias"], bias_first["scores"], bias_first["prediction"]), ([0.5, 1, 0], [-1.5, 1.5, 3], 2))

    def test_fingerprint_and_byte_order(self):
        result = self.run_json("05-models", "--fingerprint")
        payload = (DATA / "linear-parameters.f32").read_bytes()
        self.assertEqual(result["fingerprint"], hashlib.sha256(payload).hexdigest())
        self.assertTrue(result["fingerprint"].startswith("c3e79a12"))
        self.assertTrue(result["fingerprint_after_one_change"].startswith("47cfa76b"))
        self.assertEqual(result["first_bytes"], "00 00 00 3f")
        self.assertEqual(f"{result['big_endian'][0]:.3g}", "8.83e-44")

    def test_binary_and_json_must_store_the_same_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.f32"
            path.write_bytes(bytes(36))
            with patch.object(model_lab, "BINARY_FILE", path), self.assertRaisesRegex(ValueError, "参数值不一致"):
                model_lab.load_model()

    # 第 06 章

    def test_dot_batch_and_dimension_failure(self):
        result = self.run_json("06-math")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["batch"]["output"], [[0, 2.5, 1], [-1, 0.5, 1]])
        self.assertEqual(result["batch"]["predictions"], [1, 2])
        self.assertEqual([[term["product"] for term in candidate["terms"]] for candidate in result["dot_products"]], [
            [1, -1], [2, 0], [0, 2],
        ])
        self.assertEqual([candidate["dot"] for candidate in result["dot_products"]], [0, 2, 2])
        self.assertEqual(result["swapped_input"]["output"], [-1.5, 1.5, 3])

        shape_fault = self.run_json("06-math", "--fault", "shape")
        self.assertEqual(shape_fault["exit_code"], 1)
        self.assertFalse(shape_fault["shape_valid"])
        self.assertEqual(shape_fault["injected_fault"]["X_shape"], [2, 3])
        self.assertIn("X 的列数 3", shape_fault["error"])

    def test_axis_failure_still_produces_a_plausible_shape(self):
        result = self.run_json("06-math", "--fault", "axis")
        self.assertEqual(result["exit_code"], 0)
        fault = result["injected_fault"]
        batch = result["batch"]
        self.assertEqual(fault["kind"], "axis")
        self.assertTrue(fault["shape_valid"])
        self.assertTrue(fault["manual_review_required"])
        self.assertEqual(fault["X_shape"], batch["X_shape"])
        self.assertEqual(fault["actual_axis_roles"], ["feature", "sample"])
        self.assertEqual(fault["expected_axis_roles"], ["sample", "feature"])
        self.assertEqual(fault["output"], [[1, 2.5, -1], [-0.5, 1.5, 1]])
        self.assertNotEqual(fault["output"], batch["output"])

    def test_all_sixteen_rows_in_one_batch(self):
        result = self.run_json("06-math", "--all")
        self.assertEqual((result["X_shape"], result["W_shape"], result["output_shape"]), ([16, 2], [2, 3], [16, 3]))
        self.assertEqual(result["correct"], 7)
        by_id = {item["id"]: item for item in result["rows"]}
        self.assertEqual((by_id["a0-b0"]["scores"], by_id["a0-b0"]["prediction"]), ([0, 0.5, -1], 1))
        self.assertEqual((by_id["a2-b0"]["scores"], by_id["a2-b0"]["prediction"]), ([1, 2.5, -1], 1))


if __name__ == "__main__":
    unittest.main()

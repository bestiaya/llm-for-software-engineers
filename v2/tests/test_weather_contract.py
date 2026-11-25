"""验证阶段一的两版约定、第 02 章实验和第 03 章评分程序。全部离线。"""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))

from shared.weather_app import decide_v1, decide_v2, expected_problems, observe, same  # noqa: E402


def load_lab(name, chapter):
    spec = importlib.util.spec_from_file_location(name, COURSE / chapter / "lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract_lab = load_lab("contract_lab", "02-contract")
eval_lab = load_lab("eval_lab", "03-eval")
CASES = json.loads((COURSE / "02-contract/inputs/cases.json").read_text(encoding="utf-8"))["cases"]
REFERENCE = {case["id"]: case["expected"] for case in CASES}


def run_json(lab, *args):
    with redirect_stdout(io.StringIO()) as stream:
        code = lab.main([*args, "--json"])
    return code, json.loads(stream.getvalue())


def write_expected(directory, expected):
    path = Path(directory) / "mine.json"
    path.write_text(json.dumps({"expected": expected}, ensure_ascii=False), encoding="utf-8")
    return str(path)


class ContractTest(unittest.TestCase):
    def test_v2_three_results(self):
        query = observe(decide_v2('{"result":"query","city":"成都","days":3}'))
        self.assertEqual((query["outcome"], query["city"], query["days"], query["decided_by"]), ("query", "成都", 3, "model"))
        self.assertTrue(query["sends_weather_request"])
        self.assertIn("forecast_days=3", decide_v2('{"result":"query","city":"成都","days":3}')["weather_request"]["url"])
        clarify = observe(decide_v2('{"result":"clarify","missing":"city"}'))
        self.assertEqual((clarify["outcome"], clarify["sends_weather_request"], clarify["reason"]), ("clarify", False, None))
        refused = observe(decide_v2('{"result":"unsupported","reason":"涉及多个城市"}'))
        self.assertEqual(
            (refused["outcome"], refused["decided_by"], refused["reason"], refused["sends_weather_request"]),
            ("unsupported", "model", "model_unsupported", False),
        )

    def test_v2_program_rules(self):
        default = decide_v2('{"result":"query","city":"北京","days":null}')
        seen = observe(default)
        self.assertEqual((seen["outcome"], seen["days"], seen["default_disclosed"]), ("query", 1, True))
        self.assertEqual((seen["extracted_city"], seen["extracted_days"]), ("北京", None))
        self.assertIn("你没有说天数", default["message"])
        for days in (0, 17, 30):
            refused = observe(decide_v2(json.dumps({"result": "query", "city": "成都", "days": days}, ensure_ascii=False)))
            self.assertEqual(
                (refused["outcome"], refused["decided_by"], refused["reason"], refused["sends_weather_request"]),
                ("unsupported", "program", "days_out_of_range", False),
            )
            self.assertEqual((refused["extracted_city"], refused["extracted_days"]), ("成都", days))
        self.assertEqual(observe(decide_v2('{"result":"query","city":"成都","days":16}'))["outcome"], "query")
        unknown = observe(decide_v2('{"result":"query","city":"拉萨","days":2}'))
        self.assertEqual((unknown["outcome"], unknown["decided_by"], unknown["reason"]), ("unsupported", "program", "city_not_configured"))

    def test_v2_rejects_malformed_text(self):
        for text in (
            "晴", "[]", '{"city":"成都","days":3}', '{"result":"query","city":"成都","days":true}',
            '{"result":"query","city":"成都","days":"3"}', '{"result":"query","city":"成都","days":3,"extra":1}',
            '{"result":"clarify","missing":"days"}', '{"result":"unsupported","reason":""}',
            '{"result":"query","city":"成都","days":3,"days":4}', '```json\n{"result":"clarify","missing":"city"}\n```',
        ):
            with self.subTest(text=text):
                seen = observe(decide_v2(text))
                self.assertEqual((seen["outcome"], seen["reason"], seen["sends_weather_request"]), ("error", "invalid_output", False))

    def test_v1_only_knows_two_fields(self):
        self.assertEqual(observe(decide_v1('{"city":"成都","days":3}'))["outcome"], "query")
        self.assertEqual(observe(decide_v1('{"city":"成都","days":30}'))["days"], 30)
        for text in ('{"city":null,"days":1}', '{"result":"clarify","missing":"city"}', "晴", '{"city":"拉萨","days":1}'):
            with self.subTest(text=text):
                seen = observe(decide_v1(text))
                self.assertEqual((seen["outcome"], seen["sends_weather_request"]), ("error", False))

    def test_observed_facts_come_from_what_the_app_does(self):
        decision = decide_v2('{"result":"query","city":"北京","days":null}')
        decision["message"] = "查询北京 1 天的预报。"
        self.assertFalse(observe(decision)["default_disclosed"])
        decision = decide_v2('{"result":"clarify","missing":"city"}')
        decision["weather_request"] = {"city": "北京", "days": 1, "url": "x"}
        self.assertTrue(observe(decision)["sends_weather_request"])

    def test_cases_file_is_consistent(self):
        self.assertEqual([case["id"] for case in CASES], [f"w{number:02d}" for number in range(1, 11)])
        for case in CASES:
            self.assertEqual(expected_problems(case["expected"]), [], case["id"])
        self.assertEqual(REFERENCE["w07"], {
            "outcome": "unsupported", "decided_by": "program", "reason": "days_out_of_range",
            "extracted_city": "成都", "extracted_days": 30,
        })
        self.assertEqual([key for key, value in REFERENCE.items() if value["outcome"] == "unsupported"], ["w05", "w06", "w07", "w08", "w09"])

    def test_expected_behaviour_must_be_well_formed(self):
        good = {"outcome": "query", "city": "北京", "days": 1, "default_disclosed": True}
        self.assertEqual(expected_problems(good), [])
        for bad in (
            None, [], {"outcome": "ask"}, {"outcome": "query"}, {**good, "days": True}, {**good, "default_disclosed": 1},
            {**good, "decided_by": "program"}, {"outcome": "clarify", "city": "北京"},
            {"outcome": "unsupported", "reason": "太多"}, {"outcome": "unsupported", "decided_by": "user"},
            {"outcome": "unsupported", "extracted_days": "30"},
        ):
            with self.subTest(expected=bad):
                self.assertTrue(expected_problems(bad))
        self.assertFalse(same(True, 1))
        self.assertTrue(same(3, 3))


class ContractLabTest(unittest.TestCase):
    def test_samples_and_try(self):
        code, result = run_json(contract_lab)
        self.assertEqual(code, 0)
        self.assertEqual(
            [sample["decision"]["outcome"] for sample in result["samples"]],
            ["query", "query", "unsupported", "clarify", "unsupported", "error"],
        )
        self.assertEqual(result["app_rules"]["cities"], ["成都", "上海", "北京", "杭州", "重庆"])
        code, result = run_json(contract_lab, "--version", "v1", "--try", '{"city":"成都","days":30}')
        self.assertEqual((code, result["decision"]["outcome"], result["decision"]["weather_request"]["days"]), (0, "query", 30))

    def test_check_reader_expected(self):
        code, result = run_json(contract_lab, "--check", str(COURSE / "02-contract/inputs/expected-starter.json"))
        self.assertEqual(code, 1)
        self.assertEqual([entry["id"] for entry in result["results"]], ["w04", "w06", "w09"])
        self.assertEqual([entry["problems"] for entry in result["results"]], [["还没有填写"]] * 3)
        with tempfile.TemporaryDirectory() as directory:
            good = write_expected(directory, {key: REFERENCE[key] for key in ("w04", "w06", "w09")})
            self.assertEqual(run_json(contract_lab, "--check", good)[0], 0)
            wrong = write_expected(directory, {"w04": {"outcome": "query", "city": "北京", "days": 3, "default_disclosed": False}})
            code, result = run_json(contract_lab, "--check", wrong)
            self.assertEqual(code, 1)
            self.assertEqual(len(result["results"][0]["problems"]), 2)
            self.assertEqual(run_json(contract_lab, "--check", str(Path(directory) / "missing.json"))[0], 2)


class ReaderAnswersAgreeTest(unittest.TestCase):
    """读者写的期望行为，在第 02 章和第 03 章得到的反馈不能互相矛盾。"""

    def both(self, expected):
        with tempfile.TemporaryDirectory() as directory:
            path = write_expected(directory, expected)
            return run_json(contract_lab, "--check", path), run_json(eval_lab, "--expected", path)

    def test_incomplete_query_is_rejected_by_both(self):
        (check_code, check), (eval_code, _) = self.both({"w04": {"outcome": "query"}})
        self.assertEqual(check_code, 1)
        self.assertFalse(check["results"][0]["matches"])
        self.assertEqual(eval_code, 2)

    def test_extra_field_is_rejected_by_both(self):
        (check_code, check), (eval_code, _) = self.both({"w04": {**REFERENCE["w04"], "decided_by": "program"}})
        self.assertEqual((check_code, eval_code), (1, 2))
        self.assertFalse(check["results"][0]["matches"])

    def test_wrong_types_are_rejected_by_both(self):
        (check_code, _), (eval_code, _) = self.both({"w04": {"outcome": "query", "city": "北京", "days": True, "default_disclosed": 1}})
        self.assertEqual((check_code, eval_code), (1, 2))

    def test_weaker_standard_is_reported_not_hidden(self):
        (check_code, check), (eval_code, scored) = self.both({"w07": {"outcome": "unsupported"}})
        self.assertEqual(check_code, 1)
        self.assertEqual(len(check["results"][0]["problems"]), 4)
        self.assertEqual((eval_code, scored["reader_standard_ids"]), (0, ["w07"]))
        w07 = next(item for item in scored["results"] if item["id"] == "w07")
        self.assertEqual(w07["standard"], "reader")
        self.assertEqual(w07["unchecked"], ["decided_by", "reason", "extracted_city", "extracted_days"])

    def test_reference_answers_pass_both(self):
        (check_code, _), (eval_code, scored) = self.both({key: REFERENCE[key] for key in ("w04", "w06", "w09")})
        self.assertEqual((check_code, eval_code), (0, 0))
        self.assertEqual(scored["summary"]["passed"], 10)

    def test_a_different_standard_changes_the_score(self):
        _, (eval_code, scored) = self.both({"w04": {"outcome": "query", "city": "北京", "days": 3, "default_disclosed": False}})
        self.assertEqual((eval_code, scored["summary"]["failed_ids"]), (0, ["w04"]))


class ScorerTest(unittest.TestCase):
    def failed(self, name):
        code, result = run_json(eval_lab, "--synthetic", name)
        self.assertEqual(code, 0)
        self.assertTrue(all(item["source"] == "hand-made-fault" for item in result["results"]))
        return result["summary"]["failed_ids"]

    def test_scorer_catches_hand_made_faults(self):
        self.assertEqual(self.failed("always-3-days"), ["w02", "w04", "w07"])
        self.assertEqual(self.failed("fabricate-city"), ["w03"])
        self.assertEqual(self.failed("refuse-all"), ["w01", "w02", "w03", "w04", "w07", "w10"])
        self.assertEqual(self.failed("wrong-refusal"), ["w07"])
        self.assertEqual(run_json(eval_lab, "--synthetic", "no-such-set")[0], 2)

    def test_w07_needs_the_right_reason(self):
        expected = REFERENCE["w07"]
        self.assertTrue(eval_lab.score_case(decide_v2('{"result":"query","city":"成都","days":30}'), expected)["passed"])
        for text in (
            '{"result":"query","city":"拉萨","days":3}', '{"result":"query","city":"成都","days":0}',
            '{"result":"query","city":"成都","days":17}', '{"result":"unsupported","reason":"天数太多"}',
            '{"result":"query","city":"成都","days":16}',
        ):
            with self.subTest(text=text):
                self.assertFalse(eval_lab.score_case(decide_v2(text), expected)["passed"])

    def test_scorer_checks_what_the_app_actually_does(self):
        decision = decide_v2('{"result":"query","city":"北京","days":null}')
        self.assertTrue(eval_lab.score_case(decision, REFERENCE["w04"])["passed"])
        decision["message"] = "查询北京 1 天的预报。"
        self.assertFalse(eval_lab.score_case(decision, REFERENCE["w04"])["passed"])
        decision = decide_v2('{"result":"clarify","missing":"city"}')
        self.assertTrue(eval_lab.score_case(decision, REFERENCE["w03"])["passed"])
        decision["weather_request"] = {"city": "北京", "days": 1, "url": "x"}
        self.assertFalse(eval_lab.score_case(decision, REFERENCE["w03"])["passed"])

    def test_score_case_checks_written_and_implied_items(self):
        decision = decide_v2('{"result":"unsupported","reason":"超出范围"}')
        fields = [check["field"] for check in eval_lab.score_case(decision, {"outcome": "unsupported"})["checks"]]
        self.assertEqual(fields, ["outcome", "sends_weather_request"])
        self.assertTrue(eval_lab.score_case(decision, {"outcome": "unsupported"})["passed"])
        self.assertFalse(eval_lab.score_case(decision, {"outcome": "unsupported", "decided_by": "program"})["passed"])


class RealRecordsTest(unittest.TestCase):
    """随课的 20 份真实记录，以及正文引用的结果。"""

    def records(self, version):
        directory = COURSE / "03-eval/records" / version
        return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))}

    def test_twenty_records_are_complete_and_uniform(self):
        for version in ("v1", "v2"):
            records = self.records(version)
            self.assertEqual(list(records), [case["id"] for case in CASES])
            for case_id, record in records.items():
                with self.subTest(version=version, case=case_id):
                    capture = record["capture"]
                    self.assertEqual((record["case_id"], record["app_version"]), (case_id, version))
                    self.assertEqual((capture["model_requested"], capture["effort_requested"]), ("gpt-6.1-sol", "medium"))
                    self.assertEqual(capture["other_item_types"], [])
                    self.assertEqual(capture["stdin"], record["request"]["instructions"] + "\n\n" + record["request"]["input"])
                    self.assertEqual(record["response"]["text"], record["cli_message_event"]["item"]["text"])

    def test_replay_results_quoted_in_lessons(self):
        code, old = run_json(eval_lab, "--version", "v1")
        self.assertEqual((code, old["summary"]["passed"]), (0, 3))
        self.assertEqual(old["summary"]["failed_ids"], ["w03", "w04", "w05", "w06", "w07", "w08", "w09"])
        code, new = run_json(eval_lab)
        self.assertEqual((code, new["summary"]["passed"], new["summary"]["failed_ids"]), (0, 10, []))
        self.assertTrue(all(item["source"] == "real-record" and item["standard"] == "reference" for item in new["results"]))
        by_id = {item["id"]: item["observed"] for item in new["results"]}
        self.assertEqual((by_id["w04"]["days"], by_id["w04"]["default_disclosed"]), (1, True))
        self.assertEqual(
            (by_id["w07"]["outcome"], by_id["w07"]["decided_by"], by_id["w07"]["reason"], by_id["w07"]["extracted_days"]),
            ("unsupported", "program", "days_out_of_range", 30),
        )
        code, compared = run_json(eval_lab, "--compare")
        self.assertEqual(code, 0)
        self.assertEqual(compared["flips"]["fail_to_pass"], ["w03", "w04", "w05", "w06", "w07", "w08", "w09"])
        self.assertEqual(compared["flips"]["pass_to_fail"], [])

    def test_lessons_quote_the_recorded_text(self):
        lesson_02 = (COURSE / "02-contract/lesson.md").read_text(encoding="utf-8")
        lesson_03 = (COURSE / "03-eval/lesson.md").read_text(encoding="utf-8")
        old, new = self.records("v1"), self.records("v2")
        for case_id in ("w03", "w07", "w08"):
            self.assertIn(old[case_id]["response"]["text"], lesson_02)
        self.assertIn(new["w07"]["response"]["text"], lesson_02)
        self.assertIn(old["w06"]["response"]["text"], lesson_03)
        for record in new.values():
            self.assertIn(record["response"]["text"], lesson_03)

    def test_records_must_match_the_current_contract(self):
        with patch.dict(eval_lab.INSTRUCTION_FILES, {"v2": eval_lab.INSTRUCTION_FILES["v1"]}):
            self.assertEqual(run_json(eval_lab)[0], 2)

    def test_unfilled_reader_expected_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run_json(eval_lab, "--expected", write_expected(directory, {"w07": None}))[0], 2)
            self.assertEqual(run_json(eval_lab, "--expected", write_expected(directory, {"w99": {"outcome": "clarify"}}))[0], 2)


if __name__ == "__main__":
    unittest.main()

"""验证查天气实验的离线回放、错误边界和 GPT 新采集。"""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


CHAPTER = Path(__file__).resolve().parents[1] / "01-request"
SPEC = importlib.util.spec_from_file_location("first_request_lab", CHAPTER / "lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class FirstRequestTest(unittest.TestCase):
    def run_json(self, *args):
        with redirect_stdout(io.StringIO()) as stream:
            code = lab.main([*args, "--json"])
        result = json.loads(stream.getvalue())
        self.assertEqual(result["exit_code"], code)
        return result

    def test_real_records_replay_offline(self):
        saved = lab.read_json(CHAPTER / "records/chengdu-gpt.json")
        forecast = lab.read_json(CHAPTER / "records/chengdu-forecast.json")
        with patch.object(lab, "urlopen", side_effect=AssertionError("尝试联网")):
            result = self.run_json()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["mode"], "replay")
        self.assertEqual(result["request"], saved["request"])
        self.assertEqual(result["response"]["text"], saved["cli_message_event"]["item"]["text"])
        self.assertEqual(result["parameters"], {"city": "成都", "days": 3})
        weather = result["weather"]
        self.assertEqual(weather["mode"], "snapshot")
        self.assertEqual(weather["url"], forecast["url"])
        self.assertEqual(weather["retrieved_at_utc"], forecast["retrieved_at_utc"])
        self.assertEqual(len(weather["rows"]), 3)
        for index, row in enumerate(weather["rows"]):
            daily = forecast["body"]["daily"]
            self.assertEqual(row["date"], daily["time"][index])
            self.assertEqual(row["min_c"], daily["temperature_2m_min"][index])
            self.assertEqual(row["max_c"], daily["temperature_2m_max"][index])
            self.assertIn(row["date"], result["display"])

    def test_record_must_match_current_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.json"
            record = lab.read_json(CHAPTER / "records/chengdu-gpt.json")
            record["request"]["input"] = "看看别处"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(self.run_json("--record", str(path))["exit_code"], 2)

    def test_invalid_parameters_are_rejected(self):
        for text in ('{"city":"成都","days":true}', '{"city":"成都","days":3,"temp":14}',
                     '{"city":"成都","days":3,"days":4}', '[]', '{"city":2,"days":3}'):
            with self.subTest(text=text), self.assertRaises(lab.LabError):
                lab.parse_parameters(text)
        with self.assertRaises(lab.LabError):
            lab.validate_case({"city": "上海", "days": 3})

    def test_faults_stop_before_weather_and_leave_original_intact(self):
        record_path = CHAPTER / "records/chengdu-gpt.json"
        original = record_path.read_bytes()
        with patch.object(lab, "forecast_from_snapshot", side_effect=AssertionError("尝试查天气")):
            for fault in ("wrong-days", "invalid-json"):
                result = self.run_json("--fault", fault)
                self.assertEqual(result["exit_code"], 1)
                self.assertEqual(result["response"]["text"], json.loads(original)["response"]["text"])
                self.assertNotEqual(result["injected_fault"]["text"], result["response"]["text"])
                self.assertIsNone(result["weather"])
                self.assertIsNone(result["display"])
                expected = {"city": "成都", "days": 4} if fault == "wrong-days" else None
                self.assertEqual(result["parameters"], expected)
        self.assertEqual(record_path.read_bytes(), original)

    def test_snapshot_must_match_query_and_cover_three_days(self):
        snapshot = lab.read_json(CHAPTER / "records/chengdu-forecast.json")
        for mutation in ("location", "length", "unit"):
            changed = json.loads(json.dumps(snapshot))
            if mutation == "location":
                changed["url"] = changed["url"].replace("latitude=30.67", "latitude=40")
            elif mutation == "length":
                changed["body"]["daily"]["time"].pop()
            else:
                changed["body"]["daily_units"]["temperature_2m_max"] = "°F"
            with self.subTest(mutation=mutation), patch.object(lab, "read_json", return_value=changed):
                with self.assertRaises(lab.LabError):
                    lab.forecast_from_snapshot({"city": "成都", "days": 3})

    def test_live_refuses_existing_output_before_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(lab, "urlopen") as send:
            destination = Path(directory) / "existing.json"
            destination.write_text("keep this", encoding="utf-8")
            self.assertEqual(self.run_json("--live", "--out", str(destination))["exit_code"], 2)
            send.assert_not_called()
            self.assertEqual(destination.read_text(encoding="utf-8"), "keep this")

    def test_live_uses_openai_and_new_record_can_replay(self):
        text = '{"city": "成都", "days": 3}'
        envelope = {
            "status": "completed", "model": "gpt-6.1-sol",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": text}]},
            ],
            "usage": {"input_tokens": 50, "output_tokens": 12, "input_tokens_details": {"cached_tokens": 5}},
        }
        response = io.BytesIO(json.dumps(envelope).encode())
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
                patch.object(lab, "urlopen", return_value=response) as send:
            path = Path(directory) / "new.json"
            live = self.run_json("--live", "--out", str(path))
            self.assertEqual(live["exit_code"], 0)
            self.assertEqual(live["mode"], "live")
            self.assertEqual(live["weather"]["mode"], "snapshot")
            self.assertEqual(live["capture"]["model_reported"], envelope["model"])
            self.assertEqual(live["response"]["text"], text)
            self.assertEqual(live["response"]["usage"]["cache_read_input_tokens"], 5)
            replay = self.run_json("--record", str(path))
            self.assertEqual(replay["mode"], "replay")
            self.assertEqual(replay["display"], live["display"])
            send.assert_called_once()
            request = send.call_args.args[0]
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
            self.assertEqual(json.loads(request.data), {
                **live["request"], "model": "gpt-6.1-sol", "reasoning": {"effort": "medium"}, "store": False,
            })
            self.assertNotIn("test-key", path.read_text())


if __name__ == "__main__":
    unittest.main()

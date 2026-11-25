"""采集十条输入在两版约定下的真实模型回答。给课程维护者用，读者不需要运行。

每次调用都通过 Codex CLI 请求模型，交给它的全部文字是“约定的指令、空行、用户原话”。
已有的记录不会被覆盖。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile


CHAPTER = Path(__file__).resolve().parent
COURSE = CHAPTER.parent
INSTRUCTION_FILES = {
    "v1": COURSE / "01-request/inputs/weather-request.json",
    "v2": COURSE / "02-contract/inputs/contract-v2.json",
}
CASES_FILE = COURSE / "02-contract/inputs/cases.json"
INPUT_SCOPE = "下列 stdin 为课程交给 Codex CLI 的完整文本。CLI 自带的指令和服务端请求封装不在记录范围内。"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def call_codex(stdin_text, model, effort, timeout):
    with tempfile.TemporaryDirectory() as empty_directory:
        command = [
            "codex", "exec", "--ephemeral", "-s", "read-only", "--skip-git-repo-check",
            "-C", empty_directory, "-m", model, "-c", f"model_reasoning_effort={effort}", "--json", "-",
        ]
        completed = subprocess.run(
            command, input=stdin_text, capture_output=True, text=True, timeout=timeout, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"codex exec 退出码 {completed.returncode}：{completed.stderr.strip()[-300:]}")
    message_event, usage, other_items = None, {}, []
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        if event.get("type") == "item.completed":
            if item.get("type") == "agent_message":
                message_event = event
            elif item.get("type") != "reasoning":
                other_items.append(item.get("type"))
        elif event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
    if message_event is None or not isinstance(message_event["item"].get("text"), str):
        raise RuntimeError("事件流里没有助手消息。")
    return message_event, usage, other_items


def capture(version, case, instructions, cli_version, model, effort, timeout):
    request = {"instructions": instructions, "input": case["input"]}
    stdin_text = instructions + "\n\n" + case["input"]
    message_event, usage, other_items = call_codex(stdin_text, model, effort, timeout)
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "case_id": case["id"],
        "app_version": version,
        "capture": {
            "kind": "live-model",
            "transport": "codex-exec",
            "cli_version": cli_version,
            "model_requested": model,
            "model_reported": "unknown",
            "effort_requested": effort,
            "captured_at_utc": captured_at,
            "input_scope": INPUT_SCOPE,
            "stdin": stdin_text,
            "other_item_types": other_items,
        },
        "request": request,
        "response": {"text": message_event["item"]["text"], "usage": usage},
        "cli_message_event": message_event,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), action="append", help="采集哪一版约定，可重复；默认两版都采")
    parser.add_argument("--case", action="append", help="只采指定编号的输入，可重复")
    parser.add_argument("--out", type=Path, default=CHAPTER / "records", help="记录目录，其下按版本分子目录")
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--effort", default="medium", choices=("low", "medium", "high"))
    parser.add_argument("--timeout", type=int, default=240, help="单次调用的秒数上限")
    args = parser.parse_args(argv)

    cli_version = subprocess.run(["codex", "--version"], capture_output=True, text=True, check=True).stdout.split()[-1]
    cases = read_json(CASES_FILE)["cases"]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
    failures = 0
    for version in args.version or ["v1", "v2"]:
        instructions = read_json(INSTRUCTION_FILES[version])["instructions"]
        directory = args.out / version
        directory.mkdir(parents=True, exist_ok=True)
        for case in cases:
            path = directory / f"{case['id']}.json"
            if path.exists():
                print(f"{version} {case['id']} 已有记录，跳过", flush=True)
                continue
            try:
                record = capture(version, case, instructions, cli_version, args.model, args.effort, args.timeout)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures += 1
                print(f"{version} {case['id']} 失败：{error}", flush=True)
                continue
            try:
                with path.open("x", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            except FileExistsError:
                print(f"{version} {case['id']} 采集期间已有记录，保留已有的", flush=True)
                continue
            print(f"{version} {case['id']} -> {record['response']['text']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

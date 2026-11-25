"""核对第 04 到 10 章的正文和说明：引用的命令都能运行，引用的输出与实验的实际输出逐行一致，代码能执行。全部离线。"""

import math
from pathlib import Path
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ElementTree


COURSE = Path(__file__).resolve().parents[1]
ROOT = COURSE.parent
sys.path.insert(0, str(COURSE))

from shared import network as nw  # noqa: E402
from shared.numeric import linear_forward  # noqa: E402


CHAPTERS = ("04-task", "05-models", "06-math", "07-hidden", "08-loss", "09-update", "10-training")
# 这几条命令演示的是“核对发现了问题”，退出码是 1。
EXPECTED_FAILURES = {
    "v2/04-task/lab.py --fault wrong-label", "v2/04-task/lab.py --fault overlap", "v2/04-task/lab.py --leak",
    "v2/06-math/lab.py --fault shape",
}
COMMAND = re.compile(r"python3 (v2/[\w-]+/lab\.py(?: [^`<\n]*)?)")
_outputs = {}


def commands_in(text):
    return [match.strip() for match in COMMAND.findall(text)]


def run(command):
    """运行正文里引用的一条命令，返回退出码和输出。同一条命令只运行一次。"""
    if command not in _outputs:
        arguments = [part.strip('"') for part in re.findall(r'"[^"]*"|\S+', command)]
        done = subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, capture_output=True, encoding="utf-8")
        _outputs[command] = (done.returncode, done.stdout + done.stderr)
    return _outputs[command]


def blocks(text, language):
    return re.findall(rf"```{language}\n(.*?)```", text, flags=re.S)


class StageTwoLessonTest(unittest.TestCase):
    def test_every_quoted_command_runs_with_the_expected_exit_code(self):
        count = 0
        for chapter in CHAPTERS:
            for name in ("lesson.md", "README.md"):
                for command in commands_in((COURSE / chapter / name).read_text(encoding="utf-8")):
                    with self.subTest(chapter=chapter, file=name, command=command):
                        code, output = run(command)
                        self.assertEqual(code, 1 if command in EXPECTED_FAILURES else 0, output[-400:])
                        self.assertIn("exit_code:", output)
                    count += 1
        # 课程入口页里列出的阶段二命令。
        for command in commands_in((COURSE / "README.md").read_text(encoding="utf-8")):
            if command.split("/")[1] in CHAPTERS:
                with self.subTest(file="v2/README.md", command=command):
                    self.assertEqual(run(command)[0], 0)
                count += 1
        self.assertGreater(count, 70)

    def test_quoted_output_matches_the_labs_line_by_line(self):
        for chapter in CHAPTERS:
            text = (COURSE / chapter / "lesson.md").read_text(encoding="utf-8")
            produced = set()
            for command in commands_in(text):
                produced.update(line.rstrip() for line in run(command)[1].splitlines())
            for block in blocks(text, "text"):
                for line in block.splitlines():
                    if not line.strip() or line.strip() == "…":
                        continue
                    with self.subTest(chapter=chapter, line=line):
                        self.assertIn(line.rstrip(), produced)

    def test_code_in_the_lessons_compiles_and_agrees_with_the_shared_code(self):
        namespace = {"math": math}
        for chapter in CHAPTERS:
            for block in blocks((COURSE / chapter / "lesson.md").read_text(encoding="utf-8"), "python"):
                exec(compile(block, f"{chapter}/lesson.md", "exec"), namespace)
        weights, bias = [[0.5, 1.0, 0.0], [-1.0, 0.0, 2.0]], [0.0, 0.5, -1.0]
        inputs = [[a, b] for a in range(4) for b in range(4)]
        # 第 04 章的标注规则，第 05 章的前向计算和取最大，第 06 章的批量计算。
        self.assertEqual([namespace["label_for"](a, b) for a, b in inputs], [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2])
        self.assertEqual(namespace["forward"]([2, 1], weights, bias), [0, 2.5, 1])
        self.assertEqual(namespace["choose"]([0, 2.5, 1]), 1)
        self.assertEqual(namespace["choose"]([1, 1, 0]), 0)
        self.assertEqual(namespace["batch_forward"](inputs, weights, bias), linear_forward(inputs, weights, bias))
        # 第 08 章的 softmax 和平均损失。
        for scores in ([-1.0, 0.0, -1.0], [0.0, math.log(2), math.log(3)], [3.0, -2.0, 0.5]):
            for got, want in zip(namespace["softmax"](scores), nw.softmax(scores), strict=True):
                self.assertAlmostEqual(got, want, places=12)
        rows = [[0.25, 0.5, 0.25], [0.5, 0.25, 0.25], [0.125, 0.125, 0.75]]
        self.assertAlmostEqual(namespace["mean_loss"](rows, [0, 0, 2]), 0.789041, places=6)
        # 第 09 章的一次更新。
        parameters = nw.initial_parameters(1)
        task = nw.load_task(COURSE / "data/counter-task.json")
        gradient = nw.analytic_gradient(parameters, task["train"])
        by_lesson = namespace["update"](nw.flatten(parameters), nw.flatten(gradient), 0.1)
        self.assertEqual(by_lesson, nw.flatten(nw.update(parameters, gradient, 0.1)))

    def test_links_figures_and_anchors(self):
        for chapter in CHAPTERS:
            folder = COURSE / chapter
            text = (folder / "lesson.md").read_text(encoding="utf-8")
            with self.subTest(chapter=chapter):
                self.assertTrue(text.startswith(f"# K01-{chapter[:2]} "))
                for picture in re.findall(r"!\[[^\]]+\]\(([^)]+)\)", text):
                    root = ElementTree.parse(folder / picture).getroot()
                    self.assertTrue(root.tag.endswith("svg"))
                anchors = re.findall(r'<a id="([a-z-]+)"></a>', text)
                self.assertEqual(len(anchors), len(set(anchors)))
                self.assertIn("quiz", anchors)
                self.assertIn("next", anchors)
                for name in ("README.md", "LESSON_BLUEPRINT.md", "review.md"):
                    for target in re.findall(r"\]\((?!http)([^)#]+)(?:#[^)]*)?\)", (folder / name).read_text(encoding="utf-8")):
                        self.assertTrue((folder / target).exists() or target.endswith("lesson.html"), f"{name}: {target}")
                # 没有用到的图不留在目录里。
                used = set(re.findall(r"\(assets/([^)]+)\)", text))
                kept = {path.name for path in (folder / "assets").glob("*.svg")} if (folder / "assets").is_dir() else set()
                self.assertEqual(used, kept)

    def test_sources_link_to_markdown_and_built_pages_link_to_html(self):
        """源文里指向别的章和课程导读的链接写 Markdown，在 GitHub 上能跳转；构建出的页面里换成 HTML 阅读页。"""
        import json

        course = json.loads((COURSE / "course.json").read_text(encoding="utf-8"))
        sources = [COURSE / chapter["source"] for chapter in course["chapters"] if chapter["source"]] + [COURSE / "ABOUT.md"]
        self.assertEqual(len(sources), 11)
        for source in sources:
            text = source.read_text(encoding="utf-8")
            with self.subTest(source=source.name, folder=source.parent.name):
                links = re.findall(r"\]\((?!https?:)([^)#\s]+)", text)
                self.assertFalse(
                    [link for link in links if link.endswith("lesson.html") or link.endswith("about.html")],
                    "源文里不要链接到 HTML 阅读页",
                )
                page = source.with_name("about.html" if source.name == "ABOUT.md" else "lesson.html")
                built = page.read_text(encoding="utf-8")
                body = built[built.index("<main>"):]
                self.assertNotIn('ABOUT.md', body)
                self.assertFalse(re.findall(r'href="[^"]*/lesson\.md', body))

    def test_numbers_drawn_on_the_loss_chart_come_from_the_record(self):
        import json

        record = json.loads((COURSE / "data/network-training-record.json").read_text(encoding="utf-8"))
        chart = (COURSE / "10-training/assets/loss-curves.svg").read_text(encoding="utf-8")
        last = record["run"]["checkpoints"][-1]
        self.assertIn(f"训练 {last['train_loss']:.3f}", chart)
        self.assertIn(f"验证 {last['validation_loss']:.3f}", chart)
        self.assertIn(f"第 {record['run']['selected_step']} 步：验证损失最低", chart)
        self.assertIn(f"三类平分 {record['baselines']['uniform_loss']:.3f}", chart)
        self.assertEqual(chart.count("<circle"), 2 * len(record["run"]["checkpoints"]) + 2)


if __name__ == "__main__":
    unittest.main()

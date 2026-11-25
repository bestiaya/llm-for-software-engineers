"""由随课保存的训练记录生成第 10 章的曲线图 v2/10-training/assets/loss-curves.svg。只用标准库。

用法：在仓库根目录运行 python3 tools/make_loss_chart.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
record = json.loads((ROOT / "v2/data/network-training-record.json").read_text(encoding="utf-8"))
points = record["run"]["checkpoints"]
selected = record["run"]["selected_step"]
uniform = record["baselines"]["uniform_loss"]

LEFT, RIGHT, TOP, BOTTOM, TOP_VALUE = 110, 800, 124, 390, 1.4
TRAIN, VALID, SURFACE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#f4f8fa", "#142e40", "#24485e", "#cfdde5"


def x_at(index):
    return LEFT + index * (RIGHT - LEFT) / (len(points) - 1)


def y_at(value):
    return BOTTOM - value / TOP_VALUE * (BOTTOM - TOP)


def path(key):
    return "M" + "L".join(f"{x_at(i):.1f} {y_at(item[key]):.1f}" for i, item in enumerate(points))


def dots(key, color):
    return "".join(
        f'<circle cx="{x_at(i):.1f}" cy="{y_at(item[key]):.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>'
        for i, item in enumerate(points)
    )


chosen = next(i for i, item in enumerate(points) if item["step"] == selected)
last = points[-1]
grid = "".join(
    f'<path d="M{LEFT - 14} {y_at(v):.1f}H{RIGHT + 6}" stroke="{GRID}" stroke-width="1"/>'
    f'<text class="lc-tick" x="{LEFT - 22}" y="{y_at(v) + 5:.1f}" text-anchor="end">{label}</text>'
    for v, label in ((0, "0"), (0.5, "0.5"), (1.0, "1.0"))
)
ticks = "".join(
    f'<text class="lc-tick" x="{x_at(i):.1f}" y="{BOTTOM + 24}" text-anchor="middle">{item["step"]}</text>'
    for i, item in enumerate(points)
)
description = (
    "横轴是十二个检查点，从第 0 步到第 2000 步，按顺序等距排列，横向距离不代表步数差；纵轴是平均损失。"
    f"在所画的检查点上，训练损失逐个下降，从 {points[0]['train_loss']:.3f} 到 {last['train_loss']:.3f}。"
    f"验证损失从 {points[0]['validation_loss']:.3f} 降到第 {selected} 步的 {points[chosen]['validation_loss']:.3f}，"
    f"之后回升到 {last['validation_loss']:.3f}。三类平分时的损失是 {uniform:.3f}。"
)
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 460" role="img" aria-labelledby="lc-title lc-desc">
  <title id="lc-title">同一次训练里，训练损失和验证损失的变化</title>
  <desc id="lc-desc">{description}</desc>
  <defs>
    <style>
      .lc-title{{font:650 20px system-ui,"PingFang SC","Microsoft YaHei",sans-serif;fill:{INK}}}
      .lc-sub{{font:14.5px system-ui,"PingFang SC","Microsoft YaHei",sans-serif;fill:{MUTED}}}
      .lc-tick{{font:13.5px system-ui,"PingFang SC","Microsoft YaHei",sans-serif;fill:{MUTED}}}
      .lc-label{{font:650 14px system-ui,"PingFang SC","Microsoft YaHei",sans-serif;fill:{INK}}}
      .lc-line{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
    </style>
  </defs>
  <rect width="940" height="460" rx="12" fill="{SURFACE}"/>
  <text class="lc-title" x="30" y="44">同一次训练里，训练损失和验证损失的变化</text>
  <text class="lc-sub" x="30" y="68">起点种子 {record["config"]["seed"]}，学习率 {record["config"]["learning_rate"]}，共 {record["config"]["steps"]} 步；只画出留了记录的 {len(points)} 个检查点</text>
  <g>
    <path d="M30 96H58" stroke="{TRAIN}" stroke-width="2" stroke-linecap="round"/><circle cx="44" cy="96" r="4" fill="{TRAIN}"/>
    <text class="lc-sub" x="66" y="101">训练损失（10 行）</text>
    <path d="M206 96H234" stroke="{VALID}" stroke-width="2" stroke-linecap="round"/><circle cx="220" cy="96" r="4" fill="{VALID}"/>
    <text class="lc-sub" x="242" y="101">验证损失（3 行）</text>
  </g>
  {grid}
  <path d="M{LEFT - 14} {y_at(uniform):.1f}H{RIGHT + 6}" stroke="#8fa6b3" stroke-width="1"/>
  <text class="lc-tick" x="{RIGHT + 14}" y="{y_at(uniform) + 5:.1f}">三类平分 {uniform:.3f}</text>
  <path d="M{x_at(chosen):.1f} {TOP}V{BOTTOM}" stroke="#8fa6b3" stroke-width="1"/>
  <text class="lc-label" x="{x_at(chosen) - 8:.1f}" y="{TOP + 12}" text-anchor="end">第 {selected} 步：验证损失最低，选这一份</text>
  <path class="lc-line" d="{path("train_loss")}" stroke="{TRAIN}"/>
  <path class="lc-line" d="{path("validation_loss")}" stroke="{VALID}"/>
  {dots("train_loss", TRAIN)}
  {dots("validation_loss", VALID)}
  <text class="lc-label" x="{RIGHT + 14}" y="{y_at(last["validation_loss"]) + 5:.1f}">验证 {last["validation_loss"]:.3f}</text>
  <text class="lc-label" x="{RIGHT + 14}" y="{y_at(last["train_loss"]) + 2:.1f}">训练 {last["train_loss"]:.3f}</text>
  {ticks}
  <text class="lc-sub" x="{(LEFT + RIGHT) / 2:.0f}" y="{BOTTOM + 52}" text-anchor="middle">训练步数。检查点按顺序等距排列，横向距离不代表步数差</text>
  <text class="lc-sub" x="{LEFT - 22}" y="{TOP - 6}" text-anchor="end">平均损失</text>
</svg>
'''
target = ROOT / "v2/10-training/assets/loss-curves.svg"
target.write_text(svg, encoding="utf-8")
print("已生成", target.relative_to(ROOT))

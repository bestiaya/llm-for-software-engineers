# 真实运行报告

dataset=real，seed=42，参数=32192，选择第 300 步。

| 模型 | test 每目标 CE | 目标数 |
|---|---:|---:|
| model | 2.450565 | 14402 |
| unigram | 3.172494 | 14402 |
| bigram | 2.442503 | 14402 |

validation CE=2.470361；总耗时 2.333 秒。

## 配置与资料

d_model=32，heads=4，layers=2，context=64，vocab=73。
AdamW，lr=0.003，weight_decay=0.01，梯度裁剪=1.0，batch_size=16，steps=300，每 25 步及最后一步检查 validation。
CPU 单线程；窗口频率选长度桶，桶内有放回采样；无 PAD。
data/real.txt，连续原始字符 80/10/10 切分，窗口不跨分区
语料 SHA-256：93d669b42c033b246b1a959dd0c0c09d723162ea67c2afb753c5e28b1b2e6564

| 分区 | 组数 | 原字符数 | 窗口数 | 目标数 | 未知字符数 |
|---|---:|---:|---:|---:|---:|
| train | 1 | 115212 | 1801 | 115212 | 0 |
| validation | 1 | 14402 | 226 | 14402 | 4 |
| test | 1 | 14402 | 226 | 14402 | 0 |

模型按 validation 最低 CE 选定后才评价 test；完整训练轨迹与环境版本见 [real.json](real.json)。

## 生成

**输入提示（不是模型生成）**

```text
CHA
```

**模型新增内容**

```text
l the the the the the the t the the the the the the the the t
```
停止原因：context_limit

## 限制

- 有限数据和单 seed；不证明语言理解或工程应用有效。
- toy 完整组不同，但共享语法；real 相邻切分仍可能有重复表达。
- 开发期间可见的固定 test 不是永远未触碰的盲测；不要据此反复调参。
- 计数 baseline 仅使用前一个 token 或全局频率；与 decoder 的上下文长度不同。
- 无 PAD；字符而非 BPE 训练；无跨窗口记忆；绝对位置按窗口重置。
- 生成使用贪心，EOS/context/token budget 停止；不保证语义、正确性或安全性。

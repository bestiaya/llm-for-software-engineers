"""Trace scaled dot-product attention. Educational numbers; no learned model."""
from __future__ import annotations
import argparse
import json
import math

TOKENS = ['服务', '超时', '重试']
X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

def softmax(scores: list[float], allowed: list[bool]) -> list[float]:
    if len(scores) != len(allowed) or not any(allowed):
        raise ValueError('softmax requires at least one allowed key')
    visible = [s for s, ok in zip(scores, allowed) if ok]
    if not all(math.isfinite(s) for s in visible):
        raise ValueError('visible scores must be finite')
    maximum = max(visible)
    exps = [math.exp(s - maximum) if ok else 0.0 for s, ok in zip(scores, allowed)]
    total = sum(exps)
    return [v / total for v in exps]

def attention(x: list[list[float]], causal: bool = False) -> dict:
    """Wq=Wk=I, Wv=2I; all rows are valid, no padding in this example."""
    if not x or any(len(row) != 2 for row in x):
        raise ValueError('this example requires nonempty two-dimensional rows')
    if not all(math.isfinite(v) for row in x for v in row):
        raise ValueError('input values must be finite')
    q = [row[:] for row in x]
    k = [row[:] for row in x]
    values = [[2 * v for v in row] for row in x]
    scores = [[sum(a * b for a, b in zip(query, key)) / math.sqrt(2) for key in k] for query in q]
    allowed = [[not causal or j <= i for j in range(len(x))] for i in range(len(x))]
    weights = [softmax(row, mask) for row, mask in zip(scores, allowed)]
    output = [[sum(w * v[d] for w, v in zip(row, values)) for d in range(2)] for row in weights]
    return {'x':x, 'q':q, 'k':k, 'v':values, 'scores':scores, 'allowed':allowed, 'weights':weights, 'output':output}

def trace() -> dict:
    changed = [row[:] for row in X]
    changed[2] = [0.0, 0.0]
    original = attention(X, causal=True)
    after = attention(changed, causal=True)
    return {'tokens_are_presegmented_teaching_labels':TOKENS,
            'full':attention(X), 'causal':original,
            'future_change':{'changed_x':changed,
                             'causal_output':after['output'],
                             'full_output':attention(changed)['output'],
                             'earlier_rows_unchanged':original['output'][:2] == after['output'][:2]}}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='print the full numerical trace as JSON')
    args = parser.parse_args()
    result = trace()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    for name, label in [('full','允许读取所有位置'), ('causal','只读取当前位置及左侧')]:
        r = result[name]
        print('\n' + label)
        for field in ['q','k','v','scores','allowed','weights','output']:
            print(f'{field}: {r[field]}')
    changed = result['future_change']['changed_x']
    print('\n未来输入替换为：', changed)
    changed_trace = attention(changed, causal=True)
    for field in ['q', 'k', 'v']:
        print(f'修改后 {field}: {changed_trace[field]}')
    print('因果版本修改后输出：', result['future_change']['causal_output'])
    print('无遮罩版本修改后输出：', result['future_change']['full_output'])
    print('因果版本较早两行是否不变：', result['future_change']['earlier_rows_unchanged'])
    print('这是给定参数的机制实验，没有训练模型，也不证明这些标签具备故障语义。')

if __name__ == '__main__':
    main()

# LLMs for Software Engineers

A Chinese course for programmers who have not systematically studied machine learning: **from one parameter to a generated sequence**.

[Read the course](chapters/01-models.md) · [Chinese README](README.md) · [Offline site](site/index.html)

18 chapters cover the minimum mathematics, counting language models, neural networks and backpropagation, tokenization, attention and Transformer blocks, CPU training, generation, post-training concepts, KV caching, and evidence-based evaluation.

## What makes the learning path concrete

- Small, fully worked numerical examples before general formulas.
- Executable Python experiments and tests for gradients, masking, cache equivalence, target construction and evaluation denominators.
- A counting baseline alongside the decoder, including cases where the decoder does **not** win.
- Clear separation between hand-set values, synthetic grammar, and an English literary corpus.
- Exercises with solutions and deliberate failure cases.

The core experiments run on CPU; no paid API or pretrained model download is required. The final evidence-evaluation project can use a model you already have, but its outputs must be recorded honestly. A supplied retrieval script is not presented as an LLM.

## Quick experiment

```bash
python3 labs/attention.py
```

Change a future input and check that earlier causal outputs remain unchanged. This is a mechanism experiment, not a trained language model.

For training, see [setup](SETUP.md), [toy report](reports/toy.md), and [real-text report](reports/real.md). The published small real-text decoder is slightly worse than its bigram baseline and repeats text under greedy generation; those limitations are retained as teaching evidence.

## Scope and contribution

The course does not train a general assistant, establish factual reliability, or prove business value. It was developed with AI assistance and executable checks; systematic testing with real beginner learners has not yet been completed. Concrete comprehension gaps, numerical errors, and reproducibility failures are welcome.

[Contribution guide](CONTRIBUTING.md) · [Attribution](NOTICE.md) · [Data provenance](data/README.md) · [MIT license](LICENSE)

Third-party corpus material retains its own rights and notices; it is not relicensed under MIT.

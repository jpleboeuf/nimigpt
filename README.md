# nimigpt

The most atomic way to train and run inference for a GPT in pure, dependency-free Nim.
This file is the complete algorithm. Everything else is just efficiency.

Faithful port of [@karpathy](https://github.com/karpathy)'s [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) to Nim.

## Compile & Run

```bash
nim c -d:release -d:ssl -r nimigpt.nim
```

On first run, `input.txt` (a list of 32K names) is downloaded automatically.

## What it does

Trains a tiny GPT (4,192 parameters, 1 layer, 4 heads) on human names for 1,000 steps, then generates 20 new, hallucinated names.

## Why Nim?

Same clarity as the Python original, with:

- **Types that document**
- **Native speed** -- compiles to C, no interpreter overhead
- **Single file, zero dependencies** -- just the Nim standard library

## Roadmap

The initial port is intentionally literal -- a line-by-line translation of the Python original.
Future iterations will introduce idiomatic Nim tricks, notably around metaprogramming, to make the code more expressive while keeping it just as readable.

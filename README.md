# nimigpt

The most atomic way to train and run inference for a GPT in pure, dependency-free Nim.
This file is the complete algorithm. Everything else is just efficiency.

This started as my faithful Nim port of [@karpathy](https://github.com/karpathy)'s [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

The preserved literal port lives on the [`faithful-port`](/../faithful-port/README.md) branch.

The default branch is the ongoing Nim-ization of that baseline.

## Compile & Run

```bash
nim c -d:release -d:ssl -r nimigpt.nim
```

On first run, `input.txt` (a list of 32K names) is downloaded automatically.

## Regression Fixture

The faithful port baseline has a deterministic golden output fixture at `tests/testdata/nimigpt_output_42.txt`.

Generate or refresh it with:

```bash
nim c tests/gen_fixture.nim
nim c -d:release -d:ssl nimigpt.nim
./tests/gen_fixture ./nimigpt tests/testdata/nimigpt_output_42.txt
```

Check the current program against it with:

```bash
nim c tests/test_output.nim
./tests/test_output ./nimigpt tests/testdata/nimigpt_output_42.txt
```

## What it does

Trains a tiny GPT (4,192 parameters, 1 layer, 4 heads) on human names for 1,000 steps, then generates 20 new, hallucinated names.

## Why Nim?

Same clarity as the Python original, with:

- **Types that document**
- **Native speed** -- compiles to C, no interpreter overhead
- **Single file, zero dependencies** -- just the Nim standard library

## Roadmap

The initial port was intentionally literal -- a line-by-line translation of the Python original.

The active branch now evolves that baseline into more idiomatic Nim while preserving the seeded regression fixture.

Future iterations will introduce idiomatic Nim tricks, notably around metaprogramming, to make the code more expressive while keeping it just as readable.

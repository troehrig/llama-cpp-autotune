# llama.cpp Autotune

`llama.cpp Autotune` measures a local GGUF model on the hardware where it will
actually run and recommends a reproducible `llama-server` command. It is aimed
at single-user local chats and agent workloads whose context grows over time,
including Hermes-style workflows that periodically compress the conversation
and begin another growth cycle.

The project does not promise a universal optimum. It searches a bounded set of
useful combinations, records every result, survives individual candidate
failures, and produces a practical recommendation with explicit limitations.

> Status: experimental public release, version 0.19.0. Review the generated
> command and protect any server exposed to a network.

## What it optimizes

- batch and micro-batch sizes
- CPU thread counts
- Flash Attention
- context sizes and KV-cache types
- supported speculative decoding and model-native MTP variants
- stability, memory headroom, prompt throughput, generation throughput, and
  end-to-end response time

The adaptive pipeline starts with inexpensive checks and only advances useful
candidates into more expensive long-context measurements:

1. hardware, binary, capability, and GGUF discovery
2. smoke benchmark
3. batch/UBatch/Flash-Attention screening
4. thread screening
5. context/KV-cache screening
6. speculation/MTP screening
7. repeated final validation against matching non-speculative controls
8. deterministic ranking and optional explanation by the local model

## Requirements

- Linux
- Python 3.10 or newer; runtime code uses only the standard library
- a local GGUF model
- a recent `llama.cpp` build containing `llama-server` and `llama-bench`
- enough storage for logs and enough RAM/VRAM for the selected model
- NVIDIA tools are recommended for GPU telemetry; CPU-only discovery remains
  possible, but the current search policy is primarily tested on one CUDA GPU

The project does not download models or install/build `llama.cpp`.

## Quick start

Run a complete Hermes-oriented search with only the binary directory and model
path:

```bash
python3 run_autotune.py \
  /path/to/llama.cpp/build/bin \
  /path/to/model.gguf
```

Defaults are the `quick` search profile, the `hermes` objective, an optional
local-model explanation, a 131,072-token Hermes deployment context when the
model supports it, and safe loopback binding at `127.0.0.1`.

For a more reliable search, use the balanced profile:

```bash
python3 run_autotune.py \
  /path/to/llama.cpp/build/bin \
  /path/to/model.gguf \
  --profile balanced
```

The final output line is a shell-safe, copy-and-paste-ready `llama-server`
command.

### Hermes on the local network

To make the generated deployment command listen on every network interface,
request it explicitly:

```bash
python3 run_autotune.py \
  /path/to/llama.cpp/build/bin \
  /path/to/model.gguf \
  --profile balanced \
  --deployment-host 0.0.0.0
```

`0.0.0.0` can expose the API to other devices and possibly untrusted networks.
Use firewall rules, a trusted network, and an authentication/reverse-proxy
layer appropriate to your installation. Autotune's temporary measurement
servers remain local and use temporary API keys; this option changes only the
final recommended deployment command.

## Analyze a completed run

Summarize the newest run without loading a model:

```bash
python3 analyze_autotune.py
```

Or select a run and save the summary:

```bash
python3 analyze_autotune.py runs/autotune_YYYYMMDD_HHMMSS \
  --write runs/summary.md
```

The analyzer reports the environment, model, completed stages, context
coverage, deterministic winner, alternatives by objective, memory headroom,
comparison with the matching `none` baseline, relevant artifacts, and the
copy-ready command.

## Search profiles and objectives

Search scope and ranking objective are separate choices.

| Option | Purpose |
|---|---|
| `--profile quick` | Small representative search, useful for compatibility tests |
| `--profile balanced` | Broader candidate set and repeated final measurements |
| `--profile thorough` | Largest built-in search and strongest repetition count |
| `--objective hermes` | Growing chats with recurring post-compression phases |
| `--objective balanced` | General local chat/agent compromise |
| `--objective interactive` | Token output and perceived response latency |
| `--objective long-context` | End-to-end work at large context sizes |
| `--objective throughput` | Prompt and generation throughput |

The `hermes` objective weights small, medium, and large context phases equally
because a compressed chat becomes small and grows again. The duration and
quality of the compression operation itself are not measured.

## Output and reproducibility

Each run gets a timestamped directory under `runs/`. Important artifacts
include:

- `manifest.json`: environment, model, capabilities, and top-level results
- `experiment_plan.json` and `.md`: planned adaptive search
- `autotune_state.json`: checkpoints and all stage results
- `recommendation.json`: deterministic final recommendation
- `autotune_report.md`: detailed human-readable report
- per-candidate commands, logs, responses, CSV files, and GPU snapshots
- `local_ai_analysis/`: optional interpretation by the tested model

The local AI explanation cannot change scores, measurements, or the command.
The deterministic result is written first and supplied to the model as
immutable input.

Run folders can contain absolute local paths, hardware identifiers, logs, and
model responses. They are ignored by Git and should be reviewed before being
shared. Model files, common credential files, and private keys are also ignored.

## Failure handling

Each server candidate runs in its own process group. Autotune waits for
readiness, performs the request, stores diagnostics, and terminates the whole
group. One failing candidate does not normally abort the search.

Reasoning models may consume their full output budget without final content.
For truncated reasoning and known `peg-gemma4` format failures, the same
candidate is retried with a fresh server and progressively larger token budgets,
up to the configured cap. Runtime/template incompatibility is reported instead
of being silently treated as poor performance.

## Tests

The unit suite does not require a GPU or model:

```bash
python3 -m py_compile \
  llama_autotune.py run_autotune.py analyze_autotune.py \
  tests/test_gguf_metadata.py
python3 -m unittest discover -s tests -v
```

Real hardware remains necessary to validate actual performance, memory limits,
and current `llama.cpp` behavior.

## Documentation

The design history, scoring rationale, safety model, known limitations, and
lessons from real model runs are documented in
[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) (German).

## AI-assisted development

This project was developed through an iterative human–AI collaboration with
OpenAI ChatGPT/Codex. Project goals, operating constraints, hardware
experiments, and validation were directed and reviewed by a human maintainer.
AI assisted with architecture, implementation, tests, diagnostics, and
documentation. AI-generated code and explanations are treated as reviewable
work products, not as authoritative benchmark results.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. Please report
security issues according to [SECURITY.md](SECURITY.md), and never attach model
files, credentials, or unreviewed run directories to a public issue.

## License and independence

Licensed under the [Apache License 2.0](LICENSE).

This is an independent community project. It is not affiliated with or
endorsed by the `llama.cpp`, Hermes, Unsloth, Qwen, Gemma, or OpenAI projects.

# Contributing

Contributions that improve reproducibility, compatibility, diagnostics, and
safe defaults are welcome.

## Development workflow

1. Fork the repository and create a focused feature branch.
2. Keep runtime code compatible with Python 3.10+ and the standard library.
3. Add or update unit tests for behavior changes.
4. Run the complete GPU-free test suite.
5. Open a pull request describing the problem, approach, and validation.

```bash
python3 -m py_compile \
  llama_autotune.py run_autotune.py analyze_autotune.py \
  tests/test_gguf_metadata.py
python3 -m unittest discover -s tests -v
```

Hardware results are useful but should complement, not replace, deterministic
tests. State the model architecture, quantization, `llama.cpp` commit, hardware,
profile, and objective when reporting performance.

## Data safety

Do not commit or attach model files, credentials, API keys, private keys, or
complete unreviewed `runs/` directories. Run artifacts can contain absolute
paths, hardware identifiers, logs, prompts, and model responses. Reduce reports
to the smallest sanitized example that reproduces the problem.

By submitting a contribution, you agree that it is licensed under the Apache
License 2.0 used by this project.

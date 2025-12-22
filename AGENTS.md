# Repository Guidelines

## Project Structure & Module Organization
- `mini-sglang-pearl/`: FastAPI/OpenAI server; entry `minisgl_pearl/server/__main__.py`; engine glue in `engine/pearl_engine.py`.
- `nano-PEARL/`: Core speculative decoding package and kernels in `nano_pearl/`; docs/bench assets live under `docs/` and `benchmark/`.
- `mini-sglang/`: Upstream continuous batching base; treat as vendored dependency—prefer upstream bumps over ad-hoc edits.
- Root scripts: `pearls.sh` (launch + /dev/shm cleanup), `benchmark_speculative.py`, `test_*` scripts, `docker-compose*.yml`; logs land in `server*.log` and `monitor_*`.

## Build, Test, and Development Commands
- Env: `python3 -m venv .venv && source .venv/bin/activate`.
- Install editable deps: `pip install -e nano-PEARL` then `pip install -e mini-sglang-pearl`.
- Run server with gamma profiling: `bash pearls.sh` (set model paths + TP settings first).
- Containers: `docker-compose -f docker-compose-minisgl.yml up` for a local stack.
- Quick API smoke: `curl http://localhost:30000/health` and `/v1/models` once up.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent; snake_case for funcs/vars, PascalCase for classes; favor type hints and module docstrings where behavior is subtle.
- Config via CLI flags/env; mirror OpenAI-style parameter names to avoid drift.
- Shell: keep scripts executable; preserve traps/cleanup for `/dev/shm` handles.

## Testing Guidelines
- Integration scripts assume the server on `localhost:30000`.
- Streaming parity: `python test_non_streaming.py`.
- Concurrency: `python test_concurrent_2users.py` (and similar `test_*` tools) to reproduce stress.
- Post–Auto Set Gamma perf: `python benchmark_speculative.py` to record throughput/MAT; add tokens/sec and latency in PR notes.

## Commit & Pull Request Guidelines
- Commits: short, present-tense summaries (e.g., `Fix streaming fallback`, `Update mini-sglang submodule`); couple code and config when tightly related.
- PRs: describe behavior change, perf impact (TTFT/throughput/MAT), and deployment notes (model paths, TP layout, ports); link issues and attach curl/log snippets for reproduction.
- Add or refresh tests or manual steps covering the change; call out any breaking API or config default adjustments.

## Security & Configuration Tips
- Server binds to `0.0.0.0:30000` by default—scope access or adjust port for private runs.
- Keep model paths/keys out of commits; load from env or local config. `/dev/shm/*_group` files are auto-cleaned by the `pearls.sh` trap.

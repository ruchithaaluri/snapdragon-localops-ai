# Snapdragon LocalOps AI

A local-first developer diagnostic assistant for Snapdragon-powered Windows PCs.

Point it at a build log, a stack trace or a CI failure. It extracts the
evidence, proposes ranked root-cause hypotheses, suggests read-only fixes, waits
for your approval, and then verifies whether the fix actually worked.

**Everything runs on your machine.** No account, no API key, no cloud call, no
telemetry.

---

## Phase 1 status

Phase 1 is the **honest, dependency-light shell**: the whole workflow runs
end-to-end on any Windows x64 developer machine, with no model download, no
accelerator and no vendor SDK.

| Area | State |
| --- | --- |
| Evidence extraction with line citations | working |
| Rule-based diagnosis, ranked and cited | working |
| Remediation planning (proposes, never applies) | working |
| Approval gate (default: approves nothing) | working |
| Safety policy and command deny list | working |
| Verification runner (6 read-only check kinds) | working |
| PySide6 desktop shell (5 tabs, threaded) | working |
| Headless CLI (imports no Qt) | working |
| Structured JSONL logging with session ids | working |
| Mock AI backend (deterministic, offline) | working |
| SQLite run history | built, not yet wired in |
| Retrieval index feeding the pipeline | built, not yet wired in |
| Local model execution (ONNX Runtime) | Phase 2 |
| Snapdragon / NPU execution provider | Phase 3+ |

---

## Requirements

* Windows x64 (the product targets Windows; the core is portable Python)
* Python 3.11 or newer
* Disk space for the Qt toolkit - a couple of hundred megabytes

**No AI engine, accelerator or Snapdragon package is required to start the
application.**

---

## Install

```powershell
git clone <your-fork-url> snapdragon-localops
cd snapdragon-localops

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements/dev.txt
```

`requirements/dev.txt` pulls in `base.txt` (PySide6, pydantic,
pydantic-settings) plus the test tooling.

To work against an installed copy rather than the source tree:

```powershell
python -m pip install -e .
```

That also creates the `snapdragon-localops` launcher.

---

## Running

### Desktop shell

```powershell
python -m app.main
```

Opens the five-tab window: **Diagnosis**, **Evidence**, **Suggested fixes**,
**Verification**, **This machine**. The pipeline runs on a worker thread, so the
window stays responsive while a run reads files and starts subprocesses.

### Headless (SSH, CI, no display)

```powershell
python -m app.main --diagnose path\to\build.log
Get-Content build.log | python -m app.main --diagnose -
```

Prints the run outcome, every stage, the ranked hypotheses with their
confidence, the caveats and the proposed actions - then stops. Nothing is
approved and nothing is executed.

### All flags

| Flag | Meaning |
| --- | --- |
| `--config PATH` | Load a TOML config file |
| `--data-dir PATH` | Override the data directory |
| `--log-level LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `--diagnose [PATH]` | Run headlessly on a file, or stdin with `-` |
| `--question TEXT` | The question to answer in `--diagnose` mode |
| `--version` | Print the version and exit |

---

## Running the tests

```powershell
python -m pytest
```

133 tests, a couple of seconds, no network access and no model downloads. The
UI tests run on Qt's `offscreen` platform, so no display is required.

Useful subsets:

```powershell
python -m pytest -m "not ui"        # skip anything that builds Qt widgets
python -m pytest tests/test_safety.py -v
python -m pytest -q --tb=short
```

No test touches your real user profile: `tests/conftest.py` points the whole
application at a temporary data directory for every single test.

---

## The honesty rules

These are enforced in code and covered by `tests/test_honesty_rules.py`. They
are the reason the project is worth building.

* **The application never claims an NPU it has not verified.** The "This
  machine" panel reports `cpu (verified)` on a machine with no accelerator, and
  a test fails if an NPU string ever appears.
* **Retrieved text is context, never evidence.** Passages found by retrieval may
  inform a hypothesis but can never be cited as proof of anything.
* **"I do not know" is a valid result.** When the evidence does not support a
  conclusion, the run reports `unresolved` and explains why instead of guessing.
* **A generic catch-all is capped as a lead,** never promoted to a conclusion.
* **Every stage is recorded, including the ones that did not run** - a skipped
  stage says why it was skipped. A pipeline that silently omits a stage cannot
  be told apart from one where the stage legitimately found nothing.
* **Statistics need enough samples,** and every measurement carries its
  provenance.
* **Commands are never re-interpreted by a shell** - arguments are passed as an
  argv list with `shell=False`.
* **Silently truncating is not allowed.** When evidence is capped, the run says
  so.

---

## Safety model

The pipeline **stops at the approval gate**, and no configuration setting can
skip it.

1. The planner *proposes* read-only actions. It never executes.
2. The UI shows every action with its risk, its description and the exact
   command it would run.
3. You approve or deny each action individually.
4. Only approved actions run.
5. The default policy blocks mutating actions, blocks network actions, enforces
   a maximum risk level, and rejects a deny list of obvious landmines (recursive
   deletes, disk formatting, `rm -rf /` and similar).

`require_approval` cannot be set to `false` - the configuration validator
raises. Approval is recorded per action id, and `apply_plan` deliberately does
*not* rebuild the plan, so the actions you approved are the exact actions that
run.

---

## Configuration

Every setting has a working default, so no config file is required. To use one:

```powershell
python -m app.main --config config\localops.toml
```

Precedence, lowest to highest:

1. built-in defaults
2. the TOML file
3. `LOCALOPS_*` environment variables (nested with `__`, e.g.
   `LOCALOPS_UI__THEME=light`)
4. explicit overrides passed in code

Unknown keys are **rejected rather than ignored**, which turns a typo into an
error message instead of a silently misconfigured tool.

### Where files live

The data root resolves in this order: an explicit override, then
`LOCALOPS_DATA_DIR`, then `%LOCALAPPDATA%\SnapdragonLocalOps` on Windows
(`~/.local/share/snapdragon-localops` elsewhere).

```
<data root>/
    logs/       rotating JSONL application logs
    runtime/    SQLite database and mutable state
    index/      persisted retrieval indexes
    reports/    benchmark and diagnostic reports
    cache/      regenerable caches - safe to delete
    models/     model artifacts (never created automatically)
```

`models/` is intentionally **not** created on start-up, so the application can
never imply a model is present when it is not.

`config/localops.toml` is gitignored, so local experiments never get
committed.

---

## Project layout

```
app/          entry point, pipeline orchestration, PySide6 shell
core/         contracts, config, safety, errors, observability, storage
diagnostics/  evidence extraction, rule engine, remediation planner + executor
verification/ read-only check runner and check derivation
retrieval/    chunking, in-memory vector index, retriever
ai/           AI backend interface, mock backend, ONNX Runtime backend
benchmark/    measurement harness and host description
tests/        the test suite
```

`core/contracts/` defines the interfaces; each layer package supplies concrete
implementations. The desktop shell renders results and computes none of them,
so the whole flow is equally reachable from a test or the CLI.

Every provider exposes `info` and `health()`, so a stage can report *why* it is
unavailable instead of failing silently.

### Optional local inference

The local inference stack lives in `requirements/ai.txt` and is **not installed
during Phase 1**:

```powershell
python -m pip install -r requirements/ai.txt
```

ONNX Runtime is hardware-agnostic: it runs on x64 today, and on the Snapdragon
target it can be paired with a vendor execution provider. Qualcomm AI Hub
Workbench is deliberately absent - it compiles models, it does not run them, and
it belongs on a different machine.

---

## What Phase 1 deliberately does not do

* It does not download or run a language model.
* It does not claim NPU acceleration.
* It does not modify your system without an explicit per-action approval.
* It does not require a Snapdragon device, or a vendor SDK, to be useful.

Phase 2 adds real local model execution through ONNX Runtime. Phase 3+ adds the
Snapdragon execution provider on the target hardware.

---

## License

Apache-2.0



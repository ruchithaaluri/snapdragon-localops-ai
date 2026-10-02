# Snapdragon LocalOps AI

A local-first developer diagnostic assistant for Snapdragon-powered Windows PCs.

Point it at a build log, a stack trace or a CI failure. It extracts the
evidence, retrieves relevant context, proposes ranked root-cause hypotheses with
citations, suggests a fix, **waits for your explicit approval**, applies only a
predefined safe action, and then re-runs verification to show before/after.

**Everything runs on your machine.** No account, no API key, no cloud call, no
telemetry.

---

## Status at a glance

| Area | State |
| --- | --- |
| Evidence extraction with line citations | **implemented** |
| Lightweight retrieval (4-passage knowledge base) | **implemented** |
| Rule-based diagnosis, ranked and cited | **implemented** |
| Recommended fix with evidence references | **implemented** |
| Explicit per-action approval gate | **implemented** |
| Safe remediation via a fixed whitelist | **implemented** |
| Verification + before/after result | **implemented** |
| Deterministic demo scenario (FAILED -> PASSED) | **implemented** |
| Benchmark framework (real measurements only) | **implemented** |
| PySide6 desktop shell (7 tabs, threaded) | **implemented** |
| Headless CLI (`--demo`, `--benchmark`, `--diagnose`) | **implemented** |
| Mock AI backend (default, offline, no model) | **implemented** |
| LocalAIBackend on ONNX Runtime | **implemented, no model shipped** |
| SnapdragonAIBackend interface | **implemented, NOT validated** |
| NPU execution on real hardware | **target-hardware validation pending** |

> **Snapdragon deployment validation pending target hardware.**
> No NPU execution, latency or memory figure in this repository was produced on
> Snapdragon hardware. This development machine is Windows x64. The Snapdragon
> backend is structurally complete and correctly reports itself unvalidated.

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/dev.txt
```

**Launch the application:**

```powershell
python -m app.main
```

**Run the competition demo (headless, ~2 seconds):**

```powershell
python -m app.main --demo
```

**Show that nothing happens without approval:**

```powershell
python -m app.main --demo --demo-no-approve
```

**Measure this machine:**

```powershell
python -m app.main --benchmark
```

---

## The demo scenario

One deterministic story, offline, repeatable. The sample app ships with a real
defect - it imports a module that does not exist - so it genuinely cannot start.

1. The app is run and **really fails** with a `ModuleNotFoundError`.
2. The real traceback is captured to a log file.
3. Evidence extraction, retrieval and rule-based diagnosis run on it.
4. One whitelisted fix is proposed, citing the extracted evidence.
5. **Nothing is applied until approval is recorded.**
6. The fix rewrites one import line, in-process, with no shell.
7. The same check runs again.

```
BEFORE: FAILED
AFTER:  PASSED
```

Both verdicts come from a subprocess exit code, so the display cannot disagree
with what happened. The demo resets the sample app to its broken state at the
start of every run, so it works on the second run exactly as it did on the first.

With `--demo-no-approve`, the gate stops the workflow: the app stays broken and
verification **fails**. That is the safety claim demonstrated rather than
asserted.

---

## Architecture

```
Application (app/, PySide6 UI, headless CLI)
    |
    +-- AIBackend interface  (core/contracts/ai.py)
            |
            +-- MockAIBackend            deterministic, offline, no model (default)
            +-- LightweightLocalAIBackend  real ONNX Runtime, no downloads
            +-- SnapdragonAIBackend        NPU target, NOT validated
```

The application never imports a backend directly. Selection is a configuration
change (`ai.llm.name`), and an unknown name is an error rather than a silent
fallback to the mock.

```
             evidence
                |
     +----------+-----------+
     |                      |
 retrieval            diagnosis (rules)
 (context only)              |
     |                      |
     +----------+-----------+
                |
          remediation plan
                |
     ===== approval gate =====
                |
        safe action (whitelist)
                |
          verification
                |
        BEFORE / AFTER
```

### Layer map

| Package | Responsibility |
| --- | --- |
| `app/` | Entry point, pipeline orchestration, PySide6 shell |
| `core/` | Contracts, config, safety kernel, errors, logging, storage |
| `diagnostics/` | Evidence extraction, rule engine, remediation planner |
| `demo/` | The demo scenario and the whitelisted safe actions |
| `retrieval/` | Chunking, in-memory vector index, retriever |
| `ai/` | AI backends (mock, local, snapdragon), ONNX runtime |
| `benchmark/` | Measurement harness and host description |
| `verification/` | Read-only check runner and check derivation |
| `tests/` | The test suite |

`core/contracts/` defines the interfaces; each layer package supplies concrete
implementations. The desktop shell renders results and computes none of them,
so the whole flow is equally reachable from a test or the CLI.

---

## The desktop shell

```powershell
python -m app.main
```

Seven tabs: **Diagnosis**, **Evidence**, **Suggested fixes**, **Verification**,
**Demo**, **Benchmark**, **This machine**. The pipeline and the benchmark both
run on worker threads, so the window stays responsive while a run starts
subprocesses.

### The status strip

A bar across the top always states, in words:

| Indicator | Meaning |
| --- | --- |
| `LOCAL MODE` | No data leaves this machine |
| `MOCK AI` / `REAL AI` | Whether a model or the deterministic mock produced output |
| `CPU` / `GPU` / `NPU` | Where work executed |
| `VERIFIED` | That a real run backed the claim |

`NPU` appears **only** after a verified NPU inference. An advertised-but-unproven
provider renders as `NPU?` in a warning colour, never as a result. On this
development machine the strip reads `LOCAL MODE` / `MOCK AI` / `CPU`.

Every indicator is a word as well as a colour, so it stays readable in greyscale
and for a user with colour vision deficiency.

### All flags

| Flag | Meaning |
| --- | --- |
| `--config PATH` | Load a TOML config file |
| `--data-dir PATH` | Override the data directory |
| `--log-level LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `--demo` | Run the built-in scenario end to end and print the result |
| `--demo-no-approve` | With `--demo`, stop at the gate and change nothing |
| `--benchmark` | Measure this machine's real pipeline latency |
| `--diagnose [PATH]` | Run headlessly on a file, or stdin with `-` |
| `--question TEXT` | The question to answer in `--diagnose` mode |
| `--version` | Print the version and exit |

### Headless (SSH, CI, no display)

```powershell
python -m app.main --diagnose path\to\build.log
Get-Content build.log | python -m app.main --diagnose -
```

---

## Running the tests

```powershell
python -m pytest
```

185 tests, a few seconds, no network access and no model downloads. The UI tests
run on Qt's `offscreen` platform, so no display is required.

Useful subsets:

```powershell
python -m pytest -m "not ui"        # skip anything that builds Qt widgets
python -m pytest tests/test_demo_and_backends.py -v
python -m pytest -q --tb=short
```

No test touches your real user profile: `tests/conftest.py` points the whole
application at a temporary data directory for every single test.

| File | Covers |
| --- | --- |
| `test_diagnostics_flow.py` | Evidence, diagnosis, remediation, approval |
| `test_evidence.py` | Extraction patterns and line citations |
| `test_safety.py` | Approval gate, risk policy, command deny list |
| `test_honesty_rules.py` | NPU-claim refusal, benchmark provenance |
| `test_storage_and_retrieval.py` | Run history, index persistence |
| `test_verification.py` | Check kinds and failure handling |
| `test_ui.py` | The Qt shell renders and claims nothing |
| `test_config_and_cli.py` | Configuration validation, entry points |
| `test_demo_and_backends.py` | The demo, whitelist, backends, benchmark |

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

1. The planner *proposes* actions. It never executes.
2. The UI shows every action with its risk, its description and the exact
   target it would change.
3. You approve or deny each action individually.
4. Only approved actions run.
5. The default policy blocks mutating actions, blocks network actions, enforces
   a maximum risk level, and rejects a deny list of obvious landmines (recursive
   deletes, disk formatting, `rm -rf /` and similar).

`require_approval` cannot be set to `false` - the configuration validator
raises. Approval is recorded per action id, and `apply_plan` deliberately does
*not* rebuild the plan, so the actions you approved are the exact actions that
run.

### Remediations are a whitelist, not a command interpreter

`demo/safe_actions.py` holds the complete set of changes this program is
capable of making. Each entry is a reviewed **id** plus a reviewed Python
callable. There is no code path from a plan, a log line, or pasted user text to
a shell, an interpreter, `pip`, or the network.

`apply_safe_action` checks, in this order:

1. the action id must exist in the whitelist - an unknown id is **denied, not
   attempted**;
2. the safety policy must permit the action's risk;
3. the caller must have recorded an approval.

A whitelisted id is not a licence: `apply_safe_action(action, approved=False)`
raises. The target path travels inside the action, so an approval is bound to
*this* action on *this* file and cannot be replayed against another one.

Adding a capability means adding a reviewed entry to that list, which is the
point: what this tool can do to a machine is visible in one short file.

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
demo/         the demo scenario and the whitelisted safe actions
verification/ read-only check runner and check derivation
retrieval/    chunking, in-memory vector index, retriever
ai/           AI backends (mock, local, snapdragon) + ONNX Runtime layer
benchmark/    measurement harness, host description, default case suite
tests/        the test suite
```

`core/contracts/` defines the interfaces; each layer package supplies concrete
implementations. The desktop shell renders results and computes none of them,
so the whole flow is equally reachable from a test or the CLI.

Every provider exposes `info` and `health()`, so a stage can report *why* it is
unavailable instead of failing silently.

---

## AI backends

All three implement the same `AIBackend` facade. Select one in configuration:

```toml
[ai.llm]
name = "mock"        # default
```

| Name | Class | What it does |
| --- | --- | --- |
| `mock` (default) | `MockAIBackend` | Deterministic, offline, no model, no download |
| `local` | `LightweightLocalAIBackend` | Real ONNX Runtime inference on a model already on disk |
| `snapdragon` | `SnapdragonAIBackend` | NPU target path - **not validated** |

Selection never falls back silently. Choosing `local` with no model configured
reports `not configured` rather than quietly reverting to the mock, and an
unknown name raises `ProviderNotFoundError`.

**`local`** performs a real forward pass over a small encoder-style model and
reads the output. It is not a token-by-token text generator yet, and says so in
the UI. It never downloads anything: a model must already exist on disk and be
named in configuration. Install the optional stack with:

```powershell
python -m pip install -r requirements/ai.txt    # numpy + onnxruntime
```

**`snapdragon`** probes for the Qualcomm runtime and reports exactly what it
finds. On this x64 machine it reports `not a Snapdragon host`. Every capability
returns `None` and every load raises until a vendor runtime and a measured
inference exist.

---

## Snapdragon deployment architecture

```
Application
    |
 AIBackend (interface - the only thing the app knows about)
    |
 SnapdragonAIBackend
    |
 SnapdragonRuntime  ->  Qualcomm execution provider (absent until target HW)
    |
 LoadedModel  ->  ExecutionTarget  ->  verified only by a real forward pass
```

**Status: Snapdragon deployment validation pending target hardware.**

To make the NPU path real on the HP Snapdragon target:

1. Install the Qualcomm runtime **on the target machine** (never on x64).
2. Point configuration at a QNN/HTP-compiled artifact.
3. Run `python -m app.main --benchmark` on the target so a real forward pass can
   promote the target to `VERIFIED`.
4. Call `SnapdragonAIBackend.record_verified_run(target)` with that verified
   target, which is the only way the `NPU` indicator becomes available.

Until step 4 happens on real hardware, the UI shows `CPU` and this README says
validation is pending. Qualcomm AI Hub Workbench is deliberately absent - it
compiles models, it does not run them, and it belongs on a different machine.

---

## Benchmark methodology

`python -m app.main --benchmark` measures four cases - evidence extraction,
rule-based diagnosis, retrieval search, and the end-to-end pipeline - on the
same few-hundred-byte input the demo produces.

How a number is produced:

1. run `warmup_iterations` and discard them, so first-call import cost is not
   reported as steady-state cost;
2. run `iterations` timed iterations with `time.perf_counter_ns`, a monotonic
   clock;
3. compute every statistic **from those samples** - never from a typed-in
   number, never estimated.

A case that fails becomes a note saying `NOT MEASURED`. It is never dropped, and
never becomes a zero. Below five samples, the noisy aggregates (stdev, p95) are
omitted rather than presented as precision.

Every row records: method, host, runtime, iteration count, warmup count, model
id and artifact checksum, plus the execution target and how it was verified.
`Measurement` raises if a row claims an unverified NPU, so an NPU number cannot
be recorded by mistake even if the harness were modified.

The full suite runs in well under a second and touches no model, no network, and
nothing on disk.

---

## Known limitations

* **The diagnosis is rule-based.** The demo's explanation comes from a fixed
  signature table, reported as `rules_only`. It is not model inference and is
  not presented as such.
* **The embedding backend is a deterministic mock** - hashed bag-of-words. It
  matches shared words, not meaning, and says so in its own provenance. Real
  retrieval quality needs a real embedding model.
* **The demo's knowledge base is four hand-written passages**, not a corpus.
  Retrieval is demonstrated by picking the right one out of a handful.
* **No model ships with the application.** `local` is fully implemented but has
  nothing to load until you provide a model on disk.
* **The NPU path is unvalidated.** No Snapdragon execution has been performed.
* **Retrieval outside the demo is empty by default.** Nothing is indexed
  automatically - a tool that silently indexes a whole project is a tool the
  user cannot reason about. Index explicitly via `LocalRetriever.index_documents`.
* `benchmark/` measures the deterministic stages. There is no model-inference
  row, because no model is present to measure.

---

## License

Apache-2.0



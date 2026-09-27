"""Evidence extraction: raw text in, cited :class:`Evidence` out.

The extractor is deterministic pattern matching, not inference. That is a
deliberate choice for this phase: the output of a real model is harder to check,
whereas a pattern hit can be verified by reading the line. Each extracted item
records the line it came from, so a hypothesis can always be traced back to a
specific row of a specific file.

Nothing here reads a file by itself - the caller passes text it already has.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from core.contracts.diagnostics import Evidence, EvidenceKind
from core.observability import get_logger, log_event

__all__ = [
    "ERROR_PATTERNS",
    "ErrorSignature",
    "SIGNATURES",
    "dedupe",
    "evidence_from_user_input",
    "extract_evidence",
    "signature_for",
]


@dataclass(frozen=True, slots=True)
class ErrorSignature:
    """A named, recognisable error shape and the category it belongs to."""

    name: str
    category: str
    pattern: re.Pattern[str]
    kind: EvidenceKind = EvidenceKind.ERROR_MESSAGE
    #: What this error usually means: a hint for the user, not a diagnosis.
    typical_cause: str = ""

    def matches(self, line: str) -> bool:
        return bool(self.pattern.search(line))


def _signature(
    name: str,
    category: str,
    pattern: str,
    *,
    kind: EvidenceKind = EvidenceKind.ERROR_MESSAGE,
    typical_cause: str = "",
) -> ErrorSignature:
    """Build a signature, compiling the pattern once at import time."""
    return ErrorSignature(
        name=name,
        category=category,
        pattern=re.compile(pattern),
        kind=kind,
        typical_cause=typical_cause,
    )


#: Recognisable error shapes, ordered from most specific to most generic so
#: :func:`signature_for` returns the most informative match rather than a
#: catch-all. These are ordinary developer failures on a Windows workstation -
#: nothing here is specific to Snapdragon hardware.
SIGNATURES: tuple[ErrorSignature, ...] = (
    # -- Python -----------------------------------------------------------
    _signature(
        "python_module_not_found",
        "python",
        r"ModuleNotFoundError: No module named ['\"]([\w.]+)['\"]",
        typical_cause=(
            "the package is missing from the active interpreter, or the code is "
            "running under a different virtual environment than expected"
        ),
    ),
    _signature(
        "python_import_error",
        "python",
        r"\bImportError\b|cannot import name",
        typical_cause="an import target was renamed, moved, or is circular",
    ),
    _signature(
        "python_syntax_error",
        "python",
        r"\b(?:SyntaxError|IndentationError|TabError)\b",
        typical_cause="the source does not parse; check indentation and quoting",
    ),
    _signature(
        "python_attribute_error",
        "python",
        r"AttributeError: .* has no attribute",
        typical_cause="a None value was dereferenced, or a name was renamed",
    ),
    _signature(
        "python_type_error",
        "python",
        r"TypeError: ",
        typical_cause="a value of the wrong type reached a function boundary",
    ),
    _signature(
        "python_value_error",
        "python",
        r"ValueError: ",
        typical_cause="a value was well-typed but out of range or malformed",
    ),
    _signature(
        "python_key_error",
        "python",
        r"KeyError: ",
        typical_cause="a mapping was indexed with a key it does not contain",
    ),
    _signature(
        "python_file_not_found",
        "python",
        r"\bFileNotFoundError\b|\bErrno 2\b",
        typical_cause="a path is wrong, or the working directory is not what was assumed",
    ),
    _signature(
        "python_permission_error",
        "python",
        r"\bPermissionError\b|\bErrno 13\b",
        typical_cause="the process lacks rights to the file, directory, or port",
    ),
    _signature(
        "python_recursion_error",
        "python",
        r"RecursionError|maximum recursion depth",
        typical_cause="a recursive function has no terminating condition",
    ),
    _signature(
        "python_memory_error",
        "python",
        r"MemoryError|out of memory|JavaScript heap out of memory|exit code 137",
        typical_cause="the working set exceeded available RAM or a configured cap",
    ),
    _signature(
        "python_name_error",
        "python",
        r"\bNameError: |\bUnboundLocalError\b",
        typical_cause="a name is used before assignment or was never defined",
    ),
    _signature(
        "python_json_decode",
        "python",
        r"JSONDecodeError|Expecting value: line",
        typical_cause="malformed JSON input, or an empty response body",
    ),
    _signature(
        "python_unicode_error",
        "python",
        r"Unicode(?:Decode|Encode)Error|invalid start byte",
        typical_cause="text was read or written with the wrong encoding",
    ),
    _signature(
        "python_never_awaited",
        "python",
        r"was never awaited|attached to a different loop|event loop is closed",
        typical_cause="async code was called synchronously, or the loop was torn down",
    ),
    _signature(
        "python_traceback_header",
        "python",
        r"Traceback \(most recent call last\)",
        kind=EvidenceKind.STACK_TRACE,
    ),
    _signature(
        "python_frame",
        "python",
        r'^\s*File "[^"]+", line \d+',
        kind=EvidenceKind.STACK_TRACE,
        typical_cause="a stack frame pointing at the call site",
    ),
    # -- Node / TypeScript ------------------------------------------------
    _signature(
        "node_module_not_found",
        "node",
        r"Cannot find module|Cannot find package|MODULE_NOT_FOUND|ERR_MODULE_NOT_FOUND",
        typical_cause="dependencies are not installed; the lockfile is out of sync",
    ),
    _signature(
        "node_port_in_use",
        "node",
        r"EADDRINUSE|address already in use",
        typical_cause="another process already holds the port",
    ),
    _signature(
        "node_npm_error",
        "node",
        r"npm ERR!|error command failed|ELIFECYCLE",
        typical_cause="an npm lifecycle script failed",
    ),
    _signature(
        "typescript_error",
        "node",
        r"\berror TS\d+:",
        typical_cause="a TypeScript type error; the file and line are in the message",
    ),
    _signature(
        "node_unhandled_rejection",
        "node",
        r"Unhandled promise rejection|Unhandled 'error' event",
        typical_cause="an event handler threw, or a promise was never awaited",
    ),
    # -- Native build toolchains -----------------------------------------
    _signature(
        "build_failed",
        "build",
        r"BUILD FAILED|ninja: build stopped|make: \*\*\* ",
        typical_cause="a build tool ran a step that did not succeed",
    ),
    _signature(
        "compiler_error",
        "build",
        r"fatal error C\d+|\berror C\d+:|\b: (?:fatal )?error:",
        typical_cause="a compiler rejected the translation unit",
    ),
    _signature(
        "linker_error",
        "build",
        r"\bLNK\d{4}\b|undefined reference to|undefined symbol",
        typical_cause="a symbol was declared but not linked in",
    ),
    _signature(
        "cmake_error",
        "build",
        r"CMake Error|Could NOT find ",
        typical_cause="the build configuration could not find a toolchain or dependency",
    ),

    # -- Windows / system -------------------------------------------------
    _signature(
        "windows_access_denied",
        "windows",
        r"Access is denied\.|UnauthorizedAccessException",
        typical_cause="the account lacks permission, or the file is locked by another process",
    ),
    _signature(
        "windows_path_not_found",
        "windows",
        r"The system cannot find the path specified|cannot find the path",
        typical_cause="a directory component of the path does not exist",
    ),
    _signature(
        "dll_load_failure",
        "windows",
        r"DLL load failed|The code execution cannot proceed because .*\.dll was not found",
        typical_cause=(
            "a native dependency is missing from PATH or from the application "
            "directory; a missing VC++ runtime is a common cause"
        ),
    ),
    _signature(
        "access_violation",
        "windows",
        r"0x[cC]0000005|Segmentation fault|Access violation",
        typical_cause="native code dereferenced invalid memory",
    ),
    # -- Version control --------------------------------------------------
    _signature(
        "git_conflict",
        "vcs",
        r"CONFLICT \(|Automatic merge failed",
        typical_cause="both sides changed the same lines",
    ),
    _signature(
        "git_no_repository",
        "vcs",
        r"fatal: not a git repository",
        typical_cause="the command ran outside a work tree",
    ),
    _signature(
        "git_ssh_denied",
        "vcs",
        r"Permission denied \(publickey\)|Host key verification failed",
        typical_cause="SSH credentials are missing, or the known-hosts entry is absent",
    ),
    _signature(
        "git_bad_revision",
        "vcs",
        r"fatal: bad revision|pathspec .* did not match any file",
        typical_cause="a branch, tag, or path in the command does not exist",
    ),
    # -- Network ----------------------------------------------------------
    _signature(
        "connection_refused",
        "network",
        r"ECONNREFUSED|Connection refused",
        typical_cause="nothing is listening on the target address",
    ),
    _signature(
        "dns_failure",
        "network",
        r"getaddrinfo|Name or service not known|nodename nor servname",
        typical_cause="a hostname could not be resolved",
    ),
    _signature(
        "network_timeout",
        "network",
        r"ETIMEDOUT|Read timed out|timed out after \d+",
        typical_cause="a request exceeded its deadline",
    ),
    _signature(
        "tls_failure",
        "network",
        r"CERTIFICATE_VERIFY_FAILED|self signed certificate|SSL:",
        typical_cause="a TLS chain did not validate, commonly behind a proxy",
    ),
    # -- Local model runtime ---------------------------------------------
    _signature(
        "model_load_failure",
        "model_runtime",
        r"Failed to load model|Load model .* failed|is not a valid ONNX model",
        typical_cause=(
            "the artifact is missing, corrupt, or was built for a different "
            "runtime version"
        ),
    ),
    _signature(
        "accelerator_unavailable",
        "model_runtime",
        r"QNN.*(?:not available|unavailable|failed)|HTP.*(?:not available|unavailable)|"
        r"npu.*(?:not found|unavailable)",
        typical_cause=(
            "no accelerator execution provider is usable here; work must fall "
            "back to CPU, and the run must then be reported as CPU"
        ),
    ),
    # -- Generic last resorts --------------------------------------------
    _signature(
        "unhandled_exception",
        "generic",
        r"Unhandled exception|Uncaught \w*(?:Error|Exception)|^panic:",
        typical_cause="an exception escaped a handler",
    ),
    _signature(
        "fatal_error",
        "generic",
        r"\bfatal:|\bFAILED\b|\bError:\s+\S",
        typical_cause="a tool reported a failure; surrounding lines hold the detail",
    ),
)



#: Flat ``name -> pattern`` view of :data:`SIGNATURES`, for callers that want the
#: raw patterns (documentation, tests) rather than the typed descriptors.
ERROR_PATTERNS: dict[str, re.Pattern[str]] = {sig.name: sig.pattern for sig in SIGNATURES}

#: How much a match of each category is trusted. A named exception type is
#: strong evidence; a generic ``Error:`` line is a lead, not a conclusion.
_CATEGORY_CONFIDENCE: dict[str, float] = {
    "python": 0.85,
    "node": 0.85,
    "vcs": 0.85,
    "build": 0.80,
    "windows": 0.80,
    "network": 0.80,
    "model_runtime": 0.85,
    "generic": 0.50,
}

_log = get_logger(__name__)


def _confidence_for(signature: ErrorSignature) -> float:
    """Confidence that a line matching ``signature`` indicates a real fault."""
    if signature.kind is EvidenceKind.STACK_TRACE:
        return 0.60
    return _CATEGORY_CONFIDENCE.get(signature.category, 0.60)


def signature_for(line: str) -> ErrorSignature | None:
    """Return the first signature matching ``line``, or ``None``.

    :data:`SIGNATURES` is ordered specific-to-generic, so the first hit is the
    most informative one. A line that matches nothing returns ``None`` rather
    than falling back to a catch-all: "no known error here" is itself
    information, and a vague signature would erase it.
    """
    for signature in SIGNATURES:
        if signature.matches(line):
            return signature
    return None


def _evidence_key(item: Evidence) -> tuple[str, str]:
    """Normalised identity of an evidence item, used for de-duplication."""
    return (item.kind.value, " ".join(item.content.split()).lower())


def dedupe(items: Sequence[Evidence]) -> tuple[Evidence, ...]:
    """Drop repeated items, keeping the first occurrence.

    A log repeats the same error thousands of times. A diagnosis is far easier
    to read when one message appears once, carrying the first line number that
    produced it.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Evidence] = []
    for item in items:
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)



def extract_evidence(
    text: str,
    *,
    kind: EvidenceKind = EvidenceKind.LOG_FILE,
    source: str = "",
    max_items: int = 50,
) -> tuple[Evidence, ...]:
    """Scan ``text`` line by line and return the errors it contains.

    Only lines matching a known signature become evidence, plus the stack frames
    that lead to them. Ordinary log lines are not errors and are not returned -
    the caller already holds the raw text and does not need it duplicated here.

    ``max_items`` bounds the result so a huge log cannot stall the UI; truncation
    is logged rather than silently applied.

    >>> items = extract_evidence("ModuleNotFoundError: No module named 'foo'")
    >>> items[0].metadata["signature"]
    'python_module_not_found'
    """
    if max_items < 1:
        raise ValueError("max_items must be >= 1")

    found: list[Evidence] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        signature = signature_for(stripped)
        if signature is None:
            continue
        if len(found) >= max_items:
            log_event(
                _log,
                logging.WARNING,
                "evidence.truncated",
                source=source,
                max_items=max_items,
            )
            break
        found.append(
            Evidence(
                # A signature may re-type the line (a stack frame is not an
                # error message); otherwise the caller's context applies.
                kind=signature.kind if signature.kind is not EvidenceKind.ERROR_MESSAGE else kind,
                content=stripped,
                source=source,
                locator=f"line {number}",
                confidence=_confidence_for(signature),
                metadata={"signature": signature.name, "category": signature.category},
            )
        )
    return dedupe(found)


def evidence_from_user_input(text: str, *, source: str = "user input") -> tuple[Evidence, ...]:
    """Turn what the user typed or pasted into evidence.

    The text itself is always kept as one :attr:`EvidenceKind.USER_INPUT` item,
    so a diagnosis can cite the very thing that was asked. Recognised error lines
    inside the paste are extracted as well, which is what makes pasting a stack
    trace useful.
    """
    stripped = text.strip()
    if not stripped:
        return ()
    # Extracted items come first: they carry the signature and category that
    # de-duplication would otherwise discard in favour of the plain verbatim
    # copy. The verbatim text is appended afterwards, so multi-line input keeps
    # both the whole question and the specific lines found inside it.
    items: list[Evidence] = list(
        extract_evidence(stripped, kind=EvidenceKind.USER_INPUT, source=source)
    )
    items.append(
        Evidence(kind=EvidenceKind.USER_INPUT, content=stripped, source=source)
    )
    return dedupe(items)



#!/usr/bin/env python3
"""Llama Autotune: sichere Bestandsaufnahme vor dem eigentlichen Tuning."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import io
import json
import math
import os
import platform
import secrets
import shlex
import signal
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Callable


PROGRAM_VERSION = "0.19.0"

GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGUF_SCALAR_FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}

MAX_GGUF_METADATA_ITEMS = 1_000_000
MAX_GGUF_STRING_BYTES = 64 * 1024 * 1024
MAX_GGUF_TENSORS = 10_000_000
MAX_TENSOR_FEATURE_EXAMPLES = 12

KV_BYTES_PER_ELEMENT = {
    "f16": 2.0,
    "q8_0": 34 / 32,
    "q4_0": 18 / 32,
}

DEFAULT_CONTEXT_TARGETS = (8192, 32768, 65536, 131072, 262144)
DEFAULT_GROWING_TARGETS = (8192, 32768, 65536, 126000)
HERMES_DEPLOYMENT_CONTEXT = 131072
DEFAULT_DEPLOYMENT_HOST = "127.0.0.1"
LOOPBACK_DEPLOYMENT_HOSTS = {"127.0.0.1", "localhost", "::1"}
KV_ESTIMATE_MARGIN = 1.15

AUTOTUNE_PROFILES = {
    "quick": {
        "top_k": 1,
        "finalists": 1,
        "final_repetitions": 1,
        "context_policy": "representative",
    },
    "balanced": {
        "top_k": 2,
        "finalists": 2,
        "final_repetitions": 2,
        "context_policy": "all-safe",
    },
    "thorough": {
        "top_k": 3,
        "finalists": 3,
        "final_repetitions": 3,
        "context_policy": "all-safe",
    },
}

OPTIMIZATION_OBJECTIVES = {
    "hermes": {
        "description": (
            "wachsende lokale Chats mit regelmäßiger Kontextkomprimierung"
        ),
        "weights": {
            "prompt": 0.20,
            "generation": 0.35,
            "wall": 0.25,
            "stability": 0.10,
            "memory": 0.10,
        },
        "context_weighting": "equal",
        "workload_assumption": (
            "Kleine, mittlere und große Kontextphasen werden gleich gewichtet, "
            "weil der Chat nach einer Hermes-Komprimierung wieder klein beginnt."
        ),
        "not_measured": (
            "Dauer und Qualität der eigentlichen Hermes-Komprimierung werden "
            "nicht gemessen."
        ),
    },
    "balanced": {
        "description": "ausgewogener Kompromiss für lokale Chats und Agenten",
        "weights": {
            "prompt": 0.30,
            "generation": 0.30,
            "wall": 0.15,
            "stability": 0.15,
            "memory": 0.10,
        },
        "context_weighting": "equal",
    },
    "interactive": {
        "description": "schnelle Token-Ausgabe und kurze Antwortlatenz",
        "weights": {
            "prompt": 0.10,
            "generation": 0.50,
            "wall": 0.25,
            "stability": 0.10,
            "memory": 0.05,
        },
        "context_weighting": "equal",
    },
    "long-context": {
        "description": "Ende-zu-Ende-Latenz und Promptarbeit bei großen Kontexten",
        "weights": {
            "prompt": 0.25,
            "generation": 0.05,
            "wall": 0.50,
            "stability": 0.10,
            "memory": 0.10,
        },
        "context_weighting": "sqrt-context",
    },
    "throughput": {
        "description": "maximaler Prompt- und Generierungsdurchsatz",
        "weights": {
            "prompt": 0.45,
            "generation": 0.40,
            "wall": 0.05,
            "stability": 0.05,
            "memory": 0.05,
        },
        "context_weighting": "equal",
    },
}

FILLER_PARAGRAPHS = (
    (
        "Der lokale Inferenzdienst prüft Modellpfad, Kontextfenster, Batchgröße "
        "und verfügbaren Grafikspeicher vor jedem reproduzierbaren Testlauf."
    ),
    (
        "Ein Agent protokolliert Werkzeugaufrufe, Zwischenergebnisse, "
        "Fehlerzustände und Entscheidungen, damit ein späterer Lauf vollständig "
        "nachvollziehbar bleibt."
    ),
    (
        "Bei langen Dialogen werden unveränderte Präfixe wiederverwendet, während "
        "ausschließlich neu hinzugekommene Tokens durch das Modell ausgewertet "
        "werden."
    ),
    (
        "Die Messung trennt Promptverarbeitung, Textgenerierung, Cachetreffer, "
        "Gesamtlaufzeit und Speicherbelegung in eigenständige Kennzahlen."
    ),
    (
        "Sichere Standardwerte werden zuerst getestet; riskantere "
        "Konfigurationen folgen nur, wenn Modellstart und kurze Inferenz "
        "zuverlässig funktionieren."
    ),
    (
        "Für einen fairen Vergleich bleiben Modell, Gesprächsverlauf, Seed, "
        "Ausgabelänge und Hardwarezustand zwischen den Parameterläufen konstant."
    ),
    (
        "Der technische Assistent fasst Beobachtungen präzise zusammen und "
        "unterscheidet gemessene Tatsachen von Schätzungen oder Empfehlungen."
    ),
    (
        "Nach jedem Versuch beendet die Steuerung alle zugehörigen Prozesse und "
        "schreibt Rohdaten sowie normalisierte Messwerte in den Laufordner."
    ),
)

NVIDIA_QUERY = [
    "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,"
    "temperature.gpu,power.draw,power.limit,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def text_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(
    command: list[str],
    *,
    timeout: float = 30.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Führe einen reinen Diagnosebefehl aus und liefere strukturierte Daten."""
    started = dt.datetime.now(dt.timezone.utc)
    result: dict[str, Any] = {
        "command": command,
        "started_at": started.isoformat(),
        "timeout_seconds": timeout,
    }

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
        result.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": False,
            }
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "returncode": None,
                "stdout": text_output(exc.stdout),
                "stderr": text_output(exc.stderr),
                "timed_out": True,
            }
        )
    except OSError as exc:
        result.update(
            {
                "returncode": None,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
                "os_error": True,
            }
        )

    finished = dt.datetime.now(dt.timezone.utc)
    result["finished_at"] = finished.isoformat()
    result["duration_seconds"] = (finished - started).total_seconds()
    return result


def resolve_binary(llama_dir: Path, name: str, *, required: bool) -> Path | None:
    candidate = llama_dir / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate.resolve()
    if required:
        raise ValueError(f"Ausführbare Datei nicht gefunden: {candidate}")
    return None


def subprocess_environment(llama_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(llama_dir) + (f":{current}" if current else "")
    return env


def command_if_available(
    name: str,
    arguments: list[str],
    *,
    timeout: float,
) -> dict[str, Any] | None:
    executable = shutil.which(name)
    if executable is None:
        return None
    return run_command([executable, *arguments], timeout=timeout)


def read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Unerwartetes Dateiende in den GGUF-Metadaten")
    return data


def read_struct(handle: BinaryIO, byte_order: str, format_code: str) -> Any:
    size = struct.calcsize(format_code)
    return struct.unpack(byte_order + format_code, read_exact(handle, size))[0]


def read_gguf_string(handle: BinaryIO, byte_order: str) -> str:
    length = read_struct(handle, byte_order, "Q")
    if length > MAX_GGUF_STRING_BYTES:
        raise ValueError(f"Unplausibel lange GGUF-Zeichenkette: {length} Bytes")
    return read_exact(handle, length).decode("utf-8", errors="replace")


def skip_bytes(handle: BinaryIO, size: int, file_size: int) -> None:
    target = handle.tell() + size
    if target > file_size:
        raise ValueError("GGUF-Metadaten verweisen über das Dateiende hinaus")
    handle.seek(size, os.SEEK_CUR)


def skip_gguf_values(
    handle: BinaryIO,
    byte_order: str,
    value_type: int,
    count: int,
    file_size: int,
) -> None:
    if value_type in GGUF_SCALAR_FORMATS:
        item_size = struct.calcsize(GGUF_SCALAR_FORMATS[value_type])
        skip_bytes(handle, item_size * count, file_size)
        return

    if value_type == 8:
        for _ in range(count):
            length = read_struct(handle, byte_order, "Q")
            skip_bytes(handle, length, file_size)
        return

    if value_type == 9:
        for _ in range(count):
            nested_type = read_struct(handle, byte_order, "I")
            nested_count = read_struct(handle, byte_order, "Q")
            skip_gguf_values(
                handle,
                byte_order,
                nested_type,
                nested_count,
                file_size,
            )
        return

    raise ValueError(f"Unbekannter GGUF-Werttyp: {value_type}")


def read_gguf_metadata(model: Path) -> dict[str, Any]:
    """Lese GGUF-Kopf und skalare Metadaten ohne die Modell-Tensoren zu laden."""
    file_size = model.stat().st_size
    with model.open("rb") as handle:
        magic = read_exact(handle, 4)
        if magic == b"GGUF":
            byte_order = "<"
            endianness = "little"
        elif magic == b"FUGG":
            byte_order = ">"
            endianness = "big"
        else:
            raise ValueError("Dateikopf enthält keine gültige GGUF-Signatur")

        version = read_struct(handle, byte_order, "I")
        if version not in {1, 2, 3}:
            raise ValueError(f"Nicht unterstützte GGUF-Version: {version}")

        tensor_count = read_struct(handle, byte_order, "Q")
        metadata_count = read_struct(handle, byte_order, "Q")
        if tensor_count > MAX_GGUF_TENSORS:
            raise ValueError(f"Unplausibel viele GGUF-Tensoren: {tensor_count}")
        if metadata_count > MAX_GGUF_METADATA_ITEMS:
            raise ValueError(
                f"Unplausibel viele GGUF-Metadateneinträge: {metadata_count}"
            )

        metadata: dict[str, Any] = {}
        arrays: dict[str, dict[str, Any]] = {}
        for _ in range(metadata_count):
            key = read_gguf_string(handle, byte_order)
            value_type = read_struct(handle, byte_order, "I")

            if value_type == 8:
                metadata[key] = read_gguf_string(handle, byte_order)
            elif value_type == 9:
                element_type = read_struct(handle, byte_order, "I")
                element_count = read_struct(handle, byte_order, "Q")
                arrays[key] = {
                    "element_type": GGUF_VALUE_TYPES.get(
                        element_type, f"unknown-{element_type}"
                    ),
                    "length": element_count,
                }
                skip_gguf_values(
                    handle,
                    byte_order,
                    element_type,
                    element_count,
                    file_size,
                )
            elif value_type in GGUF_SCALAR_FORMATS:
                metadata[key] = read_struct(
                    handle, byte_order, GGUF_SCALAR_FORMATS[value_type]
                )
            else:
                raise ValueError(
                    f"Unbekannter GGUF-Werttyp {value_type} für Schlüssel {key!r}"
                )

        feature_patterns = {
            "mtp": ("nextn", ".mtp", "draft"),
            "vision": ("vision", "mmproj", "mm."),
        }
        tensor_features = {
            feature: {"detected": False, "count": 0, "examples": []}
            for feature in feature_patterns
        }
        for _ in range(tensor_count):
            tensor_name = read_gguf_string(handle, byte_order)
            dimension_count = read_struct(handle, byte_order, "I")
            if dimension_count > 16:
                raise ValueError(
                    f"Unplausibel viele Dimensionen für Tensor {tensor_name!r}: "
                    f"{dimension_count}"
                )
            skip_bytes(handle, dimension_count * 8, file_size)
            read_struct(handle, byte_order, "I")  # GGML-Typ
            read_struct(handle, byte_order, "Q")  # Offset im Tensor-Datenblock

            lowered_name = tensor_name.lower()
            for feature, patterns in feature_patterns.items():
                if not any(pattern in lowered_name for pattern in patterns):
                    continue
                details = tensor_features[feature]
                details["detected"] = True
                details["count"] += 1
                if len(details["examples"]) < MAX_TENSOR_FEATURE_EXAMPLES:
                    details["examples"].append(tensor_name)

        return {
            "version": version,
            "endianness": endianness,
            "tensor_count": tensor_count,
            "metadata_count": metadata_count,
            "metadata": metadata,
            "arrays": arrays,
            "tensor_features": tensor_features,
        }


def summarize_gguf(gguf: dict[str, Any]) -> dict[str, Any]:
    metadata = gguf["metadata"]
    architecture = metadata.get("general.architecture")
    prefix = f"{architecture}." if architecture else ""

    def architecture_value(name: str) -> Any:
        return metadata.get(prefix + name) if prefix else None

    head_count = architecture_value("attention.head_count")
    head_count_kv = architecture_value("attention.head_count_kv")
    gqa_ratio = None
    if (
        isinstance(head_count, int)
        and isinstance(head_count_kv, int)
        and head_count_kv > 0
    ):
        gqa_ratio = head_count / head_count_kv

    all_keys = [*metadata, *gguf["arrays"]]
    feature_terms = ("mtp", "draft", "vision", "expert", "moe")
    feature_keys = sorted(
        key for key in all_keys if any(term in key.lower() for term in feature_terms)
    )

    tensor_features = gguf.get("tensor_features", {})
    mtp_tensors = tensor_features.get("mtp", {})
    vision_tensors = tensor_features.get("vision", {})

    return {
        "name": metadata.get("general.name"),
        "architecture": architecture,
        "context_length": architecture_value("context_length"),
        "block_count": architecture_value("block_count"),
        "embedding_length": architecture_value("embedding_length"),
        "attention_head_count": head_count,
        "attention_head_count_kv": head_count_kv,
        "attention_key_length": architecture_value("attention.key_length"),
        "attention_value_length": architecture_value("attention.value_length"),
        "gqa_ratio": gqa_ratio,
        "rope_frequency_base": architecture_value("rope.freq_base"),
        "quantization_version": metadata.get("general.quantization_version"),
        "file_type": metadata.get("general.file_type"),
        "feature_metadata_keys": feature_keys,
        "mtp_detected": bool(mtp_tensors.get("detected")),
        "mtp_tensor_count": mtp_tensors.get("count", 0),
        "mtp_tensor_examples": mtp_tensors.get("examples", []),
        "vision_detected": bool(vision_tensors.get("detected")),
        "vision_tensor_count": vision_tensors.get("count", 0),
    }


def collect_hardware(timeout: float) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "lscpu": command_if_available("lscpu", ["--json"], timeout=timeout),
        "memory": command_if_available("free", ["--bytes"], timeout=timeout),
        "nvidia_smi": command_if_available(
            "nvidia-smi", NVIDIA_QUERY, timeout=timeout
        ),
    }


def collect_model(model: Path) -> dict[str, Any]:
    stat = model.stat()
    information: dict[str, Any] = {
        "path": str(model),
        "resolved_path": str(model.resolve()),
        "is_symlink": model.is_symlink(),
        "filename": model.name,
        "size_bytes": stat.st_size,
        "size_gib": round(stat.st_size / (1024**3), 3),
        "modified_at": dt.datetime.fromtimestamp(
            stat.st_mtime, tz=dt.timezone.utc
        ).isoformat(),
        "suffix": model.suffix.lower(),
    }
    try:
        gguf = read_gguf_metadata(model)
        information["gguf"] = gguf
        information["summary"] = summarize_gguf(gguf)
    except (OSError, ValueError) as exc:
        information["gguf"] = {"error": str(exc)}
        information["summary"] = None
    return information


def help_text(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    return result.get("stdout", "") + "\n" + result.get("stderr", "")


def detect_capabilities(llama: dict[str, Any]) -> dict[str, Any]:
    server_help = help_text(llama.get("server_help"))
    bench_help = help_text(llama.get("bench_help"))

    server_options = (
        "--threads",
        "--threads-batch",
        "--ctx-size",
        "--batch-size",
        "--ubatch-size",
        "--flash-attn",
        "--cache-type-k",
        "--cache-type-v",
        "--load-mode",
        "--gpu-layers",
        "--parallel",
        "--kv-unified",
        "--context-shift",
        "--cache-prompt",
        "--host",
        "--port",
        "--alias",
        "--api-key",
        "--metrics",
        "--no-context-shift",
        "--log-colors",
        "--spec-type",
        "--spec-draft-n-max",
        "--spec-ngram-mod-n-min",
        "--spec-ngram-mod-n-max",
        "--spec-ngram-mod-n-match",
    )
    bench_options = (
        "--n-prompt",
        "--n-gen",
        "--batch-size",
        "--ubatch-size",
        "--cache-type-k",
        "--cache-type-v",
        "--threads",
        "--n-gpu-layers",
        "--flash-attn",
        "--load-mode",
        "--output",
        "--repetitions",
        "--progress",
    )
    cache_types = [
        cache_type
        for cache_type in KV_BYTES_PER_ELEMENT
        if cache_type in server_help
    ]
    speculation_types = [
        spec_type
        for spec_type in ("ngram-mod", "draft-mtp")
        if spec_type in server_help
    ]

    return {
        "server_options": {
            option: option in server_help for option in server_options
        },
        "bench_options": {
            option: option in bench_help for option in bench_options
        },
        "cache_types": cache_types,
        "speculation_types": speculation_types,
    }


def parse_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_nvidia_gpus(hardware: dict[str, Any]) -> list[dict[str, Any]]:
    result = hardware.get("nvidia_smi")
    if not result or result.get("returncode") != 0:
        return []

    fields = (
        "index",
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
        "memory_free_mib",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "utilization_percent",
    )
    numeric_fields = set(fields[4:])
    gpus = []
    for row in csv.reader(io.StringIO(result.get("stdout", ""))):
        if len(row) != len(fields):
            continue
        gpu = dict(zip(fields, (value.strip() for value in row), strict=True))
        gpu["index"] = int(gpu["index"])
        for field in numeric_fields:
            gpu[field] = parse_float(gpu[field])
        gpus.append(gpu)
    return gpus


def estimate_kv_cache_bytes(
    model_summary: dict[str, Any],
    context_length: int,
    cache_type: str,
    *,
    parallel: int = 1,
) -> int | None:
    bytes_per_element = KV_BYTES_PER_ELEMENT.get(cache_type)
    block_count = model_summary.get("block_count")
    key_length = model_summary.get("attention_key_length")
    value_length = model_summary.get("attention_value_length")
    if not all(
        isinstance(value, int) and value > 0
        for value in (block_count, key_length, value_length)
    ):
        return None
    return round(
        context_length
        * block_count
        * (key_length + value_length)
        * bytes_per_element
        * parallel
    )


def gib(value: int | float) -> float:
    return round(value / (1024**3), 3)


def build_tuning_plan(
    hardware: dict[str, Any],
    model: dict[str, Any],
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    summary = model.get("summary") or {}
    native_context = summary.get("context_length")
    context_targets = [
        target
        for target in DEFAULT_CONTEXT_TARGETS
        if not isinstance(native_context, int) or target <= native_context
    ]
    if isinstance(native_context, int) and native_context not in context_targets:
        context_targets.append(native_context)
        context_targets.sort()

    gpus = parse_nvidia_gpus(hardware)
    primary_gpu = gpus[0] if gpus else None
    available_for_kv = None
    full_offload_estimated = None
    memory_budget: dict[str, Any] = {
        "method": "GGUF-Dateigröße plus Sicherheitsreserven",
        "primary_gpu": primary_gpu,
        "estimated_weight_gib": gib(model["size_bytes"] * 1.03),
        "reserve_gib": None,
        "available_for_kv_gib": None,
        "full_offload_estimated": None,
    }
    if primary_gpu and primary_gpu["memory_free_mib"] is not None:
        free_bytes = primary_gpu["memory_free_mib"] * 1024**2
        total_bytes = primary_gpu["memory_total_mib"] * 1024**2
        reserve_bytes = max(1.5 * 1024**3, total_bytes * 0.06)
        weight_bytes = model["size_bytes"] * 1.03
        available_for_kv = max(0, free_bytes - weight_bytes - reserve_bytes)
        full_offload_estimated = weight_bytes + reserve_bytes <= free_bytes
        memory_budget["reserve_gib"] = gib(reserve_bytes)
        memory_budget["available_for_kv_gib"] = gib(available_for_kv)
        memory_budget["full_offload_estimated"] = full_offload_estimated

    cache_types = [
        cache_type
        for cache_type in ("f16", "q8_0", "q4_0")
        if cache_type in capabilities["cache_types"]
    ]
    context_profiles = []
    for context_length in context_targets:
        estimates = {}
        recommended_cache = None
        for cache_type in cache_types:
            raw_bytes = estimate_kv_cache_bytes(
                summary, context_length, cache_type
            )
            guarded_bytes = (
                round(raw_bytes * KV_ESTIMATE_MARGIN)
                if raw_bytes is not None
                else None
            )
            fits = (
                guarded_bytes <= available_for_kv
                if guarded_bytes is not None and available_for_kv is not None
                else None
            )
            estimates[cache_type] = {
                "estimated_gib": gib(raw_bytes) if raw_bytes is not None else None,
                "with_margin_gib": (
                    gib(guarded_bytes) if guarded_bytes is not None else None
                ),
                "fits_estimate": fits,
            }
            if recommended_cache is None and fits is True:
                recommended_cache = cache_type

        context_profiles.append(
            {
                "context_length": context_length,
                "cache_estimates": estimates,
                "recommended_cache_type": recommended_cache,
                "default_run": recommended_cache is not None,
            }
        )

    batch_pairs = [
        {"batch_size": 512, "ubatch_size": 128},
        {"batch_size": 1024, "ubatch_size": 256},
        {"batch_size": 2048, "ubatch_size": 256},
        {"batch_size": 2048, "ubatch_size": 512},
        {"batch_size": 4096, "ubatch_size": 512},
    ]
    flash_candidates = (
        ["on", "off"]
        if capabilities["server_options"].get("--flash-attn")
        else ["auto"]
    )
    speculation_candidates: list[dict[str, Any]] = [
        {"spec_type": "none", "runtime_probe_required": False}
    ]
    supported_speculation = capabilities["speculation_types"]
    if "ngram-mod" in supported_speculation:
        speculation_candidates.append(
            {"spec_type": "ngram-mod", "runtime_probe_required": False}
        )
    if "draft-mtp" in supported_speculation:
        speculation_candidates.extend(
            {
                "spec_type": "draft-mtp",
                "spec_draft_n_max": draft_tokens,
                "runtime_probe_required": True,
            }
            for draft_tokens in (1, 2, 3)
        )
    if all(
        spec_type in supported_speculation
        for spec_type in ("ngram-mod", "draft-mtp")
    ):
        speculation_candidates.append(
            {
                "spec_type": "ngram-mod,draft-mtp",
                "spec_draft_n_max": 2,
                "runtime_probe_required": True,
            }
        )

    safe_context_profiles = [
        {
            "ctx_size": profile["context_length"],
            "cache_type_k": profile["recommended_cache_type"],
            "cache_type_v": profile["recommended_cache_type"],
        }
        for profile in context_profiles
        if profile["default_run"]
    ]
    excluded = []
    if len(gpus) <= 1:
        excluded.extend(["split_mode", "tensor_split", "main_gpu"])

    return {
        "profile": "single-user-growing-chat",
        "native_context_limit": native_context,
        "memory_budget": memory_budget,
        "kv_estimate_margin": KV_ESTIMATE_MARGIN,
        "context_profiles": context_profiles,
        "fixed_parameters": {
            "gpu_layers": "all" if full_offload_estimated else "auto",
            "load_mode": "auto",
            "parallel": 1,
            "kv_unified": True,
            "context_shift": False,
            "cache_prompt": True,
        },
        "excluded_parameters": excluded,
        "stages": [
            {
                "name": "smoke",
                "runner": "llama-bench",
                "purpose": "Modell laden und kurze Inferenz verifizieren",
                "experiments": 1,
            },
            {
                "name": "batch-screening",
                "runner": "llama-bench",
                "purpose": "Batch, UBatch und Flash Attention vergleichen",
                "batch_ubatch_pairs": batch_pairs,
                "flash_attention": flash_candidates,
                "experiments": len(batch_pairs) * len(flash_candidates),
            },
            {
                "name": "growing-context",
                "runner": "llama-server",
                "purpose": "Wachsenden Chat mit sicher geschätzten KV-Caches messen",
                "profiles": safe_context_profiles,
                "experiments": len(safe_context_profiles),
            },
            {
                "name": "speculation",
                "runner": "llama-server",
                "purpose": "N-Gram und MTP erst nach stabilem Baseline-Lauf prüfen",
                "candidates": speculation_candidates,
                "experiments": len(speculation_candidates),
            },
        ],
        "warnings": [
            "Die VRAM-Schätzung ersetzt keine Messung nach dem Modellstart.",
            "GGUF-Dateigröße und tatsächlich belegter VRAM können abweichen.",
            "MTP wird durch einen kontrollierten Laufzeit-Probeversuch erkannt.",
            "Nicht passende Kontextprofile werden standardmäßig nicht gestartet.",
        ],
    }


def autotune_thread_candidates(hardware: dict[str, Any], profile: str) -> list[int]:
    logical = hardware.get("logical_cpu_count")
    if not isinstance(logical, int) or logical <= 0:
        logical = 8
    if profile == "quick":
        values = [min(8, logical)]
    elif profile == "balanced":
        values = [min(4, logical), min(8, logical), min(16, logical)]
    else:
        values = [1, min(4, logical), min(8, logical), min(16, logical)]
        values.extend([max(1, logical // 2), logical])
    return sorted(set(values))


def select_context_variants(
    variants: list[dict[str, Any]], policy: str
) -> list[dict[str, Any]]:
    if policy == "all-safe" or len(variants) <= 3:
        return variants
    context_sizes = sorted({item["context_size"] for item in variants})
    selected_sizes = {
        context_sizes[0],
        context_sizes[len(context_sizes) // 2],
        context_sizes[-1],
    }
    return [
        item for item in variants if item["context_size"] in selected_sizes
    ]


def build_speculation_variants(
    capabilities: dict[str, Any],
    model_summary: dict[str, Any],
    profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    supported = capabilities.get("speculation_types", [])
    mtp_detected = bool(model_summary.get("mtp_detected"))
    variants: list[dict[str, Any]] = [
        {"id": "spec-none", "spec_type": "none"}
    ]
    exclusions: list[dict[str, str]] = []

    if "ngram-mod" in supported:
        if profile == "thorough":
            ngram_settings = (
                (16, 32, 12),
                (32, 48, 16),
                (48, 64, 24),
            )
        else:
            ngram_settings = ((48, 64, 24),)
        for minimum, maximum, match in ngram_settings:
            variants.append(
                {
                    "id": f"spec-ngram-{minimum}-{maximum}-{match}",
                    "spec_type": "ngram-mod",
                    "ngram_n_min": minimum,
                    "ngram_n_max": maximum,
                    "ngram_n_match": match,
                }
            )
    else:
        exclusions.append(
            {
                "parameter": "spec_type=ngram-mod",
                "reason": "vom erkannten llama-server nicht angeboten",
            }
        )

    if "draft-mtp" in supported and mtp_detected:
        for draft_tokens in (1, 2, 3):
            variants.append(
                {
                    "id": f"spec-mtp-{draft_tokens}",
                    "spec_type": "draft-mtp",
                    "draft_n_max": draft_tokens,
                }
            )
    elif "draft-mtp" not in supported:
        exclusions.append(
            {
                "parameter": "spec_type=draft-mtp",
                "reason": "vom erkannten llama-server nicht angeboten",
            }
        )
    else:
        exclusions.append(
            {
                "parameter": "spec_type=draft-mtp",
                "reason": "keine MTP-/nextn-Tensoren im Modell erkannt",
            }
        )

    if "ngram-mod" in supported and "draft-mtp" in supported and mtp_detected:
        variants.append(
            {
                "id": "spec-ngram-mtp-2",
                "spec_type": "ngram-mod,draft-mtp",
                "ngram_n_min": 48,
                "ngram_n_max": 64,
                "ngram_n_match": 24,
                "draft_n_max": 2,
            }
        )
    return variants, exclusions


def speculation_variant_label(speculation: dict[str, Any]) -> str:
    """Return a report label that distinguishes otherwise identical types."""
    variant_id = speculation.get("id")
    if variant_id:
        return str(variant_id)
    spec_type = speculation.get("spec_type", "server-default")
    draft_n_max = speculation.get("draft_n_max")
    if draft_n_max is not None:
        return f"{spec_type} (draft_n_max={draft_n_max})"
    return str(spec_type)


def build_autotune_experiment_plan(
    hardware: dict[str, Any],
    model: dict[str, Any],
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    *,
    profile: str,
    optimization_objective: str = "balanced",
) -> dict[str, Any]:
    if profile not in AUTOTUNE_PROFILES:
        raise ValueError(f"Unbekanntes Autotune-Profil: {profile}")
    if optimization_objective not in OPTIMIZATION_OBJECTIVES:
        raise ValueError(
            f"Unbekanntes Optimierungsziel: {optimization_objective}"
        )
    policy = AUTOTUNE_PROFILES[profile]
    objective_definition = OPTIMIZATION_OBJECTIVES[optimization_objective]
    summary = model.get("summary") or {}
    batch_stage = next(
        stage
        for stage in tuning_plan["stages"]
        if stage["name"] == "batch-screening"
    )

    batch_candidates = []
    candidate_number = 0
    for pair in batch_stage["batch_ubatch_pairs"]:
        if pair["ubatch_size"] > pair["batch_size"]:
            continue
        for flash_attention in batch_stage["flash_attention"]:
            candidate_number += 1
            batch_candidates.append(
                {
                    "id": f"batch-{candidate_number:02d}",
                    "batch_size": pair["batch_size"],
                    "ubatch_size": pair["ubatch_size"],
                    "flash_attention": flash_attention,
                }
            )

    context_variants = []
    context_exclusions = []
    for context_profile in tuning_plan["context_profiles"]:
        context_size = context_profile["context_length"]
        reserve = max(4096, round(context_size * 0.02))
        prompt_target = context_size - reserve
        for cache_type, estimate in context_profile["cache_estimates"].items():
            fits = estimate.get("fits_estimate")
            if fits is False:
                context_exclusions.append(
                    {
                        "parameter": (
                            f"context={context_size},cache={cache_type}"
                        ),
                        "reason": "KV-Schätzung überschreitet das VRAM-Budget",
                    }
                )
                continue
            context_variants.append(
                {
                    "id": f"ctx-{context_size}-{cache_type}",
                    "context_size": context_size,
                    "prompt_target": prompt_target,
                    "cache_type_k": cache_type,
                    "cache_type_v": cache_type,
                    "estimated_kv_gib": estimate.get("with_margin_gib"),
                    "memory_fit": (
                        "estimated-fit" if fits is True else "unknown-probe"
                    ),
                }
            )
    context_variants = select_context_variants(
        context_variants,
        policy["context_policy"],
    )

    speculation_variants, speculation_exclusions = build_speculation_variants(
        capabilities,
        summary,
        profile,
    )
    thread_candidates = [
        {"id": f"threads-{threads}", "threads": threads}
        for threads in autotune_thread_candidates(hardware, profile)
    ]
    safe_context_count = len(
        {item["context_size"] for item in context_variants}
    )
    top_k = policy["top_k"]
    estimated_runs = {
        "smoke": 1,
        "batch_screening": len(batch_candidates),
        "thread_screening": len(thread_candidates) * top_k,
        "context_cache_screening": len(context_variants) * top_k,
        "speculation_screening": len(speculation_variants) * top_k,
        "final_validation": (
            (policy["finalists"] * 2)
            * policy["final_repetitions"]
            * max(1, safe_context_count)
        ),
    }
    estimated_runs["upper_bound_total"] = sum(estimated_runs.values())

    stages = [
        {
            "id": "smoke",
            "strategy": "single-safety-gate",
            "runner": "llama-bench",
            "candidates": [{"id": "smoke-default"}],
            "advance": "nur bei erfolgreichem Modellstart und gültiger Ausgabe",
        },
        {
            "id": "batch-screening",
            "strategy": "exhaustive-within-bounds",
            "runner": "llama-bench",
            "candidates": batch_candidates,
            "advance": f"beste {top_k} nach balanciertem Durchsatzscore",
        },
        {
            "id": "thread-screening",
            "strategy": "exhaustive-on-survivors",
            "runner": "llama-bench",
            "depends_on": "batch-screening",
            "candidates": thread_candidates,
            "advance": "beste Threadzahl je übernommener Konfiguration",
        },
        {
            "id": "context-cache-screening",
            "strategy": "cross-survivors-with-safe-contexts",
            "runner": "llama-server",
            "depends_on": "thread-screening",
            "candidates": context_variants,
            "advance": f"beste {top_k} nach Chat-Score über Kontextstufen",
        },
        {
            "id": "speculation-screening",
            "strategy": "exhaustive-on-survivors",
            "runner": "llama-server",
            "depends_on": "context-cache-screening",
            "candidates": speculation_variants,
            "advance": f"beste {policy['finalists']} Gesamtkandidaten",
        },
        {
            "id": "final-validation",
            "strategy": "repeated-growing-chat",
            "runner": "llama-server",
            "depends_on": "speculation-screening",
            "candidate_source": (
                f"beste {policy['finalists']} plus passende none-Kontrollen"
            ),
            "repetitions": policy["final_repetitions"],
            "contexts": sorted(
                {item["context_size"] for item in context_variants}
            ),
            "advance": "klare Empfehlung mit Konfidenz und Alternativen",
        },
    ]

    exclusions = [
        *context_exclusions,
        *speculation_exclusions,
        *(
            {
                "parameter": parameter,
                "reason": "auf der erkannten Einzel-GPU-Hardware nicht sinnvoll",
            }
            for parameter in tuning_plan.get("excluded_parameters", [])
        ),
        {
            "parameter": "vollständiges kartesisches Produkt",
            "reason": (
                "adaptive Stufen übernehmen nur erfolgreiche Top-Kandidaten; "
                "dadurch bleiben Laufzeit und Energiebedarf begrenzt"
            ),
        },
    ]
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "profile": profile,
        "goal": "single-user-growing-chat",
        "optimization_objective": optimization_objective,
        "objective_definition": objective_definition,
        "adaptive": True,
        "policy": policy,
        "stages": stages,
        "estimated_runs": estimated_runs,
        "exclusions": exclusions,
        "hard_success_gates": [
            "Prozess startet innerhalb des Timeouts",
            "HTTP- oder Benchmark-Exitstatus ist erfolgreich",
            "Antwort enthält finalen Inhalt",
            "Finish-Reason ist stop und Kontext wurde nicht abgeschnitten",
            "Prozess wird vollständig beendet",
        ],
        "ranking": {
            "method": "normalisierter gewichteter Score plus harte Gates",
            "weights": {
                "prompt_throughput": objective_definition["weights"]["prompt"],
                "generation_throughput": objective_definition["weights"]["generation"],
                "wall_clock_latency": objective_definition["weights"]["wall"],
                "stability": objective_definition["weights"]["stability"],
                "memory_headroom": objective_definition["weights"]["memory"],
            },
            "aggregation": (
                f"geometrisches Mittel über Kontextstufen mit "
                f"{objective_definition['context_weighting']}-Gewichtung; "
                "schlechteste Stufe wird zusätzlich als Risikowert ausgewiesen"
            ),
        },
        "local_ai_role": {
            "enabled_later_by_option": True,
            "purpose": (
                "gemessene Ergebnisse erklären und Empfehlung formulieren"
            ),
            "restriction": (
                "KI darf Messwerte, harte Gates und deterministisches Ranking "
                "nicht verändern"
            ),
        },
    }


def render_autotune_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Llama Autotune – Versuchsplan",
        "",
        f"Profil: `{plan['profile']}`  ",
        f"Ziel: `{plan['goal']}`  ",
        f"Optimierungsziel: `{plan['optimization_objective']}` – "
        f"{plan['objective_definition']['description']}  ",
        "Suchverfahren: adaptiv",
        "",
        "## Stufen",
        "",
        "| Stufe | Strategie | Runner | Kandidaten/Templates | Weitergabe |",
        "|---|---|---|---:|---|",
    ]
    if plan["objective_definition"].get("workload_assumption"):
        lines.insert(
            6,
            "Annahme: "
            + plan["objective_definition"]["workload_assumption"]
            + "  ",
        )
    if plan["objective_definition"].get("not_measured"):
        lines.insert(
            7,
            "Nicht gemessen: "
            + plan["objective_definition"]["not_measured"]
            + "  ",
        )
    for stage in plan["stages"]:
        candidates = stage.get("candidates")
        candidate_count = len(candidates) if isinstance(candidates, list) else "dynamisch"
        lines.append(
            f"| `{stage['id']}` | {stage['strategy']} | `{stage['runner']}` | "
            f"{candidate_count} | {stage['advance']} |"
        )

    lines.extend(
        [
            "",
            "## Geschätzte maximale Laufzahl",
            "",
            "| Teil | Läufe |",
            "|---|---:|",
        ]
    )
    for key, value in plan["estimated_runs"].items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Ausschlüsse", ""])
    for exclusion in plan["exclusions"]:
        lines.append(
            f"- `{exclusion['parameter']}`: {exclusion['reason']}"
        )

    lines.extend(
        [
            "",
            "## Empfehlungslogik",
            "",
            plan["ranking"]["method"] + ".",
            "",
        ]
    )
    for metric, weight in plan["ranking"]["weights"].items():
        lines.append(f"- `{metric}`: {weight:.0%}")
    lines.extend(
        [
            "",
            "Die optionale lokale KI erklärt später ausschließlich die bereits "
            "berechneten Ergebnisse. Sie darf weder Messwerte noch harte "
            "Erfolgskriterien oder das deterministische Ranking verändern.",
            "",
        ]
    )
    return "\n".join(lines)


def gpu_snapshot(timeout: float) -> dict[str, Any] | None:
    return command_if_available("nvidia-smi", NVIDIA_QUERY, timeout=timeout)


def find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def redact_secret(command: list[str], secret: str) -> list[str]:
    return ["<redacted>" if argument == secret else argument for argument in command]


def build_server_smoke_command(
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    *,
    port: int,
    alias: str,
    api_key: str,
    context_size: int = 8192,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    cache_type: str = "f16",
    flash_attention: str = "on",
    threads: int | None = None,
    speculation: dict[str, Any] | None = None,
) -> list[str]:
    options = capabilities["server_options"]
    required = (
        "--host",
        "--port",
        "--alias",
        "--api-key",
        "--ctx-size",
        "--batch-size",
        "--ubatch-size",
        "--cache-type-k",
        "--cache-type-v",
        "--gpu-layers",
        "--parallel",
        "--threads",
    )
    missing = [option for option in required if not options.get(option)]
    if missing:
        raise ValueError(
            "llama-server unterstützt erforderliche Optionen nicht: "
            + ", ".join(missing)
        )
    if cache_type not in capabilities.get("cache_types", []):
        raise ValueError(
            "llama-server meldet keine Unterstützung für "
            f"{cache_type}-KV-Cache"
        )

    thread_count = threads or min(8, os.cpu_count() or 8)
    gpu_layers = (
        "all"
        if tuning_plan["fixed_parameters"]["gpu_layers"] == "all"
        else "auto"
    )
    command = [
        str(server),
        "--model",
        str(model),
        "--alias",
        alias,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--ctx-size",
        str(context_size),
        "--gpu-layers",
        gpu_layers,
        "--cache-type-k",
        cache_type,
        "--cache-type-v",
        cache_type,
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(ubatch_size),
        "--threads",
        str(thread_count),
        "--parallel",
        "1",
    ]
    if options.get("--flash-attn") and flash_attention != "auto":
        command.extend(["--flash-attn", flash_attention])
    if options.get("--threads-batch"):
        command.extend(["--threads-batch", str(thread_count)])
    if options.get("--kv-unified"):
        command.append("--kv-unified")
    if options.get("--no-context-shift"):
        command.append("--no-context-shift")
    if options.get("--cache-prompt"):
        command.append("--cache-prompt")
    if options.get("--load-mode"):
        command.extend(["--load-mode", "auto"])
    if options.get("--metrics"):
        command.append("--metrics")
    if speculation is not None:
        spec_type = speculation.get("spec_type", "none")
        if not options.get("--spec-type"):
            raise ValueError(
                "llama-server unterstützt die erforderliche Option "
                "--spec-type nicht"
            )
        requested_types = {
            item.strip() for item in str(spec_type).split(",") if item.strip()
        }
        supported_types = set(capabilities.get("speculation_types", []))
        unsupported = requested_types - supported_types - {"none"}
        if unsupported:
            raise ValueError(
                "llama-server meldet keine Unterstützung für Spekulation: "
                + ", ".join(sorted(unsupported))
            )
        command.extend(["--spec-type", str(spec_type)])
        speculation_options = (
            ("draft_n_max", "--spec-draft-n-max"),
            ("ngram_n_min", "--spec-ngram-mod-n-min"),
            ("ngram_n_max", "--spec-ngram-mod-n-max"),
            ("ngram_n_match", "--spec-ngram-mod-n-match"),
        )
        for key, option in speculation_options:
            if key not in speculation:
                continue
            if not options.get(option):
                raise ValueError(
                    f"llama-server unterstützt die erforderliche Option {option} nicht"
                )
            command.extend([option, str(speculation[key])])
    if options.get("--log-colors"):
        command.extend(["--log-colors", "off"])
    return command


def http_json_request(
    url: str,
    api_key: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    started = time.monotonic()
    status = None
    raw_body = ""
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw_body = exc.read().decode("utf-8", errors="replace")
        error = str(exc)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = str(exc)

    parsed = None
    parse_error = None
    if raw_body:
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    return {
        "status_code": status,
        "duration_seconds": round(time.monotonic() - started, 6),
        "json": parsed,
        "raw_body": raw_body if parsed is None else None,
        "error": error,
        "parse_error": parse_error,
    }


def http_text_request(
    url: str,
    api_key: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    started = time.monotonic()
    status = None
    body = ""
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
        error = str(exc)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = str(exc)
    return {
        "status_code": status,
        "duration_seconds": round(time.monotonic() - started, 6),
        "body": body,
        "error": error,
    }


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name_with_labels, raw_value = parts
        name = name_with_labels.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if "{" not in name_with_labels:
            metrics[name] = value
    return metrics


def fetch_server_metrics(
    base_url: str,
    api_key: str,
    *,
    timeout: float,
) -> dict[str, Any] | None:
    response = http_text_request(
        f"{base_url}/metrics",
        api_key,
        timeout=timeout,
    )
    if response["status_code"] != 200:
        return None
    return {
        "captured_at": utc_now(),
        "values": parse_prometheus_metrics(response["body"]),
    }


def metric_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    name: str,
) -> float | None:
    if not before or not after:
        return None
    before_value = before.get("values", {}).get(name)
    after_value = after.get("values", {}).get(name)
    if not isinstance(before_value, (int, float)) or not isinstance(
        after_value, (int, float)
    ):
        return None
    return max(0.0, after_value - before_value)


def wait_for_server(
    process: subprocess.Popen[Any],
    base_url: str,
    api_key: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + timeout
    last_probe: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            return {
                "ready": False,
                "reason": "server-exited",
                "returncode": returncode,
                "duration_seconds": round(time.monotonic() - started, 6),
                "last_probe": last_probe,
            }
        last_probe = http_json_request(
            f"{base_url}/health",
            api_key,
            timeout=min(2.0, max(0.1, deadline - time.monotonic())),
        )
        if last_probe["status_code"] == 200:
            return {
                "ready": True,
                "reason": "health-ok",
                "returncode": None,
                "duration_seconds": round(time.monotonic() - started, 6),
                "last_probe": last_probe,
            }
        time.sleep(0.25)
    return {
        "ready": False,
        "reason": "startup-timeout",
        "returncode": process.poll(),
        "duration_seconds": round(time.monotonic() - started, 6),
        "last_probe": last_probe,
    }


def stop_process_group(
    process: subprocess.Popen[Any], *, timeout: float
) -> dict[str, Any]:
    if process.poll() is not None:
        return {
            "method": "already-exited",
            "returncode": process.returncode,
        }

    method = "sigterm"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        method = "terminate"
        process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        method += "-then-sigkill"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait(timeout=max(1.0, timeout))
    return {"method": method, "returncode": process.returncode}


def summarize_chat_response(response: dict[str, Any]) -> dict[str, Any]:
    payload = response.get("json")
    choice: dict[str, Any] = {}
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        message = {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if not isinstance(content, str):
        content = ""
    if not isinstance(reasoning, str):
        reasoning = ""
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    prompt_details = usage.get("prompt_tokens_details", {})
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    timings = payload.get("timings", {}) if isinstance(payload, dict) else {}
    if not isinstance(timings, dict):
        timings = {}
    return {
        "status_code": response.get("status_code"),
        "wall_seconds": response.get("duration_seconds"),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "content_characters": len(content),
        "reasoning_content": reasoning,
        "reasoning_characters": len(reasoning),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": prompt_details.get("cached_tokens", timings.get("cache_n")),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "predicted_tokens_per_second": timings.get("predicted_per_second"),
        "error": response.get("error"),
        "parse_error": response.get("parse_error"),
    }


def chat_api_error_message(response: dict[str, Any]) -> str | None:
    """Extract a useful API error message from a recorded HTTP response."""
    payload = response.get("json")
    if not isinstance(payload, dict):
        return response.get("error")
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return response.get("error")


def classify_chat_stage(
    response: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    """Classify chat output without hiding model/runtime incompatibilities."""
    status_code = summary.get("status_code")
    finish_reason = summary.get("finish_reason")
    content = summary.get("content")
    reasoning = summary.get("reasoning_content")
    content_ok = isinstance(content, str) and bool(content.strip())
    reasoning_ok = isinstance(reasoning, str) and bool(reasoning.strip())

    if status_code != 200:
        message = (chat_api_error_message(response) or "").lower()
        if "peg-gemma4" in message:
            return "peg-format-error"
        return "request-failed"
    if content_ok and finish_reason in {"stop", "length"}:
        return "ok"
    if finish_reason == "length" and reasoning_ok:
        return "reasoning-truncated"
    if finish_reason == "length":
        return "output-truncated"
    return "invalid-response"


def adaptive_token_budgets(
    initial: int,
    maximum: int = 2048,
) -> list[int]:
    """Return deterministic doubling steps for incomplete chat responses."""
    if initial <= 0 or maximum <= 0:
        raise ValueError("Tokenbudgets müssen größer als 0 sein")
    maximum = max(initial, maximum)
    budgets = [initial]
    while budgets[-1] < maximum:
        budgets.append(min(maximum, budgets[-1] * 2))
    return budgets


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def shell_command(command: list[Any]) -> str:
    """Render an argv list as a copy-paste-safe POSIX shell command."""
    return shlex.join(str(item) for item in command)


def run_server_smoke(
    llama_dir: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    run_dir: Path,
    *,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    port = find_free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    alias = "autotune-model"
    api_key = secrets.token_urlsafe(32)
    command = build_server_smoke_command(
        server,
        model,
        capabilities,
        tuning_plan,
        port=port,
        alias=alias,
        api_key=api_key,
    )
    redacted_command = redact_secret(command, api_key)
    log_path = run_dir / "server.log"
    payload: dict[str, Any] = {
        "model": alias,
        "messages": [
            {
                "role": "system",
                "content": "Du bist ein präziser technischer Assistent.",
            },
            {
                "role": "user",
                "content": (
                    "Antworte in genau einem kurzen deutschen Satz: "
                    "Wozu dient ein KV-Cache?"
                ),
            },
        ],
        "temperature": 0,
        "seed": 12345,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning_effort": reasoning_effort,
    }
    write_json(run_dir / "chat_request.json", payload)

    before = gpu_snapshot(min(startup_timeout, 30.0))
    process: subprocess.Popen[Any] | None = None
    readiness: dict[str, Any] | None = None
    model_listing: dict[str, Any] | None = None
    requests: list[dict[str, Any]] = []
    stop_result: dict[str, Any] | None = None
    run_error = None
    status = "failed"
    started_at = utc_now()

    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env=subprocess_environment(llama_dir),
                start_new_session=True,
            )
            readiness = wait_for_server(
                process,
                base_url,
                api_key,
                timeout=startup_timeout,
            )
            if readiness["ready"]:
                model_listing = http_json_request(
                    f"{base_url}/v1/models",
                    api_key,
                    timeout=request_timeout,
                )
                write_json(run_dir / "models_response.json", model_listing)
                for number in (1, 2):
                    response = http_json_request(
                        f"{base_url}/v1/chat/completions",
                        api_key,
                        timeout=request_timeout,
                        payload=payload,
                    )
                    write_json(run_dir / f"chat_response_{number}.json", response)
                    requests.append(summarize_chat_response(response))

                if all(item["status_code"] == 200 for item in requests):
                    if all(item["content"].strip() for item in requests):
                        status = "ok"
                    else:
                        status = "no-final-content"
                else:
                    status = "request-failed"
            else:
                status = readiness["reason"]
        except (OSError, ValueError) as exc:
            run_error = str(exc)
            status = "failed"
        finally:
            if process is not None:
                stop_result = stop_process_group(
                    process,
                    timeout=shutdown_timeout,
                )

    after = gpu_snapshot(min(startup_timeout, 30.0))
    cache_reused = False
    if len(requests) == 2:
        cached_tokens = requests[1].get("cached_tokens")
        cache_reused = isinstance(cached_tokens, (int, float)) and cached_tokens > 0
    return {
        "name": "server-smoke",
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "configuration": {
            "context_size": 8192,
            "batch_size": 2048,
            "ubatch_size": 512,
            "flash_attention": "on",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "parallel": 1,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "request_count": 2,
        },
        "command": redacted_command,
        "base_url": base_url,
        "api_key": "<ephemeral-redacted>",
        "log_file": str(log_path),
        "readiness": readiness,
        "models": model_listing,
        "requests": requests,
        "prompt_cache_reused": cache_reused,
        "process_stop": stop_result,
        "error": run_error,
        "gpu_before": before,
        "gpu_after_shutdown": after,
    }


def parse_context_targets(value: str) -> list[int]:
    try:
        targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Kontextziele müssen ganze Zahlen sein") from exc
    if not targets:
        raise ValueError("Mindestens ein Kontextziel ist erforderlich")
    if any(target <= 0 for target in targets):
        raise ValueError("Kontextziele müssen größer als 0 sein")
    if targets != sorted(set(targets)):
        raise ValueError("Kontextziele müssen eindeutig und aufsteigend sein")
    return targets


def filler_text(repetitions: int) -> str:
    return "\n".join(
        FILLER_PARAGRAPHS[index % len(FILLER_PARAGRAPHS)]
        for index in range(repetitions)
    )


def growing_user_content(stage_number: int, repetitions: int) -> str:
    filler = filler_text(repetitions)
    return (
        f"Messstufe {stage_number}: Der folgende technische Dokumentationsblock "
        "ist Bestandteil eines wachsenden, reproduzierbaren Dialogs.\n\n"
        f"{filler}\n\n"
        "Aufgabe: Bestätige in genau einem kurzen deutschen Satz, dass du den "
        "bisherigen technischen Kontext berücksichtigt hast."
    )


def calibrate_growing_content(
    stage_number: int,
    target_tokens: int,
    token_counter: Callable[[str], int],
    *,
    tolerance: int = 64,
) -> dict[str, Any]:
    base_tokens = token_counter(growing_user_content(stage_number, 0))
    if base_tokens > target_tokens:
        raise ValueError(
            f"Bestehender Dialog hat bereits {base_tokens} Tokens und überschreitet "
            f"das Ziel {target_tokens}"
        )

    sample_repetitions = 16
    sample_tokens = token_counter(
        growing_user_content(stage_number, sample_repetitions)
    )
    tokens_per_repetition = max(
        1.0, (sample_tokens - base_tokens) / sample_repetitions
    )
    repetitions = max(
        0,
        round((target_tokens - base_tokens) / tokens_per_repetition),
    )
    attempts = []
    seen = set()
    for _ in range(12):
        if repetitions in seen:
            break
        seen.add(repetitions)
        content = growing_user_content(stage_number, repetitions)
        measured = token_counter(content)
        attempts.append(
            {"repetitions": repetitions, "prompt_tokens": measured}
        )
        difference = target_tokens - measured
        if 0 <= difference <= tolerance:
            break
        adjustment = round(difference / tokens_per_repetition)
        if adjustment == 0:
            adjustment = 1 if difference > 0 else -1
        repetitions = max(0, repetitions + adjustment)

    below_target = [
        attempt
        for attempt in attempts
        if attempt["prompt_tokens"] <= target_tokens
    ]
    if below_target:
        selected = max(below_target, key=lambda item: item["prompt_tokens"])
    else:
        selected = min(attempts, key=lambda item: item["prompt_tokens"])
    content = growing_user_content(stage_number, selected["repetitions"])
    return {
        "content": content,
        "prompt_tokens": selected["prompt_tokens"],
        "target_tokens": target_tokens,
        "difference_tokens": target_tokens - selected["prompt_tokens"],
        "repetitions": selected["repetitions"],
        "attempts": attempts,
    }


def chat_prompt_token_count(
    base_url: str,
    api_key: str,
    messages: list[dict[str, str]],
    *,
    timeout: float,
) -> int:
    templated = http_json_request(
        f"{base_url}/apply-template",
        api_key,
        timeout=timeout,
        payload={"messages": messages},
    )
    templated_json = templated.get("json")
    if templated["status_code"] != 200 or not isinstance(templated_json, dict):
        raise ValueError("Chat-Template konnte nicht angewendet werden")
    prompt = templated_json.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("Chat-Template lieferte keinen Prompt")

    tokenized = http_json_request(
        f"{base_url}/tokenize",
        api_key,
        timeout=timeout,
        payload={
            "content": prompt,
            "add_special": False,
            "parse_special": True,
            "with_pieces": False,
        },
    )
    tokenized_json = tokenized.get("json")
    if tokenized["status_code"] != 200 or not isinstance(tokenized_json, dict):
        raise ValueError("Prompt konnte nicht tokenisiert werden")
    tokens = tokenized_json.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("Tokenize-Endpunkt lieferte keine Tokenliste")
    return len(tokens)


def write_growing_csv(path: Path, stages: list[dict[str, Any]]) -> None:
    fields = (
        "stage",
        "target_prompt_tokens",
        "calibrated_prompt_tokens",
        "actual_prompt_tokens",
        "cached_tokens",
        "new_prompt_tokens",
        "cache_ratio",
        "prompt_tokens_per_second",
        "completion_tokens",
        "generation_tokens_per_second",
        "spec_draft_tokens",
        "spec_accepted_tokens",
        "spec_acceptance_ratio",
        "spec_drafts",
        "wall_seconds",
        "finish_reason",
        "content_characters",
        "status",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            writer.writerow({field: stage.get(field) for field in fields})


def _run_growing_chat_attempt(
    llama_dir: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    run_dir: Path,
    *,
    context_size: int,
    targets: list[int],
    cache_type: str,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    max_tokens: int,
    reasoning_effort: str,
    batch_size: int = 2048,
    ubatch_size: int = 512,
    flash_attention: str = "on",
    threads: int | None = None,
    alias: str = "autotune-growing-chat",
    speculation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    port = find_free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    api_key = secrets.token_urlsafe(32)
    command = build_server_smoke_command(
        server,
        model,
        capabilities,
        tuning_plan,
        port=port,
        alias=alias,
        api_key=api_key,
        context_size=context_size,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        cache_type=cache_type,
        flash_attention=flash_attention,
        threads=threads,
        speculation=speculation,
    )
    log_path = run_dir / "server.log"
    csv_path = run_dir / "growing_chat.csv"
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Du bist ein präziser technischer Assistent. Berücksichtige den "
                "vollständigen bisherigen Dialog und antworte knapp auf Deutsch."
            ),
        }
    ]
    stages: list[dict[str, Any]] = []
    before = gpu_snapshot(min(startup_timeout, 30.0))
    after_ready = None
    metrics_after_ready = None
    metrics_after_final_stage = None
    after_shutdown = None
    process: subprocess.Popen[Any] | None = None
    readiness: dict[str, Any] | None = None
    stop_result: dict[str, Any] | None = None
    run_error = None
    status = "failed"
    started_at = utc_now()

    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env=subprocess_environment(llama_dir),
                start_new_session=True,
            )
            readiness = wait_for_server(
                process,
                base_url,
                api_key,
                timeout=startup_timeout,
            )
            if not readiness["ready"]:
                status = readiness["reason"]
            else:
                after_ready = gpu_snapshot(min(request_timeout, 30.0))
                if capabilities["server_options"].get("--metrics"):
                    metrics_after_ready = fetch_server_metrics(
                        base_url,
                        api_key,
                        timeout=min(request_timeout, 30.0),
                    )
                for stage_number, target in enumerate(targets, 1):
                    print(
                        f"  Kontextstufe {stage_number}/{len(targets)}: "
                        f"Ziel={target} Tokens",
                        flush=True,
                    )

                    def count_candidate(content: str) -> int:
                        candidate_messages = [
                            *messages,
                            {"role": "user", "content": content},
                        ]
                        return chat_prompt_token_count(
                            base_url,
                            api_key,
                            candidate_messages,
                            timeout=request_timeout,
                        )

                    calibration = calibrate_growing_content(
                        stage_number,
                        target,
                        count_candidate,
                    )
                    messages.append(
                        {"role": "user", "content": calibration["content"]}
                    )
                    request_payload: dict[str, Any] = {
                        "model": alias,
                        "messages": messages,
                        "temperature": 0,
                        "seed": 12345,
                        "max_tokens": max_tokens,
                        "stream": False,
                        "reasoning_effort": reasoning_effort,
                    }
                    write_json(
                        run_dir / f"stage_{stage_number:02d}_request.json",
                        request_payload,
                    )
                    gpu_before_stage = gpu_snapshot(min(request_timeout, 30.0))
                    metrics_before_stage = (
                        metrics_after_final_stage or metrics_after_ready
                    )
                    response = http_json_request(
                        f"{base_url}/v1/chat/completions",
                        api_key,
                        timeout=request_timeout,
                        payload=request_payload,
                    )
                    gpu_after_stage = gpu_snapshot(min(request_timeout, 30.0))
                    metrics_after_stage = None
                    if capabilities["server_options"].get("--metrics"):
                        metrics_after_stage = fetch_server_metrics(
                            base_url,
                            api_key,
                            timeout=min(request_timeout, 30.0),
                        )
                        metrics_after_final_stage = metrics_after_stage
                    write_json(
                        run_dir / f"stage_{stage_number:02d}_response.json",
                        response,
                    )
                    summary = summarize_chat_response(response)
                    actual_prompt = summary.get("prompt_tokens")
                    cached_tokens = summary.get("cached_tokens")
                    if not isinstance(actual_prompt, int):
                        actual_prompt = None
                    if not isinstance(cached_tokens, int):
                        cached_tokens = 0
                    new_prompt = (
                        actual_prompt - cached_tokens
                        if actual_prompt is not None
                        else None
                    )
                    cache_ratio = (
                        round(cached_tokens / actual_prompt, 6)
                        if actual_prompt
                        else None
                    )
                    finish_reason = summary.get("finish_reason")
                    content = summary.get("content", "")
                    stage_status = classify_chat_stage(response, summary)
                    draft_tokens = metric_delta(
                        metrics_before_stage,
                        metrics_after_stage,
                        "llamacpp:spec_decode_num_draft_tokens_total",
                    )
                    accepted_tokens = metric_delta(
                        metrics_before_stage,
                        metrics_after_stage,
                        "llamacpp:spec_decode_num_accepted_tokens_total",
                    )
                    spec_drafts = metric_delta(
                        metrics_before_stage,
                        metrics_after_stage,
                        "llamacpp:spec_decode_num_drafts_total",
                    )
                    acceptance_ratio = (
                        round(accepted_tokens / draft_tokens, 6)
                        if isinstance(draft_tokens, (int, float))
                        and draft_tokens > 0
                        and isinstance(accepted_tokens, (int, float))
                        else None
                    )
                    stage_result = {
                        "stage": stage_number,
                        "status": stage_status,
                        "target_prompt_tokens": target,
                        "calibrated_prompt_tokens": calibration["prompt_tokens"],
                        "calibration_difference_tokens": calibration[
                            "difference_tokens"
                        ],
                        "calibration_repetitions": calibration["repetitions"],
                        "calibration_attempts": calibration["attempts"],
                        "actual_prompt_tokens": actual_prompt,
                        "cached_tokens": cached_tokens,
                        "new_prompt_tokens": new_prompt,
                        "cache_ratio": cache_ratio,
                        "prompt_tokens_per_second": summary.get(
                            "prompt_tokens_per_second"
                        ),
                        "completion_tokens": summary.get("completion_tokens"),
                        "generation_tokens_per_second": summary.get(
                            "predicted_tokens_per_second"
                        ),
                        "spec_draft_tokens": draft_tokens,
                        "spec_accepted_tokens": accepted_tokens,
                        "spec_acceptance_ratio": acceptance_ratio,
                        "spec_drafts": spec_drafts,
                        "wall_seconds": summary.get("wall_seconds"),
                        "finish_reason": finish_reason,
                        "content": content,
                        "content_characters": summary.get("content_characters"),
                        "reasoning_characters": summary.get(
                            "reasoning_characters"
                        ),
                        "output_truncated": finish_reason == "length",
                        "http_status": response.get("status_code"),
                        "response_error": response.get("error"),
                        "api_error": chat_api_error_message(response),
                        "gpu_before": gpu_before_stage,
                        "gpu_after": gpu_after_stage,
                        "metrics_before": metrics_before_stage,
                        "metrics_after": metrics_after_stage,
                    }
                    stages.append(stage_result)
                    print(
                        f"    Status={stage_status}, Prompt={actual_prompt}, "
                        f"Cache={cached_tokens}, Neu={new_prompt}, "
                        f"Generation={stage_result['generation_tokens_per_second']}",
                        flush=True,
                    )
                    if stage_status != "ok":
                        break
                    messages.append({"role": "assistant", "content": content})
                if len(stages) == len(targets) and all(
                    stage["status"] == "ok" for stage in stages
                ):
                    status = "ok"
                elif len(stages) == 1:
                    status = stages[0]["status"]
                else:
                    status = "partial"
        except (OSError, ValueError) as exc:
            run_error = str(exc)
            status = "failed"
        finally:
            if process is not None:
                stop_result = stop_process_group(
                    process,
                    timeout=shutdown_timeout,
                )

    after_shutdown = gpu_snapshot(min(startup_timeout, 30.0))
    write_growing_csv(csv_path, stages)
    return {
        "name": "growing-chat",
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "configuration": {
            "context_size": context_size,
            "targets": targets,
            "cache_type_k": cache_type,
            "cache_type_v": cache_type,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "flash_attention": flash_attention,
            "threads": threads or min(8, os.cpu_count() or 8),
            "parallel": 1,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "speculation": speculation or {"spec_type": "server-default"},
        },
        "command": redact_secret(command, api_key),
        "base_url": base_url,
        "api_key": "<ephemeral-redacted>",
        "log_file": str(log_path),
        "csv_file": str(csv_path),
        "readiness": readiness,
        "stages": stages,
        "process_stop": stop_result,
        "error": run_error,
        "gpu_before": before,
        "gpu_after_ready": after_ready,
        "gpu_after_shutdown": after_shutdown,
        "metrics_after_ready": metrics_after_ready,
        "metrics_after_final_stage": metrics_after_final_stage,
    }


def growing_chat_retry_reason(result: dict[str, Any]) -> str | None:
    """Return the retryable failure class of a growing-chat attempt."""
    stages = result.get("stages")
    if not isinstance(stages, list) or not stages:
        return None
    status = stages[-1].get("status")
    if status in {
        "reasoning-truncated",
        "output-truncated",
        "peg-format-error",
    }:
        return status
    return None


def run_growing_chat(
    llama_dir: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    run_dir: Path,
    *,
    context_size: int,
    targets: list[int],
    cache_type: str,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    max_tokens: int,
    reasoning_effort: str,
    batch_size: int,
    ubatch_size: int,
    flash_attention: str,
    threads: int | None,
    alias: str,
    speculation: dict[str, Any] | None = None,
    adaptive_max_tokens: int = 2048,
) -> dict[str, Any]:
    """Run a fresh server per adaptive token attempt to avoid cache bias."""
    budgets = adaptive_token_budgets(max_tokens, adaptive_max_tokens)
    attempts: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None

    for attempt_number, budget in enumerate(budgets, 1):
        if attempt_number == 1:
            attempt_dir = run_dir
        else:
            attempt_dir = run_dir / (
                f"adaptive_attempt_{attempt_number:02d}_tokens_{budget}"
            )
            attempt_dir.mkdir(exist_ok=True)
        result = _run_growing_chat_attempt(
            llama_dir,
            server,
            model,
            capabilities,
            tuning_plan,
            attempt_dir,
            context_size=context_size,
            targets=targets,
            cache_type=cache_type,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            shutdown_timeout=shutdown_timeout,
            max_tokens=budget,
            reasoning_effort=reasoning_effort,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            flash_attention=flash_attention,
            threads=threads,
            alias=alias,
            speculation=speculation,
        )
        reason = growing_chat_retry_reason(result)
        attempts.append(
            {
                "attempt": attempt_number,
                "max_tokens": budget,
                "status": result.get("status"),
                "retry_reason": reason,
                "run_directory": str(attempt_dir),
            }
        )
        final_result = result
        if reason is None or attempt_number == len(budgets):
            break
        next_budget = budgets[attempt_number]
        print(
            "    Antwort noch nicht verwertbar "
            f"({reason}); neuer Serverversuch mit "
            f"max_tokens={next_budget}",
            flush=True,
        )

    if final_result is None:
        raise ValueError("Growing-Chat-Versuch wurde nicht ausgeführt")
    final_result["adaptive_attempts"] = attempts
    final_result["adaptive_retry_count"] = max(0, len(attempts) - 1)
    final_result["initial_max_tokens"] = max_tokens
    final_result["effective_max_tokens"] = attempts[-1]["max_tokens"]
    write_json(run_dir / "adaptive_attempts.json", attempts)
    return final_result


def build_benchmark_command(
    bench: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    *,
    prompt_tokens: int,
    generation_tokens: int,
    batch_size: int,
    ubatch_size: int,
    flash_attention: str,
    repetitions: int,
    threads: int | None = None,
) -> list[str]:
    options = capabilities["bench_options"]
    required = (
        "--n-prompt",
        "--n-gen",
        "--batch-size",
        "--ubatch-size",
        "--output",
        "--repetitions",
    )
    missing = [option for option in required if not options.get(option)]
    if missing:
        raise ValueError(
            "llama-bench unterstützt erforderliche Optionen nicht: "
            + ", ".join(missing)
        )

    command = [
        str(bench),
        "-m",
        str(model),
        "--n-prompt",
        str(prompt_tokens),
        "--n-gen",
        str(generation_tokens),
        "--batch-size",
        str(batch_size),
        "--ubatch-size",
        str(ubatch_size),
    ]
    if (
        options.get("--cache-type-k")
        and options.get("--cache-type-v")
        and "f16" in capabilities["cache_types"]
    ):
        command.extend(["--cache-type-k", "f16", "--cache-type-v", "f16"])
    if options.get("--n-gpu-layers"):
        gpu_layers = (
            "999"
            if tuning_plan["fixed_parameters"]["gpu_layers"] == "all"
            else "-1"
        )
        command.extend(["--n-gpu-layers", gpu_layers])
    if options.get("--flash-attn"):
        command.extend(["--flash-attn", flash_attention])
    if threads is not None:
        if not options.get("--threads"):
            raise ValueError(
                "llama-bench unterstützt die erforderliche Option --threads nicht"
            )
        command.extend(["--threads", str(threads)])
    if options.get("--load-mode"):
        command.extend(["--load-mode", "auto"])
    command.extend(
        ["--repetitions", str(repetitions), "--output", "json"]
    )
    if options.get("--progress"):
        command.append("--progress")
    return command


def build_smoke_command(
    bench: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    repetitions: int,
) -> list[str]:
    return build_benchmark_command(
        bench,
        model,
        capabilities,
        tuning_plan,
        prompt_tokens=512,
        generation_tokens=64,
        batch_size=512,
        ubatch_size=128,
        flash_attention="on",
        repetitions=repetitions,
    )


def parse_benchmark_json(
    output: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(output)
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError("llama-bench-Ausgabe ist keine JSON-Liste von Messwerten")

    normalized = []
    for row in payload:
        prompt_tokens = row.get("n_prompt", 0)
        generated_tokens = row.get("n_gen", 0)
        if prompt_tokens:
            test_kind = "prompt"
        elif generated_tokens:
            test_kind = "generation"
        else:
            test_kind = "unknown"
        normalized.append(
            {
                "test_kind": test_kind,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "batch_size": row.get("n_batch"),
                "ubatch_size": row.get("n_ubatch"),
                "threads": row.get("n_threads"),
                "gpu_layers": row.get("n_gpu_layers"),
                "flash_attention": row.get("flash_attn"),
                "tokens_per_second": row.get("avg_ts"),
                "tokens_per_second_stddev": row.get("stddev_ts"),
            }
        )
    return payload, normalized


def run_recorded_benchmark(
    llama_dir: Path,
    run_dir: Path,
    command: list[str],
    *,
    name: str,
    file_stem: str,
    configuration: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    before = gpu_snapshot(min(timeout, 30.0))
    result = run_command(
        command,
        timeout=timeout,
        env=subprocess_environment(llama_dir),
    )
    after = gpu_snapshot(min(timeout, 30.0))

    stdout_path = run_dir / f"{file_stem}.json"
    stderr_path = run_dir / f"{file_stem}.log"
    stdout_path.write_text(result["stdout"], encoding="utf-8")
    stderr_path.write_text(result["stderr"], encoding="utf-8")

    status = "ok"
    parse_error = None
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    if result["timed_out"]:
        status = "timeout"
    elif result["returncode"] != 0:
        status = "failed"
    else:
        try:
            rows, metrics = parse_benchmark_json(result["stdout"])
        except (json.JSONDecodeError, ValueError) as exc:
            status = "invalid-json"
            parse_error = str(exc)

    execution = {
        key: value
        for key, value in result.items()
        if key not in {"stdout", "stderr"}
    }
    return {
        "name": name,
        "status": status,
        "configuration": configuration,
        "execution": execution,
        "stdout_file": str(stdout_path),
        "stderr_file": str(stderr_path),
        "parse_error": parse_error,
        "rows": rows,
        "metrics": metrics,
        "gpu_before": before,
        "gpu_after": after,
    }


def run_smoke_benchmark(
    llama_dir: Path,
    bench: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    run_dir: Path,
    *,
    timeout: float,
    repetitions: int,
) -> dict[str, Any]:
    command = build_smoke_command(
        bench, model, capabilities, tuning_plan, repetitions
    )
    return run_recorded_benchmark(
        llama_dir,
        run_dir,
        command,
        name="smoke",
        file_stem="smoke_benchmark",
        configuration={
            "prompt_tokens": 512,
            "generation_tokens": 64,
            "batch_size": 512,
            "ubatch_size": 128,
            "flash_attention": "on",
            "repetitions": repetitions,
        },
        timeout=timeout,
    )


def rank_screening_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for case in cases:
        if case["status"] != "ok":
            continue
        metrics = {
            metric["test_kind"]: metric for metric in case.get("metrics", [])
        }
        prompt = metrics.get("prompt", {}).get("tokens_per_second")
        generation = metrics.get("generation", {}).get("tokens_per_second")
        if not all(
            isinstance(value, (int, float)) and value > 0
            for value in (prompt, generation)
        ):
            continue
        candidates.append(
            {
                "configuration": case["configuration"],
                "prompt_tokens_per_second": prompt,
                "generation_tokens_per_second": generation,
                "prompt_stddev": metrics["prompt"].get(
                    "tokens_per_second_stddev"
                ),
                "generation_stddev": metrics["generation"].get(
                    "tokens_per_second_stddev"
                ),
            }
        )

    if not candidates:
        return []
    best_prompt = max(item["prompt_tokens_per_second"] for item in candidates)
    best_generation = max(
        item["generation_tokens_per_second"] for item in candidates
    )
    for item in candidates:
        item["balanced_score"] = round(
            math.sqrt(
                item["prompt_tokens_per_second"]
                / best_prompt
                * item["generation_tokens_per_second"]
                / best_generation
            ),
            6,
        )
    return sorted(
        candidates,
        key=lambda item: item["balanced_score"],
        reverse=True,
    )


def run_batch_screening(
    llama_dir: Path,
    bench: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    run_dir: Path,
    *,
    timeout: float,
    repetitions: int,
    prompt_tokens: int,
    generation_tokens: int,
) -> dict[str, Any]:
    stage = next(
        item
        for item in tuning_plan["stages"]
        if item["name"] == "batch-screening"
    )
    stage_dir = run_dir / "batch_screening"
    stage_dir.mkdir()
    cases = []
    case_number = 0
    total_cases = len(stage["batch_ubatch_pairs"]) * len(
        stage["flash_attention"]
    )
    for pair in stage["batch_ubatch_pairs"]:
        for flash_attention in stage["flash_attention"]:
            case_number += 1
            configuration = {
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "batch_size": pair["batch_size"],
                "ubatch_size": pair["ubatch_size"],
                "flash_attention": flash_attention,
                "repetitions": repetitions,
            }
            command = build_benchmark_command(
                bench,
                model,
                capabilities,
                tuning_plan,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                batch_size=pair["batch_size"],
                ubatch_size=pair["ubatch_size"],
                flash_attention=flash_attention,
                repetitions=repetitions,
            )
            file_stem = (
                f"{case_number:02d}_b{pair['batch_size']}_"
                f"ub{pair['ubatch_size']}_fa_{flash_attention}"
            )
            print(
                f"  Kandidat {case_number}/{total_cases}: "
                f"B={pair['batch_size']}, UB={pair['ubatch_size']}, "
                f"FA={flash_attention}",
                flush=True,
            )
            recorded = run_recorded_benchmark(
                llama_dir,
                stage_dir,
                command,
                name=file_stem,
                file_stem=file_stem,
                configuration=configuration,
                timeout=timeout,
            )
            cases.append(recorded)
            measured = {
                metric["test_kind"]: metric["tokens_per_second"]
                for metric in recorded.get("metrics", [])
            }
            print(
                f"    Status={recorded['status']}, "
                f"Prompt={measured.get('prompt', '?')}, "
                f"Generation={measured.get('generation', '?')}",
                flush=True,
            )

    successful = sum(case["status"] == "ok" for case in cases)
    if successful == len(cases):
        status = "ok"
    elif successful:
        status = "partial"
    else:
        status = "failed"
    ranking = rank_screening_cases(cases)
    return {
        "name": "batch-screening",
        "status": status,
        "successful_cases": successful,
        "total_cases": len(cases),
        "cases": cases,
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
    }


def run_thread_screening(
    llama_dir: Path,
    bench: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    experiment_plan: dict[str, Any],
    batch_result: dict[str, Any],
    run_dir: Path,
    *,
    timeout: float,
    repetitions: int,
    prompt_tokens: int,
    generation_tokens: int,
) -> dict[str, Any]:
    """Test the planned thread counts on the surviving batch candidates."""
    stage = next(
        item
        for item in experiment_plan["stages"]
        if item["id"] == "thread-screening"
    )
    top_k = experiment_plan["policy"]["top_k"]
    survivors = batch_result.get("ranking", [])[:top_k]
    thread_candidates = stage.get("candidates", [])
    stage_dir = run_dir / "thread_screening"
    stage_dir.mkdir(exist_ok=True)
    cases: list[dict[str, Any]] = []
    total_cases = len(survivors) * len(thread_candidates)
    case_number = 0

    for survivor_number, survivor in enumerate(survivors, 1):
        base = survivor["configuration"]
        for thread_candidate in thread_candidates:
            case_number += 1
            threads = thread_candidate["threads"]
            configuration = {
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "batch_size": base["batch_size"],
                "ubatch_size": base["ubatch_size"],
                "flash_attention": base["flash_attention"],
                "threads": threads,
                "repetitions": repetitions,
                "source_batch_rank": survivor_number,
            }
            file_stem = (
                f"{case_number:02d}_source{survivor_number}_"
                f"b{base['batch_size']}_ub{base['ubatch_size']}_"
                f"fa_{base['flash_attention']}_t{threads}"
            )
            print(
                f"  Thread-Kandidat {case_number}/{total_cases}: "
                f"B={base['batch_size']}, UB={base['ubatch_size']}, "
                f"FA={base['flash_attention']}, T={threads}",
                flush=True,
            )
            try:
                command = build_benchmark_command(
                    bench,
                    model,
                    capabilities,
                    tuning_plan,
                    prompt_tokens=prompt_tokens,
                    generation_tokens=generation_tokens,
                    batch_size=base["batch_size"],
                    ubatch_size=base["ubatch_size"],
                    flash_attention=base["flash_attention"],
                    repetitions=repetitions,
                    threads=threads,
                )
                recorded = run_recorded_benchmark(
                    llama_dir,
                    stage_dir,
                    command,
                    name=file_stem,
                    file_stem=file_stem,
                    configuration=configuration,
                    timeout=timeout,
                )
            except (OSError, ValueError) as exc:
                recorded = {
                    "name": file_stem,
                    "status": "configuration-error",
                    "configuration": configuration,
                    "error": str(exc),
                    "metrics": [],
                }
            cases.append(recorded)
            measured = {
                metric["test_kind"]: metric["tokens_per_second"]
                for metric in recorded.get("metrics", [])
            }
            print(
                f"    Status={recorded['status']}, "
                f"Prompt={measured.get('prompt', '?')}, "
                f"Generation={measured.get('generation', '?')}",
                flush=True,
            )

    successful = sum(case["status"] == "ok" for case in cases)
    ranking = rank_screening_cases(cases)
    if successful == len(cases) and cases:
        status = "ok"
    elif successful:
        status = "partial"
    else:
        status = "failed"
    return {
        "name": "thread-screening",
        "status": status,
        "successful_cases": successful,
        "total_cases": len(cases),
        "source_survivors": len(survivors),
        "cases": cases,
        "ranking": ranking,
        "winner": ranking[0] if ranking else None,
    }


def base_configuration_key(configuration: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        configuration.get(key)
        for key in (
            "batch_size",
            "ubatch_size",
            "flash_attention",
            "threads",
        )
    )


def rank_context_cache_cases(
    cases: list[dict[str, Any]],
    planned_contexts: list[int] | None = None,
) -> dict[str, Any]:
    """Normalize server measurements within each context and aggregate bases."""
    valid: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") != "ok":
            continue
        server_run = case.get("server_run") or {}
        stages = server_run.get("stages") or []
        if len(stages) != 1 or stages[0].get("status") != "ok":
            continue
        stage = stages[0]
        prompt = stage.get("prompt_tokens_per_second")
        generation = stage.get("generation_tokens_per_second")
        wall = stage.get("wall_seconds")
        if not all(
            isinstance(value, (int, float)) and value > 0
            for value in (prompt, generation, wall)
        ):
            continue
        valid.append(
            {
                "configuration": case["configuration"],
                "prompt_tokens_per_second": prompt,
                "generation_tokens_per_second": generation,
                "wall_seconds": wall,
                "actual_prompt_tokens": stage.get("actual_prompt_tokens"),
                "finish_reason": stage.get("finish_reason"),
            }
        )

    by_context: dict[int, list[dict[str, Any]]] = {}
    for item in valid:
        context_size = item["configuration"]["context_size"]
        by_context.setdefault(context_size, []).append(item)

    ranked_cases: list[dict[str, Any]] = []
    for context_size, items in by_context.items():
        best_prompt = max(item["prompt_tokens_per_second"] for item in items)
        best_generation = max(
            item["generation_tokens_per_second"] for item in items
        )
        best_wall = min(item["wall_seconds"] for item in items)
        for item in items:
            prompt_score = item["prompt_tokens_per_second"] / best_prompt
            generation_score = (
                item["generation_tokens_per_second"] / best_generation
            )
            latency_score = best_wall / item["wall_seconds"]
            item["relative_score"] = round(
                0.45 * prompt_score
                + 0.45 * generation_score
                + 0.10 * latency_score,
                6,
            )
            ranked_cases.append(item)

    ranked_cases.sort(
        key=lambda item: (
            item["configuration"]["context_size"],
            -item["relative_score"],
        )
    )
    winners_by_context = []
    for context_size in sorted(by_context):
        context_items = [
            item
            for item in ranked_cases
            if item["configuration"]["context_size"] == context_size
        ]
        winners_by_context.append(
            max(context_items, key=lambda item: item["relative_score"])
        )

    validated_contexts = sorted(by_context)
    all_contexts = sorted(
        set(planned_contexts)
        if planned_contexts is not None
        else set(validated_contexts)
    )
    base_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in ranked_cases:
        key = base_configuration_key(item["configuration"])
        base_groups.setdefault(key, []).append(item)
    base_ranking = []
    for key, items in base_groups.items():
        best_per_context = []
        for context_size in all_contexts:
            matching = [
                item
                for item in items
                if item["configuration"]["context_size"] == context_size
            ]
            if matching:
                best_per_context.append(
                    max(matching, key=lambda item: item["relative_score"])
                )
        if not best_per_context:
            continue
        geometric_score = math.exp(
            sum(math.log(max(item["relative_score"], 1e-12)) for item in best_per_context)
            / len(best_per_context)
        )
        coverage = len(best_per_context) / max(1, len(all_contexts))
        base_ranking.append(
            {
                "configuration": {
                    "batch_size": key[0],
                    "ubatch_size": key[1],
                    "flash_attention": key[2],
                    "threads": key[3],
                },
                "aggregate_score": round(geometric_score * coverage, 6),
                "context_coverage": len(best_per_context),
                "total_contexts": len(all_contexts),
                "best_profiles": best_per_context,
            }
        )
    base_ranking.sort(
        key=lambda item: item["aggregate_score"], reverse=True
    )
    return {
        "case_ranking": ranked_cases,
        "winners_by_context": winners_by_context,
        "base_ranking": base_ranking,
        "planned_contexts": all_contexts,
        "validated_contexts": validated_contexts,
        "missing_contexts": sorted(
            set(all_contexts) - set(validated_contexts)
        ),
        "context_coverage": len(validated_contexts),
        "total_contexts": len(all_contexts),
    }


def summarize_context_cache_screening(
    cases: list[dict[str, Any]],
    total_cases: int,
    planned_contexts: list[int] | None = None,
) -> dict[str, Any]:
    successful = sum(case.get("status") == "ok" for case in cases)
    ranking = rank_context_cache_cases(
        cases, planned_contexts=planned_contexts
    )
    if len(cases) < total_cases:
        status = "running"
    elif successful == total_cases and cases:
        status = "ok"
    elif successful:
        status = "partial"
    else:
        status = "failed"
    return {
        "name": "context-cache-screening",
        "status": status,
        "successful_cases": successful,
        "completed_cases": len(cases),
        "total_cases": total_cases,
        "cases": cases,
        **ranking,
        "winner": ranking["base_ranking"][0]
        if ranking["base_ranking"]
        else None,
    }


def run_context_cache_screening(
    llama_dir: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    experiment_plan: dict[str, Any],
    thread_result: dict[str, Any],
    run_dir: Path,
    *,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    max_tokens: int,
    reasoning_effort: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    stage = next(
        item
        for item in experiment_plan["stages"]
        if item["id"] == "context-cache-screening"
    )
    top_k = experiment_plan["policy"]["top_k"]
    survivors = thread_result.get("ranking", [])[:top_k]
    variants = stage.get("candidates", [])
    planned_contexts = sorted(
        {variant["context_size"] for variant in variants}
    )
    total_cases = len(survivors) * len(variants)
    stage_dir = run_dir / "context_cache_screening"
    stage_dir.mkdir(exist_ok=True)
    cases: list[dict[str, Any]] = []
    case_number = 0

    for survivor_number, survivor in enumerate(survivors, 1):
        base = survivor["configuration"]
        for variant in variants:
            case_number += 1
            context_size = variant["context_size"]
            cache_type = variant["cache_type_k"]
            configuration = {
                "batch_size": base["batch_size"],
                "ubatch_size": base["ubatch_size"],
                "flash_attention": base["flash_attention"],
                "threads": base["threads"],
                "context_size": context_size,
                "prompt_target": variant["prompt_target"],
                "cache_type_k": cache_type,
                "cache_type_v": variant["cache_type_v"],
                "memory_fit": variant["memory_fit"],
                "estimated_kv_gib": variant.get("estimated_kv_gib"),
                "source_thread_rank": survivor_number,
                "variant_id": variant["id"],
            }
            case_id = (
                f"{case_number:02d}_source{survivor_number}_"
                f"ctx{context_size}_{cache_type}"
            )
            case_dir = stage_dir / case_id
            case_dir.mkdir()
            print(
                f"  Kontext-Kandidat {case_number}/{total_cases}: "
                f"CTX={context_size}, Cache={cache_type}, "
                f"B={base['batch_size']}, UB={base['ubatch_size']}, "
                f"FA={base['flash_attention']}, T={base['threads']}",
                flush=True,
            )
            try:
                server_run = run_growing_chat(
                    llama_dir,
                    server,
                    model,
                    capabilities,
                    tuning_plan,
                    case_dir,
                    context_size=context_size,
                    targets=[variant["prompt_target"]],
                    cache_type=cache_type,
                    startup_timeout=startup_timeout,
                    request_timeout=request_timeout,
                    shutdown_timeout=shutdown_timeout,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    batch_size=base["batch_size"],
                    ubatch_size=base["ubatch_size"],
                    flash_attention=base["flash_attention"],
                    threads=base["threads"],
                    alias=f"autotune-context-{case_number}",
                )
                status = server_run["status"]
                error = server_run.get("error")
            except (OSError, ValueError) as exc:
                server_run = None
                status = "configuration-error"
                error = str(exc)
            case = {
                "id": case_id,
                "status": status,
                "configuration": configuration,
                "server_run": server_run,
                "error": error,
            }
            cases.append(case)
            result = summarize_context_cache_screening(
                cases, total_cases, planned_contexts
            )
            if progress_callback:
                progress_callback(result)
            server_stages = server_run.get("stages", []) if server_run else []
            stage_measurement = server_stages[0] if server_stages else {}
            print(
                f"    Status={status}, "
                f"Prompt={stage_measurement.get('prompt_tokens_per_second', '?')}, "
                f"Generation={stage_measurement.get('generation_tokens_per_second', '?')}, "
                f"Wall={stage_measurement.get('wall_seconds', '?')}",
                flush=True,
            )

    return summarize_context_cache_screening(
        cases, total_cases, planned_contexts
    )


def rank_speculation_cases(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") != "ok":
            continue
        server_run = case.get("server_run") or {}
        stages = server_run.get("stages") or []
        if len(stages) != 1 or stages[0].get("status") != "ok":
            continue
        stage = stages[0]
        prompt = stage.get("prompt_tokens_per_second")
        generation = stage.get("generation_tokens_per_second")
        wall = stage.get("wall_seconds")
        if not all(
            isinstance(value, (int, float)) and value > 0
            for value in (prompt, generation, wall)
        ):
            continue
        valid.append(
            {
                "configuration": case["configuration"],
                "prompt_tokens_per_second": prompt,
                "generation_tokens_per_second": generation,
                "wall_seconds": wall,
                "spec_draft_tokens": stage.get("spec_draft_tokens"),
                "spec_accepted_tokens": stage.get("spec_accepted_tokens"),
                "spec_acceptance_ratio": stage.get("spec_acceptance_ratio"),
                "spec_drafts": stage.get("spec_drafts"),
            }
        )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in valid:
        rank = item["configuration"]["source_context_rank"]
        grouped.setdefault(rank, []).append(item)
    ranked: list[dict[str, Any]] = []
    for source_rank, items in grouped.items():
        best_prompt = max(item["prompt_tokens_per_second"] for item in items)
        best_generation = max(
            item["generation_tokens_per_second"] for item in items
        )
        best_wall = min(item["wall_seconds"] for item in items)
        baseline = next(
            (
                item
                for item in items
                if item["configuration"]["speculation"].get("spec_type")
                == "none"
            ),
            None,
        )
        for item in items:
            prompt_score = item["prompt_tokens_per_second"] / best_prompt
            generation_score = (
                item["generation_tokens_per_second"] / best_generation
            )
            latency_score = best_wall / item["wall_seconds"]
            speculation_score = (
                0.15 * prompt_score
                + 0.75 * generation_score
                + 0.10 * latency_score
            )
            context_score = item["configuration"].get(
                "source_context_score", 1.0
            )
            item["relative_score"] = round(
                0.85 * speculation_score + 0.15 * context_score,
                6,
            )
            item["generation_speedup_vs_none"] = (
                round(
                    item["generation_tokens_per_second"]
                    / baseline["generation_tokens_per_second"],
                    6,
                )
                if baseline
                else None
            )
            item["wall_speedup_vs_none"] = (
                round(baseline["wall_seconds"] / item["wall_seconds"], 6)
                if baseline
                else None
            )
            item["source_context_rank"] = source_rank
            ranked.append(item)
    return sorted(ranked, key=lambda item: item["relative_score"], reverse=True)


def summarize_speculation_screening(
    cases: list[dict[str, Any]],
    total_cases: int,
    finalists: int,
) -> dict[str, Any]:
    successful = sum(case.get("status") == "ok" for case in cases)
    ranking = rank_speculation_cases(cases)
    if len(cases) < total_cases:
        status = "running"
    elif successful == total_cases and cases:
        status = "ok"
    elif successful:
        status = "partial"
    else:
        status = "failed"
    return {
        "name": "speculation-screening",
        "status": status,
        "successful_cases": successful,
        "completed_cases": len(cases),
        "total_cases": total_cases,
        "cases": cases,
        "ranking": ranking,
        "finalists": ranking[:finalists],
        "winner": ranking[0] if ranking else None,
    }


def run_speculation_screening(
    llama_dir: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    experiment_plan: dict[str, Any],
    context_result: dict[str, Any],
    run_dir: Path,
    *,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    max_tokens: int,
    reasoning_effort: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    stage = next(
        item
        for item in experiment_plan["stages"]
        if item["id"] == "speculation-screening"
    )
    top_k = experiment_plan["policy"]["top_k"]
    finalists = experiment_plan["policy"]["finalists"]
    survivors = context_result.get("base_ranking", [])[:top_k]
    variants = stage.get("candidates", [])
    total_cases = len(survivors) * len(variants)
    stage_dir = run_dir / "speculation_screening"
    stage_dir.mkdir(exist_ok=True)
    cases: list[dict[str, Any]] = []
    case_number = 0

    for survivor_number, survivor in enumerate(survivors, 1):
        base = survivor["configuration"]
        context_profile = max(
            survivor["best_profiles"],
            key=lambda item: item["configuration"]["context_size"],
        )
        profile_configuration = context_profile["configuration"]
        for variant in variants:
            case_number += 1
            spec_type = variant["spec_type"]
            safe_spec_id = variant["id"].replace(",", "_")
            case_id = (
                f"{case_number:02d}_source{survivor_number}_{safe_spec_id}"
            )
            case_dir = stage_dir / case_id
            case_dir.mkdir()
            configuration = {
                **base,
                "context_size": profile_configuration["context_size"],
                "prompt_target": profile_configuration["prompt_target"],
                "cache_type_k": profile_configuration["cache_type_k"],
                "cache_type_v": profile_configuration["cache_type_v"],
                "source_context_rank": survivor_number,
                "source_context_score": survivor["aggregate_score"],
                "speculation": variant,
            }
            print(
                f"  Spekulations-Kandidat {case_number}/{total_cases}: "
                f"Typ={spec_type}, CTX={configuration['context_size']}, "
                f"Cache={configuration['cache_type_k']}",
                flush=True,
            )
            try:
                server_run = run_growing_chat(
                    llama_dir,
                    server,
                    model,
                    capabilities,
                    tuning_plan,
                    case_dir,
                    context_size=configuration["context_size"],
                    targets=[configuration["prompt_target"]],
                    cache_type=configuration["cache_type_k"],
                    startup_timeout=startup_timeout,
                    request_timeout=request_timeout,
                    shutdown_timeout=shutdown_timeout,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    batch_size=base["batch_size"],
                    ubatch_size=base["ubatch_size"],
                    flash_attention=base["flash_attention"],
                    threads=base["threads"],
                    alias=f"autotune-spec-{case_number}",
                    speculation=variant,
                )
                status = server_run["status"]
                error = server_run.get("error")
            except (OSError, ValueError) as exc:
                server_run = None
                status = "configuration-error"
                error = str(exc)
            case = {
                "id": case_id,
                "status": status,
                "configuration": configuration,
                "server_run": server_run,
                "error": error,
            }
            cases.append(case)
            result = summarize_speculation_screening(
                cases, total_cases, finalists
            )
            if progress_callback:
                progress_callback(result)
            server_stages = server_run.get("stages", []) if server_run else []
            measurement = server_stages[0] if server_stages else {}
            acceptance = measurement.get("spec_acceptance_ratio")
            acceptance_text = (
                f"{acceptance * 100:.1f}%"
                if isinstance(acceptance, (int, float))
                else "-"
            )
            print(
                f"    Status={status}, "
                f"Generation={measurement.get('generation_tokens_per_second', '?')}, "
                f"Acceptance={acceptance_text}, "
                f"Wall={measurement.get('wall_seconds', '?')}",
                flush=True,
            )

    return summarize_speculation_screening(cases, total_cases, finalists)


def gpu_free_memory_mib(snapshot: dict[str, Any] | None) -> float | None:
    if not snapshot:
        return None
    gpus = parse_nvidia_gpus({"nvidia_smi": snapshot})
    values = [gpu.get("memory_free_mib") for gpu in gpus]
    numeric = [value for value in values if isinstance(value, (int, float))]
    return sum(numeric) if numeric else None


def final_validation_candidates(
    speculation_result: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = list(speculation_result.get("finalists", []))
    ranking = speculation_result.get("ranking", [])
    source_ranks = {
        item.get("source_context_rank", 1) for item in selected
    }
    for source_rank in source_ranks:
        baseline = next(
            (
                item
                for item in ranking
                if item.get("source_context_rank", 1) == source_rank
                and item["configuration"]["speculation"].get("spec_type")
                == "none"
            ),
            None,
        )
        if baseline is not None:
            selected.append(baseline)
    unique = []
    seen = set()
    for item in selected:
        speculation = item["configuration"]["speculation"]
        key = (
            item.get("source_context_rank", 1),
            speculation.get("id", speculation.get("spec_type")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def rank_final_validation_cases(
    cases: list[dict[str, Any]],
    expected_contexts: int,
    objective: str = "balanced",
) -> list[dict[str, Any]]:
    if objective not in OPTIMIZATION_OBJECTIVES:
        raise ValueError(f"Unbekanntes Optimierungsziel: {objective}")
    objective_definition = OPTIMIZATION_OBJECTIVES[objective]
    weights = objective_definition["weights"]
    measurements: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") != "ok":
            continue
        server_run = case.get("server_run") or {}
        stages = server_run.get("stages") or []
        if len(stages) != 1 or stages[0].get("status") != "ok":
            continue
        stage = stages[0]
        prompt = stage.get("prompt_tokens_per_second")
        generation = stage.get("generation_tokens_per_second")
        wall = stage.get("wall_seconds")
        if not all(
            isinstance(value, (int, float)) and value > 0
            for value in (prompt, generation, wall)
        ):
            continue
        measurements.append(
            {
                "configuration": case["configuration"],
                "prompt": prompt,
                "generation": generation,
                "wall": wall,
                "memory_free_mib": gpu_free_memory_mib(
                    server_run.get("gpu_after_ready")
                ),
                "acceptance_ratio": stage.get("spec_acceptance_ratio"),
            }
        )

    candidate_contexts: dict[tuple[Any, ...], dict[int, list[dict[str, Any]]]] = {}
    configurations: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in measurements:
        configuration = item["configuration"]
        speculation = configuration["speculation"]
        key = (
            configuration["source_context_rank"],
            speculation.get("id", speculation.get("spec_type")),
        )
        configurations[key] = configuration
        candidate_contexts.setdefault(key, {}).setdefault(
            configuration["context_size"], []
        ).append(item)

    aggregates: list[dict[str, Any]] = []
    for key, contexts in candidate_contexts.items():
        for context_size, items in contexts.items():
            prompts = [item["prompt"] for item in items]
            generations = [item["generation"] for item in items]
            walls = [item["wall"] for item in items]
            memories = [
                item["memory_free_mib"]
                for item in items
                if isinstance(item["memory_free_mib"], (int, float))
            ]
            acceptance_values = [
                item["acceptance_ratio"]
                for item in items
                if isinstance(item["acceptance_ratio"], (int, float))
            ]
            variation = 0.0
            for values in (prompts, generations, walls):
                mean = statistics.fmean(values)
                if len(values) > 1 and mean > 0:
                    variation += statistics.pstdev(values) / mean
            aggregates.append(
                {
                    "candidate_key": key,
                    "context_size": context_size,
                    "prompt_tokens_per_second": statistics.fmean(prompts),
                    "generation_tokens_per_second": statistics.fmean(
                        generations
                    ),
                    "wall_seconds": statistics.fmean(walls),
                    "memory_free_mib": (
                        statistics.fmean(memories) if memories else None
                    ),
                    "acceptance_ratio": (
                        statistics.fmean(acceptance_values)
                        if acceptance_values
                        else None
                    ),
                    "repetitions": len(items),
                    "stability_score": 1.0 / (1.0 + variation),
                }
            )

    by_context: dict[int, list[dict[str, Any]]] = {}
    for item in aggregates:
        by_context.setdefault(item["context_size"], []).append(item)
    for items in by_context.values():
        best_prompt = max(item["prompt_tokens_per_second"] for item in items)
        best_generation = max(
            item["generation_tokens_per_second"] for item in items
        )
        best_wall = min(item["wall_seconds"] for item in items)
        memory_values = [
            item["memory_free_mib"]
            for item in items
            if isinstance(item["memory_free_mib"], (int, float))
        ]
        best_memory = max(memory_values) if memory_values else None
        for item in items:
            memory_score = (
                item["memory_free_mib"] / best_memory
                if best_memory
                and isinstance(item["memory_free_mib"], (int, float))
                else 1.0
            )
            item["context_score"] = (
                weights["prompt"]
                * item["prompt_tokens_per_second"]
                / best_prompt
                + weights["generation"]
                * item["generation_tokens_per_second"]
                / best_generation
                + weights["wall"] * best_wall / item["wall_seconds"]
                + weights["stability"] * item["stability_score"]
                + weights["memory"] * memory_score
            )

    final_ranking = []
    for key, configuration in configurations.items():
        context_results = [
            item for item in aggregates if item["candidate_key"] == key
        ]
        if not context_results:
            continue
        minimum_context = min(
            item["context_size"] for item in context_results
        )
        context_weights = [
            (
                math.sqrt(item["context_size"] / minimum_context)
                if objective_definition["context_weighting"]
                == "sqrt-context"
                else 1.0
            )
            for item in context_results
        ]
        geometric_score = math.exp(
            sum(
                context_weight
                * math.log(max(item["context_score"], 1e-12))
                for item, context_weight in zip(
                    context_results, context_weights
                )
            )
            / sum(context_weights)
        )
        coverage = len(context_results) / max(1, expected_contexts)
        final_ranking.append(
            {
                "configuration": {
                    key: configuration[key]
                    for key in (
                        "batch_size",
                        "ubatch_size",
                        "flash_attention",
                        "threads",
                        "source_context_rank",
                        "speculation",
                    )
                },
                "final_score": round(geometric_score * coverage, 6),
                "worst_context_score": round(
                    min(item["context_score"] for item in context_results), 6
                ),
                "context_coverage": len(context_results),
                "total_contexts": expected_contexts,
                "optimization_objective": objective,
                "context_results": sorted(
                    context_results, key=lambda item: item["context_size"]
                ),
            }
        )
    return sorted(final_ranking, key=lambda item: item["final_score"], reverse=True)


def summarize_final_validation(
    cases: list[dict[str, Any]],
    total_cases: int,
    expected_contexts: int,
    objective: str = "balanced",
    planned_contexts: list[int] | None = None,
) -> dict[str, Any]:
    successful = sum(case.get("status") == "ok" for case in cases)
    ranking = rank_final_validation_cases(
        cases, expected_contexts, objective=objective
    )
    objective_winners = {}
    for objective_name in OPTIMIZATION_OBJECTIVES:
        objective_ranking = rank_final_validation_cases(
            cases, expected_contexts, objective=objective_name
        )
        if not objective_ranking:
            continue
        winner = objective_ranking[0]
        objective_winners[objective_name] = {
            "configuration": winner["configuration"],
            "final_score": winner["final_score"],
            "worst_context_score": winner["worst_context_score"],
        }
    if len(cases) < total_cases:
        status = "running"
    elif successful == total_cases and cases:
        status = "ok"
    elif successful:
        status = "partial"
    else:
        status = "failed"
    winner = ranking[0] if ranking else None
    coverage_complete = bool(
        winner
        and winner["context_coverage"] >= expected_contexts
        and expected_contexts > 0
    )
    if status == "ok" and not coverage_complete:
        status = "partial"
    validated_contexts = sorted(
        {
            item["context_size"]
            for item in (winner or {}).get("context_results", [])
        }
    )
    effective_planned_contexts = sorted(
        set(planned_contexts)
        if planned_contexts is not None
        else set(validated_contexts)
    )
    return {
        "name": "final-validation",
        "status": status,
        "successful_cases": successful,
        "completed_cases": len(cases),
        "total_cases": total_cases,
        "optimization_objective": objective,
        "objective_definition": OPTIMIZATION_OBJECTIVES[objective],
        "objective_winners": objective_winners,
        "cases": cases,
        "ranking": ranking,
        "winner": winner,
        "planned_contexts": effective_planned_contexts,
        "validated_contexts": validated_contexts,
        "missing_contexts": sorted(
            set(effective_planned_contexts) - set(validated_contexts)
        ),
        "coverage_complete": coverage_complete,
    }


def run_final_validation(
    llama_dir: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    experiment_plan: dict[str, Any],
    context_result: dict[str, Any],
    speculation_result: dict[str, Any],
    run_dir: Path,
    *,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    max_tokens: int,
    reasoning_effort: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    repetitions = experiment_plan["policy"]["final_repetitions"]
    objective = experiment_plan.get("optimization_objective", "balanced")
    candidates = final_validation_candidates(speculation_result)
    context_ranking = context_result.get("base_ranking", [])
    final_stage = next(
        item
        for item in experiment_plan["stages"]
        if item["id"] == "final-validation"
    )
    planned_contexts = sorted(
        set(
            context_result.get("planned_contexts")
            or final_stage.get("contexts", [])
        )
    )
    expected_contexts = len(planned_contexts)
    total_cases = sum(
        len(
            context_ranking[candidate.get("source_context_rank", 1) - 1][
                "best_profiles"
            ]
        )
        * repetitions
        for candidate in candidates
    )
    stage_dir = run_dir / "final_validation"
    stage_dir.mkdir(exist_ok=True)
    cases: list[dict[str, Any]] = []
    case_number = 0

    for candidate_number, candidate in enumerate(candidates, 1):
        source_rank = candidate.get("source_context_rank", 1)
        context_candidate = context_ranking[source_rank - 1]
        base = context_candidate["configuration"]
        speculation = candidate["configuration"]["speculation"]
        for profile in context_candidate["best_profiles"]:
            profile_configuration = profile["configuration"]
            for repetition in range(1, repetitions + 1):
                case_number += 1
                spec_id = speculation.get("id", speculation["spec_type"])
                case_id = (
                    f"{case_number:02d}_candidate{candidate_number}_"
                    f"ctx{profile_configuration['context_size']}_"
                    f"{spec_id}_r{repetition}"
                )
                case_dir = stage_dir / case_id
                case_dir.mkdir()
                configuration = {
                    **base,
                    "source_context_rank": source_rank,
                    "context_size": profile_configuration["context_size"],
                    "prompt_target": profile_configuration["prompt_target"],
                    "cache_type_k": profile_configuration["cache_type_k"],
                    "cache_type_v": profile_configuration["cache_type_v"],
                    "speculation": speculation,
                    "repetition": repetition,
                }
                print(
                    f"  Finalvalidierung {case_number}/{total_cases}: "
                    f"Variante={speculation_variant_label(speculation)}, "
                    f"CTX={configuration['context_size']}, R={repetition}",
                    flush=True,
                )
                try:
                    server_run = run_growing_chat(
                        llama_dir,
                        server,
                        model,
                        capabilities,
                        tuning_plan,
                        case_dir,
                        context_size=configuration["context_size"],
                        targets=[configuration["prompt_target"]],
                        cache_type=configuration["cache_type_k"],
                        startup_timeout=startup_timeout,
                        request_timeout=request_timeout,
                        shutdown_timeout=shutdown_timeout,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        batch_size=base["batch_size"],
                        ubatch_size=base["ubatch_size"],
                        flash_attention=base["flash_attention"],
                        threads=base["threads"],
                        alias=f"autotune-final-{case_number}",
                        speculation=speculation,
                    )
                    status = server_run["status"]
                    error = server_run.get("error")
                except (OSError, ValueError) as exc:
                    server_run = None
                    status = "configuration-error"
                    error = str(exc)
                cases.append(
                    {
                        "id": case_id,
                        "status": status,
                        "configuration": configuration,
                        "server_run": server_run,
                        "error": error,
                    }
                )
                result = summarize_final_validation(
                    cases,
                    total_cases,
                    expected_contexts,
                    objective=objective,
                    planned_contexts=planned_contexts,
                )
                if progress_callback:
                    progress_callback(result)
                server_stages = (
                    server_run.get("stages", []) if server_run else []
                )
                measurement = server_stages[0] if server_stages else {}
                print(
                    f"    Status={status}, "
                    f"Prompt={measurement.get('prompt_tokens_per_second', '?')}, "
                    f"Generation={measurement.get('generation_tokens_per_second', '?')}, "
                    f"Wall={measurement.get('wall_seconds', '?')}",
                    flush=True,
                )

    return summarize_final_validation(
        cases,
        total_cases,
        expected_contexts,
        objective=objective,
        planned_contexts=planned_contexts,
    )


def write_autotune_checkpoint(
    path: Path,
    state: dict[str, Any],
) -> None:
    """Atomically replace the execution checkpoint after every finished stage."""
    state["updated_at"] = utc_now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, state)
    temporary.replace(path)


def build_preliminary_recommendation(
    thread_result: dict[str, Any],
    context_result: dict[str, Any] | None = None,
    speculation_result: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if speculation_result and speculation_result.get("ranking"):
        winner = speculation_result["ranking"][0]
        source_rank = winner.get("source_context_rank", 1)
        context_ranking = (context_result or {}).get("base_ranking", [])
        selected_context_result = context_result
        if (
            isinstance(source_rank, int)
            and 1 <= source_rank <= len(context_ranking)
        ):
            selected_context_result = {
                **(context_result or {}),
                "base_ranking": [context_ranking[source_rank - 1]],
            }
        context_recommendation = build_preliminary_recommendation(
            thread_result,
            selected_context_result,
        )
        if context_recommendation is None:
            return None
        context_recommendation.update(
            {
                "scope": (
                    "llama-bench plus context/KV-cache and speculation screening"
                ),
                "speculation": winner["configuration"]["speculation"],
                "speculation_score": winner["relative_score"],
                "speculation_generation_tokens_per_second": winner[
                    "generation_tokens_per_second"
                ],
                "speculation_wall_seconds": winner["wall_seconds"],
                "generation_speedup_vs_none": winner[
                    "generation_speedup_vs_none"
                ],
                "wall_speedup_vs_none": winner["wall_speedup_vs_none"],
                "spec_draft_tokens": winner["spec_draft_tokens"],
                "spec_accepted_tokens": winner["spec_accepted_tokens"],
                "spec_acceptance_ratio": winner["spec_acceptance_ratio"],
                "not_yet_tested": ["wiederholte finale Validierung"],
                "warning": (
                    "Dies ist noch keine endgültige Startkommando-Empfehlung; "
                    "die wiederholte Finalvalidierung fehlt."
                ),
            }
        )
        return context_recommendation

    if context_result and context_result.get("base_ranking"):
        winner = context_result["base_ranking"][0]
        configuration = winner["configuration"]
        context_profiles = []
        for profile in winner["best_profiles"]:
            profile_configuration = profile["configuration"]
            context_profiles.append(
                {
                    "context_size": profile_configuration["context_size"],
                    "prompt_target": profile_configuration["prompt_target"],
                    "cache_type_k": profile_configuration["cache_type_k"],
                    "cache_type_v": profile_configuration["cache_type_v"],
                    "relative_score": profile["relative_score"],
                    "prompt_tokens_per_second": profile[
                        "prompt_tokens_per_second"
                    ],
                    "generation_tokens_per_second": profile[
                        "generation_tokens_per_second"
                    ],
                    "wall_seconds": profile["wall_seconds"],
                }
            )
        return {
            "status": "preliminary",
            "scope": (
                "llama-bench plus llama-server context/KV-cache screening"
            ),
            "configuration": configuration,
            "context_profiles": context_profiles,
            "context_aggregate_score": winner["aggregate_score"],
            "context_coverage": winner["context_coverage"],
            "total_contexts": winner["total_contexts"],
            "not_yet_tested": [
                "Spekulation/MTP",
                "wiederholte finale Validierung",
            ],
            "warning": (
                "Dies ist noch keine endgültige Startkommando-Empfehlung; "
                "Spekulation und wiederholte Finalvalidierung fehlen."
            ),
        }

    ranking = thread_result.get("ranking", [])
    if not ranking:
        return None
    winner = ranking[0]
    configuration = winner["configuration"]
    return {
        "status": "preliminary",
        "scope": "llama-bench batch/ubatch/flash-attention/threads",
        "configuration": {
            key: configuration[key]
            for key in (
                "batch_size",
                "ubatch_size",
                "flash_attention",
                "threads",
            )
        },
        "balanced_score": winner["balanced_score"],
        "prompt_tokens_per_second": winner["prompt_tokens_per_second"],
        "generation_tokens_per_second": winner[
            "generation_tokens_per_second"
        ],
        "not_yet_tested": [
            "wachsende Kontextgrößen",
            "KV-Cache-Typen",
            "Spekulation/MTP",
            "wiederholte finale Validierung",
        ],
        "warning": (
            "Dies ist keine endgültige Serverempfehlung; sie dient nur als "
            "Eingangskandidat für die folgenden Autotune-Stufen."
        ),
    }


def build_final_recommendation(
    final_result: dict[str, Any],
    context_result: dict[str, Any],
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
) -> dict[str, Any] | None:
    ranking = final_result.get("ranking", [])
    if not ranking:
        return None
    winner = ranking[0]
    configuration = winner["configuration"]
    source_rank = configuration["source_context_rank"]
    context_candidate = context_result["base_ranking"][source_rank - 1]
    measured_by_context = {
        item["context_size"]: item for item in winner["context_results"]
    }
    context_profiles = []
    for profile in context_candidate["best_profiles"]:
        profile_configuration = profile["configuration"]
        context_size = profile_configuration["context_size"]
        measured = measured_by_context.get(context_size, {})
        context_profiles.append(
            {
                "context_size": context_size,
                "prompt_target": profile_configuration["prompt_target"],
                "cache_type_k": profile_configuration["cache_type_k"],
                "cache_type_v": profile_configuration["cache_type_v"],
                "prompt_tokens_per_second": measured.get(
                    "prompt_tokens_per_second"
                ),
                "generation_tokens_per_second": measured.get(
                    "generation_tokens_per_second"
                ),
                "wall_seconds": measured.get("wall_seconds"),
                "stability_score": measured.get("stability_score"),
                "memory_free_mib": measured.get("memory_free_mib"),
                "repetitions": measured.get("repetitions"),
            }
        )
    highest_profile = max(
        context_profiles, key=lambda item: item["context_size"]
    )
    optimization_objective = final_result.get(
        "optimization_objective", "balanced"
    )
    native_context_limit = tuning_plan.get("native_context_limit")
    hermes_context_supported = (
        not isinstance(native_context_limit, int)
        or native_context_limit >= HERMES_DEPLOYMENT_CONTEXT
    )
    deployment_context = (
        HERMES_DEPLOYMENT_CONTEXT
        if optimization_objective == "hermes" and hermes_context_supported
        else highest_profile["context_size"]
    )
    deployment_profile = next(
        (
            profile
            for profile in context_profiles
            if profile["context_size"] == deployment_context
        ),
        highest_profile,
    )
    speculation = configuration["speculation"]
    options = capabilities["server_options"]
    deployment_host = str(
        tuning_plan.get("deployment_host", DEFAULT_DEPLOYMENT_HOST)
    )
    arguments = [
        "--model",
        str(model),
    ]
    if options.get("--host"):
        arguments.extend(["--host", deployment_host])
    arguments.extend([
        "--ctx-size",
        str(deployment_context),
        "--gpu-layers",
        (
            "all"
            if tuning_plan["fixed_parameters"]["gpu_layers"] == "all"
            else "auto"
        ),
        "--batch-size",
        str(configuration["batch_size"]),
        "--ubatch-size",
        str(configuration["ubatch_size"]),
        "--cache-type-k",
        str(deployment_profile["cache_type_k"]),
        "--cache-type-v",
        str(deployment_profile["cache_type_v"]),
        "--threads",
        str(configuration["threads"]),
        "--parallel",
        "1",
        "--spec-type",
        str(speculation["spec_type"]),
    ])
    if options.get("--flash-attn") and configuration["flash_attention"] != "auto":
        arguments.extend(
            ["--flash-attn", str(configuration["flash_attention"])]
        )
    if options.get("--threads-batch"):
        arguments.extend(["--threads-batch", str(configuration["threads"])])
    if options.get("--no-context-shift"):
        arguments.append("--no-context-shift")
    for key, option in (
        ("draft_n_max", "--spec-draft-n-max"),
        ("ngram_n_min", "--spec-ngram-mod-n-min"),
        ("ngram_n_max", "--spec-ngram-mod-n-max"),
        ("ngram_n_match", "--spec-ngram-mod-n-match"),
    ):
        if key in speculation:
            arguments.extend([option, str(speculation[key])])

    baseline = next(
        (
            item
            for item in ranking
            if item["configuration"]["source_context_rank"] == source_rank
            and item["configuration"]["speculation"].get("spec_type")
            == "none"
        ),
        None,
    )
    baseline_by_context = (
        {
            item["context_size"]: item
            for item in baseline["context_results"]
        }
        if baseline
        else {}
    )
    comparisons = []
    for item in winner["context_results"]:
        baseline_item = baseline_by_context.get(item["context_size"])
        comparisons.append(
            {
                "context_size": item["context_size"],
                "generation_speedup_vs_none": (
                    item["generation_tokens_per_second"]
                    / baseline_item["generation_tokens_per_second"]
                    if baseline_item
                    else None
                ),
                "wall_speedup_vs_none": (
                    baseline_item["wall_seconds"] / item["wall_seconds"]
                    if baseline_item
                    else None
                ),
            }
        )
    highest_measured = measured_by_context.get(
        highest_profile["context_size"], {}
    )
    highest_comparison = next(
        (
            item
            for item in comparisons
            if item["context_size"] == highest_profile["context_size"]
        ),
        {},
    )
    repetitions = min(
        (
            item.get("repetitions", 0)
            for item in winner["context_results"]
        ),
        default=0,
    )
    planned_contexts = sorted(
        set(final_result.get("planned_contexts") or measured_by_context)
    )
    validated_contexts = sorted(measured_by_context)
    missing_contexts = sorted(
        set(planned_contexts) - set(validated_contexts)
    )
    coverage_complete = bool(
        final_result.get(
            "coverage_complete",
            not missing_contexts
            and winner["context_coverage"] >= winner["total_contexts"],
        )
    )
    if not coverage_complete:
        confidence = "limited"
    elif final_result.get("status") == "ok" and repetitions >= 3:
        confidence = "high"
    elif final_result.get("status") == "ok" and repetitions >= 2:
        confidence = "medium"
    else:
        confidence = "limited"
    deployment_context_validated = deployment_context in measured_by_context
    warning_parts = [
        "Die Empfehlung gilt für die erkannte Hardware, das geprüfte Modell "
        "und die getesteten llama.cpp-Fähigkeiten."
    ]
    if missing_contexts:
        warning_parts.append(
            "Die Empfehlung ist abdeckungsbegrenzt; nicht erfolgreich "
            "validierte Kontextstufen: "
            + ", ".join(str(value) for value in missing_contexts)
            + "."
        )
    if optimization_objective == "hermes" and not deployment_context_validated:
        warning_parts.append(
            f"Der Hermes-Startbefehl konfiguriert {deployment_context} Tokens, "
            "obwohl diese Größe in diesem Lauf nicht erfolgreich validiert wurde."
        )
    if options.get("--host") and deployment_host not in LOOPBACK_DEPLOYMENT_HOSTS:
        exposure = (
            "allen Netzwerkschnittstellen"
            if deployment_host in {"0.0.0.0", "::"}
            else f"der Netzwerkadresse {deployment_host}"
        )
        warning_parts.append(
            f"Die Bind-Adresse {deployment_host} macht den Server auf "
            f"{exposure} erreichbar; Firewall und API-Schutz sind vom "
            "Betreiber festzulegen."
        )
    return {
        "status": "final",
        "confidence": confidence,
        "coverage_status": (
            "complete" if coverage_complete else "coverage-limited"
        ),
        "profile": "single-user-growing-chat",
        "optimization_objective": optimization_objective,
        "objective_definition": final_result.get(
            "objective_definition",
            OPTIMIZATION_OBJECTIVES[optimization_objective],
        ),
        "objective_winners": final_result.get("objective_winners", {}),
        "configuration": {
            key: configuration[key]
            for key in (
                "batch_size",
                "ubatch_size",
                "flash_attention",
                "threads",
            )
        },
        "context_profiles": context_profiles,
        "speculation": speculation,
        "speculation_generation_tokens_per_second": highest_measured.get(
            "generation_tokens_per_second"
        ),
        "speculation_wall_seconds": highest_measured.get("wall_seconds"),
        "spec_acceptance_ratio": highest_measured.get("acceptance_ratio"),
        "generation_speedup_vs_none": highest_comparison.get(
            "generation_speedup_vs_none"
        ),
        "wall_speedup_vs_none": highest_comparison.get(
            "wall_speedup_vs_none"
        ),
        "final_score": winner["final_score"],
        "worst_context_score": winner["worst_context_score"],
        "context_coverage": winner["context_coverage"],
        "total_contexts": winner["total_contexts"],
        "planned_contexts": planned_contexts,
        "validated_contexts": validated_contexts,
        "missing_contexts": missing_contexts,
        "coverage_complete": coverage_complete,
        "comparisons_vs_none": comparisons,
        "largest_validated_context": highest_profile["context_size"],
        "recommended_max_context": deployment_context,
        "deployment_context_validated": deployment_context_validated,
        "bind_host": deployment_host if options.get("--host") else None,
        "recommended_arguments": arguments,
        "command": [str(server), *arguments],
        "alternatives": ranking[1:3],
        "interpretation": (
            "Die Rangfolge wurde ausschließlich aus gemessenen Prompt-, "
            "Generierungs-, Wall-Clock-, Stabilitäts- und Speicherwerten "
            "berechnet."
        ),
        "warning": " ".join(warning_parts),
    }


def build_local_ai_analysis_input(
    recommendation: dict[str, Any],
    final_stage: dict[str, Any],
) -> dict[str, Any]:
    finalists = []
    for item in final_stage.get("ranking", [])[:5]:
        speculation = item["configuration"]["speculation"]
        finalists.append(
            {
                "variant": speculation_variant_label(speculation),
                "spec_type": speculation.get("spec_type"),
                "final_score": item.get("final_score"),
                "worst_context_score": item.get("worst_context_score"),
                "context_coverage": item.get("context_coverage"),
                "total_contexts": item.get("total_contexts"),
            }
        )
    objective_winners = {}
    for objective, item in recommendation.get(
        "objective_winners", {}
    ).items():
        speculation = item["configuration"]["speculation"]
        objective_winners[objective] = {
            "variant": speculation_variant_label(speculation),
            "final_score": item.get("final_score"),
            "worst_context_score": item.get("worst_context_score"),
        }
    repetitions = [
        profile.get("repetitions")
        for profile in recommendation.get("context_profiles", [])
        if isinstance(profile.get("repetitions"), int)
    ]
    minimum_repetitions = min(repetitions) if repetitions else None
    measurement_limits = [
        {
            "code": "parallelism-not-tested",
            "statement": (
                "Gemessen wurde parallel=1; über mehrere gleichzeitige "
                "Anfragen darf keine Eignungs- oder OOM-Aussage getroffen werden."
            ),
        },
        {
            "code": "memory-not-baseline-compared",
            "statement": (
                "Freier Speicher ist nur für die ausgewählte Konfiguration "
                "berichtet; ein Speichervergleich mit none ist nicht belegt."
            ),
        },
        {
            "code": "correlation-not-causation",
            "statement": (
                "Speedup-Werte beschreiben Messunterschiede, belegen aber "
                "keine technische Ursache."
            ),
        },
    ]
    if minimum_repetitions == 1:
        measurement_limits.append(
            {
                "code": "single-repetition",
                "statement": (
                    "Jeder Finalfall wurde einmal gemessen. Ein rechnerischer "
                    "Stabilitätswert von 1.0 ist nur ein Platzhalter und kein "
                    "Nachweis für geringe Varianz."
                ),
            }
        )
    if not recommendation.get("coverage_complete", True):
        measurement_limits.append(
            {
                "code": "context-coverage-limited",
                "statement": (
                    "Nicht alle geplanten Kontextstufen wurden erfolgreich "
                    "validiert. Fehlend: "
                    + ", ".join(
                        str(value)
                        for value in recommendation.get(
                            "missing_contexts", []
                        )
                    )
                    + "."
                ),
            }
        )
    return {
        "schema_version": 1,
        "immutable_deterministic_result": True,
        "optimization_objective": recommendation.get(
            "optimization_objective"
        ),
        "objective_definition": recommendation.get("objective_definition"),
        "confidence": recommendation.get("confidence"),
        "coverage_status": recommendation.get("coverage_status"),
        "planned_contexts": recommendation.get("planned_contexts", []),
        "validated_contexts": recommendation.get(
            "validated_contexts", []
        ),
        "missing_contexts": recommendation.get("missing_contexts", []),
        "selected_configuration": recommendation.get("configuration"),
        "selected_speculation": recommendation.get("speculation"),
        "final_score": recommendation.get("final_score"),
        "worst_context_score": recommendation.get("worst_context_score"),
        "context_profiles": recommendation.get("context_profiles", []),
        "comparisons_vs_none": recommendation.get(
            "comparisons_vs_none", []
        ),
        "objective_winners": objective_winners,
        "finalists": finalists,
        "recommended_command": recommendation.get("command"),
        "measurement_limits": measurement_limits,
        "minimum_final_repetitions": minimum_repetitions,
        "evidence_rule": (
            "Nur Vergleiche formulieren, für die beide Seiten im JSON "
            "vorliegen; nicht gemessene Eigenschaften ausdrücklich als "
            "nicht getestet kennzeichnen."
        ),
        "measurement_interpretation": recommendation.get("interpretation"),
        "scope_warning": recommendation.get("warning"),
    }


def run_local_ai_analysis(
    llama_dir: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    recommendation: dict[str, Any],
    final_stage: dict[str, Any],
    run_dir: Path,
    *,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    analysis_dir = run_dir / "local_ai_analysis"
    analysis_dir.mkdir(exist_ok=True)
    analysis_input = build_local_ai_analysis_input(
        recommendation, final_stage
    )
    input_path = analysis_dir / "analysis_input.json"
    request_path = analysis_dir / "analysis_request.json"
    response_path = analysis_dir / "analysis_response.json"
    report_path = analysis_dir / "local_ai_analysis.md"
    log_path = analysis_dir / "server.log"
    write_json(input_path, analysis_input)

    context_profiles = recommendation.get("context_profiles", [])
    if not context_profiles:
        raise ValueError("Keine validierten Kontextprofile für KI-Analyse")
    analysis_profile = min(
        context_profiles, key=lambda item: item["context_size"]
    )
    configuration = recommendation["configuration"]
    speculation = recommendation.get("speculation") or {
        "spec_type": "none"
    }
    port = find_free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    alias = "autotune-local-analyst"
    api_key = secrets.token_urlsafe(32)
    command = build_server_smoke_command(
        server,
        model,
        capabilities,
        tuning_plan,
        port=port,
        alias=alias,
        api_key=api_key,
        context_size=max(8192, analysis_profile["context_size"]),
        batch_size=configuration["batch_size"],
        ubatch_size=configuration["ubatch_size"],
        cache_type=analysis_profile["cache_type_k"],
        flash_attention=configuration["flash_attention"],
        threads=configuration["threads"],
        speculation=speculation,
    )
    payload = {
        "model": alias,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Du bist die erklärende Berichtsschicht eines lokalen "
                    "llama.cpp-Autotuners. Die deterministische Rangfolge im "
                    "JSON ist unveränderlich. Verwende ausschließlich die "
                    "gelieferten Messwerte, erfinde keine Zahlen oder "
                    "Parameter und ändere die Empfehlung nicht. Beachte jede "
                    "measurement_limits-Angabe wörtlich und kennzeichne nicht "
                    "getestete Eigenschaften als nicht getestet. Behaupte "
                    "keine technische Ursache nur aufgrund einer Korrelation. "
                    "Antworte vollständig in höchstens 450 Wörtern auf "
                    "Deutsch als kompaktes Markdown mit den Abschnitten "
                    "Kurzfazit, Zielkonflikte, Einsatzempfehlung, "
                    "Konfidenz und nächste sinnvolle Validierung."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Erkläre den folgenden abgeschlossenen Autotune-Lauf. "
                    "Weise ausdrücklich darauf hin, wenn andere "
                    "Optimierungsziele andere Gewinner haben. Wiederhole "
                    "keine vollständige Parameterliste und priorisiere eine "
                    "abgeschlossene Antwort vor zusätzlichen Details.\n\n"
                    + json.dumps(
                        analysis_input,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            },
        ],
        "temperature": 0,
        "seed": 12345,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning_effort": reasoning_effort,
    }
    write_json(request_path, payload)

    process: subprocess.Popen[Any] | None = None
    readiness: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    stop_result: dict[str, Any] | None = None
    error = None
    status = "failed"
    summary: dict[str, Any] = {}
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                env=subprocess_environment(llama_dir),
                start_new_session=True,
            )
            readiness = wait_for_server(
                process, base_url, api_key, timeout=startup_timeout
            )
            if not readiness["ready"]:
                status = readiness["reason"]
            else:
                for attempt_number in (1, 2):
                    attempt_payload = copy.deepcopy(payload)
                    if attempt_number == 2:
                        attempt_payload["max_tokens"] = min(
                            max(max_tokens * 2, 3072), 6144
                        )
                        attempt_payload["messages"][-1]["content"] = (
                            "Der vorige Entwurf wurde am Tokenlimit "
                            "abgeschnitten. Erstelle eine neue, vollständige "
                            "Fassung mit höchstens 350 Wörtern und beachte "
                            "alle measurement_limits.\n\n"
                            + attempt_payload["messages"][-1]["content"]
                        )
                    attempt_request_path = analysis_dir / (
                        f"analysis_request_{attempt_number}.json"
                    )
                    attempt_response_path = analysis_dir / (
                        f"analysis_response_{attempt_number}.json"
                    )
                    write_json(attempt_request_path, attempt_payload)
                    response = http_json_request(
                        f"{base_url}/v1/chat/completions",
                        api_key,
                        timeout=request_timeout,
                        payload=attempt_payload,
                    )
                    write_json(attempt_response_path, response)
                    write_json(response_path, response)
                    summary = summarize_chat_response(response)
                    attempts.append(
                        {
                            "attempt": attempt_number,
                            "max_tokens": attempt_payload["max_tokens"],
                            "status_code": summary.get("status_code"),
                            "finish_reason": summary.get("finish_reason"),
                            "content_characters": summary.get(
                                "content_characters", 0
                            ),
                            "reasoning_characters": summary.get(
                                "reasoning_characters", 0
                            ),
                            "request_file": str(attempt_request_path),
                            "response_file": str(attempt_response_path),
                        }
                    )
                    content = summary.get("content", "")
                    if summary.get("status_code") != 200:
                        status = "request-failed"
                    elif not isinstance(content, str) or not content.strip():
                        status = "no-final-content"
                    elif summary.get("finish_reason") == "stop":
                        status = "ok"
                    else:
                        status = "partial"
                    if status == "ok":
                        break
                    if summary.get("finish_reason") != "length":
                        break
                content = summary.get("content", "")
                if isinstance(content, str) and content.strip():
                    report_path.write_text(
                        content.strip() + "\n", encoding="utf-8"
                    )
        except (OSError, ValueError) as exc:
            error = str(exc)
            status = "failed"
        finally:
            if process is not None:
                stop_result = stop_process_group(
                    process, timeout=shutdown_timeout
                )

    return {
        "name": "local-ai-analysis",
        "status": status,
        "immutable_deterministic_result": True,
        "analysis": summary.get("content", ""),
        "finish_reason": summary.get("finish_reason"),
        "content_characters": summary.get("content_characters", 0),
        "reasoning_characters": summary.get("reasoning_characters", 0),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "command": redact_secret(command, api_key),
        "input_file": str(input_path),
        "request_file": str(request_path),
        "response_file": str(response_path),
        "analysis_file": str(report_path) if report_path.is_file() else None,
        "log_file": str(log_path),
        "readiness": readiness,
        "process_stop": stop_result,
        "error": error,
    }


def render_autotune_execution_markdown(
    execution: dict[str, Any],
) -> str:
    lines = [
        "# Llama Autotune – Ausführungsbericht",
        "",
        f"Profil: `{execution['profile']}`  ",
        f"Optimierungsziel: `{execution.get('optimization_objective', 'balanced')}`  ",
        f"Status: `{execution['status']}`  ",
        (
            "Implementierte Reichweite: vollständiger adaptiver Lauf mit "
            "Finalvalidierung"
        ),
        "",
        "## Stufen",
        "",
        "| Stufe | Status | Erfolgreich | Gesamt |",
        "|---|---|---:|---:|",
    ]
    for stage_id in (
        "smoke",
        "batch-screening",
        "thread-screening",
        "context-cache-screening",
        "speculation-screening",
        "final-validation",
    ):
        stage = execution.get("stages", {}).get(stage_id)
        if not stage:
            lines.append(f"| `{stage_id}` | übersprungen | 0 | 0 |")
            continue
        if stage_id == "smoke":
            successful = int(stage.get("status") == "ok")
            total = 1
        else:
            successful = stage.get("successful_cases", 0)
            total = stage.get("total_cases", 0)
        lines.append(
            f"| `{stage_id}` | {stage.get('status')} | "
            f"{successful} | {total} |"
        )

    recommendation = execution.get("recommendation") or execution.get(
        "preliminary_recommendation"
    )
    heading = (
        "## Endgültige deterministische Empfehlung"
        if recommendation and recommendation.get("status") == "final"
        else "## Vorläufiger Eingangskandidat"
    )
    lines.extend(["", heading, ""])
    if recommendation:
        configuration = recommendation["configuration"]
        lines.extend(
            [
                f"- Batch: `{configuration['batch_size']}`",
                f"- UBatch: `{configuration['ubatch_size']}`",
                f"- Flash Attention: `{configuration['flash_attention']}`",
                f"- Threads: `{configuration['threads']}`",
            ]
        )
        if "final_score" in recommendation:
            objective = recommendation.get(
                "optimization_objective", "balanced"
            )
            objective_definition = recommendation.get(
                "objective_definition", OPTIMIZATION_OBJECTIVES[objective]
            )
            lines.extend(
                [
                    f"- Bewertungsziel: `{objective}` – "
                    f"{objective_definition['description']}",
                    f"- Finaler Score: `{recommendation['final_score']:.6f}`",
                    f"- Konfidenz: `{recommendation['confidence']}`",
                    "- Kontextabdeckung: "
                    f"`{recommendation.get('context_coverage', 0)}/"
                    f"{recommendation.get('total_contexts', 0)}` "
                    f"(`{recommendation.get('coverage_status', '-')}`)",
                    "- Schwächster Kontextscore: "
                    f"`{recommendation['worst_context_score']:.6f}`",
                ]
            )
            if recommendation.get("missing_contexts"):
                lines.append(
                    "- Nicht erfolgreich validiert: `"
                    + ", ".join(
                        str(value)
                        for value in recommendation["missing_contexts"]
                    )
                    + "`"
                )
            if objective_definition.get("workload_assumption"):
                lines.append(
                    "- Workload-Annahme: "
                    + objective_definition["workload_assumption"]
                )
            if objective_definition.get("not_measured"):
                lines.append(
                    "- Nicht gemessen: "
                    + objective_definition["not_measured"]
                )
        elif "context_aggregate_score" in recommendation:
            lines.append(
                "- Kontext-Gesamtscore: "
                f"`{recommendation['context_aggregate_score']:.6f}`"
            )
        else:
            lines.append(
                f"- Balancierter Score: `{recommendation['balanced_score']:.6f}`"
            )
        if recommendation.get("context_profiles"):
            lines.extend(
                [
                    "",
                    "### Beste gemessene KV-Profile",
                    "",
                    "| Kontext | Cache K/V | Prompt/s | Generation/s | Wall |",
                    "|---:|---|---:|---:|---:|",
                ]
            )
            for profile in recommendation["context_profiles"]:
                prompt_value = profile.get("prompt_tokens_per_second")
                generation_value = profile.get(
                    "generation_tokens_per_second"
                )
                wall_value = profile.get("wall_seconds")
                prompt_text = (
                    f"{prompt_value:.3f}"
                    if isinstance(prompt_value, (int, float))
                    else "-"
                )
                generation_text = (
                    f"{generation_value:.3f}"
                    if isinstance(generation_value, (int, float))
                    else "-"
                )
                wall_text = (
                    f"{wall_value:.3f} s"
                    if isinstance(wall_value, (int, float))
                    else "-"
                )
                lines.append(
                    f"| {profile['context_size']} | "
                    f"`{profile['cache_type_k']}` | "
                    f"{prompt_text} | {generation_text} | {wall_text} |"
                )
        if "speculation" in recommendation:
            speculation = recommendation["speculation"]
            acceptance = recommendation.get("spec_acceptance_ratio")
            acceptance_text = (
                f"{acceptance * 100:.1f}%"
                if isinstance(acceptance, (int, float))
                else "nicht verfügbar"
            )
            speedup = recommendation.get("generation_speedup_vs_none")
            speedup_text = (
                f"{speedup:.3f}x"
                if isinstance(speedup, (int, float))
                else "nicht verfügbar"
            )
            lines.extend(
                [
                    "",
                    "### Beste Spekulationsvariante",
                    "",
                    f"- Typ: `{speculation['spec_type']}`",
                    f"- Variante: `{speculation_variant_label(speculation)}`",
                    "- Generierung: "
                    f"`{recommendation['speculation_generation_tokens_per_second']:.3f}` Tokens/s",
                    "- Speedup gegenüber `none`: "
                    f"`{speedup_text}`",
                    f"- Acceptance-Rate: `{acceptance_text}`",
                ]
            )
            speculation_stage = execution.get("stages", {}).get(
                "speculation-screening", {}
            )
            lines.extend(
                [
                    "",
                    "| Rang | Variante | Generation/s | Speedup | Acceptance | Score |",
                    "|---:|---|---:|---:|---:|---:|",
                ]
            )
            for rank, item in enumerate(
                speculation_stage.get("ranking", [])[:10], 1
            ):
                item_acceptance = item.get("spec_acceptance_ratio")
                item_acceptance_text = (
                    f"{item_acceptance * 100:.1f}%"
                    if isinstance(item_acceptance, (int, float))
                    else "-"
                )
                item_speedup = item.get("generation_speedup_vs_none")
                item_speedup_text = (
                    f"{item_speedup:.3f}x"
                    if isinstance(item_speedup, (int, float))
                    else "-"
                )
                lines.append(
                    f"| {rank} | "
                    f"`{speculation_variant_label(item['configuration']['speculation'])}` | "
                    f"{item['generation_tokens_per_second']:.3f} | "
                    f"{item_speedup_text} | "
                    f"{item_acceptance_text} | {item['relative_score']:.6f} |"
                )
            comparisons = recommendation.get("comparisons_vs_none", [])
            if comparisons:
                lines.extend(
                    [
                        "",
                        "### Finalvergleich mit identischer `none`-Baseline",
                        "",
                        "| Kontext | Generation-Speedup | Wall-Clock-Speedup |",
                        "|---:|---:|---:|",
                    ]
                )
                for comparison in comparisons:
                    generation_speedup = comparison.get(
                        "generation_speedup_vs_none"
                    )
                    wall_speedup = comparison.get("wall_speedup_vs_none")
                    generation_speedup_text = (
                        f"{generation_speedup:.3f}x"
                        if isinstance(generation_speedup, (int, float))
                        else "-"
                    )
                    wall_speedup_text = (
                        f"{wall_speedup:.3f}x"
                        if isinstance(wall_speedup, (int, float))
                        else "-"
                    )
                    lines.append(
                        f"| {comparison['context_size']} | "
                        f"{generation_speedup_text} | "
                        f"{wall_speedup_text} |"
                    )
                lines.extend(
                    [
                        "",
                        "Bei beiden Speedups bedeutet ein Wert über `1.0x`, "
                        "dass die empfohlene Variante schneller als `none` war.",
                    ]
                )
            objective_winners = recommendation.get("objective_winners", {})
            if objective_winners:
                lines.extend(
                    [
                        "",
                        "### Gewinner bei alternativen Bewertungszielen",
                        "",
                        "| Bewertungsziel | Variante | Finalscore | Schwächster Kontext |",
                        "|---|---|---:|---:|",
                    ]
                )
                for objective_name, winner in objective_winners.items():
                    winner_speculation = winner["configuration"]["speculation"]
                    lines.append(
                        f"| `{objective_name}` | "
                        f"`{speculation_variant_label(winner_speculation)}` | "
                        f"{winner['final_score']:.6f} | "
                        f"{winner['worst_context_score']:.6f} |"
                    )
        if recommendation.get("status") == "final":
            command_heading = (
                "### Empfohlene Argumente für Hermes mit 131k Kontext"
                if recommendation.get("optimization_objective") == "hermes"
                else "### Empfohlene Argumente für den größten validierten Kontext"
            )
            lines.extend(
                [
                    "",
                    command_heading,
                    "",
                    "```text",
                    " ".join(recommendation["recommended_arguments"]),
                    "```",
                ]
            )
        lines.extend(["", recommendation["warning"]])
    else:
        lines.append("Kein Kandidat erfüllte die Erfolgskriterien.")
    if not recommendation or recommendation.get("status") != "final":
        lines.extend(
            [
                "",
                "## Noch ausstehende Stufen",
                "",
                "- wiederholte Finalvalidierung und endgültige Empfehlung",
                "",
            ]
        )
    local_ai = execution.get("local_ai_analysis")
    if local_ai:
        lines.extend(
            [
                "",
                "## Optionale Erläuterung durch die lokale KI",
                "",
                f"Status: `{local_ai.get('status')}`  ",
                f"Versuche: `{local_ai.get('attempt_count', 0)}`  ",
                (
                    "Diese Erläuterung ist nicht Teil des Rankings und kann "
                    "weder Messwerte noch die deterministische Empfehlung "
                    "verändern."
                ),
                "",
            ]
        )
        analysis = local_ai.get("analysis")
        if isinstance(analysis, str) and analysis.strip():
            lines.append(analysis.strip())
        else:
            lines.append(
                "Die lokale KI hat keine verwendbare finale Erläuterung geliefert."
            )
    return "\n".join(lines)


def run_autotune_foundation(
    llama_dir: Path,
    bench: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    experiment_plan: dict[str, Any],
    run_dir: Path,
    *,
    benchmark_timeout: float,
    smoke_repetitions: int,
    screening_repetitions: int,
    prompt_tokens: int,
    generation_tokens: int,
    server_start_timeout: float,
    server_request_timeout: float,
    server_shutdown_timeout: float,
    server_max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Execute adaptive benchmark, context/KV and speculation screening."""
    checkpoint_path = run_dir / "autotune_state.json"
    report_path = run_dir / "autotune_report.md"
    recommendation_path = run_dir / "recommendation.json"
    execution: dict[str, Any] = {
        "schema_version": 4,
        "status": "running",
        "profile": experiment_plan["profile"],
        "optimization_objective": experiment_plan.get(
            "optimization_objective", "balanced"
        ),
        "objective_definition": experiment_plan.get(
            "objective_definition", OPTIMIZATION_OBJECTIVES["balanced"]
        ),
        "started_at": utc_now(),
        "implemented_through": "final-validation",
        "completed_stages": [],
        "stages": {},
        "preliminary_recommendation": None,
        "recommendation": None,
    }
    write_autotune_checkpoint(checkpoint_path, execution)

    print("Llama Autotune – Autonomer Smoke-Test wird gestartet ...")
    try:
        smoke = run_smoke_benchmark(
            llama_dir,
            bench,
            model,
            capabilities,
            tuning_plan,
            run_dir,
            timeout=benchmark_timeout,
            repetitions=smoke_repetitions,
        )
    except (OSError, ValueError) as exc:
        smoke = {
            "name": "smoke",
            "status": "configuration-error",
            "error": str(exc),
        }
    execution["stages"]["smoke"] = smoke
    execution["completed_stages"].append("smoke")
    write_autotune_checkpoint(checkpoint_path, execution)

    if smoke["status"] != "ok":
        execution["status"] = "blocked"
        execution["stop_reason"] = "Smoke-Test war nicht erfolgreich"
    else:
        print("Llama Autotune – Autonomes Batch-Screening wird gestartet ...")
        try:
            batch = run_batch_screening(
                llama_dir,
                bench,
                model,
                capabilities,
                tuning_plan,
                run_dir,
                timeout=benchmark_timeout,
                repetitions=screening_repetitions,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
            )
        except (OSError, ValueError) as exc:
            batch = {
                "name": "batch-screening",
                "status": "configuration-error",
                "error": str(exc),
                "ranking": [],
            }
        execution["stages"]["batch-screening"] = batch
        execution["completed_stages"].append("batch-screening")
        write_autotune_checkpoint(checkpoint_path, execution)

        if not batch.get("ranking"):
            execution["status"] = "blocked"
            execution["stop_reason"] = "Kein erfolgreicher Batch-Kandidat"
        else:
            print("Llama Autotune – Thread-Screening wird gestartet ...")
            thread = run_thread_screening(
                llama_dir,
                bench,
                model,
                capabilities,
                tuning_plan,
                experiment_plan,
                batch,
                run_dir,
                timeout=benchmark_timeout,
                repetitions=screening_repetitions,
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
            )
            execution["stages"]["thread-screening"] = thread
            execution["completed_stages"].append("thread-screening")
            if not thread.get("ranking"):
                execution["status"] = "blocked"
                execution["stop_reason"] = "Kein erfolgreicher Thread-Kandidat"
            else:
                print(
                    "Llama Autotune – Kontext-/KV-Cache-Screening wird gestartet ..."
                )

                def save_context_progress(result: dict[str, Any]) -> None:
                    execution["stages"]["context-cache-screening"] = result
                    write_autotune_checkpoint(checkpoint_path, execution)

                try:
                    context = run_context_cache_screening(
                        llama_dir,
                        server,
                        model,
                        capabilities,
                        tuning_plan,
                        experiment_plan,
                        thread,
                        run_dir,
                        startup_timeout=server_start_timeout,
                        request_timeout=server_request_timeout,
                        shutdown_timeout=server_shutdown_timeout,
                        max_tokens=server_max_tokens,
                        reasoning_effort=reasoning_effort,
                        progress_callback=save_context_progress,
                    )
                except (OSError, ValueError) as exc:
                    context = {
                        "name": "context-cache-screening",
                        "status": "configuration-error",
                        "error": str(exc),
                        "base_ranking": [],
                    }
                execution["stages"]["context-cache-screening"] = context
                execution["completed_stages"].append(
                    "context-cache-screening"
                )
                if not context.get("base_ranking"):
                    execution["status"] = "blocked"
                    execution["stop_reason"] = (
                        "Kein erfolgreicher Kontext-/KV-Cache-Kandidat"
                    )
                else:
                    print(
                        "Llama Autotune – Spekulations-Screening wird gestartet ..."
                    )

                    def save_speculation_progress(
                        result: dict[str, Any],
                    ) -> None:
                        execution["stages"]["speculation-screening"] = result
                        write_autotune_checkpoint(checkpoint_path, execution)

                    try:
                        speculation = run_speculation_screening(
                            llama_dir,
                            server,
                            model,
                            capabilities,
                            tuning_plan,
                            experiment_plan,
                            context,
                            run_dir,
                            startup_timeout=server_start_timeout,
                            request_timeout=server_request_timeout,
                            shutdown_timeout=server_shutdown_timeout,
                            max_tokens=server_max_tokens,
                            reasoning_effort=reasoning_effort,
                            progress_callback=save_speculation_progress,
                        )
                    except (OSError, ValueError) as exc:
                        speculation = {
                            "name": "speculation-screening",
                            "status": "configuration-error",
                            "error": str(exc),
                            "ranking": [],
                        }
                    execution["stages"]["speculation-screening"] = speculation
                    execution["completed_stages"].append(
                        "speculation-screening"
                    )
                    execution["preliminary_recommendation"] = (
                        build_preliminary_recommendation(
                            thread,
                            context,
                            speculation,
                        )
                    )
                    if speculation.get("ranking"):
                        print(
                            "Llama Autotune – Finale Kontrollläufe werden gestartet ..."
                        )

                        def save_final_progress(
                            result: dict[str, Any],
                        ) -> None:
                            execution["stages"]["final-validation"] = result
                            write_autotune_checkpoint(checkpoint_path, execution)

                        try:
                            final_validation = run_final_validation(
                                llama_dir,
                                server,
                                model,
                                capabilities,
                                tuning_plan,
                                experiment_plan,
                                context,
                                speculation,
                                run_dir,
                                startup_timeout=server_start_timeout,
                                request_timeout=server_request_timeout,
                                shutdown_timeout=server_shutdown_timeout,
                                max_tokens=server_max_tokens,
                                reasoning_effort=reasoning_effort,
                                progress_callback=save_final_progress,
                            )
                        except (OSError, ValueError, IndexError) as exc:
                            final_validation = {
                                "name": "final-validation",
                                "status": "configuration-error",
                                "error": str(exc),
                                "ranking": [],
                            }
                        execution["stages"]["final-validation"] = (
                            final_validation
                        )
                        execution["completed_stages"].append(
                            "final-validation"
                        )
                        execution["recommendation"] = (
                            build_final_recommendation(
                                final_validation,
                                context,
                                server,
                                model,
                                capabilities,
                                tuning_plan,
                            )
                        )
                        if execution["recommendation"]:
                            execution["status"] = (
                                "final-validation-complete"
                                if final_validation.get("status") == "ok"
                                else "final-validation-partial"
                            )
                        else:
                            execution["status"] = "blocked"
                            execution["stop_reason"] = (
                                "Kein erfolgreicher Finalkandidat"
                            )
                    else:
                        execution["status"] = "blocked"
                        execution["stop_reason"] = (
                            "Kein erfolgreicher Spekulationskandidat"
                        )

    execution["finished_at"] = utc_now()
    write_autotune_checkpoint(checkpoint_path, execution)
    recommendation = execution.get("recommendation") or execution.get(
        "preliminary_recommendation"
    )
    write_json(
        recommendation_path,
        recommendation
        or {
            "status": "unavailable",
            "reason": execution.get("stop_reason", "unbekannt"),
        },
    )
    report_path.write_text(
        render_autotune_execution_markdown(execution),
        encoding="utf-8",
    )
    execution["checkpoint_file"] = str(checkpoint_path)
    execution["recommendation_file"] = str(recommendation_path)
    execution["report_file"] = str(report_path)
    return execution


def collect_llama(
    llama_dir: Path,
    server: Path,
    bench: Path | None,
    timeout: float,
) -> dict[str, Any]:
    env = subprocess_environment(llama_dir)
    information: dict[str, Any] = {
        "directory": str(llama_dir.resolve()),
        "server": str(server),
        "bench": str(bench) if bench else None,
        "server_version": run_command(
            [str(server), "--version"], timeout=timeout, env=env
        ),
        "server_help": run_command(
            [str(server), "--help"], timeout=timeout, env=env
        ),
        "devices": run_command(
            [str(server), "--list-devices"], timeout=timeout, env=env
        ),
    }
    if bench:
        information["bench_help"] = run_command(
            [str(bench), "--help"], timeout=timeout, env=env
        )
    return information


def reanalyze_existing_run(
    source_run_dir: Path,
    llama_dir: Path,
    server: Path,
    model: Path,
    capabilities: dict[str, Any],
    tuning_plan: dict[str, Any],
    *,
    optimization_objective: str,
    startup_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    source_run_dir = source_run_dir.expanduser().resolve()
    checkpoint_path = source_run_dir / "autotune_state.json"
    state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    source_final = state.get("stages", {}).get("final-validation", {})
    cases = source_final.get("cases", [])
    if not cases:
        raise ValueError("Der ausgewählte Lauf enthält keine Finalmessungen")
    contexts = {
        case["configuration"]["context_size"]
        for case in cases
        if case.get("status") == "ok"
    }
    if not contexts:
        raise ValueError("Der ausgewählte Lauf enthält keine gültigen Kontexte")
    final_stage = summarize_final_validation(
        cases,
        source_final.get("total_cases", len(cases)),
        len(contexts),
        objective=optimization_objective,
    )
    context_stage = state.get("stages", {}).get(
        "context-cache-screening", {}
    )
    recommendation = build_final_recommendation(
        final_stage,
        context_stage,
        server,
        model,
        capabilities,
        tuning_plan,
    )
    if recommendation is None:
        raise ValueError("Aus den Finalmessungen ließ sich keine Empfehlung bilden")

    output_dir = make_run_directory(
        source_run_dir, f"reanalysis_{optimization_objective}"
    )
    execution = copy.deepcopy(state)
    execution["optimization_objective"] = optimization_objective
    execution["objective_definition"] = OPTIMIZATION_OBJECTIVES[
        optimization_objective
    ]
    execution["stages"]["final-validation"] = final_stage
    execution["recommendation"] = recommendation
    execution["reanalysis_source"] = str(source_run_dir)
    execution["status"] = (
        "final-validation-complete"
        if final_stage["status"] == "ok"
        else "final-validation-partial"
    )
    local_ai = run_local_ai_analysis(
        llama_dir,
        server,
        model,
        capabilities,
        tuning_plan,
        recommendation,
        final_stage,
        output_dir,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
        shutdown_timeout=shutdown_timeout,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    execution["local_ai_analysis"] = local_ai
    execution_path = output_dir / "reanalysis_state.json"
    recommendation_path = output_dir / "recommendation.json"
    report_path = output_dir / "autotune_report.md"
    execution["checkpoint_file"] = str(execution_path)
    execution["recommendation_file"] = str(recommendation_path)
    execution["report_file"] = str(report_path)
    write_json(execution_path, execution)
    write_json(recommendation_path, recommendation)
    report_path.write_text(
        render_autotune_execution_markdown(execution), encoding="utf-8"
    )
    return {
        "status": local_ai["status"],
        "source_run": str(source_run_dir),
        "output_dir": str(output_dir),
        "optimization_objective": optimization_objective,
        "recommendation": recommendation,
        "local_ai_analysis": local_ai,
        "state_file": str(execution_path),
        "recommendation_file": str(recommendation_path),
        "report_file": str(report_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Untersucht Hardware, llama.cpp-Binaries und ein lokales GGUF-Modell. "
            "Erstellt einen Testplan und kann Benchmarks, einen isolierten "
            "Server-Smoke-Test oder einen wachsenden Chat ausführen."
        )
    )
    parser.add_argument(
        "--llama-dir",
        type=Path,
        required=True,
        help="Verzeichnis, das llama-server und möglichst llama-bench enthält",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Pfad zur lokalen GGUF-Modelldatei",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Basisverzeichnis für Laufdaten (Standard: ./runs)",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=30.0,
        help="Timeout für einzelne Diagnosebefehle in Sekunden",
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="führt einen kurzen, protokollierten llama-bench-Test aus",
    )
    parser.add_argument(
        "--run-screening",
        action="store_true",
        help="führt Smoke-Test und Batch-/UBatch-Screening aus",
    )
    parser.add_argument(
        "--run-server-smoke",
        action="store_true",
        help=(
            "startet llama-server lokal, sendet zwei Chat-Requests und beendet ihn"
        ),
    )
    parser.add_argument(
        "--run-growing-chat",
        action="store_true",
        help="misst einen deterministisch wachsenden Chat in einer Serverinstanz",
    )
    parser.add_argument(
        "--plan-autotune",
        action="store_true",
        help="erstellt den adaptiven vollständigen Versuchsplan ohne Rechenläufe",
    )
    parser.add_argument(
        "--run-autotune",
        action="store_true",
        help=(
            "führt den vollständigen adaptiven Executor einschließlich "
            "Finalvalidierung und deterministischer Empfehlung aus"
        ),
    )
    parser.add_argument(
        "--autotune-profile",
        choices=tuple(AUTOTUNE_PROFILES),
        default="balanced",
        help="Umfang der adaptiven Suche (Standard: balanced)",
    )
    parser.add_argument(
        "--optimization-objective",
        choices=tuple(OPTIMIZATION_OBJECTIVES),
        default="balanced",
        help=(
            "Bewertungsziel für die finale Empfehlung "
            "(Standard: balanced)"
        ),
    )
    parser.add_argument(
        "--deployment-host",
        default=DEFAULT_DEPLOYMENT_HOST,
        help=(
            "Bind-Adresse im empfohlenen llama-server-Befehl; 0.0.0.0 macht "
            "den Server im Netzwerk erreichbar (Standard: 127.0.0.1)"
        ),
    )
    parser.add_argument(
        "--local-ai-analysis",
        action="store_true",
        help=(
            "lässt das lokale Modell die unveränderliche finale Empfehlung "
            "nach dem Autotune-Lauf erklären"
        ),
    )
    parser.add_argument(
        "--analyze-existing-run",
        type=Path,
        help=(
            "bewertet einen vorhandenen Autotune-Lauf neu und lässt ihn "
            "ohne erneute Benchmarks lokal erklären"
        ),
    )
    parser.add_argument(
        "--local-ai-max-tokens",
        type=int,
        default=3072,
        help="Maximale Ausgabetokens der lokalen KI-Erklärung (Standard: 3072)",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=float,
        default=600.0,
        help="Timeout für den Smoke-Benchmark in Sekunden (Standard: 600)",
    )
    parser.add_argument(
        "--smoke-repetitions",
        type=int,
        default=2,
        help="Wiederholungen je Smoke-Test (Standard: 2)",
    )
    parser.add_argument(
        "--screening-repetitions",
        type=int,
        default=2,
        help="Wiederholungen je Screening-Kandidat (Standard: 2)",
    )
    parser.add_argument(
        "--screening-prompt-tokens",
        type=int,
        default=4096,
        help="Prompt-Tokens im Batch-Screening (Standard: 4096)",
    )
    parser.add_argument(
        "--screening-generation-tokens",
        type=int,
        default=64,
        help="Generierungstokens im Batch-Screening (Standard: 64)",
    )
    parser.add_argument(
        "--server-start-timeout",
        type=float,
        default=180.0,
        help="Timeout für den Serverstart in Sekunden (Standard: 180)",
    )
    parser.add_argument(
        "--server-request-timeout",
        type=float,
        default=180.0,
        help="Timeout je Chat-Request in Sekunden (Standard: 180)",
    )
    parser.add_argument(
        "--server-shutdown-timeout",
        type=float,
        default=15.0,
        help="Wartezeit beim kontrollierten Serverstopp (Standard: 15)",
    )
    parser.add_argument(
        "--server-max-tokens",
        type=int,
        default=256,
        help="Maximale Ausgabetokens je Server-Request (Standard: 256)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
        help="Reasoning-Aufwand für Server-Tests (Standard: low)",
    )
    parser.add_argument(
        "--growing-context-size",
        type=int,
        default=131072,
        help="Kontextfenster des Growing-Chat-Servers (Standard: 131072)",
    )
    parser.add_argument(
        "--growing-targets",
        default=",".join(str(target) for target in DEFAULT_GROWING_TARGETS),
        help=(
            "aufsteigende Prompt-Ziele, kommagetrennt "
            "(Standard: 8192,32768,65536,126000)"
        ),
    )
    parser.add_argument(
        "--growing-cache-type",
        choices=("f16", "q8_0", "q4_0"),
        default="q4_0",
        help="KV-Cache-Typ für den Growing Chat (Standard: q4_0)",
    )
    parser.add_argument(
        "--growing-max-tokens",
        type=int,
        default=256,
        help="maximale Ausgabetokens je Kontextstufe (Standard: 256)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PROGRAM_VERSION}",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    llama_dir = args.llama_dir.expanduser().resolve()
    model = args.model.expanduser().absolute()

    if not llama_dir.is_dir():
        raise ValueError(f"llama.cpp-Verzeichnis nicht gefunden: {llama_dir}")
    if not model.is_file():
        raise ValueError(f"Modelldatei nicht gefunden: {model}")
    if model.suffix.lower() != ".gguf":
        raise ValueError(f"Das Modell ist keine GGUF-Datei: {model}")
    if args.command_timeout <= 0:
        raise ValueError("--command-timeout muss größer als 0 sein")
    if not args.deployment_host.strip() or any(
        character.isspace() for character in args.deployment_host
    ):
        raise ValueError("--deployment-host muss eine einzelne Adresse sein")
    if args.benchmark_timeout <= 0:
        raise ValueError("--benchmark-timeout muss größer als 0 sein")
    if args.smoke_repetitions <= 0:
        raise ValueError("--smoke-repetitions muss größer als 0 sein")
    if args.screening_repetitions <= 0:
        raise ValueError("--screening-repetitions muss größer als 0 sein")
    if args.screening_prompt_tokens <= 0:
        raise ValueError("--screening-prompt-tokens muss größer als 0 sein")
    if args.screening_generation_tokens <= 0:
        raise ValueError("--screening-generation-tokens muss größer als 0 sein")
    if args.server_start_timeout <= 0:
        raise ValueError("--server-start-timeout muss größer als 0 sein")
    if args.server_request_timeout <= 0:
        raise ValueError("--server-request-timeout muss größer als 0 sein")
    if args.server_shutdown_timeout <= 0:
        raise ValueError("--server-shutdown-timeout muss größer als 0 sein")
    if args.server_max_tokens <= 0:
        raise ValueError("--server-max-tokens muss größer als 0 sein")
    if args.local_ai_max_tokens <= 0:
        raise ValueError("--local-ai-max-tokens muss größer als 0 sein")
    if args.local_ai_analysis and not args.run_autotune:
        raise ValueError(
            "--local-ai-analysis kann nur mit --run-autotune verwendet werden"
        )
    if args.analyze_existing_run:
        source_run = args.analyze_existing_run.expanduser().resolve()
        if not (source_run / "autotune_state.json").is_file():
            raise ValueError(
                "--analyze-existing-run benötigt einen Laufordner mit "
                "autotune_state.json"
            )
    if args.growing_context_size <= 0:
        raise ValueError("--growing-context-size muss größer als 0 sein")
    if args.growing_max_tokens <= 0:
        raise ValueError("--growing-max-tokens muss größer als 0 sein")
    growing_targets = parse_context_targets(args.growing_targets)
    if growing_targets[-1] + args.growing_max_tokens >= args.growing_context_size:
        raise ValueError(
            "Das größte Growing-Chat-Ziel plus Ausgabe muss kleiner als "
            "--growing-context-size sein"
        )
    execution_modes = sum(
        bool(mode)
        for mode in (
            args.run_smoke,
            args.run_screening,
            args.run_server_smoke,
            args.run_growing_chat,
            args.plan_autotune,
            args.run_autotune,
            args.analyze_existing_run,
        )
    )
    if execution_modes > 1:
        raise ValueError(
            "Es kann jeweils nur ein Ausführungsmodus gewählt werden"
        )

    server = resolve_binary(llama_dir, "llama-server", required=True)
    assert server is not None
    bench = resolve_binary(llama_dir, "llama-bench", required=False)
    benchmark_needed = (
        args.run_smoke
        or args.run_screening
        or args.plan_autotune
        or args.run_autotune
    )
    if benchmark_needed and bench is None:
        raise ValueError(
            "Benchmark-Ausführung benötigt eine ausführbare llama-bench-Datei"
        )
    return llama_dir, model, server, bench


def make_run_directory(base: Path, prefix: str = "discovery") -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base.expanduser().resolve() / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = candidate.with_name(f"{prefix}_{timestamp}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        llama_dir, model, server, bench = validate_args(args)
    except ValueError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2

    hardware = collect_hardware(args.command_timeout)
    model_information = collect_model(model)
    llama_information = collect_llama(
        llama_dir, server, bench, args.command_timeout
    )
    capabilities = detect_capabilities(llama_information)
    tuning_plan = build_tuning_plan(
        hardware, model_information, capabilities
    )
    tuning_plan["deployment_host"] = args.deployment_host

    if args.analyze_existing_run:
        print("Llama Autotune – Vorhandener Lauf wird neu bewertet ...")
        try:
            reanalysis = reanalyze_existing_run(
                args.analyze_existing_run,
                llama_dir,
                server,
                model,
                capabilities,
                tuning_plan,
                optimization_objective=args.optimization_objective,
                startup_timeout=args.server_start_timeout,
                request_timeout=args.server_request_timeout,
                shutdown_timeout=args.server_shutdown_timeout,
                max_tokens=args.local_ai_max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Fehler: {exc}", file=sys.stderr)
            return 1
        recommendation = reanalysis["recommendation"]
        speculation = recommendation["speculation"]
        print("Llama Autotune – Neuauswertung abgeschlossen")
        print(f"  Quelle       : {reanalysis['source_run']}")
        print(
            "  Bewertungsziel: "
            f"{reanalysis['optimization_objective']}"
        )
        print(
            "  Empfehlung   : "
            f"{speculation_variant_label(speculation)}, "
            f"Score={recommendation['final_score']:.6f}"
        )
        local_ai = reanalysis["local_ai_analysis"]
        print(f"  Lokale KI    : {local_ai['status']}")
        if local_ai.get("analysis_file"):
            print(f"  KI-Bericht   : {local_ai['analysis_file']}")
        print(f"  Gesamtbericht: {reanalysis['report_file']}")
        print(f"  Ergebnis     : {reanalysis['output_dir']}")
        return 0 if local_ai["status"] in {"ok", "partial"} else 1

    run_smoke_requested = args.run_smoke or args.run_screening
    if args.run_autotune:
        run_prefix = "autotune"
    elif args.run_screening:
        run_prefix = "screening"
    elif args.plan_autotune:
        run_prefix = "autotune_plan"
    elif args.run_growing_chat:
        run_prefix = "growing_chat"
    elif args.run_server_smoke:
        run_prefix = "server_smoke"
    elif args.run_smoke:
        run_prefix = "smoke"
    else:
        run_prefix = "discovery"
    run_dir = make_run_directory(args.output_dir, run_prefix)
    benchmark_runs = []
    server_runs = []
    experiment_plan = None
    experiment_plan_json_path = None
    experiment_plan_markdown_path = None
    autotune_execution = None
    if args.plan_autotune or args.run_autotune:
        experiment_plan = build_autotune_experiment_plan(
            hardware,
            model_information,
            capabilities,
            tuning_plan,
            profile=args.autotune_profile,
            optimization_objective=args.optimization_objective,
        )
        experiment_plan_json_path = run_dir / "experiment_plan.json"
        experiment_plan_markdown_path = run_dir / "experiment_plan.md"
        write_json(experiment_plan_json_path, experiment_plan)
        experiment_plan_markdown_path.write_text(
            render_autotune_plan_markdown(experiment_plan),
            encoding="utf-8",
        )
    if args.run_autotune:
        assert bench is not None
        assert experiment_plan is not None
        autotune_execution = run_autotune_foundation(
            llama_dir,
            bench,
            server,
            model,
            capabilities,
            tuning_plan,
            experiment_plan,
            run_dir,
            benchmark_timeout=args.benchmark_timeout,
            smoke_repetitions=args.smoke_repetitions,
            screening_repetitions=args.screening_repetitions,
            prompt_tokens=args.screening_prompt_tokens,
            generation_tokens=args.screening_generation_tokens,
            server_start_timeout=args.server_start_timeout,
            server_request_timeout=args.server_request_timeout,
            server_shutdown_timeout=args.server_shutdown_timeout,
            server_max_tokens=args.server_max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        if args.local_ai_analysis:
            recommendation = autotune_execution.get("recommendation")
            final_stage = autotune_execution.get("stages", {}).get(
                "final-validation", {}
            )
            if recommendation and recommendation.get("status") == "final":
                print(
                    "Llama Autotune – Lokale KI erläutert die fertige "
                    "Empfehlung ..."
                )
                try:
                    local_ai_analysis = run_local_ai_analysis(
                        llama_dir,
                        server,
                        model,
                        capabilities,
                        tuning_plan,
                        recommendation,
                        final_stage,
                        run_dir,
                        startup_timeout=args.server_start_timeout,
                        request_timeout=args.server_request_timeout,
                        shutdown_timeout=args.server_shutdown_timeout,
                        max_tokens=args.local_ai_max_tokens,
                        reasoning_effort=args.reasoning_effort,
                    )
                except (OSError, ValueError) as exc:
                    local_ai_analysis = {
                        "name": "local-ai-analysis",
                        "status": "configuration-error",
                        "immutable_deterministic_result": True,
                        "analysis": "",
                        "error": str(exc),
                    }
            else:
                local_ai_analysis = {
                    "name": "local-ai-analysis",
                    "status": "skipped",
                    "immutable_deterministic_result": True,
                    "analysis": "",
                    "error": "Keine endgültige deterministische Empfehlung",
                }
            autotune_execution["local_ai_analysis"] = local_ai_analysis
            write_autotune_checkpoint(
                Path(autotune_execution["checkpoint_file"]),
                autotune_execution,
            )
            Path(autotune_execution["report_file"]).write_text(
                render_autotune_execution_markdown(autotune_execution),
                encoding="utf-8",
            )
    if run_smoke_requested:
        assert bench is not None
        print("Llama Autotune – Smoke-Benchmark wird gestartet ...")
        try:
            smoke_run = run_smoke_benchmark(
                llama_dir,
                bench,
                model,
                capabilities,
                tuning_plan,
                run_dir,
                timeout=args.benchmark_timeout,
                repetitions=args.smoke_repetitions,
            )
        except (OSError, ValueError) as exc:
            smoke_run = {
                "name": "smoke",
                "status": "configuration-error",
                "error": str(exc),
            }
        benchmark_runs.append(smoke_run)
        if args.run_screening:
            if smoke_run["status"] == "ok":
                print("Llama Autotune – Batch-Screening wird gestartet ...")
                try:
                    screening_run = run_batch_screening(
                        llama_dir,
                        bench,
                        model,
                        capabilities,
                        tuning_plan,
                        run_dir,
                        timeout=args.benchmark_timeout,
                        repetitions=args.screening_repetitions,
                        prompt_tokens=args.screening_prompt_tokens,
                        generation_tokens=args.screening_generation_tokens,
                    )
                except (OSError, ValueError) as exc:
                    screening_run = {
                        "name": "batch-screening",
                        "status": "configuration-error",
                        "error": str(exc),
                    }
            else:
                screening_run = {
                    "name": "batch-screening",
                    "status": "skipped",
                    "reason": "Smoke-Test war nicht erfolgreich",
                }
            benchmark_runs.append(screening_run)

    if args.run_server_smoke:
        print("Llama Autotune – Server-Smoke-Test wird gestartet ...")
        try:
            server_run = run_server_smoke(
                llama_dir,
                server,
                model,
                capabilities,
                tuning_plan,
                run_dir,
                startup_timeout=args.server_start_timeout,
                request_timeout=args.server_request_timeout,
                shutdown_timeout=args.server_shutdown_timeout,
                max_tokens=args.server_max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
        except (OSError, ValueError) as exc:
            server_run = {
                "name": "server-smoke",
                "status": "configuration-error",
                "error": str(exc),
            }
        server_runs.append(server_run)

    if args.run_growing_chat:
        native_context = (model_information.get("summary") or {}).get(
            "context_length"
        )
        if (
            isinstance(native_context, int)
            and args.growing_context_size > native_context
        ):
            growing_run = {
                "name": "growing-chat",
                "status": "configuration-error",
                "error": (
                    f"Kontextgröße {args.growing_context_size} überschreitet das "
                    f"native Modelllimit {native_context}"
                ),
            }
        else:
            print("Llama Autotune – Growing-Chat-Benchmark wird gestartet ...")
            try:
                growing_run = run_growing_chat(
                    llama_dir,
                    server,
                    model,
                    capabilities,
                    tuning_plan,
                    run_dir,
                    context_size=args.growing_context_size,
                    targets=parse_context_targets(args.growing_targets),
                    cache_type=args.growing_cache_type,
                    startup_timeout=args.server_start_timeout,
                    request_timeout=args.server_request_timeout,
                    shutdown_timeout=args.server_shutdown_timeout,
                    max_tokens=args.growing_max_tokens,
                    reasoning_effort=args.reasoning_effort,
                )
            except (OSError, ValueError) as exc:
                growing_run = {
                    "name": "growing-chat",
                    "status": "configuration-error",
                    "error": str(exc),
                }
        server_runs.append(growing_run)

    manifest = {
        "schema_version": 10,
        "program": {"name": "llama-autotune", "version": PROGRAM_VERSION},
        "created_at": utc_now(),
        "mode": (
            "adaptive-autotune-foundation-execution"
            if args.run_autotune
            else (
                "adaptive-autotune-planning"
                if args.plan_autotune
                else (
                    "discovery-planning-and-growing-chat"
                    if args.run_growing_chat
                    else (
                        "discovery-planning-and-server-smoke"
                        if args.run_server_smoke
                        else (
                            "discovery-planning-and-screening"
                            if args.run_screening
                            else (
                                "discovery-planning-and-smoke"
                                if args.run_smoke
                                else "discovery-and-planning"
                            )
                        )
                    )
                )
            )
        ),
        "hardware": hardware,
        "model": model_information,
        "llama_cpp": llama_information,
        "capabilities": capabilities,
        "tuning_plan": tuning_plan,
        "benchmark_runs": benchmark_runs,
        "server_runs": server_runs,
        "experiment_plan": experiment_plan,
        "autotune_execution": autotune_execution,
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Llama Autotune – Bestandsaufnahme abgeschlossen")
    print(f"  llama-server : {server}")
    print(f"  llama-bench  : {bench or 'nicht gefunden'}")
    print(f"  Modell       : {model}")
    print(f"  Modellgröße  : {manifest['model']['size_gib']} GiB")
    model_summary = manifest["model"]["summary"]
    if model_summary:
        print(f"  Modellname   : {model_summary['name'] or 'nicht angegeben'}")
        print(
            "  Architektur  : "
            f"{model_summary['architecture'] or 'nicht angegeben'}"
        )
        print(
            "  Kontextlimit : "
            f"{model_summary['context_length'] or 'nicht angegeben'}"
        )
        if model_summary.get("mtp_detected"):
            print(
                "  MTP-Tensoren : "
                f"erkannt ({model_summary['mtp_tensor_count']})"
            )
    else:
        print(f"  GGUF-Metadaten: {manifest['model']['gguf']['error']}")
    budget = tuning_plan["memory_budget"]["available_for_kv_gib"]
    print(f"  KV-VRAM-Budget: {budget if budget is not None else 'unbekannt'} GiB")
    print("  Kontextplan  :")
    for profile in tuning_plan["context_profiles"]:
        recommendation = profile["recommended_cache_type"] or "nicht automatisch"
        print(f"    {profile['context_length']:>6} Tokens -> {recommendation}")
    if benchmark_runs:
        smoke_run = benchmark_runs[0]
        print(f"  Smoke-Test   : {smoke_run['status']}")
        for metric in smoke_run.get("metrics", []):
            throughput = metric["tokens_per_second"]
            throughput_text = (
                f"{throughput:.3f}" if isinstance(throughput, (int, float)) else "?"
            )
            print(
                f"    {metric['test_kind']:<10} -> {throughput_text} Tokens/s"
            )
    if len(benchmark_runs) > 1:
        screening_run = benchmark_runs[1]
        print(
            "  Screening    : "
            f"{screening_run['status']} "
            f"({screening_run.get('successful_cases', 0)}/"
            f"{screening_run.get('total_cases', 0)})"
        )
        for rank, item in enumerate(screening_run.get("ranking", [])[:3], 1):
            config = item["configuration"]
            print(
                f"    {rank}. B={config['batch_size']}, "
                f"UB={config['ubatch_size']}, "
                f"FA={config['flash_attention']} -> "
                f"Score {item['balanced_score']:.4f}"
            )
    if server_runs:
        server_run = server_runs[0]
        if server_run.get("name") == "growing-chat":
            print(f"  Growing Chat : {server_run['status']}")
            for stage in server_run.get("stages", []):
                generation = stage.get("generation_tokens_per_second")
                generation_text = (
                    f"{generation:.3f} Tokens/s"
                    if isinstance(generation, (int, float))
                    else "?"
                )
                ratio = stage.get("cache_ratio")
                ratio_text = (
                    f"{ratio * 100:.1f}%"
                    if isinstance(ratio, (int, float))
                    else "?"
                )
                print(
                    f"    {stage['actual_prompt_tokens']:>6} Tokens -> "
                    f"Cache {ratio_text}, Neu {stage['new_prompt_tokens']}, "
                    f"{generation_text}"
                )
            if server_run.get("csv_file"):
                print(f"    CSV        -> {server_run['csv_file']}")
        else:
            print(f"  Server-Test  : {server_run['status']}")
            readiness = server_run.get("readiness") or {}
            startup_seconds = readiness.get("duration_seconds")
            if startup_seconds is not None:
                print(f"    Startzeit  -> {startup_seconds:.3f} Sekunden")
            for number, request in enumerate(server_run.get("requests", []), 1):
                generation_speed = request.get("predicted_tokens_per_second")
                speed_text = (
                    f"{generation_speed:.3f} Tokens/s"
                    if isinstance(generation_speed, (int, float))
                    else "?"
                )
                print(
                    f"    Request {number} -> "
                    f"final={request['content_characters']} Zeichen, "
                    f"reasoning={request['reasoning_characters']} Zeichen, "
                    f"cache={request.get('cached_tokens')}, {speed_text}"
                )
            print(
                "    Prompt-Cache -> "
                + (
                    "wiederverwendet"
                    if server_run.get("prompt_cache_reused")
                    else "nicht nachgewiesen"
                )
            )
        if server_run.get("log_file"):
            print(f"    Server-Log -> {server_run['log_file']}")
    if experiment_plan:
        runs = experiment_plan["estimated_runs"]
        print(f"  Autotune-Plan: {experiment_plan['profile']}")
        print(
            "    Bewertungsziel -> "
            f"{experiment_plan['optimization_objective']}"
        )
        print(
            "    Obergrenze -> "
            f"{runs['upper_bound_total']} adaptive Experimente"
        )
        for stage in experiment_plan["stages"]:
            candidates = stage.get("candidates")
            count = len(candidates) if isinstance(candidates, list) else "dynamisch"
            print(f"    {stage['id']:<25} {count}")
        print(f"    JSON       -> {experiment_plan_json_path}")
        print(f"    Bericht    -> {experiment_plan_markdown_path}")
    if autotune_execution:
        print(
            "  Autotune-Lauf: "
            f"{autotune_execution['status']} "
            "(vollständiger adaptiver Lauf)"
        )
        print(
            "    Bewertungsziel -> "
            f"{autotune_execution.get('optimization_objective', 'balanced')}"
        )
        recommendation = autotune_execution.get("recommendation") or (
            autotune_execution.get("preliminary_recommendation")
        )
        if recommendation:
            config = recommendation["configuration"]
            label = (
                "Empfehlung"
                if recommendation.get("status") == "final"
                else "Vorläufig"
            )
            print(
                f"    {label} -> "
                f"B={config['batch_size']}, UB={config['ubatch_size']}, "
                f"FA={config['flash_attention']}, T={config['threads']}"
            )
            for profile in recommendation.get("context_profiles", []):
                prompt = profile.get("prompt_tokens_per_second")
                generation = profile.get("generation_tokens_per_second")
                prompt_text = (
                    f"{prompt:.3f}"
                    if isinstance(prompt, (int, float))
                    else "?"
                )
                generation_text = (
                    f"{generation:.3f}"
                    if isinstance(generation, (int, float))
                    else "?"
                )
                print(
                    f"      CTX={profile['context_size']}, "
                    f"Cache={profile['cache_type_k']} -> "
                    f"Prompt={prompt_text}, Generation={generation_text}"
                )
            speculation = recommendation.get("speculation")
            if speculation:
                speedup = recommendation.get("generation_speedup_vs_none")
                speedup_text = (
                    f"{speedup:.3f}x"
                    if isinstance(speedup, (int, float))
                    else "nicht verfügbar"
                )
                print(
                    "    Spekulation -> "
                    f"{speculation_variant_label(speculation)}, "
                    f"{recommendation.get('speculation_generation_tokens_per_second', 0):.3f} "
                    f"Tokens/s, Speedup={speedup_text}"
                )
            if recommendation.get("status") != "final":
                print("    Noch offen -> wiederholte Finalvalidierung")
        local_ai = autotune_execution.get("local_ai_analysis")
        if local_ai:
            print(
                "    Lokale KI   -> "
                f"{local_ai.get('status')}, "
                f"{local_ai.get('content_characters', 0)} Zeichen, "
                f"{local_ai.get('attempt_count', 0)} Versuch(e)"
            )
            if local_ai.get("analysis_file"):
                print(f"      Analyse  -> {local_ai['analysis_file']}")
            if local_ai.get("log_file"):
                print(f"      Serverlog -> {local_ai['log_file']}")
        print(f"    Zustand    -> {autotune_execution['checkpoint_file']}")
        print(f"    Bericht    -> {autotune_execution['report_file']}")
    print(f"  Ergebnis     : {manifest_path}")
    exit_code = 0
    if (
        autotune_execution
        and autotune_execution["status"]
        not in {
            "final-validation-complete",
            "final-validation-partial",
        }
    ):
        exit_code = 1
    if any(run["status"] != "ok" for run in [*benchmark_runs, *server_runs]):
        exit_code = 1
    final_recommendation = (
        autotune_execution.get("recommendation")
        if autotune_execution
        else None
    )
    if (
        exit_code == 0
        and final_recommendation
        and final_recommendation.get("status") == "final"
    ):
        print()
        print("Direkt nutzbarer llama.cpp-Startbefehl:")
        print(shell_command(final_recommendation["command"]))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

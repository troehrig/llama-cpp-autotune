#!/usr/bin/env python3
"""Deterministische Zusammenfassung eines vorhandenen Autotune-Laufs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llama_autotune import (
    parse_nvidia_gpus,
    shell_command,
    speculation_variant_label,
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Datei nicht gefunden: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ungültiges JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Erwartetes JSON-Objekt fehlt in {path}")
    return data


def find_latest_run(runs_dir: Path) -> Path:
    candidates = [
        path
        for path in runs_dir.expanduser().resolve().glob("autotune_*")
        if path.is_dir()
        and (path / "autotune_state.json").is_file()
        and (path / "recommendation.json").is_file()
    ]
    if not candidates:
        raise ValueError(
            f"Kein vollständiger Autotune-Lauf unter {runs_dir} gefunden"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_run_directory(
    run_dir: Path | None, runs_dir: Path
) -> Path:
    resolved = (
        run_dir.expanduser().resolve()
        if run_dir is not None
        else find_latest_run(runs_dir)
    )
    if not resolved.is_dir():
        raise ValueError(f"Laufordner nicht gefunden: {resolved}")
    return resolved


def load_run(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir / "autotune_state.json"
    if not state_path.is_file():
        state_path = run_dir / "reanalysis_state.json"
    state = read_json(state_path)
    recommendation = read_json(run_dir / "recommendation.json")

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file() and state.get("reanalysis_source"):
        manifest_path = Path(state["reanalysis_source"]) / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    return {
        "run_dir": run_dir,
        "state_path": state_path,
        "manifest_path": manifest_path if manifest_path.is_file() else None,
        "state": state,
        "recommendation": recommendation,
        "manifest": manifest,
    }


def number(value: Any, digits: int = 3, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return "-"


def first_output_line(result: Any) -> str:
    if not isinstance(result, dict):
        return "-"
    output = result.get("stdout") or result.get("stderr") or ""
    lines = str(output).strip().splitlines()
    return lines[0] if lines else "-"


def render_summary(bundle: dict[str, Any]) -> str:
    run_dir: Path = bundle["run_dir"]
    state = bundle["state"]
    recommendation = bundle["recommendation"]
    manifest = bundle["manifest"]
    lines = [
        "# Llama Autotune – Laufzusammenfassung",
        "",
        f"Laufordner: `{run_dir}`",
        f"Status: `{state.get('status', '-')}`",
        f"Suchprofil: `{state.get('profile', '-')}`",
        "Optimierungsziel: "
        f"`{recommendation.get('optimization_objective', state.get('optimization_objective', '-'))}`",
        f"Konfidenz: `{recommendation.get('confidence', '-')}`",
    ]
    planned_contexts = recommendation.get("planned_contexts", [])
    validated_contexts = recommendation.get("validated_contexts", [])
    missing_contexts = recommendation.get("missing_contexts", [])
    if planned_contexts:
        lines.append(
            "Kontextabdeckung: "
            f"`{len(validated_contexts)}/{len(planned_contexts)}` "
            f"(`{recommendation.get('coverage_status', '-')}`)"
        )
    if missing_contexts:
        lines.append(
            "Nicht erfolgreich validiert: `"
            + ", ".join(str(value) for value in missing_contexts)
            + "`"
        )
    deployment_context = recommendation.get("recommended_max_context")
    if deployment_context is not None:
        validation_text = (
            "validiert"
            if recommendation.get("deployment_context_validated")
            else "nicht validiert"
        )
        lines.append(
            f"Hermes-Startkontext: `{deployment_context}` "
            f"(`{validation_text}`)"
        )
    if recommendation.get("bind_host"):
        bind_host = str(recommendation["bind_host"])
        if bind_host in {"127.0.0.1", "localhost", "::1"}:
            exposure = "nur lokal"
        elif bind_host in {"0.0.0.0", "::"}:
            exposure = "alle Schnittstellen"
        else:
            exposure = "konfigurierte Netzwerkadresse"
        lines.append(
            f"Netzwerkbindung: `{bind_host}` ({exposure})"
        )
    objective_definition = recommendation.get("objective_definition") or {}
    if objective_definition.get("workload_assumption"):
        lines.append(
            "Workload-Annahme: "
            + objective_definition["workload_assumption"]
        )
    if objective_definition.get("not_measured"):
        lines.append(
            "Nicht gemessen: " + objective_definition["not_measured"]
        )

    hardware = manifest.get("hardware", {})
    model = manifest.get("model", {})
    model_summary = model.get("summary") or {}
    llama_cpp = manifest.get("llama_cpp", {})
    gpus = parse_nvidia_gpus(hardware) if hardware else []
    lines.extend(["", "## Umgebung", ""])
    lines.append(f"- Betriebssystem: `{hardware.get('platform', '-')}`")
    lines.append(
        f"- Logische CPU-Threads: `{hardware.get('logical_cpu_count', '-')}`"
    )
    if gpus:
        for gpu in gpus:
            lines.append(
                f"- GPU {gpu['index']}: `{gpu['name']}`, "
                f"{number(gpu.get('memory_total_mib'), 0, ' MiB')} VRAM, "
                f"Treiber `{gpu.get('driver_version', '-')}`"
            )
    else:
        lines.append("- GPU: `nicht im Manifest verfügbar`")
    lines.append(
        "- llama.cpp: `"
        + first_output_line(llama_cpp.get("server_version"))
        + "`"
    )
    lines.extend(["", "## Modell", ""])
    lines.append(f"- Name: `{model_summary.get('name', '-')}`")
    lines.append(f"- Datei: `{model.get('path', '-')}`")
    lines.append(f"- Größe: `{model.get('size_gib', '-')} GiB`")
    lines.append(
        f"- Architektur: `{model_summary.get('architecture', '-')}`"
    )
    lines.append(
        f"- Natives Kontextlimit: `{model_summary.get('context_length', '-')}`"
    )
    lines.append(
        "- MTP: `"
        + (
            f"erkannt ({model_summary.get('mtp_tensor_count', 0)} Tensoren)"
            if model_summary.get("mtp_detected")
            else "nicht erkannt"
        )
        + "`"
    )

    lines.extend(
        [
            "",
            "## Ausgeführte Stufen",
            "",
            "| Stufe | Status | Erfolgreich | Gesamt |",
            "|---|---|---:|---:|",
        ]
    )
    for stage_name, stage in state.get("stages", {}).items():
        if not isinstance(stage, dict):
            continue
        if stage_name == "smoke":
            successful = int(stage.get("status") == "ok")
            total = 1
        else:
            successful = stage.get("successful_cases", 0)
            total = stage.get("total_cases", 0)
        lines.append(
            f"| `{stage_name}` | {stage.get('status', '-')} | "
            f"{successful} | {total} |"
        )

    configuration = recommendation.get("configuration", {})
    speculation = recommendation.get("speculation") or {
        "spec_type": "none"
    }
    lines.extend(["", "## Deterministische Empfehlung", ""])
    lines.append(f"- Batch/UBatch: `{configuration.get('batch_size', '-')}` / `"
                 f"{configuration.get('ubatch_size', '-')}`")
    lines.append(
        f"- Flash Attention: `{configuration.get('flash_attention', '-')}`"
    )
    lines.append(f"- Threads: `{configuration.get('threads', '-')}`")
    lines.append(
        f"- Spekulation: `{speculation_variant_label(speculation)}`"
    )
    lines.append(
        f"- Finalscore: `{number(recommendation.get('final_score'), 6)}`"
    )
    lines.append(
        "- Schwächster Kontextscore: `"
        + number(recommendation.get("worst_context_score"), 6)
        + "`"
    )

    lines.extend(
        [
            "",
            "### Kontextprofile",
            "",
            "| Kontext | Cache K/V | Prompt/s | Generation/s | Wall | Speicher frei |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for profile in recommendation.get("context_profiles", []):
        lines.append(
            f"| {profile.get('context_size', '-')} | "
            f"`{profile.get('cache_type_k', '-')}` | "
            f"{number(profile.get('prompt_tokens_per_second'))} | "
            f"{number(profile.get('generation_tokens_per_second'))} | "
            f"{number(profile.get('wall_seconds'), suffix=' s')} | "
            f"{number(profile.get('memory_free_mib'), 0, ' MiB')} |"
        )

    comparisons = recommendation.get("comparisons_vs_none", [])
    if comparisons:
        lines.extend(
            [
                "",
                "### Vergleich mit `none`",
                "",
                "| Kontext | Generation-Speedup | Wall-Clock-Speedup |",
                "|---:|---:|---:|",
            ]
        )
        for comparison in comparisons:
            lines.append(
                f"| {comparison.get('context_size', '-')} | "
                f"{number(comparison.get('generation_speedup_vs_none'), suffix='x')} | "
                f"{number(comparison.get('wall_speedup_vs_none'), suffix='x')} |"
            )

    objective_winners = recommendation.get("objective_winners", {})
    if objective_winners:
        lines.extend(
            [
                "",
                "### Gewinner nach Bewertungsziel",
                "",
                "| Ziel | Variante | Score |",
                "|---|---|---:|",
            ]
        )
        for objective, winner in objective_winners.items():
            winner_speculation = winner["configuration"]["speculation"]
            lines.append(
                f"| `{objective}` | "
                f"`{speculation_variant_label(winner_speculation)}` | "
                f"{number(winner.get('final_score'), 6)} |"
            )

    local_ai = state.get("local_ai_analysis")
    if isinstance(local_ai, dict):
        lines.extend(["", "## Lokale KI-Erklärung", ""])
        lines.append(f"- Status: `{local_ai.get('status', '-')}`")
        lines.append(f"- Versuche: `{local_ai.get('attempt_count', 0)}`")
        lines.append(
            f"- Finish-Reason: `{local_ai.get('finish_reason', '-')}`"
        )
        if local_ai.get("analysis_file"):
            lines.append(f"- Bericht: `{local_ai['analysis_file']}`")

    lines.extend(["", "## Wichtige Artefakte", ""])
    for label, path in (
        ("Zustand", bundle["state_path"]),
        ("Manifest", bundle["manifest_path"]),
        ("Empfehlung", run_dir / "recommendation.json"),
        ("Gesamtbericht", run_dir / "autotune_report.md"),
    ):
        if path is not None and Path(path).is_file():
            lines.append(f"- {label}: `{path}`")

    command = recommendation.get("command")
    if isinstance(command, list) and command:
        lines.extend(
            [
                "",
                "## Direkt nutzbarer llama.cpp-Startbefehl",
                "",
                shell_command(command),
            ]
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fasst einen vorhandenen Autotune-Lauf zusammen, ohne ein Modell "
            "oder einen Server zu starten."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="Laufordner; ohne Angabe wird der neueste unter --runs-dir gewählt",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Basisverzeichnis für die automatische Auswahl (Standard: ./runs)",
    )
    parser.add_argument(
        "--write",
        type=Path,
        help="schreibt dieselbe Zusammenfassung zusätzlich in eine Datei",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = resolve_run_directory(args.run_dir, args.runs_dir)
        summary = render_summary(load_run(run_dir))
    except ValueError as exc:
        print(f"Fehler: {exc}")
        return 2
    if args.write:
        output = args.write.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

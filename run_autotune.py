#!/usr/bin/env python3
"""Einfacher Einstiegspunkt für einen vollständigen Llama-Autotune-Lauf."""

from __future__ import annotations

import argparse
from pathlib import Path

from llama_autotune import (
    AUTOTUNE_PROFILES,
    DEFAULT_DEPLOYMENT_HOST,
    OPTIMIZATION_OBJECTIVES,
    main as engine_main,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Startet einen vollständigen, Hermes-orientierten Autotune-Lauf. "
            "Am Ende wird ein direkt kopierbarer llama-server-Befehl ausgegeben."
        )
    )
    parser.add_argument(
        "llama_dir",
        type=Path,
        help="Verzeichnis mit llama-server und llama-bench",
    )
    parser.add_argument("model", type=Path, help="lokale GGUF-Modelldatei")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Basisverzeichnis der Laufdaten (Standard: ./runs)",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(AUTOTUNE_PROFILES),
        default="quick",
        help="Suchumfang: quick, balanced oder thorough (Standard: quick)",
    )
    parser.add_argument(
        "--objective",
        choices=tuple(OPTIMIZATION_OBJECTIVES),
        default="hermes",
        help="Bewertungsziel (Standard: hermes)",
    )
    parser.add_argument(
        "--deployment-host",
        default=DEFAULT_DEPLOYMENT_HOST,
        help=(
            "Bind-Adresse im empfohlenen Startbefehl; 0.0.0.0 gibt den "
            "Server im Netzwerk frei (Standard: 127.0.0.1)"
        ),
    )
    parser.add_argument(
        "--no-local-ai",
        action="store_true",
        help="überspringt nur die optionale lokale KI-Erklärung",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=float,
        default=600.0,
        help="Timeout je llama-bench-Aufruf (Standard: 600)",
    )
    parser.add_argument(
        "--server-start-timeout",
        type=float,
        default=300.0,
        help="Timeout je Serverstart (Standard: 300)",
    )
    parser.add_argument(
        "--server-request-timeout",
        type=float,
        default=600.0,
        help="Timeout je Serverrequest (Standard: 600)",
    )
    parser.add_argument(
        "--server-max-tokens",
        type=int,
        default=256,
        help=(
            "initiales Antwortbudget; bei abgeschnittenem Reasoning wird "
            "automatisch bis 2048 erhöht (Standard: 256)"
        ),
    )
    parser.add_argument(
        "--local-ai-max-tokens",
        type=int,
        default=3072,
        help="Tokenbudget für die optionale Erklärung (Standard: 3072)",
    )
    return parser.parse_args(argv)


def build_engine_arguments(args: argparse.Namespace) -> list[str]:
    arguments = [
        "--llama-dir",
        str(args.llama_dir),
        "--model",
        str(args.model),
        "--output-dir",
        str(args.output_dir),
        "--run-autotune",
        "--autotune-profile",
        args.profile,
        "--optimization-objective",
        args.objective,
        "--deployment-host",
        args.deployment_host,
        "--benchmark-timeout",
        str(args.benchmark_timeout),
        "--server-start-timeout",
        str(args.server_start_timeout),
        "--server-request-timeout",
        str(args.server_request_timeout),
        "--server-max-tokens",
        str(args.server_max_tokens),
        "--local-ai-max-tokens",
        str(args.local_ai_max_tokens),
        "--reasoning-effort",
        "low",
    ]
    if not args.no_local_ai:
        arguments.append("--local-ai-analysis")
    return arguments


def main(argv: list[str] | None = None) -> int:
    return engine_main(build_engine_arguments(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

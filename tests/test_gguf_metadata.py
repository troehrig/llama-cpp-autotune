import json
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from llama_autotune import (
    adaptive_token_budgets,
    build_server_smoke_command,
    build_smoke_command,
    build_autotune_experiment_plan,
    build_final_recommendation,
    build_local_ai_analysis_input,
    build_preliminary_recommendation,
    build_speculation_variants,
    build_tuning_plan,
    calibrate_growing_content,
    chat_prompt_token_count,
    classify_chat_stage,
    detect_capabilities,
    estimate_kv_cache_bytes,
    FILLER_PARAGRAPHS,
    final_validation_candidates,
    OPTIMIZATION_OBJECTIVES,
    parse_benchmark_json,
    parse_prometheus_metrics,
    parse_context_targets,
    rank_context_cache_cases,
    rank_final_validation_cases,
    rank_screening_cases,
    rank_speculation_cases,
    read_gguf_metadata,
    redact_secret,
    reanalyze_existing_run,
    run_batch_screening,
    run_autotune_foundation,
    run_command,
    run_context_cache_screening,
    run_growing_chat,
    run_local_ai_analysis,
    run_speculation_screening,
    run_smoke_benchmark,
    run_thread_screening,
    render_autotune_plan_markdown,
    select_context_variants,
    shell_command,
    summarize_chat_response,
    summarize_final_validation,
    summarize_gguf,
    speculation_variant_label,
    write_autotune_checkpoint,
)
from analyze_autotune import load_run, render_summary
from run_autotune import (
    build_engine_arguments,
    parse_args as parse_runner_args,
)


def encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def metadata_entry(key: str, value_type: int, value: bytes) -> bytes:
    return encode_string(key) + struct.pack("<I", value_type) + value


def tensor_descriptor(name: str, dimensions: tuple[int, ...]) -> bytes:
    return (
        encode_string(name)
        + struct.pack("<I", len(dimensions))
        + b"".join(struct.pack("<Q", dimension) for dimension in dimensions)
        + struct.pack("<IQ", 0, 0)
    )


class GgufMetadataTest(unittest.TestCase):
    def test_hermes_objective_models_repeating_context_phases(self) -> None:
        objective = OPTIMIZATION_OBJECTIVES["hermes"]
        self.assertAlmostEqual(sum(objective["weights"].values()), 1.0)
        self.assertEqual(objective["context_weighting"], "equal")
        self.assertIn("Komprimierung", objective["workload_assumption"])
        self.assertIn("nicht gemessen", objective["not_measured"])

    def test_simple_runner_builds_a_complete_hermes_invocation(self) -> None:
        args = parse_runner_args(
            ["/opt/llama/bin", "/models/test model.gguf"]
        )
        engine_arguments = build_engine_arguments(args)

        self.assertIn("--run-autotune", engine_arguments)
        self.assertEqual(
            engine_arguments[
                engine_arguments.index("--optimization-objective") + 1
            ],
            "hermes",
        )
        self.assertEqual(
            engine_arguments[engine_arguments.index("--autotune-profile") + 1],
            "quick",
        )
        self.assertIn("--local-ai-analysis", engine_arguments)
        self.assertEqual(
            engine_arguments[
                engine_arguments.index("--server-max-tokens") + 1
            ],
            "256",
        )
        self.assertEqual(
            engine_arguments[
                engine_arguments.index("--deployment-host") + 1
            ],
            "127.0.0.1",
        )

    def test_simple_runner_forwards_explicit_network_binding(self) -> None:
        args = parse_runner_args(
            [
                "/opt/llama/bin",
                "/models/test.gguf",
                "--deployment-host",
                "0.0.0.0",
            ]
        )
        engine_arguments = build_engine_arguments(args)
        self.assertEqual(
            engine_arguments[
                engine_arguments.index("--deployment-host") + 1
            ],
            "0.0.0.0",
        )

    def test_classifies_gemma_reasoning_truncation_and_peg_error(self) -> None:
        truncated_response = {"status_code": 200, "json": {}}
        truncated_summary = {
            "status_code": 200,
            "finish_reason": "length",
            "content": "",
            "reasoning_content": "noch nicht abgeschlossene Überlegung",
        }
        self.assertEqual(
            classify_chat_stage(truncated_response, truncated_summary),
            "reasoning-truncated",
        )

        peg_response = {
            "status_code": 500,
            "json": {
                "error": {
                    "message": (
                        "The model produced output that does not match the "
                        "expected peg-gemma4 format"
                    )
                }
            },
        }
        self.assertEqual(
            classify_chat_stage(
                peg_response,
                {
                    "status_code": 500,
                    "finish_reason": None,
                    "content": "",
                    "reasoning_content": "",
                },
            ),
            "peg-format-error",
        )

    def test_accepts_usable_content_when_token_limit_is_reached(self) -> None:
        self.assertEqual(
            classify_chat_stage(
                {"status_code": 200, "json": {}},
                {
                    "status_code": 200,
                    "finish_reason": "length",
                    "content": "Eine bereits verwertbare Antwort.",
                    "reasoning_content": "",
                },
            ),
            "ok",
        )

    def test_adaptive_token_budgets_double_to_the_cap(self) -> None:
        self.assertEqual(
            adaptive_token_budgets(256),
            [256, 512, 1024, 2048],
        )
        self.assertEqual(adaptive_token_budgets(3072), [3072])

    def test_growing_chat_restarts_server_with_larger_budget(self) -> None:
        attempts = [
            {
                "status": "reasoning-truncated",
                "stages": [{"status": "reasoning-truncated"}],
            },
            {"status": "ok", "stages": [{"status": "ok"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with (
                patch(
                    "llama_autotune._run_growing_chat_attempt",
                    side_effect=attempts,
                ) as mocked_attempt,
                redirect_stdout(StringIO()),
            ):
                result = run_growing_chat(
                    Path("/opt/llama"),
                    Path("/opt/llama/llama-server"),
                    Path("/models/gemma.gguf"),
                    {},
                    {},
                    run_dir,
                    context_size=8192,
                    targets=[4096],
                    cache_type="f16",
                    startup_timeout=10,
                    request_timeout=10,
                    shutdown_timeout=5,
                    max_tokens=256,
                    reasoning_effort="low",
                    batch_size=2048,
                    ubatch_size=512,
                    flash_attention="on",
                    threads=8,
                    alias="test-gemma",
                )

        self.assertEqual(mocked_attempt.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["max_tokens"]
                for call in mocked_attempt.call_args_list
            ],
            [256, 512],
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["adaptive_retry_count"], 1)
        self.assertEqual(result["effective_max_tokens"], 512)

    def test_shell_command_quotes_paths_for_copy_and_paste(self) -> None:
        rendered = shell_command(
            [
                "/opt/llama build/llama-server",
                "--model",
                "/models/test model.gguf",
                "--threads",
                8,
            ]
        )
        self.assertEqual(
            rendered,
            "'/opt/llama build/llama-server' --model "
            "'/models/test model.gguf' --threads 8",
        )

    def test_analyzer_summarizes_a_saved_run_and_ends_with_command(self) -> None:
        state = {
            "status": "final-validation-complete",
            "profile": "quick",
            "optimization_objective": "hermes",
            "stages": {
                "smoke": {"status": "ok"},
                "final-validation": {
                    "status": "ok",
                    "successful_cases": 2,
                    "total_cases": 2,
                },
            },
            "local_ai_analysis": {
                "status": "ok",
                "attempt_count": 1,
                "finish_reason": "stop",
            },
        }
        recommendation = {
            "status": "final",
            "optimization_objective": "hermes",
            "confidence": "limited",
            "configuration": {
                "batch_size": 4096,
                "ubatch_size": 512,
                "flash_attention": "on",
                "threads": 8,
            },
            "speculation": {"id": "spec-none", "spec_type": "none"},
            "final_score": 0.95,
            "worst_context_score": 0.90,
            "context_profiles": [
                {
                    "context_size": 8192,
                    "cache_type_k": "f16",
                    "prompt_tokens_per_second": 2000.0,
                    "generation_tokens_per_second": 40.0,
                    "wall_seconds": 3.0,
                    "memory_free_mib": 3000.0,
                }
            ],
            "comparisons_vs_none": [],
            "objective_winners": {},
            "bind_host": "127.0.0.1",
            "command": [
                "/opt/llama build/llama-server",
                "--model",
                "/models/test model.gguf",
            ],
        }
        manifest = {
            "hardware": {
                "platform": "Test Linux",
                "logical_cpu_count": 16,
            },
            "model": {
                "path": "/models/test model.gguf",
                "size_gib": 10.0,
                "summary": {
                    "name": "Test Model",
                    "architecture": "test",
                    "context_length": 8192,
                    "mtp_detected": False,
                },
            },
            "llama_cpp": {
                "server_version": {
                    "stdout": "version: test-build\nmore",
                }
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "autotune_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            (run_dir / "recommendation.json").write_text(
                json.dumps(recommendation), encoding="utf-8"
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            summary = render_summary(load_run(run_dir))

        self.assertIn("Test Model", summary)
        self.assertIn("final-validation", summary)
        self.assertIn("Optimierungsziel: `hermes`", summary)
        self.assertIn(
            "Netzwerkbindung: `127.0.0.1` (nur lokal)", summary
        )
        self.assertTrue(
            summary.endswith(
                "'/opt/llama build/llama-server' --model "
                "'/models/test model.gguf'"
            )
        )

    def test_speculation_label_distinguishes_draft_lengths(self) -> None:
        self.assertEqual(
            speculation_variant_label(
                {"spec_type": "draft-mtp", "draft_n_max": 3}
            ),
            "draft-mtp (draft_n_max=3)",
        )
        self.assertEqual(
            speculation_variant_label(
                {
                    "id": "spec-mtp-5",
                    "spec_type": "draft-mtp",
                    "draft_n_max": 5,
                }
            ),
            "spec-mtp-5",
        )

    def test_builds_compact_immutable_local_ai_input(self) -> None:
        recommendation = {
            "optimization_objective": "balanced",
            "objective_definition": {"weights": {"generation": 0.3}},
            "confidence": "limited",
            "configuration": {"batch_size": 4096},
            "speculation": {
                "id": "spec-mtp-3",
                "spec_type": "draft-mtp",
                "draft_n_max": 3,
            },
            "final_score": 0.93,
            "worst_context_score": 0.90,
            "context_profiles": [
                {"context_size": 8192, "repetitions": 1}
            ],
            "comparisons_vs_none": [
                {
                    "context_size": 8192,
                    "generation_speedup_vs_none": 1.8,
                    "wall_speedup_vs_none": 1.06,
                }
            ],
            "objective_winners": {},
            "command": ["llama-server", "--ctx-size", "8192"],
        }
        final_stage = {
            "cases": [{"large_raw_measurement": True}],
            "ranking": [
                {
                    "configuration": {
                        "speculation": recommendation["speculation"]
                    },
                    "final_score": 0.93,
                    "worst_context_score": 0.90,
                    "context_coverage": 1,
                    "total_contexts": 1,
                }
            ],
        }

        result = build_local_ai_analysis_input(
            recommendation, final_stage
        )

        self.assertTrue(result["immutable_deterministic_result"])
        self.assertEqual(result["finalists"][0]["variant"], "spec-mtp-3")
        self.assertNotIn("cases", result)
        limit_codes = {
            item["code"] for item in result["measurement_limits"]
        }
        self.assertIn("single-repetition", limit_codes)
        self.assertIn("parallelism-not-tested", limit_codes)
        self.assertIn("memory-not-baseline-compared", limit_codes)

    def test_local_ai_analysis_records_output_and_stops_server(self) -> None:
        required_options = {
            option: True
            for option in (
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
                "--spec-type",
            )
        }
        recommendation = {
            "optimization_objective": "long-context",
            "objective_definition": {
                "description": "große Kontexte",
                "weights": {"wall": 0.5},
            },
            "confidence": "limited",
            "configuration": {
                "batch_size": 4096,
                "ubatch_size": 512,
                "flash_attention": "on",
                "threads": 8,
            },
            "speculation": {"id": "spec-none", "spec_type": "none"},
            "final_score": 0.97,
            "worst_context_score": 0.94,
            "context_profiles": [
                {
                    "context_size": 8192,
                    "cache_type_k": "f16",
                    "cache_type_v": "f16",
                }
            ],
            "comparisons_vs_none": [],
            "objective_winners": {},
            "command": ["/opt/llama-server"],
        }
        final_stage = {"ranking": []}
        response = {
            "status_code": 200,
            "duration_seconds": 1.0,
            "json": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "# Kurzfazit\n\nDie Messung ist eindeutig.",
                            "reasoning_content": "Kurze Prüfung.",
                        },
                    }
                ],
                "usage": {"completion_tokens": 20},
                "timings": {},
            },
        }
        partial_response = json.loads(json.dumps(response))
        partial_response["json"]["choices"][0]["finish_reason"] = "length"
        partial_response["json"]["choices"][0]["message"][
            "content"
        ] = "# Kurzfazit\n\nAbgeschnittener Entwurf"

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("llama_autotune.find_free_local_port", return_value=18080),
                patch("llama_autotune.subprocess.Popen", return_value=object()),
                patch(
                    "llama_autotune.wait_for_server",
                    return_value={"ready": True, "reason": "ok"},
                ),
                patch(
                    "llama_autotune.http_json_request",
                    side_effect=[partial_response, response],
                ),
                patch(
                    "llama_autotune.stop_process_group",
                    return_value={"method": "sigterm", "returncode": 0},
                ) as stop_mock,
            ):
                result = run_local_ai_analysis(
                    Path("/opt"),
                    Path("/opt/llama-server"),
                    Path("/models/test.gguf"),
                    {
                        "server_options": required_options,
                        "cache_types": ["f16"],
                        "speculation_types": [],
                    },
                    {"fixed_parameters": {"gpu_layers": "all"}},
                    recommendation,
                    final_stage,
                    Path(directory),
                    startup_timeout=10,
                    request_timeout=10,
                    shutdown_timeout=5,
                    max_tokens=512,
                    reasoning_effort="low",
                )

            self.assertEqual(result["status"], "ok")
            self.assertTrue(Path(result["analysis_file"]).is_file())
            self.assertIn("Kurzfazit", result["analysis"])
            self.assertNotIn("api-key", result["analysis"])
            self.assertEqual(result["attempt_count"], 2)
            self.assertEqual(result["attempts"][1]["max_tokens"], 3072)
            request = json.loads(
                Path(result["request_file"]).read_text(encoding="utf-8")
            )
            self.assertIn(
                "höchstens 450 Wörtern",
                request["messages"][0]["content"],
            )
            self.assertIn(
                "measurement_limits",
                request["messages"][1]["content"],
            )
            stop_mock.assert_called_once()

    def test_reanalyzes_existing_measurements_without_benchmarks(self) -> None:
        base = {
            "batch_size": 4096,
            "ubatch_size": 512,
            "flash_attention": "on",
            "threads": 8,
        }

        def case(spec_id: str, spec_type: str, generation: float) -> dict:
            return {
                "status": "ok",
                "configuration": {
                    **base,
                    "source_context_rank": 1,
                    "context_size": 8192,
                    "prompt_target": 4096,
                    "cache_type_k": "f16",
                    "cache_type_v": "f16",
                    "speculation": {
                        "id": spec_id,
                        "spec_type": spec_type,
                    },
                    "repetition": 1,
                },
                "server_run": {
                    "gpu_after_ready": None,
                    "stages": [
                        {
                            "status": "ok",
                            "prompt_tokens_per_second": 2500.0,
                            "generation_tokens_per_second": generation,
                            "wall_seconds": 3.0,
                        }
                    ],
                },
            }

        profile = {
            "configuration": {
                **base,
                "context_size": 8192,
                "prompt_target": 4096,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
            }
        }
        state = {
            "profile": "quick",
            "status": "final-validation-complete",
            "stages": {
                "context-cache-screening": {
                    "base_ranking": [
                        {
                            "configuration": base,
                            "best_profiles": [profile],
                        }
                    ]
                },
                "final-validation": {
                    "total_cases": 2,
                    "cases": [
                        case("spec-none", "none", 40.0),
                        case("spec-mtp-3", "draft-mtp", 60.0),
                    ],
                },
            },
        }
        local_ai_result = {
            "name": "local-ai-analysis",
            "status": "ok",
            "analysis": "# Kurzfazit\n\nErklärung.",
            "analysis_file": "/tmp/analysis.md",
        }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "autotune_source"
            source.mkdir()
            (source / "autotune_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            with patch(
                "llama_autotune.run_local_ai_analysis",
                return_value=local_ai_result,
            ) as analysis_mock:
                result = reanalyze_existing_run(
                    source,
                    Path("/opt"),
                    Path("/opt/llama-server"),
                    Path("/models/test.gguf"),
                    {
                        "server_options": {
                            "--flash-attn": True,
                            "--threads-batch": True,
                            "--no-context-shift": True,
                        }
                    },
                    {"fixed_parameters": {"gpu_layers": "all"}},
                    optimization_objective="interactive",
                    startup_timeout=10,
                    request_timeout=10,
                    shutdown_timeout=5,
                    max_tokens=512,
                    reasoning_effort="low",
                )

            self.assertEqual(result["optimization_objective"], "interactive")
            self.assertTrue(Path(result["recommendation_file"]).is_file())
            self.assertTrue(Path(result["report_file"]).is_file())
            analysis_mock.assert_called_once()

    def test_reads_scalars_and_skips_arrays(self) -> None:
        entries = [
            metadata_entry("general.architecture", 8, encode_string("qwen35")),
            metadata_entry("general.name", 8, encode_string("Testmodell")),
            metadata_entry(
                "qwen35.context_length", 10, struct.pack("<Q", 262144)
            ),
            metadata_entry(
                "qwen35.attention.head_count", 4, struct.pack("<I", 24)
            ),
            metadata_entry(
                "qwen35.attention.head_count_kv", 4, struct.pack("<I", 4)
            ),
            metadata_entry(
                "qwen35.attention.key_length", 4, struct.pack("<I", 256)
            ),
            metadata_entry(
                "qwen35.attention.value_length", 4, struct.pack("<I", 256)
            ),
            metadata_entry(
                "tokenizer.ggml.tokens",
                9,
                struct.pack("<IQ", 8, 2)
                + encode_string("eins")
                + encode_string("zwei"),
            ),
        ]
        tensors = [
            tensor_descriptor("blk.0.attn_q.weight", (5120, 5120)),
            tensor_descriptor("blk.64.nextn.eh_proj.weight", (5120, 5120)),
        ]
        content = (
            b"GGUF"
            + struct.pack("<IQQ", 3, len(tensors), len(entries))
            + b"".join(entries)
            + b"".join(tensors)
        )

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "test.gguf"
            model.write_bytes(content)
            parsed = read_gguf_metadata(model)

        summary = summarize_gguf(parsed)
        self.assertEqual(parsed["version"], 3)
        self.assertEqual(parsed["tensor_count"], 2)
        self.assertEqual(
            parsed["arrays"]["tokenizer.ggml.tokens"]["length"], 2
        )
        self.assertEqual(summary["name"], "Testmodell")
        self.assertEqual(summary["context_length"], 262144)
        self.assertEqual(summary["gqa_ratio"], 6)
        self.assertEqual(summary["attention_key_length"], 256)
        self.assertEqual(summary["attention_value_length"], 256)
        self.assertTrue(summary["mtp_detected"])
        self.assertEqual(summary["mtp_tensor_count"], 1)
        self.assertEqual(
            summary["mtp_tensor_examples"],
            ["blk.64.nextn.eh_proj.weight"],
        )

    def test_builds_a_conservative_single_gpu_plan(self) -> None:
        model = {
            "size_bytes": round(16.692 * 1024**3),
            "summary": {
                "context_length": 262144,
                "block_count": 65,
                "attention_key_length": 256,
                "attention_value_length": 256,
            },
        }
        hardware = {
            "nvidia_smi": {
                "returncode": 0,
                "stdout": (
                    "0, NVIDIA GeForce RTX 4090, GPU-test, 595.84, "
                    "24564, 22267, 40, 26.48, 450.00, 5\n"
                ),
            }
        }
        llama = {
            "server_help": {
                "stdout": (
                    "--ctx-size --batch-size --ubatch-size --flash-attn "
                    "--cache-type-k f16 q8_0 q4_0 --cache-type-v "
                    "--load-mode --gpu-layers --parallel --kv-unified "
                    "--context-shift --cache-prompt "
                    "--spec-type none,ngram-mod,draft-mtp"
                ),
                "stderr": "",
            },
            "bench_help": {
                "stdout": (
                    "--n-prompt --n-gen --batch-size --ubatch-size "
                    "--cache-type-k --cache-type-v --threads "
                    "--n-gpu-layers --flash-attn --load-mode "
                    "--output --repetitions --progress"
                ),
                "stderr": "",
            },
        }

        capabilities = detect_capabilities(llama)
        plan = build_tuning_plan(hardware, model, capabilities)
        profiles = {
            profile["context_length"]: profile
            for profile in plan["context_profiles"]
        }

        self.assertEqual(capabilities["speculation_types"], ["ngram-mod", "draft-mtp"])
        self.assertEqual(profiles[131072]["recommended_cache_type"], "q4_0")
        self.assertIsNone(profiles[262144]["recommended_cache_type"])
        self.assertFalse(profiles[262144]["default_run"])
        self.assertEqual(
            plan["excluded_parameters"],
            ["split_mode", "tensor_split", "main_gpu"],
        )

    def test_estimates_qwen38_kv_cache(self) -> None:
        summary = {
            "block_count": 65,
            "attention_key_length": 256,
            "attention_value_length": 256,
        }
        estimated = estimate_kv_cache_bytes(summary, 131072, "q4_0")
        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(estimated / 1024**3, 2.285, places=3)

    def test_builds_and_parses_the_smoke_benchmark(self) -> None:
        capabilities = {
            "cache_types": ["f16", "q8_0", "q4_0"],
            "bench_options": {
                option: True
                for option in (
                    "--n-prompt",
                    "--n-gen",
                    "--batch-size",
                    "--ubatch-size",
                    "--cache-type-k",
                    "--cache-type-v",
                    "--n-gpu-layers",
                    "--flash-attn",
                    "--load-mode",
                    "--output",
                    "--repetitions",
                    "--progress",
                )
            },
        }
        plan = {"fixed_parameters": {"gpu_layers": "all"}}
        command = build_smoke_command(
            Path("/opt/llama-bench"),
            Path("/models/test.gguf"),
            capabilities,
            plan,
            2,
        )

        self.assertIn("--n-prompt", command)
        self.assertIn("--progress", command)
        self.assertEqual(
            command[command.index("--n-gpu-layers") + 1], "999"
        )

        output = json.dumps(
            [
                {
                    "n_prompt": 512,
                    "n_gen": 0,
                    "n_batch": 512,
                    "n_ubatch": 128,
                    "n_threads": 8,
                    "n_gpu_layers": 999,
                    "flash_attn": 1,
                    "avg_ts": 2300.344847,
                    "stddev_ts": 0.295523,
                },
                {
                    "n_prompt": 0,
                    "n_gen": 64,
                    "avg_ts": 44.553095,
                    "stddev_ts": 0.379692,
                },
            ]
        )
        rows, metrics = parse_benchmark_json(output)
        self.assertEqual(len(rows), 2)
        self.assertEqual(metrics[0]["test_kind"], "prompt")
        self.assertEqual(metrics[1]["test_kind"], "generation")
        self.assertEqual(metrics[1]["tokens_per_second"], 44.553095)

    def test_builds_a_secured_server_smoke_command(self) -> None:
        server_options = {
            option: True
            for option in (
                "--host",
                "--port",
                "--alias",
                "--api-key",
                "--ctx-size",
                "--batch-size",
                "--ubatch-size",
                "--flash-attn",
                "--cache-type-k",
                "--cache-type-v",
                "--gpu-layers",
                "--parallel",
                "--threads",
                "--threads-batch",
                "--kv-unified",
                "--no-context-shift",
                "--cache-prompt",
                "--load-mode",
                "--metrics",
                "--log-colors",
                "--spec-type",
                "--spec-draft-n-max",
                "--spec-ngram-mod-n-min",
                "--spec-ngram-mod-n-max",
                "--spec-ngram-mod-n-match",
            )
        }
        command = build_server_smoke_command(
            Path("/opt/llama-server"),
            Path("/models/test.gguf"),
            {
                "server_options": server_options,
                "cache_types": ["f16", "q4_0"],
                "speculation_types": ["ngram-mod", "draft-mtp"],
            },
            {"fixed_parameters": {"gpu_layers": "all"}},
            port=18080,
            alias="test-alias",
            api_key="temporary-secret",
            cache_type="q4_0",
            flash_attention="off",
            threads=16,
            speculation={
                "spec_type": "ngram-mod,draft-mtp",
                "ngram_n_min": 48,
                "ngram_n_max": 64,
                "ngram_n_match": 24,
                "draft_n_max": 2,
            },
        )

        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "18080")
        self.assertEqual(command[command.index("--batch-size") + 1], "2048")
        self.assertEqual(command[command.index("--ubatch-size") + 1], "512")
        self.assertEqual(command[command.index("--flash-attn") + 1], "off")
        self.assertEqual(command[command.index("--threads") + 1], "16")
        self.assertEqual(
            command[command.index("--spec-type") + 1],
            "ngram-mod,draft-mtp",
        )
        self.assertEqual(
            command[command.index("--spec-draft-n-max") + 1], "2"
        )
        self.assertIn("--api-key", command)
        self.assertIn("--no-context-shift", command)
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q4_0")
        self.assertNotIn(
            "temporary-secret", redact_secret(command, "temporary-secret")
        )

    def test_summarizes_reasoning_and_final_content_separately(self) -> None:
        response = {
            "status_code": 200,
            "duration_seconds": 1.25,
            "json": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "Interne Überlegung",
                            "content": "Ein KV-Cache beschleunigt die Dekodierung.",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 82,
                    "completion_tokens": 90,
                    "prompt_tokens_details": {"cached_tokens": 78},
                },
                "timings": {"predicted_per_second": 43.6},
            },
            "error": None,
            "parse_error": None,
        }

        summary = summarize_chat_response(response)
        self.assertEqual(summary["finish_reason"], "stop")
        self.assertEqual(summary["cached_tokens"], 78)
        self.assertGreater(summary["reasoning_characters"], 0)
        self.assertGreater(summary["content_characters"], 0)

    def test_parses_unlabelled_prometheus_speculation_counters(self) -> None:
        metrics = parse_prometheus_metrics(
            """
# HELP llamacpp:spec_decode_num_draft_tokens_total Draft tokens
llamacpp:spec_decode_num_draft_tokens_total 232
llamacpp:spec_decode_num_accepted_tokens_total 180
llamacpp:metric_with_labels{slot=\"0\"} 12
invalid line
"""
        )
        self.assertEqual(
            metrics["llamacpp:spec_decode_num_draft_tokens_total"], 232.0
        )
        self.assertEqual(
            metrics["llamacpp:spec_decode_num_accepted_tokens_total"], 180.0
        )
        self.assertNotIn("llamacpp:metric_with_labels", metrics)

    def test_parses_and_validates_growing_context_targets(self) -> None:
        self.assertEqual(
            parse_context_targets("8192, 32768,65536"),
            [8192, 32768, 65536],
        )
        with self.assertRaisesRegex(ValueError, "aufsteigend"):
            parse_context_targets("32768,8192")
        with self.assertRaisesRegex(ValueError, "ganze Zahlen"):
            parse_context_targets("8192,viel")

    def test_calibrates_growing_content_below_the_target(self) -> None:
        def fake_counter(content: str) -> int:
            repetitions = sum(
                content.count(paragraph) for paragraph in FILLER_PARAGRAPHS
            )
            return 100 + repetitions * 20

        result = calibrate_growing_content(
            1,
            1000,
            fake_counter,
            tolerance=10,
        )
        self.assertEqual(result["prompt_tokens"], 1000)
        self.assertEqual(result["difference_tokens"], 0)
        self.assertEqual(result["repetitions"], 45)

    def test_counts_chat_tokens_via_template_and_tokenize(self) -> None:
        responses = [
            {
                "status_code": 200,
                "json": {"prompt": "<chat>Test</chat>"},
            },
            {
                "status_code": 200,
                "json": {"tokens": [1, 2, 3, 4]},
            },
        ]
        with patch(
            "llama_autotune.http_json_request",
            side_effect=responses,
        ) as request:
            count = chat_prompt_token_count(
                "http://127.0.0.1:1234",
                "secret",
                [{"role": "user", "content": "Test"}],
                timeout=10,
            )

        self.assertEqual(count, 4)
        self.assertEqual(request.call_count, 2)
        self.assertTrue(request.call_args_list[1].kwargs["payload"]["parse_special"])

    def test_builds_an_adaptive_autotune_plan(self) -> None:
        hardware = {"logical_cpu_count": 32}
        model = {"summary": {"mtp_detected": True}}
        capabilities = {
            "speculation_types": ["ngram-mod", "draft-mtp"],
        }
        tuning_plan = {
            "excluded_parameters": ["tensor_split"],
            "stages": [
                {
                    "name": "batch-screening",
                    "batch_ubatch_pairs": [
                        {"batch_size": 512, "ubatch_size": 128},
                        {"batch_size": 2048, "ubatch_size": 512},
                    ],
                    "flash_attention": ["on", "off"],
                }
            ],
            "context_profiles": [
                {
                    "context_length": 8192,
                    "cache_estimates": {
                        "f16": {"fits_estimate": True, "with_margin_gib": 0.3},
                        "q8_0": {"fits_estimate": True, "with_margin_gib": 0.2},
                        "q4_0": {"fits_estimate": True, "with_margin_gib": 0.1},
                    },
                },
                {
                    "context_length": 131072,
                    "cache_estimates": {
                        "f16": {"fits_estimate": False, "with_margin_gib": 9.0},
                        "q8_0": {"fits_estimate": False, "with_margin_gib": 5.0},
                        "q4_0": {"fits_estimate": True, "with_margin_gib": 2.7},
                    },
                },
            ],
        }

        plan = build_autotune_experiment_plan(
            hardware,
            model,
            capabilities,
            tuning_plan,
            profile="balanced",
            optimization_objective="long-context",
        )
        stages = {stage["id"]: stage for stage in plan["stages"]}
        speculation = stages["speculation-screening"]["candidates"]
        context = stages["context-cache-screening"]["candidates"]

        self.assertTrue(plan["adaptive"])
        self.assertEqual(plan["optimization_objective"], "long-context")
        self.assertEqual(
            plan["ranking"]["weights"]["wall_clock_latency"], 0.50
        )
        self.assertEqual(len(stages["batch-screening"]["candidates"]), 4)
        self.assertEqual(len(context), 4)
        self.assertTrue(any(item["spec_type"] == "draft-mtp" for item in speculation))
        self.assertFalse(
            any(
                item["context_size"] == 131072
                and item["cache_type_k"] == "f16"
                for item in context
            )
        )
        self.assertGreater(plan["estimated_runs"]["upper_bound_total"], 20)
        self.assertIn("## Stufen", render_autotune_plan_markdown(plan))

    def test_excludes_mtp_without_matching_model_tensors(self) -> None:
        variants, exclusions = build_speculation_variants(
            {"speculation_types": ["ngram-mod", "draft-mtp"]},
            {"mtp_detected": False},
            "balanced",
        )
        self.assertFalse(any(item["spec_type"] == "draft-mtp" for item in variants))
        self.assertTrue(
            any("nextn" in item["reason"] for item in exclusions)
        )

    def test_quick_context_policy_keeps_representative_sizes(self) -> None:
        variants = [
            {"context_size": size, "cache_type_k": "q4_0"}
            for size in (4096, 8192, 32768, 65536, 131072)
        ]
        selected = select_context_variants(variants, "representative")
        self.assertEqual(
            [item["context_size"] for item in selected],
            [4096, 32768, 131072],
        )

    def test_timeout_output_remains_json_serializable(self) -> None:
        error = subprocess.TimeoutExpired(
            ["llama-bench"], 1, output=b"teilweise", stderr=b"timeout"
        )
        with patch("llama_autotune.subprocess.run", side_effect=error):
            result = run_command(["llama-bench"], timeout=1)

        self.assertTrue(result["timed_out"])
        self.assertEqual(result["stdout"], "teilweise")
        json.dumps(result)

    def test_smoke_runner_records_success_failure_and_timeout(self) -> None:
        capabilities = {
            "cache_types": ["f16"],
            "bench_options": {
                option: True
                for option in (
                    "--n-prompt",
                    "--n-gen",
                    "--batch-size",
                    "--ubatch-size",
                    "--cache-type-k",
                    "--cache-type-v",
                    "--n-gpu-layers",
                    "--flash-attn",
                    "--load-mode",
                    "--output",
                    "--repetitions",
                    "--progress",
                )
            },
        }
        plan = {"fixed_parameters": {"gpu_layers": "all"}}
        valid_output = json.dumps(
            [{"n_prompt": 512, "n_gen": 0, "avg_ts": 2300.0}]
        )
        cases = (
            (0, False, valid_output, "ok"),
            (1, False, "", "failed"),
            (None, True, "teilweise", "timeout"),
        )

        for returncode, timed_out, stdout, expected_status in cases:
            with self.subTest(status=expected_status):
                execution = {
                    "command": ["llama-bench"],
                    "started_at": "start",
                    "timeout_seconds": 10,
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": "diagnose",
                    "timed_out": timed_out,
                    "finished_at": "ende",
                    "duration_seconds": 1.0,
                }
                with tempfile.TemporaryDirectory() as directory:
                    with (
                        patch(
                            "llama_autotune.gpu_snapshot",
                            return_value={"returncode": 0},
                        ),
                        patch(
                            "llama_autotune.run_command",
                            return_value=execution,
                        ),
                    ):
                        recorded = run_smoke_benchmark(
                            Path("/opt"),
                            Path("/opt/llama-bench"),
                            Path("/models/test.gguf"),
                            capabilities,
                            plan,
                            Path(directory),
                            timeout=10,
                            repetitions=2,
                        )

                    self.assertEqual(recorded["status"], expected_status)
                    self.assertTrue(
                        Path(recorded["stderr_file"]).is_file()
                    )

    def test_ranks_screening_with_a_balanced_score(self) -> None:
        cases = [
            {
                "status": "ok",
                "configuration": {"batch_size": 512},
                "metrics": [
                    {"test_kind": "prompt", "tokens_per_second": 2000.0},
                    {
                        "test_kind": "generation",
                        "tokens_per_second": 45.0,
                    },
                ],
            },
            {
                "status": "ok",
                "configuration": {"batch_size": 2048},
                "metrics": [
                    {"test_kind": "prompt", "tokens_per_second": 2400.0},
                    {
                        "test_kind": "generation",
                        "tokens_per_second": 44.0,
                    },
                ],
            },
        ]

        ranking = rank_screening_cases(cases)
        self.assertEqual(ranking[0]["configuration"]["batch_size"], 2048)
        self.assertGreater(ranking[0]["balanced_score"], 0.98)

    def test_screening_continues_after_a_failed_candidate(self) -> None:
        options = {
            option: True
            for option in (
                "--n-prompt",
                "--n-gen",
                "--batch-size",
                "--ubatch-size",
                "--cache-type-k",
                "--cache-type-v",
                "--n-gpu-layers",
                "--flash-attn",
                "--load-mode",
                "--output",
                "--repetitions",
                "--progress",
            )
        }
        capabilities = {
            "cache_types": ["f16"],
            "bench_options": options,
        }
        plan = {
            "fixed_parameters": {"gpu_layers": "all"},
            "stages": [
                {
                    "name": "batch-screening",
                    "batch_ubatch_pairs": [
                        {"batch_size": 512, "ubatch_size": 128},
                        {"batch_size": 1024, "ubatch_size": 256},
                    ],
                    "flash_attention": ["on", "off"],
                }
            ],
        }
        call_count = 0

        def fake_record(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            configuration = kwargs["configuration"]
            if call_count == 1:
                return {
                    "status": "failed",
                    "configuration": configuration,
                    "metrics": [],
                }
            return {
                "status": "ok",
                "configuration": configuration,
                "metrics": [
                    {"test_kind": "prompt", "tokens_per_second": 2200.0},
                    {
                        "test_kind": "generation",
                        "tokens_per_second": 44.0,
                    },
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "llama_autotune.run_recorded_benchmark",
                side_effect=fake_record,
            ):
                with redirect_stdout(StringIO()):
                    result = run_batch_screening(
                        Path("/opt"),
                        Path("/opt/llama-bench"),
                        Path("/models/test.gguf"),
                        capabilities,
                        plan,
                        Path(directory),
                        timeout=10,
                        repetitions=2,
                        prompt_tokens=4096,
                        generation_tokens=64,
                    )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["successful_cases"], 3)
        self.assertEqual(result["total_cases"], 4)
        self.assertEqual(call_count, 4)

    def test_thread_screening_combines_survivors_and_planned_threads(self) -> None:
        capabilities = {
            "cache_types": ["f16"],
            "bench_options": {
                option: True
                for option in (
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
            },
        }
        tuning_plan = {"fixed_parameters": {"gpu_layers": "all"}}
        experiment_plan = {
            "policy": {"top_k": 2},
            "stages": [
                {
                    "id": "thread-screening",
                    "candidates": [
                        {"id": "threads-4", "threads": 4},
                        {"id": "threads-8", "threads": 8},
                    ],
                }
            ],
        }
        batch_result = {
            "ranking": [
                {
                    "configuration": {
                        "batch_size": 2048,
                        "ubatch_size": 512,
                        "flash_attention": "on",
                    }
                },
                {
                    "configuration": {
                        "batch_size": 1024,
                        "ubatch_size": 256,
                        "flash_attention": "off",
                    }
                },
            ]
        }

        def fake_record(*args, **kwargs):
            configuration = kwargs["configuration"]
            return {
                "status": "ok",
                "configuration": configuration,
                "metrics": [
                    {
                        "test_kind": "prompt",
                        "tokens_per_second": 2000.0 + configuration["threads"],
                    },
                    {
                        "test_kind": "generation",
                        "tokens_per_second": 40.0 + configuration["threads"],
                    },
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "llama_autotune.run_recorded_benchmark",
                side_effect=fake_record,
            ):
                with redirect_stdout(StringIO()):
                    result = run_thread_screening(
                        Path("/opt"),
                        Path("/opt/llama-bench"),
                        Path("/models/test.gguf"),
                        capabilities,
                        tuning_plan,
                        experiment_plan,
                        batch_result,
                        Path(directory),
                        timeout=10,
                        repetitions=2,
                        prompt_tokens=4096,
                        generation_tokens=64,
                    )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total_cases"], 4)
        self.assertEqual(result["source_survivors"], 2)
        self.assertEqual(result["winner"]["configuration"]["threads"], 8)

    def test_preliminary_recommendation_is_explicitly_limited(self) -> None:
        recommendation = build_preliminary_recommendation(
            {
                "ranking": [
                    {
                        "configuration": {
                            "batch_size": 2048,
                            "ubatch_size": 512,
                            "flash_attention": "on",
                            "threads": 8,
                        },
                        "balanced_score": 1.0,
                        "prompt_tokens_per_second": 2900.0,
                        "generation_tokens_per_second": 44.0,
                    }
                ]
            }
        )
        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual(recommendation["status"], "preliminary")
        self.assertIn("wachsende Kontextgrößen", recommendation["not_yet_tested"])

    def test_ranks_context_cache_cases_within_each_context(self) -> None:
        def case(
            context_size: int,
            cache_type: str,
            batch_size: int,
            prompt: float,
            generation: float,
            wall: float,
        ) -> dict:
            return {
                "status": "ok",
                "configuration": {
                    "context_size": context_size,
                    "prompt_target": context_size - 1024,
                    "cache_type_k": cache_type,
                    "cache_type_v": cache_type,
                    "batch_size": batch_size,
                    "ubatch_size": 512,
                    "flash_attention": "on",
                    "threads": 8,
                },
                "server_run": {
                    "stages": [
                        {
                            "status": "ok",
                            "prompt_tokens_per_second": prompt,
                            "generation_tokens_per_second": generation,
                            "wall_seconds": wall,
                            "actual_prompt_tokens": context_size - 1024,
                            "finish_reason": "stop",
                        }
                    ]
                },
            }

        cases = [
            case(8192, "f16", 2048, 2500.0, 44.0, 5.0),
            case(8192, "q8_0", 4096, 2600.0, 44.2, 4.8),
            case(65536, "q8_0", 2048, 1800.0, 36.0, 25.0),
            case(65536, "q4_0", 4096, 1900.0, 37.0, 23.0),
        ]
        ranking = rank_context_cache_cases(cases)

        self.assertEqual(len(ranking["winners_by_context"]), 2)
        self.assertEqual(
            ranking["base_ranking"][0]["configuration"]["batch_size"],
            4096,
        )
        self.assertEqual(
            ranking["base_ranking"][0]["context_coverage"],
            2,
        )

    def test_context_ranking_penalizes_missing_planned_context(self) -> None:
        cases = [
            {
                "status": "ok",
                "configuration": {
                    "context_size": context_size,
                    "prompt_target": context_size - 1024,
                    "cache_type_k": "q4_0",
                    "cache_type_v": "q4_0",
                    "batch_size": 4096,
                    "ubatch_size": 512,
                    "flash_attention": "on",
                    "threads": 8,
                },
                "server_run": {
                    "stages": [
                        {
                            "status": "ok",
                            "prompt_tokens_per_second": 2000.0,
                            "generation_tokens_per_second": 100.0,
                            "wall_seconds": 5.0,
                        }
                    ]
                },
            }
            for context_size in (8192, 32768)
        ]

        ranking = rank_context_cache_cases(
            cases, planned_contexts=[8192, 32768, 65536]
        )

        self.assertEqual(ranking["validated_contexts"], [8192, 32768])
        self.assertEqual(ranking["missing_contexts"], [65536])
        self.assertEqual(ranking["base_ranking"][0]["total_contexts"], 3)
        self.assertAlmostEqual(
            ranking["base_ranking"][0]["aggregate_score"],
            2 / 3,
            places=5,
        )

    def test_context_screening_checkpoints_and_continues_after_failure(self) -> None:
        experiment_plan = {
            "policy": {"top_k": 1},
            "stages": [
                {
                    "id": "context-cache-screening",
                    "candidates": [
                        {
                            "id": "ctx-8192-f16",
                            "context_size": 8192,
                            "prompt_target": 4096,
                            "cache_type_k": "f16",
                            "cache_type_v": "f16",
                            "memory_fit": "estimated-fit",
                            "estimated_kv_gib": 0.5,
                        },
                        {
                            "id": "ctx-8192-q8_0",
                            "context_size": 8192,
                            "prompt_target": 4096,
                            "cache_type_k": "q8_0",
                            "cache_type_v": "q8_0",
                            "memory_fit": "estimated-fit",
                            "estimated_kv_gib": 0.3,
                        },
                    ],
                }
            ],
        }
        thread_result = {
            "ranking": [
                {
                    "configuration": {
                        "batch_size": 4096,
                        "ubatch_size": 512,
                        "flash_attention": "on",
                        "threads": 8,
                    }
                }
            ]
        }
        calls = 0

        def fake_growing(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"status": "failed", "stages": [], "error": "OOM"}
            return {
                "status": "ok",
                "stages": [
                    {
                        "status": "ok",
                        "prompt_tokens_per_second": 2000.0,
                        "generation_tokens_per_second": 40.0,
                        "wall_seconds": 5.0,
                        "actual_prompt_tokens": 4096,
                        "finish_reason": "stop",
                    }
                ],
                "error": None,
            }

        checkpoints = []
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "llama_autotune.run_growing_chat",
                side_effect=fake_growing,
            ):
                with redirect_stdout(StringIO()):
                    result = run_context_cache_screening(
                        Path("/opt"),
                        Path("/opt/llama-server"),
                        Path("/models/test.gguf"),
                        {},
                        {},
                        experiment_plan,
                        thread_result,
                        Path(directory),
                        startup_timeout=10,
                        request_timeout=10,
                        shutdown_timeout=5,
                        max_tokens=64,
                        reasoning_effort="low",
                        progress_callback=lambda state: checkpoints.append(
                            state["completed_cases"]
                        ),
                    )

        self.assertEqual(calls, 2)
        self.assertEqual(checkpoints, [1, 2])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["successful_cases"], 1)
        self.assertIsNotNone(result["winner"])

    def test_ranks_speculation_against_none_baseline(self) -> None:
        def case(spec_type: str, generation: float, wall: float) -> dict:
            return {
                "status": "ok",
                "configuration": {
                    "source_context_rank": 1,
                    "source_context_score": 1.0,
                    "speculation": {"spec_type": spec_type},
                },
                "server_run": {
                    "stages": [
                        {
                            "status": "ok",
                            "prompt_tokens_per_second": 1800.0,
                            "generation_tokens_per_second": generation,
                            "wall_seconds": wall,
                            "spec_draft_tokens": 200.0,
                            "spec_accepted_tokens": 150.0,
                            "spec_acceptance_ratio": 0.75,
                            "spec_drafts": 80.0,
                        }
                    ]
                },
            }

        ranking = rank_speculation_cases(
            [
                case("none", 40.0, 10.0),
                case("draft-mtp", 48.0, 8.0),
            ]
        )
        self.assertEqual(
            ranking[0]["configuration"]["speculation"]["spec_type"],
            "draft-mtp",
        )
        self.assertEqual(ranking[0]["generation_speedup_vs_none"], 1.2)

    def test_final_candidates_always_include_matching_none_control(self) -> None:
        mtp = {
            "source_context_rank": 1,
            "configuration": {
                "speculation": {
                    "id": "spec-mtp-3",
                    "spec_type": "draft-mtp",
                }
            },
        }
        baseline = {
            "source_context_rank": 1,
            "configuration": {
                "speculation": {"id": "spec-none", "spec_type": "none"}
            },
        }
        selected = final_validation_candidates(
            {"finalists": [mtp], "ranking": [mtp, baseline]}
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {item["configuration"]["speculation"]["spec_type"] for item in selected},
            {"draft-mtp", "none"},
        )

    def test_final_ranking_balances_generation_wall_and_contexts(self) -> None:
        def case(
            spec_id: str,
            spec_type: str,
            context_size: int,
            prompt: float,
            generation: float,
            wall: float,
        ) -> dict:
            return {
                "status": "ok",
                "configuration": {
                    "batch_size": 4096,
                    "ubatch_size": 512,
                    "flash_attention": "on",
                    "threads": 8,
                    "source_context_rank": 1,
                    "context_size": context_size,
                    "speculation": {"id": spec_id, "spec_type": spec_type},
                },
                "server_run": {
                    "gpu_after_ready": None,
                    "stages": [
                        {
                            "status": "ok",
                            "prompt_tokens_per_second": prompt,
                            "generation_tokens_per_second": generation,
                            "wall_seconds": wall,
                            "spec_acceptance_ratio": 0.6
                            if spec_type != "none"
                            else None,
                        }
                    ],
                },
            }

        cases = [
            case("spec-none", "none", 8192, 2600.0, 42.0, 3.0),
            case("spec-none", "none", 131072, 1800.0, 30.0, 72.0),
            case("spec-mtp-3", "draft-mtp", 8192, 2580.0, 50.0, 2.8),
            case("spec-mtp-3", "draft-mtp", 131072, 1780.0, 45.0, 77.0),
        ]
        ranking = rank_final_validation_cases(cases, expected_contexts=2)
        self.assertEqual(
            ranking[0]["configuration"]["speculation"]["spec_type"],
            "draft-mtp",
        )
        self.assertEqual(ranking[0]["context_coverage"], 2)
        self.assertGreater(ranking[0]["final_score"], ranking[1]["final_score"])

    def test_final_validation_is_partial_when_a_planned_context_is_missing(
        self,
    ) -> None:
        cases = [
            {
                "status": "ok",
                "configuration": {
                    "batch_size": 4096,
                    "ubatch_size": 512,
                    "flash_attention": "on",
                    "threads": 8,
                    "source_context_rank": 1,
                    "context_size": context_size,
                    "speculation": {"id": "spec-none", "spec_type": "none"},
                },
                "server_run": {
                    "stages": [
                        {
                            "status": "ok",
                            "prompt_tokens_per_second": 2000.0,
                            "generation_tokens_per_second": 100.0,
                            "wall_seconds": 5.0,
                        }
                    ]
                },
            }
            for context_size in (8192, 32768)
        ]

        result = summarize_final_validation(
            cases,
            total_cases=2,
            expected_contexts=3,
            objective="hermes",
            planned_contexts=[8192, 32768, 65536],
        )

        self.assertEqual(result["status"], "partial")
        self.assertFalse(result["coverage_complete"])
        self.assertEqual(result["missing_contexts"], [65536])

    def test_hermes_command_uses_131k_and_all_network_interfaces(self) -> None:
        context_results = [
            {
                "context_size": context_size,
                "prompt_tokens_per_second": 2000.0,
                "generation_tokens_per_second": 100.0,
                "wall_seconds": 5.0,
                "memory_free_mib": 3000.0,
                "acceptance_ratio": None,
                "repetitions": 2,
                "stability_score": 1.0,
                "context_score": 1.0,
            }
            for context_size in (8192, 32768)
        ]
        configuration = {
            "batch_size": 4096,
            "ubatch_size": 512,
            "flash_attention": "on",
            "threads": 8,
            "source_context_rank": 1,
            "speculation": {"id": "spec-none", "spec_type": "none"},
        }
        final_result = {
            "status": "partial",
            "optimization_objective": "hermes",
            "objective_definition": OPTIMIZATION_OBJECTIVES["hermes"],
            "planned_contexts": [8192, 32768, 65536],
            "validated_contexts": [8192, 32768],
            "missing_contexts": [65536],
            "coverage_complete": False,
            "ranking": [
                {
                    "configuration": configuration,
                    "final_score": 0.66,
                    "worst_context_score": 1.0,
                    "context_coverage": 2,
                    "total_contexts": 3,
                    "context_results": context_results,
                }
            ],
        }
        context_result = {
            "base_ranking": [
                {
                    "best_profiles": [
                        {
                            "configuration": {
                                "context_size": context_size,
                                "prompt_target": context_size - 4096,
                                "cache_type_k": "q4_0",
                                "cache_type_v": "q4_0",
                            }
                        }
                        for context_size in (8192, 32768)
                    ]
                }
            ]
        }

        recommendation = build_final_recommendation(
            final_result,
            context_result,
            Path("/opt/llama-server"),
            Path("/models/qwen.gguf"),
            {
                "server_options": {
                    "--host": True,
                    "--flash-attn": True,
                    "--threads-batch": True,
                    "--no-context-shift": True,
                }
            },
            {
                "native_context_limit": 262144,
                "deployment_host": "0.0.0.0",
                "fixed_parameters": {"gpu_layers": "all"},
            },
        )

        assert recommendation is not None
        command = recommendation["command"]
        self.assertEqual(command[command.index("--host") + 1], "0.0.0.0")
        self.assertEqual(command[command.index("--ctx-size") + 1], "131072")
        self.assertEqual(recommendation["confidence"], "limited")
        self.assertEqual(
            recommendation["coverage_status"], "coverage-limited"
        )
        self.assertFalse(recommendation["deployment_context_validated"])

        local_recommendation = build_final_recommendation(
            final_result,
            context_result,
            Path("/opt/llama-server"),
            Path("/models/qwen.gguf"),
            {
                "server_options": {
                    "--host": True,
                    "--flash-attn": True,
                    "--threads-batch": True,
                    "--no-context-shift": True,
                }
            },
            {
                "native_context_limit": 262144,
                "fixed_parameters": {"gpu_layers": "all"},
            },
        )
        assert local_recommendation is not None
        local_command = local_recommendation["command"]
        self.assertEqual(
            local_command[local_command.index("--host") + 1],
            "127.0.0.1",
        )
        self.assertNotIn(
            "Netzwerkschnittstellen", local_recommendation["warning"]
        )

    def test_final_ranking_changes_with_the_optimization_objective(self) -> None:
        def case(
            spec_id: str,
            spec_type: str,
            context_size: int,
            prompt: float,
            generation: float,
            wall: float,
        ) -> dict:
            return {
                "status": "ok",
                "configuration": {
                    "batch_size": 4096,
                    "ubatch_size": 512,
                    "flash_attention": "on",
                    "threads": 8,
                    "source_context_rank": 1,
                    "context_size": context_size,
                    "speculation": {
                        "id": spec_id,
                        "spec_type": spec_type,
                    },
                },
                "server_run": {
                    "gpu_after_ready": None,
                    "stages": [
                        {
                            "status": "ok",
                            "prompt_tokens_per_second": prompt,
                            "generation_tokens_per_second": generation,
                            "wall_seconds": wall,
                        }
                    ],
                },
            }

        cases = []
        measurements = (
            (8192, 2300.0, 44.0, 3.0, 2290.0, 80.0, 2.82),
            (65536, 2150.0, 37.0, 29.3, 2135.0, 58.0, 30.38),
            (131072, 1720.0, 31.3, 71.9, 1709.0, 46.3, 76.34),
        )
        for values in measurements:
            context_size, none_prompt, none_generation, none_wall, mtp_prompt, mtp_generation, mtp_wall = values
            cases.extend(
                [
                    case(
                        "spec-none",
                        "none",
                        context_size,
                        none_prompt,
                        none_generation,
                        none_wall,
                    ),
                    case(
                        "spec-mtp-3",
                        "draft-mtp",
                        context_size,
                        mtp_prompt,
                        mtp_generation,
                        mtp_wall,
                    ),
                ]
            )

        interactive = rank_final_validation_cases(
            cases, expected_contexts=3, objective="interactive"
        )
        long_context = rank_final_validation_cases(
            cases, expected_contexts=3, objective="long-context"
        )

        self.assertEqual(
            interactive[0]["configuration"]["speculation"]["spec_type"],
            "draft-mtp",
        )
        self.assertEqual(
            long_context[0]["configuration"]["speculation"]["spec_type"],
            "none",
        )

    def test_speculation_screening_continues_after_failed_variant(self) -> None:
        experiment_plan = {
            "policy": {"top_k": 1, "finalists": 1},
            "stages": [
                {
                    "id": "speculation-screening",
                    "candidates": [
                        {"id": "spec-none", "spec_type": "none"},
                        {
                            "id": "spec-mtp-2",
                            "spec_type": "draft-mtp",
                            "draft_n_max": 2,
                        },
                    ],
                }
            ],
        }
        base = {
            "batch_size": 4096,
            "ubatch_size": 512,
            "flash_attention": "on",
            "threads": 8,
        }
        profile = {
            "configuration": {
                **base,
                "context_size": 8192,
                "prompt_target": 4096,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
            }
        }
        context_result = {
            "base_ranking": [
                {
                    "configuration": base,
                    "aggregate_score": 1.0,
                    "best_profiles": [profile],
                }
            ]
        }
        calls = 0

        def fake_growing(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return {"status": "failed", "stages": [], "error": "probe"}
            return {
                "status": "ok",
                "stages": [
                    {
                        "status": "ok",
                        "prompt_tokens_per_second": 2000.0,
                        "generation_tokens_per_second": 40.0,
                        "wall_seconds": 5.0,
                        "spec_draft_tokens": 0.0,
                        "spec_accepted_tokens": 0.0,
                        "spec_acceptance_ratio": None,
                        "spec_drafts": 0.0,
                    }
                ],
                "error": None,
            }

        checkpoints = []
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "llama_autotune.run_growing_chat", side_effect=fake_growing
            ):
                with redirect_stdout(StringIO()):
                    result = run_speculation_screening(
                        Path("/opt"),
                        Path("/opt/llama-server"),
                        Path("/models/test.gguf"),
                        {},
                        {},
                        experiment_plan,
                        context_result,
                        Path(directory),
                        startup_timeout=10,
                        request_timeout=10,
                        shutdown_timeout=5,
                        max_tokens=64,
                        reasoning_effort="low",
                        progress_callback=lambda state: checkpoints.append(
                            state["completed_cases"]
                        ),
                    )

        self.assertEqual(calls, 2)
        self.assertEqual(checkpoints, [1, 2])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["successful_cases"], 1)
        self.assertEqual(
            result["winner"]["configuration"]["speculation"]["spec_type"],
            "none",
        )

    def test_autotune_executor_reaches_final_checkpoint(self) -> None:
        base = {
            "batch_size": 4096,
            "ubatch_size": 512,
            "flash_attention": "on",
            "threads": 8,
        }
        profile = {
            "configuration": {
                **base,
                "context_size": 8192,
                "prompt_target": 4096,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
            },
            "relative_score": 1.0,
            "prompt_tokens_per_second": 2500.0,
            "generation_tokens_per_second": 44.0,
            "wall_seconds": 5.0,
        }
        batch = {"status": "ok", "ranking": [{"configuration": base}]}
        thread = {"status": "ok", "ranking": [{"configuration": base}]}
        context = {
            "status": "ok",
            "base_ranking": [
                {
                    "configuration": base,
                    "aggregate_score": 1.0,
                    "context_coverage": 1,
                    "total_contexts": 1,
                    "best_profiles": [profile],
                }
            ],
        }
        speculation_item = {
            "configuration": {
                **base,
                "speculation": {"id": "spec-none", "spec_type": "none"},
            },
            "relative_score": 1.0,
            "generation_tokens_per_second": 44.0,
            "wall_seconds": 5.0,
            "generation_speedup_vs_none": 1.0,
            "wall_speedup_vs_none": 1.0,
            "spec_draft_tokens": 0.0,
            "spec_accepted_tokens": 0.0,
            "spec_acceptance_ratio": None,
        }
        speculation = {"status": "ok", "ranking": [speculation_item]}
        final_validation = {
            "status": "ok",
            "ranking": [
                {
                    "configuration": {
                        **base,
                        "source_context_rank": 1,
                        "speculation": {
                            "id": "spec-none",
                            "spec_type": "none",
                        },
                    },
                    "final_score": 1.0,
                    "worst_context_score": 1.0,
                    "context_coverage": 1,
                    "total_contexts": 1,
                    "context_results": [
                        {
                            "context_size": 8192,
                            "prompt_tokens_per_second": 2500.0,
                            "generation_tokens_per_second": 44.0,
                            "wall_seconds": 5.0,
                            "stability_score": 1.0,
                            "memory_free_mib": 2000.0,
                            "acceptance_ratio": None,
                            "repetitions": 1,
                        }
                    ],
                }
            ],
        }
        experiment_plan = {
            "profile": "quick",
            "policy": {"top_k": 1},
        }

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with (
                patch(
                    "llama_autotune.run_smoke_benchmark",
                    return_value={"status": "ok"},
                ),
                patch(
                    "llama_autotune.run_batch_screening",
                    return_value=batch,
                ),
                patch(
                    "llama_autotune.run_thread_screening",
                    return_value=thread,
                ),
                patch(
                    "llama_autotune.run_context_cache_screening",
                    return_value=context,
                ),
                patch(
                    "llama_autotune.run_speculation_screening",
                    return_value=speculation,
                ),
                patch(
                    "llama_autotune.run_final_validation",
                    return_value=final_validation,
                ),
                redirect_stdout(StringIO()),
            ):
                result = run_autotune_foundation(
                    Path("/opt"),
                    Path("/opt/llama-bench"),
                    Path("/opt/llama-server"),
                    Path("/models/test.gguf"),
                    {
                        "server_options": {
                            "--flash-attn": True,
                            "--threads-batch": True,
                            "--no-context-shift": True,
                        }
                    },
                    {"fixed_parameters": {"gpu_layers": "all"}},
                    experiment_plan,
                    run_dir,
                    benchmark_timeout=10,
                    smoke_repetitions=1,
                    screening_repetitions=1,
                    prompt_tokens=4096,
                    generation_tokens=64,
                    server_start_timeout=10,
                    server_request_timeout=10,
                    server_shutdown_timeout=5,
                    server_max_tokens=64,
                    reasoning_effort="low",
                )

            self.assertEqual(
                result["status"], "final-validation-complete"
            )
            self.assertTrue((run_dir / "autotune_state.json").is_file())
            self.assertTrue((run_dir / "recommendation.json").is_file())
            self.assertTrue((run_dir / "autotune_report.md").is_file())
            report = (run_dir / "autotune_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Optimierungsziel: `balanced`", report)
            self.assertIn(
                "Finalvergleich mit identischer `none`-Baseline", report
            )

    def test_writes_an_atomic_autotune_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune_state.json"
            state = {"status": "running"}
            write_autotune_checkpoint(path, state)
            loaded = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(loaded["status"], "running")
            self.assertIn("updated_at", loaded)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_rejects_a_file_without_gguf_magic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "invalid.gguf"
            model.write_bytes(b"kein GGUF")
            with self.assertRaisesRegex(ValueError, "GGUF-Signatur"):
                read_gguf_metadata(model)


if __name__ == "__main__":
    unittest.main()

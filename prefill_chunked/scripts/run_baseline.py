#!/usr/bin/env python3
"""Start the baseline vLLM OpenAI-compatible server for submission."""

from __future__ import annotations

import copy
import concurrent.futures
import json
import os
import signal
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - Python 3.9+ images should have zoneinfo.
    ZoneInfo = None  # type: ignore[assignment]

    class ZoneInfoNotFoundError(Exception):
        pass

try:
    import yaml
except ImportError as exc:  # pragma: no cover - depends on the base image.
    raise RuntimeError(
        "PyYAML is required to load the baseline YAML config. "
        "Install pyyaml in the base image or provide an image that includes it."
    ) from exc


DEFAULT_CONFIG_PATH = Path("/app/configs/baseline.yaml")
CONFIGS_DIR = Path("/app/configs")
LOCAL_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PROFILES: dict[str, str] = {
    "short_long": "config_tp2_dp2_short_long.yaml",
    "dp3_short": "config_dp3_short_dp1_long.yaml",
    "dp3_1": "config_dp3_short_dp1_long.yaml",
    "v1b": "config_v1b_2short_2long.yaml",
    "v1b_2short_2long": "config_v1b_2short_2long.yaml",
    "jit": "config_tp2_triton_attn.yaml",
    "triton_attn": "config_tp2_triton_attn.yaml",
    "native": "config_tp2_dp2_native.yaml",
    "native_dp": "config_tp2_dp2_native.yaml",
    "tp4": "config_5.yaml",
}
LOCAL_CONFIG_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml",
    Path(__file__).resolve().parents[1] / "configs" / "config_5.yaml",
    Path(__file__).resolve().parents[1] / "configs" / "config_dp3_short_dp1_long.yaml",
    Path(__file__).resolve().parents[1] / "configs" / "config_v1b_2short_2long.yaml",
    Path(__file__).resolve().parents[1] / "configs" / "config_tp2_dp2_short_long.yaml",
    Path(__file__).resolve().parents[1] / "configs" / "config_tp2_dp2_native.yaml",
    Path(__file__).resolve().parents[1] / "configs" / "config_tp2_triton_attn.yaml",
)
PATCH_SCRIPT_PATH = Path(__file__).resolve().parent / "patch_vllm_concurrent_prefill.py"
DOCKER_PATCH_SCRIPT_PATH = Path("/app/scripts/patch_vllm_concurrent_prefill.py")
DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "model_path": "/models/gpt-oss-20b",
        "served_model_name": None,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "internal_port": 8001,
    },
    "deployment": {
        "mode": "single",
        "split_prompt_tokens": 8192,
        "pools": {},
    },
    "vllm": {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "max_model_len": 28672,
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 64,
        "max_num_partial_prefills": 1,
        "enable_prefix_caching": True,
        "enable_concurrent_prefill_patch": True,
        "extra_args": [],
        # Attention backend: None = vLLM default (FLASH_ATTN).
        # Set to "triton_attn" to switch to Triton path for A/B testing.
        # Env override: BASELINE_ATTENTION_BACKEND=triton_attn
        "attention_backend": None,
        # Arbitrary extra env vars injected only into the vLLM subprocess.
        # Useful for compile-config overrides, NCCL tuning, etc.
        # Example: {"VLLM_COMPILE_CONFIG": '{"pass_config":{"fuse_gemm_comms":true}}'}
        "extra_env": {},
    },
    "logging": {
        "log_dir": "/app/logs",
        "timezone": "Asia/Ho_Chi_Minh",
        "datetime_format": "%d/%m/%Y %H:%M:%S",
    },
    "default_chat_template_kwargs": {},
    "warmup": {
        "enabled": False,
        "readiness_gate": True,
        "health_timeout_sec": 1800,
        "health_poll_interval_sec": 5,
        "request_timeout_sec": 300,
        "concurrency": 4,
        "rounds": 1,
        "stream": True,
        "token_tolerance_pct": 0.05,
        "scenarios": [],
        # Optional second phase: pre-compile Triton JIT kernels at batch sizes 1/2/4/8.
        # Scenarios with jit_compile: true (or name prefix jit_) are replayed at each size.
        "jit_compile": {
            "enabled": False,
            "batch_sizes": [1, 2, 4, 8],
            "rounds": 1,
        },
    },
}

WARMUP_SAMPLING_FIELDS = ("temperature", "top_p", "stop", "seed")


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean value: true/false, 1/0, yes/no, or on/off"
    )


def env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    return default if raw_value is None else int(raw_value)


def env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    return default if raw_value is None else float(raw_value)


def coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(
        f"{name} must be a boolean value: true/false, 1/0, yes/no, or on/off"
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config_profile_path(profile: str) -> Path | None:
    normalized = profile.strip().lower()
    filename = CONFIG_PROFILES.get(normalized)
    if not filename:
        return None
    for base in (CONFIGS_DIR, LOCAL_CONFIGS_DIR):
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


def config_path() -> Path:
    raw_path = os.environ.get("BASELINE_CONFIG_PATH")
    if raw_path:
        return Path(raw_path)

    if env_bool("BASELINE_JIT_MODE", False):
        jit_path = resolve_config_profile_path("jit")
        if jit_path is not None:
            return jit_path

    profile = os.environ.get("BASELINE_CONFIG_PROFILE", "").strip()
    if profile:
        profile_path = resolve_config_profile_path(profile)
        if profile_path is None:
            raise ValueError(
                f"Unknown BASELINE_CONFIG_PROFILE={profile!r}; "
                f"expected one of: {', '.join(sorted(CONFIG_PROFILES))}"
            )
        return profile_path

    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    for candidate in LOCAL_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    return LOCAL_CONFIG_CANDIDATES[0]


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        if os.environ.get("BASELINE_CONFIG_PATH"):
            raise FileNotFoundError(f"Baseline config not found: {path}")
        return {}
    loaded = yaml.safe_load(path.read_text("utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Baseline config must be a YAML mapping: {path}")
    return loaded


def normalize_extra_args(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("vllm.extra_args must be a list or shell-style string")


def extra_args_has_flag(extra_args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra_args)


def extra_args_flag_value(extra_args: list[str], flag: str) -> Any:
    value = None
    for index, arg in enumerate(extra_args):
        if arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
        elif arg == flag and index + 1 < len(extra_args):
            value = extra_args[index + 1]
    return value


def append_arg_unless_extra(
    command: list[str], extra_args: list[str], flag: str, value: Any
) -> None:
    if not extra_args_has_flag(extra_args, flag):
        command.extend([flag, str(value)])


def command_flag_value(command: list[str], flag: str) -> str | None:
    for index, arg in enumerate(command):
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
        if arg == flag and index + 1 < len(command):
            return command[index + 1]
    return None


def validate_serving_command(config: dict[str, Any], command: list[str]) -> list[str]:
    vllm = config["vllm"]
    extra_args = normalize_extra_args(vllm.get("extra_args"))
    warnings: list[str] = []
    checks = (
        ("--tensor-parallel-size", str(vllm["tensor_parallel_size"])),
        ("--max-num-partial-prefills", str(vllm["max_num_partial_prefills"])),
        ("--max-num-seqs", str(vllm["max_num_seqs"])),
        ("--max-model-len", str(vllm["max_model_len"])),
    )
    for flag, expected in checks:
        actual = command_flag_value(command, flag)
        if actual is None:
            warnings.append(f"missing {flag} (expected {expected})")
        elif actual != expected:
            warnings.append(f"{flag}={actual} (expected {expected})")

    patch_enabled = coerce_bool(
        vllm.get("enable_concurrent_prefill_patch", True),
        "vllm.enable_concurrent_prefill_patch",
    )
    partial_prefills = int(vllm.get("max_num_partial_prefills", 1))
    if patch_enabled and partial_prefills <= 1:
        warnings.append(
            "concurrent prefill patch is enabled but max_num_partial_prefills<=1; "
            "scheduler patch will not activate and TTFT under queue pressure will suffer"
        )
    if partial_prefills > 1:
        if "--enable-chunked-prefill" not in command:
            warnings.append(
                "max_num_partial_prefills>1 requires --enable-chunked-prefill"
            )
        if not patch_enabled:
            warnings.append(
                "concurrent prefill patch disabled while max_num_partial_prefills>1"
            )
        if not extra_args_has_flag(extra_args, "--long-prefill-token-threshold"):
            warnings.append(
                "max_num_partial_prefills>1 without --long-prefill-token-threshold; "
                "long prefills may starve short requests in the waiting queue"
            )
    return warnings


def apply_extra_arg_overrides(vllm: dict[str, Any]) -> None:
    extra_args = normalize_extra_args(vllm.get("extra_args"))
    numeric_overrides = (
        ("--tensor-parallel-size", "tensor_parallel_size", int),
        ("--max-model-len", "max_model_len", int),
        ("--gpu-memory-utilization", "gpu_memory_utilization", float),
        ("--max-num-seqs", "max_num_seqs", int),
        ("--max-num-partial-prefills", "max_num_partial_prefills", int),
        ("--data-parallel-size", "data_parallel_size", int),
    )
    for flag, key, parser in numeric_overrides:
        value = extra_args_flag_value(extra_args, flag)
        if value is not None:
            vllm[key] = parser(value)

    if extra_args_has_flag(extra_args, "--enable-prefix-caching"):
        vllm["enable_prefix_caching"] = True
    if extra_args_has_flag(extra_args, "--no-enable-prefix-caching"):
        vllm["enable_prefix_caching"] = False


def normalize_warmup_config(config: dict[str, Any]) -> None:
    warmup = config.get("warmup")
    if warmup is None:
        warmup = copy.deepcopy(DEFAULT_CONFIG["warmup"])
        config["warmup"] = warmup
    if not isinstance(warmup, dict):
        raise ValueError("warmup must be a YAML mapping")

    enabled = coerce_bool(warmup.get("enabled", False), "warmup.enabled")
    warmup["enabled"] = env_bool("BASELINE_WARMUP_ENABLED", enabled)
    gate_default = bool(warmup["enabled"])
    warmup["readiness_gate"] = env_bool(
        "BASELINE_READINESS_GATE",
        coerce_bool(warmup.get("readiness_gate", gate_default), "warmup.readiness_gate"),
    )
    if warmup["enabled"] and not warmup["readiness_gate"]:
        warmup["readiness_gate"] = True
    warmup["health_timeout_sec"] = max(
        1, int(warmup.get("health_timeout_sec", 1800))
    )
    warmup["health_poll_interval_sec"] = max(
        1, int(warmup.get("health_poll_interval_sec", 5))
    )
    warmup["request_timeout_sec"] = max(1, int(warmup.get("request_timeout_sec", 120)))
    warmup["concurrency"] = max(1, int(warmup.get("concurrency", 1)))
    warmup["rounds"] = max(0, int(warmup.get("rounds", 1)))
    warmup["stream"] = coerce_bool(warmup.get("stream", True), "warmup.stream")
    warmup["token_tolerance_pct"] = max(
        0.0, float(warmup.get("token_tolerance_pct", 0.05))
    )

    scenarios = warmup.get("scenarios") or []
    if not isinstance(scenarios, list):
        raise ValueError("warmup.scenarios must be a list")
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"warmup.scenarios[{index}] must be a YAML mapping")
        scenario["weight"] = max(1, int(scenario.get("weight", 1)))
        if "stream" in scenario:
            scenario["stream"] = coerce_bool(
                scenario["stream"], f"warmup.scenarios[{index}].stream"
            )
    warmup["scenarios"] = scenarios

    jit_default = bool(warmup["enabled"]) and env_bool("BASELINE_JIT_MODE", False)
    jit_compile = warmup.get("jit_compile")
    if jit_compile is None:
        jit_compile = copy.deepcopy(DEFAULT_CONFIG["warmup"]["jit_compile"])
        warmup["jit_compile"] = jit_compile
    if not isinstance(jit_compile, dict):
        raise ValueError("warmup.jit_compile must be a YAML mapping")
    jit_compile["enabled"] = env_bool(
        "BASELINE_JIT_WARMUP",
        coerce_bool(jit_compile.get("enabled", jit_default), "warmup.jit_compile.enabled"),
    )
    raw_batch_sizes = jit_compile.get("batch_sizes") or [1, 2, 4, 8]
    if not isinstance(raw_batch_sizes, list) or not raw_batch_sizes:
        raise ValueError("warmup.jit_compile.batch_sizes must be a non-empty list")
    jit_compile["batch_sizes"] = sorted(
        {max(1, int(size)) for size in raw_batch_sizes}
    )
    jit_compile["rounds"] = max(0, int(jit_compile.get("rounds", 1)))


def normalize_attention_backend(vllm: dict[str, Any]) -> None:
    env_backend = os.environ.get("BASELINE_ATTENTION_BACKEND", "").strip().upper()
    if env_backend:
        vllm["attention_backend"] = env_backend
    elif vllm.get("attention_backend"):
        vllm["attention_backend"] = str(vllm["attention_backend"]).strip().upper() or None
    else:
        vllm["attention_backend"] = None


def merge_extra_env(vllm: dict[str, Any]) -> None:
    if not isinstance(vllm.get("extra_env"), dict):
        vllm["extra_env"] = {}
    raw_extra_env = os.environ.get("BASELINE_EXTRA_ENV_JSON", "").strip()
    if not raw_extra_env:
        return
    import json as _json

    try:
        injected = _json.loads(raw_extra_env)
    except Exception:
        return
    if isinstance(injected, dict):
        vllm["extra_env"] = deep_merge(vllm["extra_env"], injected)


def load_effective_config() -> tuple[dict[str, Any], Path]:
    path = config_path()
    config = deep_merge(DEFAULT_CONFIG, load_yaml_config(path))

    model = config["model"]
    server = config["server"]
    vllm = config["vllm"]
    logging_config = config["logging"]

    model["model_path"] = env("MODEL_PATH", str(model["model_path"]))
    model["served_model_name"] = os.environ.get(
        "SERVED_MODEL_NAME", model.get("served_model_name")
    )
    if not model["served_model_name"]:
        model["served_model_name"] = model["model_path"]
    server["host"] = env("HOST", str(server["host"]))
    server["port"] = env_int("PORT", int(server["port"]))
    vllm["tensor_parallel_size"] = env_int(
        "TENSOR_PARALLEL_SIZE", int(vllm["tensor_parallel_size"])
    )
    vllm["max_model_len"] = env_int("MAX_MODEL_LEN", int(vllm["max_model_len"]))
    vllm["gpu_memory_utilization"] = env_float(
        "GPU_MEMORY_UTILIZATION", float(vllm["gpu_memory_utilization"])
    )
    vllm["max_num_seqs"] = env_int("MAX_NUM_SEQS", int(vllm["max_num_seqs"]))
    vllm["max_num_partial_prefills"] = env_int(
        "MAX_NUM_PARTIAL_PREFILLS", int(vllm["max_num_partial_prefills"])
    )
    vllm["data_parallel_size"] = env_int(
        "DATA_PARALLEL_SIZE", int(vllm.get("data_parallel_size", 1))
    )
    vllm["enable_prefix_caching"] = env_bool(
        "ENABLE_PREFIX_CACHING", bool(vllm["enable_prefix_caching"])
    )
    if "VLLM_EXTRA_ARGS" in os.environ:
        vllm["extra_args"] = shlex.split(os.environ["VLLM_EXTRA_ARGS"])
    else:
        vllm["extra_args"] = normalize_extra_args(vllm.get("extra_args"))
    apply_extra_arg_overrides(vllm)

    normalize_attention_backend(vllm)
    merge_extra_env(vllm)

    logging_config["log_dir"] = env("BASELINE_LOG_DIR", str(logging_config["log_dir"]))
    logging_config["timezone"] = env("BASELINE_TIMEZONE", str(logging_config["timezone"]))
    logging_config["datetime_format"] = env(
        "BASELINE_DATETIME_FORMAT", str(logging_config["datetime_format"])
    )
    normalize_warmup_config(config)
    normalize_server_config(config)
    normalize_deployment_config(config)
    return config, path


def normalize_server_config(config: dict[str, Any]) -> None:
    server = config["server"]
    public_port = int(server["port"])
    internal_default = int(server.get("internal_port", public_port + 1))
    server["internal_port"] = env_int("INTERNAL_PORT", internal_default)
    if server["internal_port"] == public_port:
        server["internal_port"] = public_port + 1


def deployment_mode(config: dict[str, Any]) -> str:
    deployment = config.get("deployment") or {}
    mode = str(deployment.get("mode", "single")).strip().lower()
    if mode in {"single", "native_dp", "short_long", "replica_short_long"}:
        return mode
    raise ValueError(
        "deployment.mode must be one of: single, native_dp, short_long, replica_short_long"
    )


def normalize_deployment_config(config: dict[str, Any]) -> None:
    deployment = config.get("deployment")
    if deployment is None:
        deployment = copy.deepcopy(DEFAULT_CONFIG["deployment"])
        config["deployment"] = deployment
    if not isinstance(deployment, dict):
        raise ValueError("deployment must be a YAML mapping")

    mode = deployment_mode(config)
    deployment["mode"] = mode
    deployment["split_prompt_tokens"] = max(
        1, int(deployment.get("split_prompt_tokens", 8192))
    )

    pools = deployment.get("pools") or {}
    if not isinstance(pools, dict):
        raise ValueError("deployment.pools must be a YAML mapping")
    if mode == "short_long":
        for pool_name in ("short", "long"):
            if pool_name not in pools:
                raise ValueError(
                    f"deployment.pools.{pool_name} is required for short_long mode"
                )
            pool = pools[pool_name]
            if not isinstance(pool, dict):
                raise ValueError(f"deployment.pools.{pool_name} must be a mapping")
            pool["internal_port"] = int(pool.get("internal_port", 8001 if pool_name == "short" else 8002))
            if "cuda_visible_devices" not in pool:
                pool["cuda_visible_devices"] = "0,1" if pool_name == "short" else "2,3"
            pool_vllm = pool.get("vllm")
            if pool_vllm is not None and not isinstance(pool_vllm, dict):
                raise ValueError(f"deployment.pools.{pool_name}.vllm must be a mapping")
    if mode == "replica_short_long":
        workers = deployment.get("workers") or {}
        if not isinstance(workers, dict) or not workers:
            raise ValueError("deployment.workers is required for replica_short_long mode")
        for worker_id, worker in workers.items():
            if not isinstance(worker, dict):
                raise ValueError(f"deployment.workers.{worker_id} must be a mapping")
            pool_name = str(worker.get("pool", "")).strip().lower()
            if pool_name not in {"short", "long"}:
                raise ValueError(
                    f"deployment.workers.{worker_id}.pool must be short or long"
                )
            worker["pool"] = pool_name
            worker["internal_port"] = int(worker["internal_port"])
            if "cuda_visible_devices" not in worker:
                raise ValueError(
                    f"deployment.workers.{worker_id}.cuda_visible_devices is required"
                )
            worker_vllm = worker.get("vllm")
            if worker_vllm is not None and not isinstance(worker_vllm, dict):
                raise ValueError(f"deployment.workers.{worker_id}.vllm must be a mapping")
        deployment["workers"] = workers
        router = deployment.get("router") or {}
        if router is not None and not isinstance(router, dict):
            raise ValueError("deployment.router must be a YAML mapping")
        deployment["router"] = router or {}
    deployment["pools"] = pools


def pool_effective_config(config: dict[str, Any], pool_name: str) -> dict[str, Any]:
    pool_config = copy.deepcopy(config)
    deployment = pool_config["deployment"]
    pool = deployment["pools"][pool_name]
    base_vllm = copy.deepcopy(config["vllm"])
    pool_vllm = copy.deepcopy(pool.get("vllm") or {})
    merged_vllm = deep_merge(base_vllm, pool_vllm)
    merged_vllm["extra_args"] = normalize_extra_args(merged_vllm.get("extra_args"))
    apply_extra_arg_overrides(merged_vllm)
    normalize_attention_backend(merged_vllm)
    merge_extra_env(merged_vllm)
    pool_config["vllm"] = merged_vllm
    return pool_config


def pool_backend_url(config: dict[str, Any], pool_name: str) -> str:
    port = int(config["deployment"]["pools"][pool_name]["internal_port"])
    return f"http://127.0.0.1:{port}"


def deployment_workers(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    workers = config["deployment"].get("workers") or {}
    if not isinstance(workers, dict):
        raise ValueError("deployment.workers must be a YAML mapping")
    return workers


def worker_effective_config(config: dict[str, Any], worker_id: str) -> dict[str, Any]:
    worker_config = copy.deepcopy(config)
    worker = deployment_workers(config)[worker_id]
    base_vllm = copy.deepcopy(config["vllm"])
    worker_vllm = copy.deepcopy(worker.get("vllm") or {})
    merged_vllm = deep_merge(base_vllm, worker_vllm)
    merged_vllm["extra_args"] = normalize_extra_args(merged_vllm.get("extra_args"))
    apply_extra_arg_overrides(merged_vllm)
    normalize_attention_backend(merged_vllm)
    merge_extra_env(merged_vllm)
    worker_config["vllm"] = merged_vllm
    return worker_config


def worker_backend_url(config: dict[str, Any], worker_id: str) -> str:
    port = int(deployment_workers(config)[worker_id]["internal_port"])
    return f"http://127.0.0.1:{port}"


def scenario_target_pool(
    scenario: dict[str, Any], config: dict[str, Any]
) -> str:
    explicit = scenario.get("pool")
    if explicit in {"short", "long"}:
        return str(explicit)
    prompt_tokens = scenario.get("prompt_tokens")
    if prompt_tokens is not None:
        split = int(config["deployment"]["split_prompt_tokens"])
        return "long" if int(prompt_tokens) >= split else "short"
    return "short"


def uses_readiness_gate(config: dict[str, Any]) -> bool:
    warmup = config.get("warmup") or {}
    return bool(warmup.get("enabled")) and bool(warmup.get("readiness_gate", True))


def vllm_listen_target(config: dict[str, Any]) -> tuple[str, int]:
    server = config["server"]
    if uses_readiness_gate(config):
        return "127.0.0.1", int(server["internal_port"])
    return str(server["host"]), int(server["port"])


def inference_backend_url(config: dict[str, Any]) -> str:
    host, port = vllm_listen_target(config)
    return f"http://{host}:{port}"


def public_server_url(config: dict[str, Any]) -> str:
    return f"http://127.0.0.1:{int(config['server']['port'])}"


def timezone_for(name: str) -> tzinfo:
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            pass
    if name == "Asia/Ho_Chi_Minh":
        return timezone(timedelta(hours=7), name)
    return timezone.utc


def patch_script_path() -> Path | None:
    for candidate in (DOCKER_PATCH_SCRIPT_PATH, PATCH_SCRIPT_PATH):
        if candidate.exists():
            return candidate
    return None


def apply_vllm_runtime_patches(config: dict[str, Any]) -> None:
    vllm = config.get("vllm") or {}
    if not coerce_bool(vllm.get("enable_concurrent_prefill_patch", True), "vllm.enable_concurrent_prefill_patch"):
        log_startup(config, "Concurrent prefill patch disabled")
        return
    script = patch_script_path()
    if script is None:
        log_startup(config, "Concurrent prefill patch script not found; skipping")
        return
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (result.stdout or "").splitlines():
        if line.strip():
            log_startup(config, line)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(f"failed to patch vLLM for concurrent partial prefill: {details}")

    check = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    patch_status = (check.stdout or "").strip()
    log_startup(config, f"Concurrent prefill patch status: {patch_status or 'unknown'}")
    max_partial_prefills = int(config.get("vllm", {}).get("max_num_partial_prefills", 1))
    if max_partial_prefills > 1 and check.returncode != 0:
        raise RuntimeError(
            "concurrent partial prefill patch is required but not active "
            f"(max_num_partial_prefills={max_partial_prefills})"
        )


_VALID_ATTENTION_BACKENDS = frozenset({"FLASH_ATTN", "TRITON_ATTN"})


def server_process_env(vllm_config: dict[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # Keep collectives predictable and avoid CPU oversubscription.
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("NCCL_NVLS_ENABLE", "1")

    if vllm_config:
        # Attention backend override (maps to VLLM_ATTENTION_BACKEND).
        attn_backend = (vllm_config.get("attention_backend") or "").strip().upper()
        if attn_backend:
            if attn_backend not in _VALID_ATTENTION_BACKENDS:
                raise ValueError(
                    f"vllm.attention_backend must be one of "
                    f"{sorted(_VALID_ATTENTION_BACKENDS)}, got: {attn_backend!r}"
                )
            env["VLLM_ATTENTION_BACKEND"] = attn_backend

        # Arbitrary extra env vars for the vLLM subprocess.
        extra_env = vllm_config.get("extra_env") or {}
        if not isinstance(extra_env, dict):
            raise ValueError("vllm.extra_env must be a YAML mapping")
        for key, value in extra_env.items():
            env[str(key)] = str(value)

    return env


def server_process_env_for_pool(
    cuda_visible_devices: str | None = None,
    vllm_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    env = server_process_env(vllm_config)
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def start_vllm_process(
    command: list[str],
    *,
    cuda_visible_devices: str | None = None,
    vllm_config: dict[str, Any] | None = None,
) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        command,
        env=server_process_env_for_pool(cuda_visible_devices, vllm_config),
    )


def terminate_servers(servers: list[subprocess.Popen[Any]]) -> None:
    for server in servers:
        terminate_server(server)


def install_signal_forwarders_multi(
    servers: list[subprocess.Popen[Any]],
    gate_server: Any | None = None,
    *,
    gate_kind: str = "readiness",
) -> dict[int, signal.Handlers]:
    previous_handlers: dict[int, signal.Handlers] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}; stopping vLLM servers", flush=True)
        if gate_server is not None:
            if gate_kind == "length_router":
                from length_router import stop_length_router

                stop_length_router(gate_server)
            elif gate_kind == "slo_router":
                from slo_router import stop_slo_router

                stop_slo_router(gate_server)
            else:
                from readiness_gate import stop_readiness_gate

                stop_readiness_gate(gate_server)
        terminate_servers(servers)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)
    return previous_handlers


def run_short_long_deployment(config: dict[str, Any], path: Path) -> int:
    deployment = config["deployment"]
    split = int(deployment["split_prompt_tokens"])
    pools = deployment["pools"]
    commands: dict[str, list[str]] = {}
    warnings: list[str] = []

    for pool_name in ("short", "long"):
        pool_cfg = pool_effective_config(config, pool_name)
        port = int(pools[pool_name]["internal_port"])
        command = build_command(
            pool_cfg, listen_host="127.0.0.1", listen_port=port
        )
        commands[pool_name] = command
        warnings.extend(
            f"[{pool_name}] {item}"
            for item in validate_serving_command(pool_cfg, command)
        )

    experiment = write_startup_artifacts(
        config,
        path,
        commands["short"] + ["# long:"] + commands["long"],
    )
    jit_compile = (config.get("warmup") or {}).get("jit_compile") or {}
    if jit_compile.get("enabled"):
        log_startup(
            config,
            (
                "JIT compile warmup enabled: "
                f"batch_sizes={jit_compile.get('batch_sizes')}"
            ),
        )
    patch_config = copy.deepcopy(config)
    patch_config["vllm"]["max_num_partial_prefills"] = max(
        int(pool_effective_config(config, pool)["vllm"]["max_num_partial_prefills"])
        for pool in ("short", "long")
    )
    apply_vllm_runtime_patches(patch_config)

    log_startup(
        config,
        (
            f"Short/long deployment: split>={split} tokens -> long pool; "
            f"short GPUs={pools['short']['cuda_visible_devices']} "
            f"TP={pool_effective_config(config, 'short')['vllm']['tensor_parallel_size']}"
            f" DP={pool_effective_config(config, 'short')['vllm'].get('data_parallel_size', 1)} "
            f"port={pools['short']['internal_port']}, "
            f"long GPUs={pools['long']['cuda_visible_devices']} "
            f"TP={pool_effective_config(config, 'long')['vllm']['tensor_parallel_size']}"
            f" DP={pool_effective_config(config, 'long')['vllm'].get('data_parallel_size', 1)} "
            f"port={pools['long']['internal_port']}"
        ),
    )

    servers: list[subprocess.Popen[Any]] = []
    router_server = None
    gate_state = None
    try:
        for pool_name in ("short", "long"):
            pool = pools[pool_name]
            command = commands[pool_name]
            pool_vllm = pool_effective_config(config, pool_name)["vllm"]
            attn_backend = pool_vllm.get("attention_backend")
            if attn_backend:
                log_startup(
                    config,
                    f"[{pool_name}] Attention backend: VLLM_ATTENTION_BACKEND={attn_backend}",
                )
            log_startup(config, f"Starting {pool_name} pool: {shlex.join(command)}")
            servers.append(
                start_vllm_process(
                    command,
                    cuda_visible_devices=str(pool["cuda_visible_devices"]),
                    vllm_config=pool_vllm,
                )
            )

        from length_router import ReadinessState, start_length_router

        gate_state = ReadinessState()
        short_port = int(pools["short"]["internal_port"])
        long_port = int(pools["long"]["internal_port"])
        model_name = str(
            config["model"].get("served_model_name") or config["model"]["model_path"]
        )
        router_server = start_length_router(
            public_host=str(config["server"]["host"]),
            public_port=int(config["server"]["port"]),
            short_host="127.0.0.1",
            short_port=short_port,
            long_host="127.0.0.1",
            long_port=long_port,
            split_prompt_tokens=split,
            model_name=model_name,
            state=gate_state,
            log=lambda message: log_startup(config, message),
        )

        for warning in warnings:
            log_startup(config, f"Config warning: {warning}")

        previous_handlers = install_signal_forwarders_multi(
            servers, router_server, gate_kind="length_router"
        )
        try:
            warmup = config["warmup"]
            if warmup["enabled"]:
                for pool_name in ("short", "long"):
                    backend_url = pool_backend_url(config, pool_name)
                    log_startup(
                        config,
                        f"Waiting for {pool_name} pool health at {backend_url}/health",
                    )
                    wait_for_health(
                        backend_url,
                        int(warmup["health_timeout_sec"]),
                        int(warmup["health_poll_interval_sec"]),
                        servers[["short", "long"].index(pool_name)],
                    )

                for pool_name in ("short", "long"):
                    backend_url = pool_backend_url(config, pool_name)
                    log_startup(config, f"Warmup {pool_name} pool at {backend_url}")
                    run_all_warmup_phases(
                        config,
                        server_url=backend_url,
                        scenario_filter=lambda scenario, pool=pool_name: (
                            scenario_target_pool(scenario, config) == pool
                        ),
                        label=f"Warmup {pool_name}",
                    )

                gate_state.open()
                log_startup(
                    config,
                    (
                        "Length router open: "
                        f"http://{config['server']['host']}:{config['server']['port']}/health"
                    ),
                )
            else:
                log_startup(config, "Warmup disabled")

            exit_code = servers[0].wait()
            for server in servers[1:]:
                code = server.wait()
                if exit_code == 0:
                    exit_code = code
            return exit_code
        except Exception:
            terminate_servers(servers)
            raise
        finally:
            restore_signal_handlers(previous_handlers)
    finally:
        if router_server is not None:
            from length_router import stop_length_router

            stop_length_router(router_server)


def run_replica_short_long_deployment(config: dict[str, Any], path: Path) -> int:
    deployment = config["deployment"]
    split = int(deployment["split_prompt_tokens"])
    workers = deployment_workers(config)
    router_cfg = deployment.get("router") or {}
    prefix_locality = bool(router_cfg.get("prefix_locality", True))
    commands: dict[str, list[str]] = {}
    warnings: list[str] = []

    for worker_id, worker in workers.items():
        worker_cfg = worker_effective_config(config, worker_id)
        port = int(worker["internal_port"])
        command = build_command(
            worker_cfg, listen_host="127.0.0.1", listen_port=port
        )
        commands[worker_id] = command
        warnings.extend(
            f"[{worker_id}] {item}"
            for item in validate_serving_command(worker_cfg, command)
        )

    experiment = write_startup_artifacts(
        config,
        path,
        [f"{worker_id}: {shlex.join(command)}" for worker_id, command in commands.items()],
    )
    patch_config = copy.deepcopy(config)
    patch_config["vllm"]["max_num_partial_prefills"] = max(
        int(worker_effective_config(config, worker_id)["vllm"]["max_num_partial_prefills"])
        for worker_id in workers
    )
    apply_vllm_runtime_patches(patch_config)

    worker_summary = ", ".join(
        f"{worker_id} pool={workers[worker_id]['pool']} "
        f"GPUs={workers[worker_id]['cuda_visible_devices']} "
        f"port={workers[worker_id]['internal_port']}"
        for worker_id in workers
    )
    log_startup(
        config,
        (
            f"Replica short/long deployment: split>={split} tokens; "
            f"SLO router prefix_locality={prefix_locality}; {worker_summary}"
        ),
    )

    servers: list[subprocess.Popen[Any]] = []
    router_server = None
    gate_state = None
    worker_ids = list(workers.keys())
    try:
        for worker_id in worker_ids:
            worker = workers[worker_id]
            command = commands[worker_id]
            worker_vllm = worker_effective_config(config, worker_id)["vllm"]
            attn_backend = worker_vllm.get("attention_backend")
            if attn_backend:
                log_startup(
                    config,
                    f"[{worker_id}] Attention backend: VLLM_ATTENTION_BACKEND={attn_backend}",
                )
            log_startup(config, f"Starting {worker_id}: {shlex.join(command)}")
            servers.append(
                start_vllm_process(
                    command,
                    cuda_visible_devices=str(worker["cuda_visible_devices"]),
                    vllm_config=worker_vllm,
                )
            )

        from routing_utils import WorkerEndpoint
        from slo_router import ReadinessState, start_slo_router

        gate_state = ReadinessState()
        model_name = str(
            config["model"].get("served_model_name") or config["model"]["model_path"]
        )
        worker_endpoints = [
            WorkerEndpoint(
                worker_id=worker_id,
                pool=str(workers[worker_id]["pool"]),
                host="127.0.0.1",
                port=int(workers[worker_id]["internal_port"]),
            )
            for worker_id in worker_ids
        ]
        router_server = start_slo_router(
            public_host=str(config["server"]["host"]),
            public_port=int(config["server"]["port"]),
            workers=worker_endpoints,
            split_prompt_tokens=split,
            model_name=model_name,
            state=gate_state,
            prefix_locality=prefix_locality,
            log=lambda message: log_startup(config, message),
        )

        for warning in warnings:
            log_startup(config, f"Config warning: {warning}")

        previous_handlers = install_signal_forwarders_multi(
            servers, router_server, gate_kind="slo_router"
        )
        try:
            warmup = config["warmup"]
            if warmup["enabled"]:
                for index, worker_id in enumerate(worker_ids):
                    backend_url = worker_backend_url(config, worker_id)
                    log_startup(
                        config,
                        f"Waiting for {worker_id} health at {backend_url}/health",
                    )
                    wait_for_health(
                        backend_url,
                        int(warmup["health_timeout_sec"]),
                        int(warmup["health_poll_interval_sec"]),
                        servers[index],
                    )

                for worker_id in worker_ids:
                    target_pool = str(workers[worker_id]["pool"])
                    backend_url = worker_backend_url(config, worker_id)
                    log_startup(config, f"Warmup {worker_id} at {backend_url}")
                    run_all_warmup_phases(
                        config,
                        server_url=backend_url,
                        scenario_filter=lambda scenario, pool=target_pool: (
                            scenario_target_pool(scenario, config) == pool
                        ),
                        label=f"Warmup {worker_id}",
                    )

                gate_state.open()
                log_startup(
                    config,
                    (
                        "SLO router open: "
                        f"http://{config['server']['host']}:{config['server']['port']}/health"
                    ),
                )
            else:
                log_startup(config, "Warmup disabled")

            exit_code = servers[0].wait()
            for server in servers[1:]:
                code = server.wait()
                if exit_code == 0:
                    exit_code = code
            return exit_code
        except Exception:
            terminate_servers(servers)
            raise
        finally:
            restore_signal_handlers(previous_handlers)
    finally:
        if router_server is not None:
            from slo_router import stop_slo_router

            stop_slo_router(router_server)


def build_command(
    config: dict[str, Any],
    *,
    listen_host: str | None = None,
    listen_port: int | None = None,
) -> list[str]:
    model = config["model"]
    server = config["server"]
    vllm = config["vllm"]
    model_path = str(model["model_path"])
    served_model_name = model.get("served_model_name") or model_path
    extra_args = normalize_extra_args(vllm.get("extra_args"))
    if listen_host is None or listen_port is None:
        listen_host, listen_port = vllm_listen_target(config)

    command = [
        "vllm",
        "serve",
        model_path,
        "--host",
        str(listen_host),
        "--port",
        str(listen_port),
        "--served-model-name",
        str(served_model_name),
    ]
    append_arg_unless_extra(
        command, extra_args, "--tensor-parallel-size", vllm["tensor_parallel_size"]
    )
    append_arg_unless_extra(
        command,
        extra_args,
        "--data-parallel-size",
        vllm.get("data_parallel_size", 1),
    )
    append_arg_unless_extra(command, extra_args, "--max-model-len", vllm["max_model_len"])
    append_arg_unless_extra(
        command, extra_args, "--gpu-memory-utilization", vllm["gpu_memory_utilization"]
    )
    append_arg_unless_extra(command, extra_args, "--max-num-seqs", vllm["max_num_seqs"])
    append_arg_unless_extra(
        command,
        extra_args,
        "--max-num-partial-prefills",
        vllm["max_num_partial_prefills"],
    )
    if not (
        extra_args_has_flag(extra_args, "--enable-prefix-caching")
        or extra_args_has_flag(extra_args, "--no-enable-prefix-caching")
    ):
        command.append(
            "--enable-prefix-caching"
            if vllm["enable_prefix_caching"]
            else "--no-enable-prefix-caching"
        )
    command.extend(extra_args)
    return command


def write_startup_artifacts(
    config: dict[str, Any], config_path_used: Path, command: list[str]
) -> dict[str, Any]:
    logging_config = config["logging"]
    log_dir = Path(str(logging_config["log_dir"]))
    log_dir.mkdir(parents=True, exist_ok=True)

    tz_name = str(logging_config["timezone"])
    now = datetime.now(timezone_for(tz_name))
    datetime_format = str(logging_config["datetime_format"])
    experiment = {
        "experiment_id": (
            f"vllm-baseline-{now.strftime('%Y%m%d_%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        ),
        "started_at": now.strftime(datetime_format),
        "started_date": now.strftime("%d/%m/%Y"),
        "started_time": now.strftime("%H:%M:%S"),
        "timezone": tz_name,
        "config_path": str(config_path_used),
        "command": command,
        "command_string": shlex.join(command),
    }

    (log_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), "utf-8"
    )
    (log_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    (log_dir / "startup.log").write_text(
        "\n".join(
            [
                f"started_at={experiment['started_at']}",
                f"timezone={experiment['timezone']}",
                f"config_path={experiment['config_path']}",
                f"command={experiment['command_string']}",
                "",
            ]
        ),
        "utf-8",
    )
    return experiment


def append_startup_log(config: dict[str, Any], lines: list[str]) -> None:
    log_dir = Path(str(config["logging"]["log_dir"]))
    with (log_dir / "startup.log").open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def log_startup(config: dict[str, Any], message: str) -> None:
    print(message, flush=True)
    append_startup_log(config, [message])


def local_server_url(config: dict[str, Any]) -> str:
    return inference_backend_url(config)


def wait_for_health(
    server_url: str,
    timeout_sec: int,
    poll_interval_sec: int,
    server_process: subprocess.Popen[Any] | None = None,
) -> None:
    health_url = f"{server_url.rstrip('/')}/health"
    deadline = time.monotonic() + timeout_sec
    last_error = ""

    while time.monotonic() < deadline:
        if server_process is not None and server_process.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited before becoming healthy "
                f"(code={server_process.returncode})"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=poll_interval_sec) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(poll_interval_sec)

    raise TimeoutError(
        f"Timed out waiting {timeout_sec}s for {health_url}; last error: {last_error}"
    )


def warmup_prompt(prompt_chars: int, max_tokens: int | None = None) -> str:
    prompt_chars = max(1, int(prompt_chars))
    budget = (
        f"Write close to {max(1, int(max_tokens))} output tokens. "
        if max_tokens is not None
        else ""
    )
    seed = (
        "Please produce a steady warmup answer after reading this prompt. "
        f"{budget}Use the requested response budget so the decode path is exercised. "
    )
    repeats = (prompt_chars // len(seed)) + 1
    return (seed * repeats)[:prompt_chars]


def warmup_prompt_for_tokens(
    target_tokens: int,
    max_tokens: int,
    server_url: str | None = None,
    config: dict[str, Any] | None = None,
    timeout_sec: int = 30,
    token_tolerance_pct: float = 0.05,
    prompt_cache: dict[int, str] | None = None,
) -> tuple[str, str | None]:
    target_tokens = max(1, int(target_tokens))
    if prompt_cache is not None and target_tokens in prompt_cache:
        return prompt_cache[target_tokens], None

    prompt = warmup_prompt(target_tokens * 4, max_tokens)
    if server_url is None or config is None:
        if prompt_cache is not None:
            prompt_cache[target_tokens] = prompt
        return prompt, "tokenize unavailable; used char estimate"

    tolerance = max(1, int(target_tokens * max(0.0, token_tolerance_pct)))
    warning = None
    low_chars = 1
    high_chars = max(64, target_tokens * 8)
    last_prompt = prompt

    try:
        low_count = tokenize_prompt(
            server_url, config, warmup_prompt(low_chars, max_tokens), timeout_sec
        )
        high_count = tokenize_prompt(
            server_url, config, warmup_prompt(high_chars, max_tokens), timeout_sec
        )
        while high_count < target_tokens and high_chars < target_tokens * 32:
            low_chars = high_chars
            low_count = high_count
            high_chars *= 2
            high_count = tokenize_prompt(
                server_url, config, warmup_prompt(high_chars, max_tokens), timeout_sec
            )

        for _ in range(12):
            mid_chars = max(1, (low_chars + high_chars) // 2)
            candidate = warmup_prompt(mid_chars, max_tokens)
            count = tokenize_prompt(server_url, config, candidate, timeout_sec)
            last_prompt = candidate
            if abs(count - target_tokens) <= tolerance:
                if prompt_cache is not None:
                    prompt_cache[target_tokens] = candidate
                return candidate, None
            if count < target_tokens:
                low_chars = mid_chars + 1
                low_count = count
            else:
                high_chars = mid_chars - 1

        low_delta = abs(low_count - target_tokens)
        high_delta = abs(high_count - target_tokens)
        prompt = warmup_prompt(
            low_chars if low_delta <= high_delta else high_chars, max_tokens
        )
        warning = (
            f"token calibration ended outside tolerance for target={target_tokens}"
        )
    except Exception as exc:
        prompt = warmup_prompt(target_tokens * 4, max_tokens)
        warning = f"token calibration failed for target={target_tokens}: {exc}"

    if not prompt:
        prompt = last_prompt
    if prompt_cache is not None:
        prompt_cache[target_tokens] = prompt
    return prompt, warning


def tokenize_prompt(
    server_url: str,
    config: dict[str, Any],
    prompt: str,
    timeout_sec: int,
) -> int:
    model = config["model"]
    model_name = model.get("served_model_name") or model["model_path"]
    messages = [{"role": "user", "content": prompt}]
    payloads = [
        {"model": str(model_name), "messages": messages},
        {"model": str(model_name), "prompt": prompt},
    ]
    last_error: Exception | None = None
    for payload in payloads:
        try:
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                f"{server_url.rstrip('/')}/tokenize",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                body = response.read().decode("utf-8")
                if not (200 <= response.status < 300):
                    raise RuntimeError(f"HTTP {response.status}")
            return parse_token_count(json.loads(body))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error) if last_error is not None else "tokenize failed")


def parse_token_count(payload: Any) -> int:
    if not isinstance(payload, dict):
        raise ValueError("tokenize response must be a JSON object")
    for key in ("count", "num_tokens", "token_count"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    for key in ("tokens", "token_ids", "input_ids"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    raise ValueError("tokenize response does not include a token count")


def build_warmup_payload(
    config: dict[str, Any],
    scenario: dict[str, Any],
    server_url: str | None = None,
    timeout_sec: int = 30,
    prompt_cache: dict[int, str] | None = None,
) -> dict[str, Any]:
    model = config["model"]
    warmup = config.get("warmup") or {}
    model_name = model.get("served_model_name") or model["model_path"]
    max_tokens = int(scenario.get("max_tokens", 16))
    if "prompt_tokens" in scenario:
        prompt, warning = warmup_prompt_for_tokens(
            int(scenario["prompt_tokens"]),
            max_tokens,
            server_url,
            config,
            timeout_sec,
            float(warmup.get("token_tolerance_pct", 0.05)),
            prompt_cache,
        )
        if warning:
            scenario["_warmup_warning"] = warning
    else:
        prompt = warmup_prompt(int(scenario.get("prompt_chars", 400)), max_tokens)

    payload: dict[str, Any] = {
        "model": str(model_name),
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "stream": bool(scenario.get("stream", warmup.get("stream", True))),
    }
    for field in WARMUP_SAMPLING_FIELDS:
        if field in scenario:
            payload[field] = scenario[field]

    default_kwargs = config.get("default_chat_template_kwargs") or {}
    if not isinstance(default_kwargs, dict):
        raise ValueError("default_chat_template_kwargs must be a YAML mapping")
    chat_template_kwargs = copy.deepcopy(default_kwargs)
    if "reasoning_effort" in scenario:
        chat_template_kwargs["reasoning_effort"] = scenario["reasoning_effort"]
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    return payload


def send_warmup_request(
    server_url: str,
    config: dict[str, Any],
    scenario: dict[str, Any],
    timeout_sec: int,
    prompt_cache: dict[int, str] | None = None,
) -> dict[str, Any]:
    scenario_name = str(scenario.get("name", "unnamed"))
    payload = build_warmup_payload(
        config, scenario, server_url, timeout_sec, prompt_cache
    )
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            response.read()
            if 200 <= response.status < 300:
                result = {
                    "ok": True,
                    "scenario": scenario_name,
                    "status": response.status,
                }
                if scenario.get("_warmup_warning"):
                    result["warning"] = scenario["_warmup_warning"]
                return result
            return {
                "ok": False,
                "scenario": scenario_name,
                "error": f"HTTP {response.status}",
            }
    except Exception as exc:  # Warmup failures should not kill the server.
        return {"ok": False, "scenario": scenario_name, "error": str(exc)}


def is_jit_warmup_scenario(scenario: dict[str, Any]) -> bool:
    if "jit_compile" in scenario:
        return coerce_bool(scenario["jit_compile"], "warmup.scenario.jit_compile")
    name = str(scenario.get("name", ""))
    return name.startswith("jit_seed_") or name.startswith("jit_")


def empty_warmup_summary() -> dict[str, Any]:
    return {
        "total": 0,
        "ok": 0,
        "failed": 0,
        "failures": [],
        "warnings": [],
        "elapsed_sec": 0.0,
    }


def summarize_warmup_results(
    results: list[dict[str, Any]], started: float
) -> dict[str, Any]:
    failures = [result for result in results if not result["ok"]]
    warnings = [result for result in results if result.get("warning")]
    return {
        "total": len(results),
        "ok": sum(1 for result in results if result["ok"]),
        "failed": len(failures),
        "failures": failures,
        "warnings": warnings,
        "elapsed_sec": time.monotonic() - started,
    }


def log_warmup_summary(
    config: dict[str, Any],
    label: str,
    summary: dict[str, Any],
) -> None:
    log_startup(
        config,
        (
            f"{label}: ok={summary['ok']} failed={summary['failed']} "
            f"total={summary['total']} elapsed_sec={summary['elapsed_sec']:.2f}"
        ),
    )
    for failure in summary["failures"][:10]:
        log_startup(
            config,
            (
                f"{label} warning: scenario={failure['scenario']} "
                f"error={failure.get('error', '')}"
            ),
        )
    if len(summary["failures"]) > 10:
        log_startup(
            config,
            f"{label} warning: {len(summary['failures']) - 10} more failures",
        )


def run_warmup(
    config: dict[str, Any],
    *,
    server_url: str | None = None,
    scenario_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    warmup = config["warmup"]
    scenarios = warmup["scenarios"]
    if scenario_filter is not None:
        scenarios = [scenario for scenario in scenarios if scenario_filter(scenario)]
    jit_compile = warmup.get("jit_compile") or {}
    if jit_compile.get("enabled"):
        scenarios = [
            scenario for scenario in scenarios if not is_jit_warmup_scenario(scenario)
        ]
    rounds = int(warmup["rounds"])
    if rounds <= 0 or not scenarios:
        return empty_warmup_summary()

    backend_url = server_url or local_server_url(config)
    timeout_sec = int(warmup["request_timeout_sec"])
    prompt_cache: dict[int, str] = {}
    for scenario in scenarios:
        if "prompt_tokens" not in scenario:
            continue
        _, warning = warmup_prompt_for_tokens(
            int(scenario["prompt_tokens"]),
            int(scenario.get("max_tokens", 16)),
            backend_url,
            config,
            timeout_sec,
            float(warmup.get("token_tolerance_pct", 0.05)),
            prompt_cache,
        )
        if warning:
            scenario["_warmup_warning"] = warning

    tasks = [
        scenario
        for _ in range(rounds)
        for scenario in scenarios
        for _ in range(int(scenario.get("weight", 1)))
    ]
    max_workers = min(int(warmup["concurrency"]), len(tasks))
    started = time.monotonic()
    results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                send_warmup_request,
                backend_url,
                config,
                scenario,
                timeout_sec,
                prompt_cache,
            )
            for scenario in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return summarize_warmup_results(results, started)


def run_jit_compile_warmup(
    config: dict[str, Any],
    *,
    server_url: str | None = None,
    scenario_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    warmup = config["warmup"]
    jit_compile = warmup.get("jit_compile") or {}
    if not jit_compile.get("enabled"):
        return empty_warmup_summary()

    scenarios = [
        scenario
        for scenario in warmup["scenarios"]
        if is_jit_warmup_scenario(scenario)
    ]
    if scenario_filter is not None:
        scenarios = [scenario for scenario in scenarios if scenario_filter(scenario)]
    rounds = int(jit_compile.get("rounds", 1))
    batch_sizes = [int(size) for size in jit_compile.get("batch_sizes") or []]
    if rounds <= 0 or not scenarios or not batch_sizes:
        return empty_warmup_summary()

    backend_url = server_url or local_server_url(config)
    timeout_sec = int(warmup["request_timeout_sec"])
    prompt_cache: dict[int, str] = {}
    for scenario in scenarios:
        if "prompt_tokens" not in scenario:
            continue
        _, warning = warmup_prompt_for_tokens(
            int(scenario["prompt_tokens"]),
            int(scenario.get("max_tokens", 16)),
            backend_url,
            config,
            timeout_sec,
            float(warmup.get("token_tolerance_pct", 0.05)),
            prompt_cache,
        )
        if warning:
            scenario["_warmup_warning"] = warning

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        for _ in range(rounds):
            for scenario in scenarios:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=batch_size
                ) as executor:
                    futures = [
                        executor.submit(
                            send_warmup_request,
                            backend_url,
                            config,
                            scenario,
                            timeout_sec,
                            prompt_cache,
                        )
                        for _ in range(batch_size)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        result["jit_batch_size"] = batch_size
                        results.append(result)

    return summarize_warmup_results(results, started)


def run_all_warmup_phases(
    config: dict[str, Any],
    *,
    server_url: str | None = None,
    scenario_filter: Callable[[dict[str, Any]], bool] | None = None,
    label: str = "Warmup",
) -> None:
    summary = run_warmup(
        config,
        server_url=server_url,
        scenario_filter=scenario_filter,
    )
    log_warmup_summary(config, label, summary)

    jit_compile = (config.get("warmup") or {}).get("jit_compile") or {}
    if not jit_compile.get("enabled"):
        return

    jit_summary = run_jit_compile_warmup(
        config,
        server_url=server_url,
        scenario_filter=scenario_filter,
    )
    batch_sizes = jit_compile.get("batch_sizes") or []
    log_warmup_summary(
        config,
        f"{label} JIT compile (batch_sizes={batch_sizes})",
        jit_summary,
    )


def terminate_server(server: subprocess.Popen[Any]) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def install_signal_forwarders(
    server: subprocess.Popen[Any],
    gate_server: Any | None = None,
) -> dict[int, signal.Handlers]:
    previous_handlers: dict[int, signal.Handlers] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        print(f"Received signal {signum}; stopping vLLM server", flush=True)
        if gate_server is not None:
            from readiness_gate import stop_readiness_gate

            stop_readiness_gate(gate_server)
        terminate_server(server)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)
    return previous_handlers


def restore_signal_handlers(previous_handlers: dict[int, signal.Handlers]) -> None:
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def run_single_deployment(config: dict[str, Any], path: Path) -> int:
    command = build_command(config)
    warnings = validate_serving_command(config, command)
    experiment = write_startup_artifacts(config, path, command)

    apply_vllm_runtime_patches(config)
    jit_compile = (config.get("warmup") or {}).get("jit_compile") or {}
    if jit_compile.get("enabled"):
        log_startup(
            config,
            (
                "JIT compile warmup enabled: "
                f"batch_sizes={jit_compile.get('batch_sizes')}"
            ),
        )

    gate_enabled = uses_readiness_gate(config)
    if gate_enabled:
        public_port = int(config["server"]["port"])
        internal_port = int(config["server"]["internal_port"])
        log_startup(
            config,
            (
                "Readiness gate enabled: vLLM listens on "
                f"127.0.0.1:{internal_port}, public port {public_port} "
                "returns 503 until warmup completes"
            ),
        )

    mode = deployment_mode(config)
    if mode == "native_dp":
        dp = int(config["vllm"].get("data_parallel_size", 1))
        log_startup(
            config,
            f"Native DP deployment: tensor_parallel={config['vllm']['tensor_parallel_size']} "
            f"data_parallel={dp} (each DP rank has its own KV cache)",
        )

    print("Starting vLLM server:", shlex.join(command), flush=True)
    for warning in warnings:
        log_startup(config, f"Config warning: {warning}")
    attn_backend = (config.get("vllm") or {}).get("attention_backend")
    if attn_backend:
        log_startup(config, f"Attention backend override: VLLM_ATTENTION_BACKEND={attn_backend}")
    print(
        "Experiment log:",
        Path(str(config["logging"]["log_dir"])),
        experiment["started_at"],
        flush=True,
    )
    server = subprocess.Popen(command, env=server_process_env(config.get("vllm")))
    gate_server = None
    gate_state = None
    try:
        if gate_enabled:
            from readiness_gate import ReadinessState, start_readiness_gate

            gate_state = ReadinessState()
            gate_server = start_readiness_gate(
                public_host=str(config["server"]["host"]),
                public_port=int(config["server"]["port"]),
                upstream_host="127.0.0.1",
                upstream_port=int(config["server"]["internal_port"]),
                state=gate_state,
                log=lambda message: log_startup(config, message),
            )

        previous_handlers = install_signal_forwarders(server, gate_server)
        try:
            warmup = config["warmup"]
            if warmup["enabled"]:
                backend_url = inference_backend_url(config)
                log_startup(config, f"Waiting for vLLM health at {backend_url}/health")
                wait_for_health(
                    backend_url,
                    int(warmup["health_timeout_sec"]),
                    int(warmup["health_poll_interval_sec"]),
                    server,
                )
                log_startup(config, "vLLM backend is healthy; starting warmup")
                run_all_warmup_phases(config, label="Warmup")
                if gate_state is not None:
                    gate_state.open()
                    log_startup(
                        config,
                        (
                            "Readiness gate open: "
                            f"http://{config['server']['host']}:{config['server']['port']}/health"
                        ),
                    )
            else:
                log_startup(config, "Warmup disabled")
            return server.wait()
        except Exception:
            terminate_server(server)
            raise
        finally:
            restore_signal_handlers(previous_handlers)
    finally:
        if gate_server is not None:
            from readiness_gate import stop_readiness_gate

            stop_readiness_gate(gate_server)


def main() -> int:
    config, path = load_effective_config()
    mode = deployment_mode(config)
    if mode == "short_long":
        return run_short_long_deployment(config, path)
    if mode == "replica_short_long":
        return run_replica_short_long_deployment(config, path)
    return run_single_deployment(config, path)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Failed to start vLLM server: {exc}", file=sys.stderr, flush=True)
        raise

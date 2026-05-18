#!/usr/bin/env python3
"""Launch a checked-in Hugging Face Job spec safely.

The launcher enforces the agent leash (`.ml-intern/cli_agent_config.json`)
and adds two production guardrails on top of the basic `hf jobs run` wrapper:

* ``--image-tag TAG`` rewrites the GHCR image reference at submission time so
  a release tag (e.g. ``v0.1.0``) can be pinned without editing the YAML.
* ``--cost-cap-usd USD`` performs a pre-flight cost estimate using the
  hardware flavour and the YAML ``timeout`` value, refusing to submit when the
  *worst-case* spend would exceed the cap. Combined with the post-hoc cost
  ledger check this gives both upper and lower bounds on accidental spend.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEASH_PATH = ROOT / ".ml-intern" / "cli_agent_config.json"
ENV_EXPR_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")
IMAGE_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

DEFAULT_COST_CAP_USD = Decimal("20.00")  # per-session soft cap (CLAUDE.md)


class LaunchError(RuntimeError):
    """A non-retryable refusal: the request is unsafe to submit."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="path to jobs/*.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    parser.add_argument("--detach", action="store_true", help="submit and return the HF Job id")
    parser.add_argument(
        "--allow-approval-required",
        action="store_true",
        help="allow jobs listed as human-approval-required in the leash config",
    )
    parser.add_argument(
        "--image-tag",
        default=os.environ.get("LEWM_IMAGE_TAG"),
        help=(
            "override the GHCR image tag (e.g. v0.1.0). Defaults to the value in "
            "the YAML (typically `latest` for dev). Reads $LEWM_IMAGE_TAG when "
            "set; the explicit flag wins."
        ),
    )
    parser.add_argument(
        "--cost-cap-usd",
        type=parse_cost_cap,
        default=DEFAULT_COST_CAP_USD,
        help=(
            "worst-case (hardware-rate * YAML-timeout) USD ceiling that aborts the "
            "submission before contacting HF. Default: 20.00 (per-session soft "
            "cap from CLAUDE.md). Set to 0 to disable."
        ),
    )
    args = parser.parse_args()

    try:
        job_path = resolve_job_path(args.job)
        job = parse_job(job_path)
        leash = load_leash()
        validate_job(job_path, job, leash, args.allow_approval_required)
        if args.image_tag:
            job["image"] = rewrite_image_tag(str(job["image"]), args.image_tag)
        check_cost_cap(job_path, job, args.cost_cap_usd)
        command = render_command(job, detach=args.detach)
    except LaunchError as error:
        print(f"launch_hf_job.py: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(shlex.join(command))
        return 0

    if shutil.which("hf") is None:
        print("launch_hf_job.py: missing required command: hf", file=sys.stderr)
        return 1

    return subprocess.run(command, check=False).returncode


def parse_cost_cap(raw: str) -> Decimal:
    """Parse the ``--cost-cap-usd`` value (Decimal, two-place)."""
    try:
        amount = Decimal(raw)
    except Exception as error:
        raise argparse.ArgumentTypeError(f"invalid USD amount: {raw!r}") from error
    if amount < 0:
        raise argparse.ArgumentTypeError("cost cap must not be negative")
    return amount.quantize(Decimal("0.01"))


def rewrite_image_tag(image: str, new_tag: str) -> str:
    """Replace the ``:tag`` segment of ``image`` with ``new_tag``."""
    if not IMAGE_TAG_RE.fullmatch(new_tag):
        raise LaunchError(f"invalid image tag: {new_tag!r}")
    if "@" in image:
        raise LaunchError("image already pins a digest; refuse to overwrite")
    repo, _, _existing = image.rpartition(":")
    if not repo:
        # No tag in the source string; treat the whole thing as the repo.
        repo = image
    return f"{repo}:{new_tag}"


def check_cost_cap(path: Path, job: dict[str, object], cap_usd: Decimal) -> None:
    """Refuse to submit if the worst-case spend would exceed ``cap_usd``."""
    if cap_usd == 0:
        return

    hardware = str(job["hardware"])
    timeout = str(job["timeout"])
    hours = parse_timeout_hours(timeout)
    price = lookup_price(hardware)
    if price is None:
        # Unknown flavour — be conservative and refuse rather than under-estimate.
        raise LaunchError(
            f"{path.relative_to(ROOT)}: unknown hardware flavour for cost cap: {hardware!r}"
        )

    worst_case = (price * hours).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if worst_case > cap_usd:
        raise LaunchError(
            f"{path.relative_to(ROOT)}: pre-flight cost ${worst_case} exceeds cap "
            f"${cap_usd} (hardware {hardware!r}, timeout {timeout!r}, "
            f"{price} USD/h * {hours} h). Pass --cost-cap-usd to raise the cap, "
            "or shorten the timeout."
        )


def parse_timeout_hours(timeout: str) -> Decimal:
    """Convert an HF Jobs timeout string (`90m`, `2h`, `12h`) to hours."""
    text = timeout.strip().lower()
    if not text:
        raise LaunchError("empty timeout")
    suffix = text[-1]
    try:
        amount = Decimal(text[:-1])
    except Exception as error:
        raise LaunchError(f"unparseable timeout: {timeout!r}") from error
    if suffix == "h":
        return amount
    if suffix == "m":
        return (amount / Decimal(60)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if suffix == "s":
        return (amount / Decimal(3600)).quantize(Decimal("0.0001"), rounding=ROUND_CEILING)
    raise LaunchError(f"unrecognised timeout suffix in {timeout!r}; expected s/m/h")


def lookup_price(hardware: str) -> Decimal | None:
    """Look up the per-hour USD price for an HF hardware flavour."""
    python_dir = ROOT / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    try:
        from hf_pricing import HF_HARDWARE_PRICE_USD_PER_HOUR  # type: ignore[import-not-found]
    except ImportError:
        return None
    return HF_HARDWARE_PRICE_USD_PER_HOUR.get(hardware)


def resolve_job_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else ROOT / path
    if not candidate.is_file():
        raise LaunchError(f"missing job spec: {path}")
    try:
        candidate.relative_to(ROOT / "jobs")
    except ValueError as error:
        raise LaunchError("job spec must live under jobs/") from error
    return candidate


def parse_job(path: Path) -> dict[str, object]:
    top: dict[str, object] = {}
    env: dict[str, str] = {}
    command_lines: list[str] = []
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if raw_line.startswith("  ") and section == "env":
            key, value = split_key_value(raw_line.strip(), path)
            env[key] = value
            continue

        if raw_line.startswith("  ") and section == "command":
            command_lines.append(raw_line.strip())
            continue

        section = None
        key, value = split_key_value(raw_line, path)
        if key == "env":
            section = "env"
            top["env"] = env
        elif key == "command":
            section = "command"
            top["command"] = ""
        else:
            top[key] = value

    if command_lines:
        top["command"] = "\n".join(command_lines)
    return top


def split_key_value(line: str, path: Path) -> tuple[str, str]:
    if ":" not in line:
        raise LaunchError(f"{path.relative_to(ROOT)}: invalid YAML line: {line!r}")
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    if value in {">", "|"}:
        value = ""
    if not key:
        raise LaunchError(f"{path.relative_to(ROOT)}: empty YAML key in line: {line!r}")
    return key, value


def load_leash() -> dict[str, object]:
    try:
        payload = json.loads(LEASH_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LaunchError(f"missing leash config: {LEASH_PATH.relative_to(ROOT)}") from error
    except json.JSONDecodeError as error:
        raise LaunchError(f"{LEASH_PATH.relative_to(ROOT)} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LaunchError("leash config must be a JSON object")
    return payload


def validate_job(
    path: Path,
    job: dict[str, object],
    leash: dict[str, object],
    allow_approval_required: bool,
) -> None:
    required = ("hardware", "timeout", "namespace", "image", "env", "command")
    missing = [key for key in required if key not in job]
    if missing:
        raise LaunchError(f"{path.relative_to(ROOT)} missing keys: {', '.join(missing)}")

    namespace = str(job["namespace"])
    if namespace != leash.get("namespace"):
        raise LaunchError(f"{path.relative_to(ROOT)} namespace {namespace!r} does not match leash")

    hardware = str(job["hardware"])
    allowed_hardware = as_string_set(leash.get("hardware_allowed"))
    if hardware not in allowed_hardware:
        raise LaunchError(f"{path.relative_to(ROOT)} hardware {hardware!r} is not allowed")

    for denied in as_string_set(leash.get("hardware_denied")):
        if denied and denied in hardware:
            raise LaunchError(f"{path.relative_to(ROOT)} hardware {hardware!r} is denied")

    job_name = path.name
    approval_required = as_string_set(leash.get("jobs_human_approval_required"))
    allowed_jobs = as_string_set(leash.get("jobs_allowed"))
    if job_name in approval_required and not allow_approval_required:
        raise LaunchError(f"{job_name} requires --allow-approval-required")
    if job_name not in allowed_jobs and job_name not in approval_required:
        raise LaunchError(f"{job_name} is not listed in the leash config")

    if not str(job["timeout"]):
        raise LaunchError(f"{path.relative_to(ROOT)} must set timeout")

    try:
        shlex.split(str(job["command"]))
    except ValueError as error:
        raise LaunchError(f"{path.relative_to(ROOT)} command is not shell-parseable: {error}") from error

    validate_job_env_contract(path, job)


def validate_job_env_contract(path: Path, job: dict[str, object]) -> None:
    """Validate job-specific environment values that must be known before launch."""
    if path.name != "train_so100_warmstart.yaml":
        return

    env = job.get("env", {})
    if not isinstance(env, dict):
        raise LaunchError(f"{path.relative_to(ROOT)} env must be a mapping")

    raw_source = env.get("LEWM_PUSHT_WARMSTART_MPK")
    if raw_source is None:
        raise LaunchError(
            f"{path.relative_to(ROOT)} must define LEWM_PUSHT_WARMSTART_MPK in env"
        )
    source = resolve_env_value(str(raw_source))
    if not source:
        raise LaunchError(
            "train_so100_warmstart.yaml requires LEWM_PUSHT_WARMSTART_MPK to name a "
            "compatible PushT .mpk source path"
        )
    if Path(source).is_absolute() or ".." in Path(source).parts:
        raise LaunchError(
            "LEWM_PUSHT_WARMSTART_MPK must be a repo-relative Hub path, not an absolute "
            "path or parent traversal"
        )
    if not source.endswith(".mpk"):
        raise LaunchError("LEWM_PUSHT_WARMSTART_MPK must end in .mpk")


def render_command(job: dict[str, object], *, detach: bool) -> list[str]:
    command = [
        "hf",
        "jobs",
        "run",
        "--namespace",
        str(job["namespace"]),
        "--flavor",
        str(job["hardware"]),
        "--timeout",
        str(job["timeout"]),
    ]
    if detach:
        command.append("--detach")

    env = job.get("env", {})
    if not isinstance(env, dict):
        raise LaunchError("job env must be a mapping")
    for key, raw_value in sorted(env.items()):
        if key == "HF_TOKEN" and raw_value == "${HF_TOKEN}":
            command.extend(["--secrets", "HF_TOKEN"])
            continue
        command.extend(["--env", f"{key}={resolve_env_value(str(raw_value))}"])

    command.append(str(job["image"]))
    command.append("--")
    command.extend(shlex.split(str(job["command"])))
    return command


def resolve_env_value(value: str) -> str:
    match = ENV_EXPR_RE.fullmatch(value)
    if not match:
        return value

    name, default = match.groups()
    return os.environ.get(name, default or "")


def as_string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


if __name__ == "__main__":
    raise SystemExit(main())

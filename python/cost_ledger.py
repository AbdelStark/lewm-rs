"""Maintain reports/cost.md for Hugging Face Jobs spend."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from hf_pricing import estimate_cost_usd, format_usd, format_wall, parse_timestamp

DEFAULT_LEDGER = Path("reports/cost.md")
DEFAULT_CAP_USD = Decimal("200.00")
HEADER = """# `lewm-rs` cost ledger

> Updated automatically by `lewm-hub::cost_ledger::append_entry` at every job termination.
> Manual entries are forbidden; use `cost_ledger::backfill --from <job_url>` to import.

| Date (UTC)          | Phase | Job ID            | Hardware     | Wall   | Cost (USD) | Cumulative (USD) | Notes |
|---------------------|-------|-------------------|--------------|--------|-----------:|----------------:|-------|
"""


@dataclass(frozen=True)
class LedgerRow:
    """One cost-ledger row."""

    date_utc: str
    phase: str
    job_id: str
    hardware: str
    wall: str
    cost_usd: Decimal
    cumulative_usd: Decimal
    notes: str


def parse_usd(value: str) -> Decimal:
    """Parse a USD amount at cent precision."""
    amount = Decimal(value.strip()).quantize(Decimal("0.01"))
    if amount < 0:
        raise ValueError(f"negative USD amount: {value}")
    return amount


def read_ledger(path: Path) -> list[LedgerRow]:
    """Read a Markdown cost ledger."""
    if not path.exists():
        return []

    rows: list[LedgerRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells == [
            "Date (UTC)",
            "Phase",
            "Job ID",
            "Hardware",
            "Wall",
            "Cost (USD)",
            "Cumulative (USD)",
            "Notes",
        ]:
            continue
        if len(cells) == 8 and set(cells[0]) == {"-"}:
            continue
        if len(cells) != 8:
            raise ValueError(f"expected 8 cells in ledger row: {line}")
        rows.append(
            LedgerRow(
                date_utc=cells[0],
                phase=cells[1],
                job_id=cells[2],
                hardware=cells[3],
                wall=cells[4],
                cost_usd=parse_usd(cells[5]),
                cumulative_usd=parse_usd(cells[6]),
                notes=cells[7],
            )
        )
    verify_rows(rows, DEFAULT_CAP_USD)
    return rows


def verify_rows(rows: Sequence[LedgerRow], cap_usd: Decimal) -> None:
    """Verify cumulative values and the configured spending cap."""
    cumulative = Decimal("0.00")
    for index, row in enumerate(rows, start=1):
        cumulative += row.cost_usd
        cumulative = cumulative.quantize(Decimal("0.01"))
        if row.cumulative_usd != cumulative:
            raise ValueError(
                f"row {index} cumulative mismatch: stored {row.cumulative_usd}, expected {cumulative}"
            )
        if row.cumulative_usd > cap_usd:
            raise ValueError(f"cost cap exceeded at row {index}: {row.cumulative_usd} > {cap_usd}")


def write_ledger(path: Path, rows: Iterable[LedgerRow]) -> None:
    """Write a Markdown cost ledger with recomputed cumulative values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cumulative = Decimal("0.00")
    rendered = HEADER
    for row in rows:
        cumulative += row.cost_usd
        cumulative = cumulative.quantize(Decimal("0.01"))
        rendered += (
            f"| {cell(row.date_utc, 'date_utc')} | {cell(row.phase, 'phase')} | "
            f"{cell(row.job_id, 'job_id')} | {cell(row.hardware, 'hardware')} | "
            f"{cell(row.wall, 'wall')} | {format_usd(row.cost_usd)} | "
            f"{format_usd(cumulative)} | {cell(row.notes, 'notes', allow_empty=True)} |\n"
        )
    path.write_text(rendered, encoding="utf-8")


def cell(value: str, field: str, *, allow_empty: bool = False) -> str:
    """Validate and normalize a Markdown table cell."""
    normalized = value.strip()
    if (not allow_empty and not normalized) or "|" in normalized or "\n" in normalized or "\r" in normalized:
        raise ValueError(f"invalid ledger field {field}: {value!r}")
    return normalized


def append_row(path: Path, row: LedgerRow, cap_usd: Decimal) -> list[LedgerRow]:
    """Append a row, recompute cumulative values, and write the ledger."""
    existing = read_ledger(path)
    rows = [*existing, row_without_cumulative(row)]
    recomputed = recompute(rows, cap_usd)
    write_ledger(path, recomputed)
    return recomputed


def row_without_cumulative(row: LedgerRow) -> LedgerRow:
    """Drop a stale cumulative value before recomputing."""
    return LedgerRow(
        date_utc=row.date_utc,
        phase=row.phase,
        job_id=row.job_id,
        hardware=row.hardware,
        wall=row.wall,
        cost_usd=row.cost_usd,
        cumulative_usd=Decimal("0.00"),
        notes=row.notes,
    )


def recompute(rows: Sequence[LedgerRow], cap_usd: Decimal) -> list[LedgerRow]:
    """Recompute cumulative values for all rows."""
    cumulative = Decimal("0.00")
    recomputed: list[LedgerRow] = []
    for index, row in enumerate(rows, start=1):
        cumulative += row.cost_usd
        cumulative = cumulative.quantize(Decimal("0.01"))
        if cumulative > cap_usd:
            raise ValueError(f"cost cap exceeded at row {index}: {cumulative} > {cap_usd}")
        recomputed.append(
            LedgerRow(
                date_utc=row.date_utc,
                phase=row.phase,
                job_id=row.job_id,
                hardware=row.hardware,
                wall=row.wall,
                cost_usd=row.cost_usd,
                cumulative_usd=cumulative,
                notes=row.notes,
            )
        )
    return recomputed


def row_from_job_record(record: Mapping[str, object]) -> LedgerRow:
    """Convert an HF Jobs record to a ledger row."""
    job_id = str(record.get("job_id") or record.get("id") or "").strip()
    hardware = str(record.get("hardware_flavor") or record.get("hardware") or "").strip()
    started_at = str(record.get("started_at") or record.get("startedAt") or "").strip()
    ended_at = str(record.get("ended_at") or record.get("endedAt") or "").strip()
    if not job_id or not hardware or not started_at or not ended_at:
        raise ValueError(f"job record is missing required fields: {record}")
    phase = str(record.get("phase") or "unknown").strip()
    exit_code = record.get("exit_code", record.get("exitCode", "unknown"))
    notes = str(record.get("notes") or f"exit_code={exit_code}").strip()
    return LedgerRow(
        date_utc=parse_timestamp(ended_at).strftime("%Y-%m-%d %H:%M:%S"),
        phase=phase,
        job_id=job_id,
        hardware=hardware,
        wall=format_wall(started_at, ended_at),
        cost_usd=estimate_cost_usd(hardware, started_at, ended_at),
        cumulative_usd=Decimal("0.00"),
        notes=notes,
    )


def load_job_records(
    input_json: Path | None,
    since: str | None,
    org: str,
    from_url: str | None,
) -> list[Mapping[str, object]]:
    """Load HF Jobs records from JSON or by shelling out to `hf jobs list`."""
    if input_json is not None:
        payload = json.loads(input_json.read_text(encoding="utf-8"))
    elif from_url is not None:
        command = ["hf", "jobs", "inspect", from_url, "--json"]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        payload = json.loads(completed.stdout)
    else:
        if since is None:
            raise ValueError("backfill requires --since, --from, or --input-json")
        command = ["hf", "jobs", "list", "--org", org, "--since", since, "--json"]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        payload = json.loads(completed.stdout)
    if isinstance(payload, dict):
        if "jobs" in payload:
            payload = payload["jobs"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("HF Jobs payload must be a list or an object with a jobs list")
    return [record for record in payload if isinstance(record, Mapping)]


def check_command(args: argparse.Namespace) -> None:
    """Run the ledger integrity and cap check."""
    rows = read_ledger(args.path)
    verify_rows(rows, Decimal(args.cap_usd).quantize(Decimal("0.01")))
    print(f"cost ledger ok: {len(rows)} rows, cap {Decimal(args.cap_usd).quantize(Decimal('0.01'))} USD")


def append_command(args: argparse.Namespace) -> None:
    """Append one job row."""
    if args.cost_usd is not None:
        if args.wall is None:
            raise ValueError("--wall is required with --cost-usd")
        cost_usd = parse_usd(args.cost_usd)
        wall = args.wall
    else:
        if args.started_at is None or args.ended_at is None:
            raise ValueError("--started-at and --ended-at are required when --cost-usd is absent")
        cost_usd = estimate_cost_usd(args.hardware, args.started_at, args.ended_at)
        wall = format_wall(args.started_at, args.ended_at)

    row = LedgerRow(
        date_utc=args.date_utc,
        phase=args.phase,
        job_id=args.job_id,
        hardware=args.hardware,
        wall=wall,
        cost_usd=cost_usd,
        cumulative_usd=Decimal("0.00"),
        notes=args.notes,
    )
    rows = append_row(args.path, row, Decimal(args.cap_usd).quantize(Decimal("0.01")))
    print(f"appended {args.job_id}; rows={len(rows)}")


def backfill_command(args: argparse.Namespace) -> None:
    """Backfill missing jobs into the ledger."""
    rows = read_ledger(args.path)
    known_job_ids = {row.job_id for row in rows}
    new_rows = [row_without_cumulative(row) for row in rows]
    appended = 0
    for record in load_job_records(args.input_json, args.since, args.org, args.from_url):
        row = row_from_job_record(record)
        if row.job_id in known_job_ids:
            continue
        new_rows.append(row)
        known_job_ids.add(row.job_id)
        appended += 1
    recomputed = recompute(new_rows, Decimal(args.cap_usd).quantize(Decimal("0.01")))
    write_ledger(args.path, recomputed)
    print(f"backfilled {appended} jobs; rows={len(recomputed)}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    check = subcommands.add_parser("check", help="verify cumulative values and cap")
    check.add_argument("--path", type=Path, default=DEFAULT_LEDGER)
    check.add_argument("--cap-usd", default=str(DEFAULT_CAP_USD))
    check.set_defaults(func=check_command)

    append = subcommands.add_parser("append", help="append one ledger row")
    append.add_argument("--path", type=Path, default=DEFAULT_LEDGER)
    append.add_argument("--cap-usd", default=str(DEFAULT_CAP_USD))
    append.add_argument("--date-utc", required=True)
    append.add_argument("--phase", required=True)
    append.add_argument("--job-id", required=True)
    append.add_argument("--hardware", required=True)
    append.add_argument("--started-at")
    append.add_argument("--ended-at")
    append.add_argument("--wall")
    append.add_argument("--cost-usd")
    append.add_argument("--notes", default="")
    append.set_defaults(func=append_command)

    backfill = subcommands.add_parser("backfill", help="backfill missing HF Jobs rows")
    backfill.add_argument("--path", type=Path, default=DEFAULT_LEDGER)
    backfill.add_argument("--cap-usd", default=str(DEFAULT_CAP_USD))
    backfill.add_argument("--since")
    backfill.add_argument("--from", dest="from_url")
    backfill.add_argument("--org", default="AbdelStark")
    backfill.add_argument("--input-json", type=Path)
    backfill.set_defaults(func=backfill_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

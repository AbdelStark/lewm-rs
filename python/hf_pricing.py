"""Hugging Face Jobs pricing helpers for the cost ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING

HF_HARDWARE_PRICE_USD_PER_HOUR = {
    "cpu-basic": Decimal("0.00"),
    "cpu-xl": Decimal("1.00"),
    "l4": Decimal("0.80"),
    "a10g-small": Decimal("1.00"),
    "a10g-large": Decimal("1.50"),
    "l40s": Decimal("1.80"),
    "a100-large": Decimal("2.50"),
    "h100": Decimal("8.00"),
}


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp as an aware UTC datetime."""
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def rounded_billable_minutes(started_at: str, ended_at: str) -> int:
    """Return conservative billable minutes for a job interval."""
    started = parse_timestamp(started_at)
    ended = parse_timestamp(ended_at)
    seconds = Decimal(str((ended - started).total_seconds()))
    if seconds < 0:
        raise ValueError("ended_at must not be earlier than started_at")
    return int((seconds / Decimal(60)).to_integral_value(rounding=ROUND_CEILING))


def format_wall(started_at: str, ended_at: str) -> str:
    """Format a job interval as H:MM:SS."""
    started = parse_timestamp(started_at)
    ended = parse_timestamp(ended_at)
    seconds = int(
        Decimal(str((ended - started).total_seconds())).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    if seconds < 0:
        raise ValueError("ended_at must not be earlier than started_at")
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def estimate_cost_usd(hardware_flavor: str, started_at: str, ended_at: str) -> Decimal:
    """Estimate USD cost rounded up to the nearest minute and cent."""
    try:
        price_per_hour = HF_HARDWARE_PRICE_USD_PER_HOUR[hardware_flavor]
    except KeyError as error:
        raise ValueError(f"unknown HF hardware flavor: {hardware_flavor}") from error
    minutes = Decimal(rounded_billable_minutes(started_at, ended_at))
    cost = (minutes / Decimal(60)) * price_per_hour
    return cost.quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def format_usd(amount: Decimal) -> str:
    """Format a Decimal USD amount with two decimal places."""
    return str(amount.quantize(Decimal("0.01")))

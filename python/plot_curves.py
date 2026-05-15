#!/usr/bin/env python3
"""Generate training loss curve plots and CSV summaries from train_report.json.

Usage:
    uv run python plot_curves.py --report /path/to/train_report.json
    uv run python plot_curves.py --report /path/to/train_report.json --output paper/figures/
    uv run python plot_curves.py --report /path/to/train_report.json --csv-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_rows(losses: list[dict], n: int = 200) -> list[dict]:
    if len(losses) <= n:
        return losses
    step = len(losses) // n
    sampled = losses[::step]
    if sampled[-1] != losses[-1]:
        sampled.append(losses[-1])
    return sampled


def write_csv(losses: list[dict], out_path: Path, sample: int = 500) -> None:
    rows = sample_rows(losses, sample)
    lines = ["step,total_loss,sigreg_proxy_loss,pred_loss,learning_rate,grad_norm_post"]
    for r in rows:
        lines.append(
            f"{r['step']},{r['loss']:.6e},{r['sigreg_proxy_loss']:.6e},"
            f"{r['pred_loss']:.6e},{r['learning_rate']:.6e},"
            f"{r.get('grad_norm_post', r.get('grad_norm_pre', 0.0)):.6e}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CSV written: {out_path}")


def write_ascii_plot(losses: list[dict], title: str, out_path: Path | None = None) -> None:
    rows = sample_rows(losses, 60)
    total = [r["loss"] for r in rows]
    min_v, max_v = min(total), max(total)
    height = 20
    width = len(rows)

    canvas = [[" "] * width for _ in range(height)]
    for col, v in enumerate(total):
        if max_v > min_v:
            row = height - 1 - int((v - min_v) / (max_v - min_v) * (height - 1))
        else:
            row = 0
        canvas[row][col] = "█"

    lines = [f"\n{title}"]
    lines.append(f"  y: [{min_v:.2e}, {max_v:.2e}]  x: step 1 → {rows[-1]['step']}")
    lines.append("  " + "─" * (width + 2))
    for row in canvas:
        lines.append("  │" + "".join(row) + "│")
    lines.append("  " + "─" * (width + 2))

    text = "\n".join(lines)
    print(text)
    if out_path is not None:
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"ASCII plot written: {out_path}")


def plot_matplotlib(losses: list[dict], report: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not available — skipping PNG plots", file=sys.stderr)
        return

    rows = sample_rows(losses, 1000)
    steps = [r["step"] for r in rows]
    total = [r["loss"] for r in rows]
    sigreg = [r["sigreg_proxy_loss"] for r in rows]
    pred = [r["pred_loss"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"lewm-rs Training — {report.get('mode', 'unknown')} "
        f"({report.get('steps_completed', '?')}k steps, {report.get('device', '?')})",
        fontsize=12,
    )

    ax1.semilogy(steps, total, label="Total", color="#1f77b4", linewidth=0.8)
    ax1.semilogy(steps, sigreg, label="SIGReg", color="#ff7f0e", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Loss (log scale)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(steps, pred, label="Pred loss", color="#2ca02c", linewidth=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Pred loss (log scale)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    out_path = out_dir / f"training_curves_{report.get('mode', 'unknown').replace('-', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PNG written: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Path to train_report.json")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for plots")
    parser.add_argument("--csv-only", action="store_true", help="Only write CSV, no plots")
    parser.add_argument("--sample", type=int, default=500, help="CSV sample points (default 500)")
    args = parser.parse_args()

    if not args.report.exists():
        print(f"error: {args.report} not found", file=sys.stderr)
        return 1

    report = load_report(args.report)
    losses = report.get("losses", [])
    if not losses:
        print("error: report contains no loss rows", file=sys.stderr)
        return 1

    out_dir = args.output or args.report.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = report.get("mode", "training")
    steps = report.get("steps_completed", len(losses))

    csv_path = out_dir / f"losses_{mode.replace('-', '_')}.csv"
    write_csv(losses, csv_path, sample=args.sample)

    if not args.csv_only:
        write_ascii_plot(losses, f"Total loss — {mode} ({steps} steps)")
        plot_matplotlib(losses, report, out_dir)

    print(f"\nSummary: mode={mode}, steps={steps}, "
          f"initial={losses[0]['loss']:.4e}, final={losses[-1]['loss']:.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

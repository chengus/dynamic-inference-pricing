"""Small plotting helpers for already-saved experiment results."""

from __future__ import annotations

import json
from pathlib import Path


def plot_report(input_path: str | Path, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    names, revenue, feasible = [], [], []
    for name, policy in data.get("policies", {}).items():
        rows = policy.get("evaluation", [])
        names.append(name.replace("_", " "))
        vals = [r["metrics"].get("revenue_per_gpu_hour") for r in rows]
        vals = [v for v in vals if v is not None]
        revenue.append(sum(vals) / len(vals) if vals else 0.0)
        feasible.append(bool(rows) and all(r["metrics"]["slo"]["all_segments_feasible"] for r in rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2b8cbe" if ok else "#bdbdbd" for ok in feasible]
    ax.bar(names, revenue, color=colors)
    ax.set_ylabel("Revenue per GPU-hour (synthetic server)")
    ax.set_title(f"{data.get('experiment', 'Experiment')}: frozen evaluation")
    ax.tick_params(axis="x", rotation=25)
    ax.text(.01, .98, "Blue = every evaluation seed passed all segment SLOs; gray = failed or unevaluated",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target

"""Command line interface. Work execution always requires explicit opt-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments.config import config_summary, load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="auction-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate YAML and print a static summary")
    validate.add_argument("--config", default="configs/default.yaml")
    serve = sub.add_parser("serve", help="start the local live token-depth dashboard")
    serve.add_argument("--config", default="configs/live.yaml")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--seed", type=int, default=42)
    serve.add_argument("--speed", type=float, default=1.0,
                       help="simulated seconds per wall-clock second")
    serve.add_argument("--allow-simulation", action="store_true",
                       help="required acknowledgement before the live simulation starts")
    for command in ("run", "calibrate"):
        help_text = "calibrate policy choices on calibration seeds only" if command == "calibrate" else "run a configured calibration and evaluation experiment"
        run = sub.add_parser(command, help=help_text)
        run.add_argument("--config", default="configs/default.yaml")
        run.add_argument("--output", help="JSON output file; defaults to config output or results/<name>.json")
        run.add_argument("--allow-simulation", action="store_true",
                         help="required acknowledgement before simulation or calibration executes")
    sweep = sub.add_parser("sweep", help="run a paired load, reserve, burstiness, or reservation-slack sweep")
    sweep.add_argument("--config", default="configs/default.yaml")
    sweep.add_argument("--family", choices=("load", "reserve", "burstiness", "reservation_slack"), required=True)
    sweep.add_argument("--values", required=True, help="comma-separated values; load also accepts low,medium,high")
    sweep.add_argument("--output", help="JSON output file")
    sweep.add_argument("--allow-simulation", action="store_true",
                       help="required acknowledgement before sweep simulations execute")
    plot = sub.add_parser("plot", help="plot an existing experiment JSON file")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps(config_summary(load_config(args.config)), indent=2))
        return 0
    if args.command == "plot":
        from .plots import plot_report
        print(plot_report(args.input, args.output))
        return 0
    if args.command == "serve":
        if not args.allow_simulation:
            parser.error("serve starts a live simulation; pass --allow-simulation explicitly")
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        if args.seed < 0:
            parser.error("--seed must be nonnegative")
        if not 0.05 <= args.speed <= 20:
            parser.error("--speed must be finite and between 0.05 and 20")
        from .live import serve_dashboard
        serve_dashboard(load_config(args.config), host="127.0.0.1", port=args.port,
                        seed=args.seed, speed=args.speed)
        return 0
    if args.command == "sweep":
        if not args.allow_simulation:
            parser.error("sweep executes simulations; pass --allow-simulation explicitly")
        values = []
        for item in args.values.split(","):
            item = item.strip()
            if not item:
                parser.error("--values must not contain empty entries")
            if args.family == "load" and item in {"low", "medium", "high"}:
                values.append(item)
            else:
                try:
                    values.append(float(item))
                except ValueError:
                    parser.error(f"invalid value in --values: {item}")
        config = load_config(args.config)
        from .experiments.sweeps import run_sweep
        report = run_sweep(config, args.family, values, allow_simulation=True)
        output = Path(args.output or f"results/{config.get('name', 'experiment')}-{args.family}-sweep.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved {output}")
        return 0
    if not args.allow_simulation:
        parser.error(f"{args.command} executes simulation/calibration; pass --allow-simulation explicitly")
    config = load_config(args.config)
    from .experiments.runner import run_experiment
    report = run_experiment(config, calibration_only=args.command == "calibrate")
    if args.output:
        output = Path(args.output)
    elif args.command == "calibrate":
        configured = Path(config.get("output", f"results/{config.get('name', 'experiment')}.json"))
        output = configured.with_name(f"{configured.stem}-calibration{configured.suffix or '.json'}")
    else:
        output = Path(config.get("output", f"results/{config.get('name', 'experiment')}.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

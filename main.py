"""modsim — 2D sensor coverage modeling and simulation tool.

DEPRECATED: This script is deprecated and will be removed in v0.3.0.
Please use the new CLI: `modsim run scenario.yaml`

Old Usage:
    python main.py scenario.yaml [--no-viz] [--animate] [--export results.json]

New Usage:
    modsim run scenario.yaml [--no-viz] [--animate] [--export results.json]
    modsim validate scenario.yaml
    modsim version

Python API (unchanged):
    from modsim import load_scenario, run, print_summary

    cfg = load_scenario("scenario.yaml")
    state = run(cfg)
    print_summary(state)
"""

from __future__ import annotations

import argparse
import warnings

from modsim.config import load_scenario
from modsim.sim import run
from modsim.report import print_summary, export_json
from modsim.viz import animate_scenario, plot_final


def main():
    # Show deprecation warning
    warnings.warn(
        "\n"
        "=" * 70 + "\n"
        "DEPRECATION WARNING: Direct execution of main.py is deprecated.\n"
        "Please use the new CLI instead:\n\n"
        "  Old: python main.py scenario.yaml\n"
        "  New: modsim run scenario.yaml\n\n"
        "This script will be removed in modsim v0.3.0.\n"
        "See MIGRATION.md for details.\n"
        "=" * 70,
        DeprecationWarning,
        stacklevel=2,
    )
    print()  # Add spacing after warning
    parser = argparse.ArgumentParser(description="2D Sensor Coverage Simulator")
    parser.add_argument("scenario", help="Path to YAML scenario file")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument(
        "--animate", action="store_true", help="Show animation instead of static plot"
    )
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    parser.add_argument(
        "--export-mode",
        choices=["default", "comms", "states", "both"],
        default="default",
        help="Export format: default (original), comms (Locust timeline), states (object states at every timestep), both",
    )
    args = parser.parse_args()

    cfg = load_scenario(args.scenario)
    print(f"Loaded scenario: {cfg.name}")
    print(f"  Area: {cfg.area.width_km} x {cfg.area.height_km} km")
    print(f"  Groups: {len(cfg.groups)}, Detection: {cfg.detection_mode.value}")

    if args.animate and not args.no_viz:
        animate_scenario(cfg)
    else:
        state = run(cfg)
        print_summary(state)

        if args.export:
            export_json(state, args.export, export_mode=args.export_mode)
            print(f"Results exported to {args.export} (mode: {args.export_mode})")

        if not args.no_viz:
            plot_final(state, cfg)


if __name__ == "__main__":
    main()

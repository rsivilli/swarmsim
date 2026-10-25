"""Click-based CLI for modsim."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from . import __version__
from .config import load_scenario
from .errors import ConfigError, ModsimError, ValidationError
from .models import SimState, TimestepSnapshot
from .report import export_json, print_summary
from .sim import _build_sensor_metadata, build_sensors, run, run_iter
from .utils import format_duration, format_number
from .validators import validate_scenario_structure
from .viz import animate_scenario, plot_final


@click.group()
@click.version_option(version=__version__, prog_name="modsim")
def cli():
    """modsim — 2D/3D sensor coverage modeling and simulation tool.

    Simulate sensor detection coverage with support for binary and probabilistic
    detection models, multiple sensor layouts, and object motion patterns.
    """
    # Output uses ✓/✗/⚠ symbols; legacy Windows consoles default to cp1252 and
    # raise UnicodeEncodeError on them
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@cli.command()
@click.argument("scenario", type=click.Path(exists=True))
@click.option("--no-viz", is_flag=True, help="Skip visualization (faster)")
@click.option("--animate", is_flag=True, help="Show animated playback instead of final plot")
@click.option(
    "--save-animation",
    "save_animation",
    type=click.Path(),
    default=None,
    help="Save animation to file (.gif, .mp4, .mov, .html)",
)
@click.option(
    "--frame-step",
    type=click.IntRange(min=1),
    default=1,
    help="Animate every Nth timestep (speeds up saving long scenarios)",
)
@click.option(
    "--map-style",
    type=click.Choice(["terrain", "satellite", "street", "toner", "basic"]),
    default="terrain",
    help="Map background style (requires cartopy)",
)
@click.option(
    "--export",
    "export_path",
    type=click.Path(),
    help="Export results to JSON file (default: output/{scenario}_{mode}.json)",
)
@click.option(
    "--export-mode",
    type=click.Choice(["default", "comms", "states", "both"]),
    default=None,
    help="Export format: default (stats+detections), comms (Locust timeline), states (object positions), both (all)",
)
@click.option(
    "--helm-values", is_flag=True, help="Generate Helm values YAML (only with --export-mode comms)"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output with timing information")
def run_cmd(
    scenario: str,
    no_viz: bool,
    animate: bool,
    save_animation: str | None,
    frame_step: int,
    map_style: str,
    export_path: str | None,
    export_mode: str | None,
    helm_values: bool,
    verbose: bool,
):
    """Run simulation from scenario YAML file.

    Example:
        modsim run scenarios/basic_grid.yaml
        modsim run scenario.yaml --no-viz --export results.json
        modsim run scenario.yaml --animate --verbose
        modsim run scenario.yaml --map-style satellite
    """
    try:
        import time

        start_time = time.time()

        # Load scenario
        if verbose:
            click.echo(f"Loading scenario: {scenario}")

        cfg = load_scenario(scenario)

        if verbose:
            load_time = time.time() - start_time
            click.secho(f"✓ Loaded scenario: {cfg.name}", fg="green")
            click.echo(f"  Area: {cfg.area.width_km} × {cfg.area.height_km} km")
            if cfg.reference_origin:
                click.echo(
                    f"  Reference Origin: {cfg.reference_origin[0]:.4f}°, {cfg.reference_origin[1]:.4f}° (lat, lon)"
                )
            click.echo(f"  Duration: {cfg.duration}s (dt={cfg.dt}s)")
            click.echo(f"  Groups: {len(cfg.groups)}")
            if cfg.sensors:
                click.echo(f"  Sensors: {cfg.sensors.layout.value}")
            click.echo(f"  Load time: {format_duration(load_time)}")
            click.echo()

        # Run simulation
        num_steps = int(cfg.duration / cfg.dt)

        if verbose:
            click.echo(f"Running simulation ({format_number(num_steps)} timesteps)...")
            sim_start = time.time()

            # Accumulate state from run_iter (single pass, no re-run)
            detection_log = []
            track_state_log = []
            timestep_stats = []
            object_snapshots = []
            last_objects = []

            with click.progressbar(
                length=num_steps,
                label="Simulating",
                show_eta=True,
                show_percent=True,
            ) as bar:
                for step, time_sec_val, snapshot, detections, track_states, stats in run_iter(cfg):
                    bar.update(1)
                    detection_log.extend(detections)
                    track_state_log.extend(track_states)
                    timestep_stats.append(stats)
                    object_snapshots.append(
                        TimestepSnapshot(
                            timestep=step,
                            time_sec=time_sec_val,
                            objects=snapshot,
                        )
                    )
                    last_objects = snapshot

            sensor_positions, sensor_ranges = build_sensors(cfg)
            sensor_metadata = []
            if cfg.communication and cfg.communication.enabled:
                sensor_metadata = _build_sensor_metadata(cfg, sensor_positions)

            state = SimState(
                objects=last_objects,
                sensors=sensor_positions,
                sensor_range=sensor_ranges,
                timestep=timestep_stats[-1].timestep if timestep_stats else 0,
                time_sec=timestep_stats[-1].time_sec if timestep_stats else 0.0,
                detection_log=detection_log,
                timestep_stats=timestep_stats,
                track_state_log=track_state_log,
                sensor_metadata=sensor_metadata,
                object_snapshots=object_snapshots,
            )

            sim_time = time.time() - sim_start
            click.secho("✓ Simulation complete", fg="green", bold=True)
            click.echo(f"  Simulation time: {format_duration(sim_time)}")
            click.echo()
        else:
            # Non-verbose: run without progress bar
            state = run(cfg)
            click.secho("✓ Simulation complete", fg="green")

        # Print summary
        if not verbose:
            click.echo()
        print_summary(state)

        # Export if requested or if export_mode is specified
        should_export = export_path or export_mode
        if should_export:
            # Generate default export path if not provided
            if not export_path:
                # Extract scenario name from path
                scenario_path = Path(scenario)
                scenario_name = scenario_path.stem  # filename without extension

                # Use export_mode or default to "default"
                mode_suffix = export_mode or "default"

                # Create output directory
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)

                # Generate filename
                export_path = str(output_dir / f"{scenario_name}_{mode_suffix}.json")

            # Set export_mode to default if not specified
            if not export_mode:
                export_mode = "default"

            if verbose:
                click.echo(f"Exporting to {export_path} (mode: {export_mode})...")
            actual_path = export_json(state, export_path, export_mode, config=cfg)
            click.secho(f"✓ Exported to: {actual_path}", fg="green")

            # Generate Helm values if requested
            if helm_values:
                if export_mode != "comms":
                    click.secho(
                        "⚠ Warning: --helm-values only works with --export-mode comms", fg="yellow"
                    )
                else:
                    from .comm.helm_values import generate_helm_values

                    # Generate helm values path based on actual export path
                    helm_values_path = (
                        Path(actual_path).parent / f"{Path(actual_path).stem}_values.yaml"
                    )
                    if verbose:
                        click.echo(f"Generating Helm values to {helm_values_path}...")
                    generate_helm_values(actual_path, str(helm_values_path))
                    click.secho(f"✓ Helm values: {helm_values_path}", fg="green")

        # Save animation to file (works independently of --no-viz)
        if save_animation:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend for saving
            if verbose:
                click.echo(f"Saving animation to {save_animation} (map style: {map_style})...")
            animate_scenario(
                cfg, map_style=map_style, save_path=save_animation, frame_step=frame_step
            )
            click.secho(f"✓ Animation saved to: {save_animation}", fg="green")

        # Interactive visualization
        if not no_viz and not save_animation:
            if animate:
                if verbose:
                    click.echo(f"Starting animation (map style: {map_style})...")
                animate_scenario(cfg, map_style=map_style, frame_step=frame_step)
            else:
                if verbose:
                    click.echo(f"Generating plot (map style: {map_style})...")
                plot_final(state, cfg, map_style=map_style)

        # Final timing
        if verbose:
            total_time = time.time() - start_time
            click.echo()
            click.secho(f"✓ Total time: {format_duration(total_time)}", fg="cyan", bold=True)

    except ConfigError as e:
        click.secho("✗ Configuration error:", fg="red", bold=True)
        click.secho(f"  {e}", fg="red")
        sys.exit(1)
    except ValidationError as e:
        click.secho("✗ Validation error:", fg="red", bold=True)
        click.echo(str(e))
        sys.exit(1)
    except ModsimError as e:
        click.secho("✗ Simulation error:", fg="red", bold=True)
        click.secho(f"  {e}", fg="red")
        sys.exit(1)
    except KeyboardInterrupt:
        click.secho("\n✗ Interrupted by user", fg="yellow")
        sys.exit(130)
    except (OSError, ValueError, TypeError, RuntimeError) as e:
        click.secho("✗ Unexpected error:", fg="red", bold=True)
        click.secho(f"  {e}", fg="red")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("scenario", type=click.Path(exists=True))
@click.option("--strict", is_flag=True, help="Treat warnings as errors")
def validate(scenario: str, strict: bool):
    """Validate scenario YAML file without running simulation.

    Checks for:
    - Valid YAML syntax
    - Required fields present
    - Valid enum values with typo suggestions
    - Numerical ranges and constraints

    Example:
        modsim validate scenarios/basic_grid.yaml
        modsim validate my_scenario.yaml --strict
    """
    try:
        import yaml

        click.echo(f"Validating: {scenario}")
        click.echo()

        # Load and parse YAML
        with open(scenario) as f:
            raw = yaml.safe_load(f)

        # Pre-validation
        errors = validate_scenario_structure(raw)

        if errors:
            click.secho("✗ Validation failed:", fg="red", bold=True)
            for error in errors:
                click.secho(f"  • {error}", fg="red")
            sys.exit(1)

        # Try to load scenario (full validation)
        try:
            cfg = load_scenario(scenario)
        except (ValueError, TypeError, KeyError, OSError) as e:
            click.secho("✗ Failed to load scenario:", fg="red", bold=True)
            click.secho(f"  {e}", fg="red")
            sys.exit(1)

        # Success
        click.secho("✓ Validation passed", fg="green", bold=True)
        click.echo()
        click.echo(f"  Name: {cfg.name}")
        click.echo(f"  Area: {cfg.area.width_km} × {cfg.area.height_km} km")
        if cfg.reference_origin:
            click.echo(
                f"  Reference Origin: {cfg.reference_origin[0]:.4f}°, {cfg.reference_origin[1]:.4f}° (lat, lon)"
            )
        click.echo(f"  Duration: {cfg.duration}s (dt={cfg.dt}s)")
        click.echo(f"  Groups: {len(cfg.groups)}")
        click.echo(f"  Detection: {cfg.detection_mode.value}")

        if cfg.sensors:
            click.echo(f"  Sensors: {cfg.sensors.layout.value}")

        if cfg.communication and cfg.communication.enabled:
            click.echo(f"  Communication: enabled ({cfg.communication.strategy.strategy_name})")

        if cfg.rf:
            _echo_rf_summary(cfg)

        sys.exit(0)

    except FileNotFoundError:
        click.secho(f"✗ File not found: {scenario}", fg="red")
        sys.exit(1)
    except yaml.YAMLError as e:
        click.secho("✗ Invalid YAML:", fg="red", bold=True)
        click.secho(f"  {e}", fg="red")
        sys.exit(1)
    except KeyboardInterrupt:
        click.secho("\n✗ Interrupted by user", fg="yellow")
        sys.exit(130)


@cli.command()
@click.argument("comms_export", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    help="Output path for Helm values YAML (default: helm-values-generated.yaml)",
)
def helm_values(comms_export: str, output_path: str | None):
    """Generate Helm values YAML from comms export JSON.

    Takes a comms export JSON (from --export-mode comms) and generates
    a Helm values YAML file ready for deploying Locust load tests.

    Example:
        modsim run scenario.yaml --export-mode comms
        modsim helm-values output/scenario_comms.json -o values.yaml
    """
    try:
        from .comm.helm_values import generate_helm_values, print_deployment_instructions

        if not output_path:
            output_path = "helm-values-generated.yaml"

        click.echo(f"Generating Helm values from: {comms_export}")
        generate_helm_values(comms_export, output_path)
        print_deployment_instructions(comms_export, output_path)

    except FileNotFoundError:
        click.secho(f"✗ File not found: {comms_export}", fg="red")
        sys.exit(1)
    except ValueError as e:
        click.secho("✗ Invalid comms export file:", fg="red", bold=True)
        click.secho(f"  {e}", fg="red")
        sys.exit(1)
    except (OSError, RuntimeError, TypeError, KeyError) as e:
        click.secho("✗ Unexpected error:", fg="red", bold=True)
        click.secho(f"  {e}", fg="red")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@cli.command("animate-all")
@click.argument("scenario_dir", type=click.Path(exists=True), default="scenarios")
@click.option(
    "--output-dir", "-o", default="output/animations", help="Output directory for animation files"
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["mp4", "gif", "mov", "html"]),
    default="mp4",
    help="Animation output format",
)
@click.option("--mp4", "force_mp4", is_flag=True, help="Shorthand for --format mp4")
@click.option(
    "--map-style",
    type=click.Choice(["terrain", "satellite", "street", "toner", "basic"]),
    default="terrain",
    help="Map background style",
)
@click.option(
    "--frame-step",
    type=click.IntRange(min=1),
    default=1,
    help="Animate every Nth timestep (speeds up saving long scenarios)",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def animate_all(
    scenario_dir: str,
    output_dir: str,
    fmt: str,
    force_mp4: bool,
    map_style: str,
    frame_step: int,
    verbose: bool,
):
    """Generate animations for all scenario YAML files in a directory.

    Iterates through every .yaml file in SCENARIO_DIR (default: scenarios/),
    runs the simulation, and saves an animation to OUTPUT_DIR.

    Example:
        modsim animate-all
        modsim animate-all scenarios/ --mp4
        modsim animate-all scenarios/ -o output/videos --format gif
        modsim animate-all --map-style satellite --verbose
    """
    import time
    import matplotlib

    matplotlib.use("Agg")

    if force_mp4:
        fmt = "mp4"

    scenario_path = Path(scenario_dir)
    scenarios = sorted(scenario_path.glob("*.yaml"))

    if not scenarios:
        click.secho(f"No .yaml files found in {scenario_dir}", fg="yellow")
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    click.secho(f"Found {len(scenarios)} scenarios in {scenario_dir}", fg="cyan", bold=True)
    click.echo(f"Output: {out}/ (format: .{fmt}, map: {map_style})")
    click.echo()

    succeeded = []
    failed = []

    for i, sc in enumerate(scenarios, 1):
        label = f"[{i}/{len(scenarios)}]"
        output_file = out / f"{sc.stem}.{fmt}"
        click.echo(f"{label} {sc.name} → {output_file.name} ... ", nl=False)

        try:
            t0 = time.time()
            cfg = load_scenario(str(sc))
            animate_scenario(
                cfg, map_style=map_style, save_path=str(output_file), frame_step=frame_step
            )
            elapsed = time.time() - t0
            click.secho(f"done ({format_duration(elapsed)})", fg="green")
            succeeded.append(sc.name)
        except KeyboardInterrupt:
            click.secho("\n✗ Interrupted by user", fg="yellow")
            sys.exit(130)
        except Exception as e:
            click.secho("FAILED", fg="red")
            if verbose:
                click.secho(f"    {e}", fg="red")
            failed.append((sc.name, str(e)))

    # Summary
    click.echo()
    click.secho(f"✓ {len(succeeded)}/{len(scenarios)} animations generated", fg="green", bold=True)
    if failed:
        click.secho(f"✗ {len(failed)} failed:", fg="red")
        for name, err in failed:
            click.secho(f"  • {name}: {err}", fg="red")
        sys.exit(1)


def _echo_rf_summary(cfg, out_dir: str = "output/rf") -> None:
    """Print the RF section summary and the IQ storage preflight for `validate`."""
    from .rf.sizing import format_bytes, free_bytes, recordings_for_sensor

    rfs = cfg.rf
    window = rfs.rf.window or (0.0, float(cfg.duration))
    links = sum(len(g.links) for g in rfs.groups.values())
    click.echo(
        f"  RF: {len(rfs.sensors)} sensor(s), {len(rfs.groups)} group(s) with {links} link(s), "
        f"{len(rfs.rf.background)} background source(s)"
    )
    recordings = [r for s in rfs.sensors for r in recordings_for_sensor(s, window[1] - window[0])]
    total = sum(r.total_bytes for r in recordings)
    free = free_bytes(out_dir)
    click.echo(
        f"  RF IQ estimate: {format_bytes(total)} across {len(recordings)} recording(s) "
        f"for window {window[0]:g}-{window[1]:g}s ({format_bytes(free)} free at {out_dir})"
    )
    if total > free:
        click.secho("  ⚠ Estimated IQ output exceeds free disk space", fg="yellow")


def _parse_window(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        start, end = (float(v) for v in value.split(":"))
    except ValueError:
        raise click.BadParameter("expected START:END in seconds, e.g. 0:60") from None
    return start, end


@cli.command("rf")
@click.argument("scenario", type=click.Path(exists=True))
@click.option(
    "--out",
    "out_dir",
    type=click.Path(),
    default=None,
    help="Output directory (default: output/rf/<scenario file stem>)",
)
@click.option("--window", default=None, help="Time window to synthesize, START:END seconds")
@click.option("--sensors", default=None, help="Comma-separated RF sensor ids (default: all)")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Schedule emissions, compute link budgets, write truth tables and sizing; no IQ",
)
@click.option(
    "--truth-format",
    type=click.Choice(["auto", "parquet", "csv"]),
    default="auto",
    help="Truth table format (auto: parquet when pyarrow is installed, else csv)",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Parallel synthesis processes (default: CPU count)",
)
@click.option("--force", is_flag=True, help="Synthesize even if the IQ estimate exceeds free space")
def rf_cmd(
    scenario: str,
    out_dir: str | None,
    window: str | None,
    sensors: str | None,
    dry_run: bool,
    truth_format: str,
    workers: int | None,
    force: bool,
):
    """Simulate RF emitters and RF sensors, writing SigMF recordings.

    Writes ground-truth tables (emitters, emissions, links) and a manifest, then
    synthesizes IQ as SigMF (.sigmf-data / .sigmf-meta, plus a .sigmf-collection
    per sensor): one recording per band for fixed/multi tuning, one per sensor
    when scanning. --dry-run stops before synthesis.

    Example:
        modsim rf scenarios/rf_fpv_mix.yaml --dry-run
        modsim rf scenarios/rf_fpv_mix.yaml --window 0:5 --sensors RF-01 --workers 4
    """
    import json

    from .rf.iq import synthesize
    from .rf.runner import plan_rf, write_outputs
    from .rf.sizing import format_bytes, free_bytes

    try:
        cfg = load_scenario(scenario)
        if cfg.rf is None:
            raise ConfigError("Scenario has no RF configuration (rf / rf_sensors / groups[].rf)")
        out = Path(out_dir) if out_dir else Path("output") / "rf" / Path(scenario).stem
        sensor_ids = [s.strip() for s in sensors.split(",")] if sensors else None

        plan = plan_rf(cfg, window=_parse_window(window), sensor_ids=sensor_ids)
        iq_bytes = plan.total_bytes

        free = free_bytes(out)
        click.echo(f"RF plan: {cfg.name}")
        click.echo(f"  Window:     {plan.window[0]:g}-{plan.window[1]:g}s")
        click.echo(f"  Emitters:   {format_number(len(plan.emitters))}")
        click.echo(f"  Emissions:  {format_number(len(plan.emissions))}")
        det = plan.links["detectable"]
        click.echo(
            f"  Links:      {format_number(len(det))} above cull floor, "
            f"{format_number(int(det.sum()))} detectable"
        )
        click.echo("  Recordings:")
        for r in plan.recordings:
            click.echo(
                f"    {r.recording}: {r.sample_rate / 1e6:g} MS/s × {r.num_channels} ch "
                f"{r.sample_format} → {format_bytes(r.total_bytes)}"
            )
        click.echo(f"  IQ total:   {format_bytes(iq_bytes)} ({format_bytes(free)} free)")
        if iq_bytes > free:
            if not dry_run and not force:
                raise ConfigError(
                    "Estimated IQ output exceeds free disk space; narrow --window/--sensors "
                    "or pass --force"
                )
            click.secho("  ⚠ Estimated IQ output exceeds free disk space", fg="yellow")
        if not dry_run and not plan.recordings:
            raise ConfigError("Scenario has no RF sensors to synthesize")

        files, manifest_path, manifest = write_outputs(plan, out, scenario, truth_format)
        click.echo()
        for f in files:
            click.secho(f"✓ {f}", fg="green")

        if not dry_run:
            total = sum(
                int(round((plan.window[1] - plan.window[0]) * r.sample_rate))
                for r in plan.recordings
            )
            with click.progressbar(length=total, label="Synthesizing", show_eta=True) as bar:
                results, timing = synthesize(plan, out, workers=workers, progress=bar.update)
            for r in results:
                click.secho(f"✓ {r.meta_path} ({r.annotations} annotations)", fg="green")
                if r.clipped:
                    click.secho(f"  ⚠ {r.clipped} clipped I/Q components", fg="yellow")
            click.echo(
                f"  {timing['samples'] / 1e6:.1f} MS (all channels) in {timing['render_s']:.1f}s "
                f"({timing['msps']:.2f} MS/s, {timing['workers']} workers)"
            )
            manifest["mode"] = "iq"
            manifest["synthesis"] = {
                "recordings": [
                    {
                        "recording": r.name,
                        "data": str(r.data_path.relative_to(out)),
                        "meta": str(r.meta_path.relative_to(out)),
                        "samples": r.n_samples,
                        "bytes": r.bytes,
                        "emissions_rendered": r.rows,
                        "annotations": r.annotations,
                        "clipped_components": r.clipped,
                    }
                    for r in results
                ],
                "timing": {
                    k: round(v, 3) if isinstance(v, float) else v for k, v in timing.items()
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
        click.secho(f"✓ {manifest_path}", fg="green")

    except (ConfigError, ValueError, RuntimeError) as e:
        click.secho(f"✗ {e}", fg="red")
        sys.exit(1)
    except KeyboardInterrupt:
        click.secho("\n✗ Interrupted by user", fg="yellow")
        sys.exit(130)


@cli.command()
def version():
    """Show version and dependency information.

    Example:
        modsim version
    """
    import importlib.metadata

    import numpy as np
    import matplotlib
    import yaml

    click.echo(f"modsim {__version__}")
    click.echo()
    click.echo("Dependencies:")
    click.echo(f"  Python:      {sys.version.split()[0]}")
    click.echo(f"  NumPy:       {np.__version__}")
    click.echo(f"  Matplotlib:  {matplotlib.__version__}")
    click.echo(f"  PyYAML:      {yaml.__version__}")
    click.echo(f"  Click:       {importlib.metadata.version('click')}")
    click.echo()
    click.echo("For more information: https://github.com/rsivilli/swarmsim")


if __name__ == "__main__":
    cli()

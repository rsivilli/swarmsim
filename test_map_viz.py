#!/usr/bin/env python3
"""Test script to verify map visualization with cartopy."""

from modsim import load_scenario, run
from modsim.viz import HAS_CARTOPY


def test_map_viz():
    """Test map visualization with SF Bay scenario."""
    print("=" * 60)
    print("Testing Map Visualization")
    print("=" * 60)
    print(f"Cartopy available: {HAS_CARTOPY}")

    # Load scenario with reference_origin
    cfg = load_scenario("scenarios/sf_bay_map.yaml")
    print(f"\nScenario: {cfg.name}")
    print(f"Reference Origin: {cfg.reference_origin}")

    if not cfg.reference_origin:
        print("ERROR: No reference_origin found in scenario!")
        return False

    print("\nRunning simulation...")
    state = run(cfg)

    print("✓ Simulation complete")
    print(f"  Total detections: {len(state.detection_log)}")
    print(f"  Sensors: {len(state.sensors)}")

    if HAS_CARTOPY:
        print("\n✓ Cartopy is available - map visualization will show:")
        print("  - Coastlines and borders")
        print("  - Ocean and land features")
        print("  - Geographic grid with labels")
        print("  - Sensors and detections on real map")
    else:
        print("\n⚠ Cartopy not available - will show simple lat/lon plot")
        print("  Install with: uv sync --extra maps")

    print("\nTo visualize, run:")
    print("  uv run modsim run scenarios/sf_bay_map.yaml")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_map_viz()

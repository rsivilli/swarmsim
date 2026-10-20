#!/usr/bin/env python3
"""Test animation with geographic projection."""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

from modsim import load_scenario
from modsim.viz import animate_scenario, HAS_CARTOPY


def test_animation():
    print("Testing animation with geographic projection...")
    print(f"Cartopy available: {HAS_CARTOPY}")

    cfg = load_scenario("scenarios/sf_bay_map.yaml")
    print(f"Loaded scenario: {cfg.name}")
    print(f"Reference origin: {cfg.reference_origin}")

    # Save animation to file
    output_path = "output/sf_bay_animation.gif"
    print(f"\nSaving animation to: {output_path}")

    try:
        animate_scenario(cfg, interval_ms=100, save_path=output_path)
        print("✓ Animation saved successfully!")
    except Exception as e:
        print(f"✗ Animation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    test_animation()

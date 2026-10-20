"""Validation logic for scenario configuration with friendly error messages."""

from __future__ import annotations

from difflib import get_close_matches

from .models import GroupMode, SensorLayout, SpacingProfile


def suggest_typo_fixes(key: str, valid_keys: list[str], cutoff: float = 0.6) -> str:
    """Suggest fixes for typos in configuration keys.

    Args:
        key: The unknown key
        valid_keys: List of valid keys
        cutoff: Similarity threshold (0-1)

    Returns:
        Error message with suggestions if matches found
    """
    matches = get_close_matches(key, valid_keys, n=3, cutoff=cutoff)
    if matches:
        suggestions = ", ".join(f"'{m}'" for m in matches)
        return f"Unknown key '{key}'. Did you mean: {suggestions}?"
    return f"Unknown key '{key}'"


def validate_scenario_structure(raw: dict) -> list[str]:
    """Pre-validate scenario structure before parsing.

    Args:
        raw: Raw YAML dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required top-level fields
    required_fields = ["duration"]
    for field in required_fields:
        if field not in raw:
            errors.append(f"Missing required field: '{field}'")

    # Validate area if present
    if "area" in raw:
        area_errors = validate_area_config(raw["area"])
        errors.extend(area_errors)

    # Validate groups if present
    if "groups" in raw:
        for i, group in enumerate(raw.get("groups", [])):
            group_errors = validate_group_config(group, i)
            errors.extend(group_errors)

    # Validate sensors if present
    if "sensors" in raw:
        sensor_errors = validate_sensor_config(raw["sensors"])
        errors.extend(sensor_errors)

    return errors


def validate_area_config(area: dict) -> list[str]:
    """Validate area configuration.

    Args:
        area: Area configuration dictionary

    Returns:
        List of error messages
    """
    errors = []

    if "width_km" not in area:
        errors.append("Area missing 'width_km'")
    elif not isinstance(area["width_km"], (int, float)) or area["width_km"] <= 0:
        errors.append(f"Area width_km must be positive, got {area.get('width_km')}")

    if "height_km" not in area:
        errors.append("Area missing 'height_km'")
    elif not isinstance(area["height_km"], (int, float)) or area["height_km"] <= 0:
        errors.append(f"Area height_km must be positive, got {area.get('height_km')}")

    return errors


def validate_group_config(group: dict, index: int) -> list[str]:
    """Validate object group configuration.

    Args:
        group: Group configuration dictionary
        index: Group index for error messages

    Returns:
        List of error messages
    """
    errors = []
    prefix = f"Group {index}"

    # Check required fields
    required = ["group_id", "count", "mode", "heading", "spawn_center", "spawn_radius"]
    for field in required:
        if field not in group:
            errors.append(f"{prefix}: Missing required field '{field}'")

    # Validate mode
    if "mode" in group:
        try:
            GroupMode(group["mode"])
        except ValueError:
            valid_modes = [m.value for m in GroupMode]
            suggestion = suggest_typo_fixes(group["mode"], valid_modes)
            errors.append(f"{prefix}: Invalid mode. {suggestion}")

    # Validate spacing_profile if present
    if "spacing_profile" in group:
        try:
            SpacingProfile(group["spacing_profile"])
        except ValueError:
            valid_profiles = [p.value for p in SpacingProfile]
            suggestion = suggest_typo_fixes(group["spacing_profile"], valid_profiles)
            errors.append(f"{prefix}: Invalid spacing_profile. {suggestion}")

    # Validate count
    if "count" in group:
        if not isinstance(group["count"], int) or group["count"] <= 0:
            errors.append(f"{prefix}: count must be a positive integer, got {group.get('count')}")

    # Validate speed values
    if "speed" in group and group["speed"] is not None:
        if not isinstance(group["speed"], (int, float)) or group["speed"] < 0:
            errors.append(f"{prefix}: speed must be non-negative, got {group.get('speed')}")

    return errors


def validate_sensor_config(sensors: dict) -> list[str]:
    """Validate sensor configuration.

    Args:
        sensors: Sensor configuration dictionary

    Returns:
        List of error messages
    """
    errors = []

    # Check layout
    if "layout" not in sensors:
        errors.append("Sensors missing 'layout'")
    else:
        try:
            layout = SensorLayout(sensors["layout"])

            # Layout-specific validation
            if layout == SensorLayout.MANUAL:
                if "positions" not in sensors:
                    errors.append("Manual sensor layout requires 'positions'")
                elif not isinstance(sensors["positions"], list) or len(sensors["positions"]) == 0:
                    errors.append("Manual sensor positions must be a non-empty list")
            elif layout in (SensorLayout.GRID, SensorLayout.HEX):
                if "spacing" not in sensors:
                    errors.append(f"{layout.value} sensor layout requires 'spacing'")
            elif layout == SensorLayout.RANDOM:
                if "count" not in sensors:
                    errors.append("Random sensor layout requires 'count'")

        except ValueError:
            valid_layouts = [layout.value for layout in SensorLayout]
            suggestion = suggest_typo_fixes(sensors["layout"], valid_layouts)
            errors.append(f"Invalid sensor layout. {suggestion}")

    # Validate detection_range
    if "detection_range" in sensors:
        if (
            not isinstance(sensors["detection_range"], (int, float))
            or sensors["detection_range"] <= 0
        ):
            errors.append(
                f"Sensor detection_range must be positive, got {sensors.get('detection_range')}"
            )

    return errors


def validate_range(
    value: float, min_val: float | None = None, max_val: float | None = None, name: str = "value"
) -> str | None:
    """Validate a numeric value is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the value for error messages

    Returns:
        Error message if invalid, None if valid
    """
    if min_val is not None and value < min_val:
        return f"{name} must be >= {min_val}, got {value}"
    if max_val is not None and value > max_val:
        return f"{name} must be <= {max_val}, got {value}"
    return None

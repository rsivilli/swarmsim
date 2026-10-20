"""Tests for data models (Vec2, Vec3, enums, dataclasses)."""

from __future__ import annotations

import pytest

from modsim.models import (
    DetectionMode,
    GroupMode,
    SensorLayout,
    SpacingProfile,
    Vec2,
    Vec3,
)


class TestModels:
    """Test data model classes and enums."""

    def test_vec2_frozen(self):
        """Vec2 should be immutable (frozen dataclass)."""
        v = Vec2(1.0, 2.0)
        with pytest.raises(AttributeError):
            v.x = 3.0  # type: ignore[misc]

    def test_vec3_frozen(self):
        """Vec3 should be immutable (frozen dataclass)."""
        v = Vec3(1.0, 2.0, 3.0)
        with pytest.raises(AttributeError):
            v.x = 4.0  # type: ignore[misc]

    def test_vec2_to_vec3(self):
        """Vec2.to_vec3() should create Vec3 with default z=0."""
        v2 = Vec2(1.0, 2.0)
        v3 = v2.to_vec3()
        assert v3.x == 1.0
        assert v3.y == 2.0
        assert v3.z == 0.0

    def test_vec2_to_vec3_custom_z(self):
        """Vec2.to_vec3(z) should use specified z value."""
        v2 = Vec2(1.0, 2.0)
        v3 = v2.to_vec3(z=5.0)
        assert v3.z == 5.0

    def test_vec3_to_vec2(self):
        """Vec3.to_vec2() should drop z coordinate."""
        v3 = Vec3(1.0, 2.0, 3.0)
        v2 = v3.to_vec2()
        assert v2.x == 1.0
        assert v2.y == 2.0

    def test_vec3_at_ground(self):
        """Vec3.at_ground() should return Vec3 with z=0."""
        v3 = Vec3(1.0, 2.0, 3.0)
        ground = v3.at_ground()
        assert ground.x == 1.0
        assert ground.y == 2.0
        assert ground.z == 0.0

    def test_enum_values(self):
        """Enums should be constructible from string values."""
        assert GroupMode("formation") is GroupMode.FORMATION
        assert DetectionMode("probabilistic") is DetectionMode.PROBABILISTIC
        assert SensorLayout("hex") is SensorLayout.HEX
        assert SpacingProfile("gaussian") is SpacingProfile.GAUSSIAN

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class TrackManager:
    """Manages track IDs and status transitions across simulation timesteps."""

    _track_ids: dict[str, str] = field(default_factory=dict)  # obj_id -> UUID
    _first_seen: dict[str, int] = field(default_factory=dict)  # obj_id -> timestep
    _last_seen: dict[str, int] = field(default_factory=dict)  # obj_id -> timestep

    initiating_window: int = 3  # INITIATING for first N timesteps
    dropping_window: int = 5  # DROPPING if not seen in last N timesteps

    def get_or_create_track_id(self, obj_id: str) -> str:
        """Get existing track ID or create new UUID."""
        if obj_id not in self._track_ids:
            self._track_ids[obj_id] = str(uuid.uuid4())
        return self._track_ids[obj_id]

    def update_detection(self, obj_id: str, timestep: int) -> None:
        """Record that object was detected at this timestep."""
        if obj_id not in self._first_seen:
            self._first_seen[obj_id] = timestep
        self._last_seen[obj_id] = timestep

    def get_track_status(self, obj_id: str, current_timestep: int) -> str:
        """Compute trkStat: INITIATING | MAINTAINING | DROPPING."""
        if obj_id not in self._first_seen:
            return "INITIATING"

        first = self._first_seen[obj_id]
        last = self._last_seen[obj_id]

        if current_timestep - first < self.initiating_window:
            return "INITIATING"
        if current_timestep - last > self.dropping_window:
            return "DROPPING"
        return "MAINTAINING"

    def generate_track_num(self, obj_id: str) -> str:
        """Generate 5-char alphanumeric track number from obj_id hash."""
        h = hash(obj_id) % (36**5)
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = []
        for _ in range(5):
            result.append(chars[h % 36])
            h //= 36
        return "".join(result)

"""Communication emulation module.

Optional feature for generating network traffic timelines for load testing.

This module provides:
- Track state management with persistent IDs
- Track.json message generation (defense data standard)
- Network topology modeling (relay hierarchy, bundling)
- Message timeline generation for Locust load testing
"""

from __future__ import annotations

from .export import export_locust_format, export_locust_jsonl, generate_message_timeline
from .track_manager import TrackManager
from .track_messages import generate_track_message

__all__ = [
    "generate_message_timeline",
    "export_locust_format",
    "export_locust_jsonl",
    "TrackManager",
    "generate_track_message",
]

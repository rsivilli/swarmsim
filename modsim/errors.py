"""Custom exception hierarchy for modsim."""

from __future__ import annotations


class ModsimError(Exception):
    """Base exception for all modsim errors."""

    pass


class ConfigError(ModsimError):
    """Configuration or YAML parsing errors."""

    pass


class ValidationError(ModsimError):
    """Validation errors with helpful suggestions."""

    def __init__(self, message: str, suggestion: str | None = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        if self.suggestion:
            return f"{self.message}\n  Suggestion: {self.suggestion}"
        return self.message


class SimulationError(ModsimError):
    """Runtime simulation errors."""

    pass

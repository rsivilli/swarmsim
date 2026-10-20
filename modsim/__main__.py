"""Entry point for python -m modsim execution."""

from __future__ import annotations


def main():
    """Main entry point for python -m modsim."""
    from modsim.cli import cli

    cli()


if __name__ == "__main__":
    main()

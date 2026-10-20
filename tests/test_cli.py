"""Tests for Click CLI commands."""

from __future__ import annotations


from modsim.cli import cli


class TestCLI:
    """Test Click command-line interface."""

    def test_cli_help(self, cli_runner):
        """CLI should show help message."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "modsim" in result.output.lower()
        assert "run" in result.output
        assert "validate" in result.output

    def test_cli_version(self, cli_runner):
        """CLI should show version."""
        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.output

    def test_version_command(self, cli_runner):
        """Version subcommand should show dependency info."""
        result = cli_runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "modsim 0.2.0" in result.output
        assert "NumPy" in result.output
        assert "Matplotlib" in result.output
        assert "PyYAML" in result.output
        assert "Click" in result.output

    def test_run_command_help(self, cli_runner):
        """Run command should have detailed help."""
        result = cli_runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "scenario" in result.output.lower()
        assert "--no-viz" in result.output
        assert "--animate" in result.output
        assert "--export" in result.output
        assert "--frame-step" in result.output

    def test_run_command_rejects_zero_frame_step(self, cli_runner, basic_scenario):
        """--frame-step must be at least 1."""
        result = cli_runner.invoke(cli, ["run", str(basic_scenario), "--frame-step", "0"])
        assert result.exit_code != 0

    def test_run_command_nonexistent_file(self, cli_runner):
        """Run command should fail gracefully for missing files."""
        result = cli_runner.invoke(cli, ["run", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_run_command_basic_scenario(self, cli_runner, basic_scenario):
        """Run command should execute basic scenario."""
        result = cli_runner.invoke(cli, ["run", str(basic_scenario), "--no-viz"])
        assert result.exit_code == 0
        assert "Simulation complete" in result.output or "Simulation Complete" in result.output

    def test_validate_command_help(self, cli_runner):
        """Validate command should have help."""
        result = cli_runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "scenario" in result.output.lower()
        assert "--strict" in result.output

    def test_validate_command_valid_file(self, cli_runner, basic_scenario):
        """Validate command should pass for valid scenario."""
        result = cli_runner.invoke(cli, ["validate", str(basic_scenario)])
        assert result.exit_code == 0
        assert "passed" in result.output.lower()

    def test_validate_command_invalid_yaml(self, cli_runner, tmp_path):
        """Validate command should fail for invalid YAML."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("invalid: yaml: syntax:")
        result = cli_runner.invoke(cli, ["validate", str(bad_file)])
        assert result.exit_code != 0

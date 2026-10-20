"""Generate Helm values YAML from comms export (.json or .jsonl)."""

from __future__ import annotations

import json
from pathlib import Path


# Name of the ConfigMap the generated values expect the timeline to live in
TIMELINE_CONFIGMAP = "locust-timeline"


def _timeline_filename(comms_path: str | Path) -> str:
    """Return the file name the timeline is mounted as; the extension selects JSON vs JSONL."""
    return "timeline.jsonl" if str(comms_path).endswith(".jsonl") else "timeline.json"


def _load_comms_header(comms_path: str | Path) -> dict:
    """Load metadata from a comms export file (.json or .jsonl).

    For JSONL files, only reads the first line (header) — O(1) memory.
    For JSON files, loads the full file as before.
    """
    path_str = str(comms_path)
    if path_str.endswith(".jsonl"):
        with open(comms_path) as f:
            return json.loads(f.readline())
    else:
        with open(comms_path) as f:
            return json.load(f)


def generate_helm_values(comms_json_path: str | Path, output_path: str | Path) -> None:
    """Generate Helm values YAML from a comms export file (.json or .jsonl).

    Args:
        comms_json_path: Path to the comms export JSON or JSONL
        output_path: Path to write the Helm values YAML

    Raises:
        ValueError: If the file doesn't contain required fields
    """
    data = _load_comms_header(comms_json_path)

    # Validate structure
    if "config" not in data or "recommended_deployment" not in data["config"]:
        raise ValueError(
            "Invalid comms export file. Missing 'config.recommended_deployment' section."
        )

    deployment = data["config"]["recommended_deployment"]
    timeline_file = _timeline_filename(comms_json_path)
    metadata = data.get("metadata", {})
    scenario_name = metadata.get("scenario", "modsim-scenario")

    # Clean scenario name for use in comments (alphanumeric + hyphen)
    safe_scenario_name = "".join(
        c if c.isalnum() or c in "-_ " else "" for c in scenario_name
    ).strip()

    # Build Helm values YAML
    helm_values = f"""# Helm values auto-generated from modsim comms export
# Scenario: {safe_scenario_name}
# Duration: {metadata.get("duration_sec", "N/A")}s
# Total Messages: {metadata.get("total_messages", "N/A")}
# Message Rate: {deployment.get("statistics", {}).get("avg_messages_per_sec", "N/A")} msg/sec (avg)

image:
  repository: modsim-locust
  tag: latest
  pullPolicy: IfNotPresent

locust:
  master:
    replicas: 1
    resources:
      requests:
        cpu: 500m
        memory: 512Mi
      limits:
        cpu: 1000m
        memory: 1Gi

  worker:
    replicas: {deployment.get("worker_replicas", 3)}
    resources:
      requests:
        cpu: 1000m
        memory: 1Gi
      limits:
        cpu: 2000m
        memory: 2Gi

# Timeline configuration
# Create the ConfigMap first (< 1MiB):
#   kubectl create configmap {TIMELINE_CONFIGMAP} --from-file={timeline_file}=<comms export>
# For larger timelines, clear existingConfigMap and set hostPath instead.
timeline:
  path: /data/{timeline_file}
  existingConfigMap: {TIMELINE_CONFIGMAP}
  hostPath: ""

# Backend configuration
backend:
  host: "{deployment.get("backend_host", "http://backend-service:8080")}"

# Test configuration
# These values are derived from the simulation parameters
test:
  users: {deployment.get("users", 10)}
  spawnRate: {deployment.get("spawn_rate", 5)}
  runTime: {deployment.get("run_time", "60s")}
  headless: false  # Set true for automated runs

service:
  type: ClusterIP
  port: 8089

# Simulation statistics (for reference)
# Sensors: {deployment.get("statistics", {}).get("num_sensors", "N/A")}
# Unique Senders: {deployment.get("statistics", {}).get("unique_senders", "N/A")}
# Peak Message Rate: {deployment.get("statistics", {}).get("peak_messages_per_sec", "N/A")} msg/sec
"""

    with open(output_path, "w") as f:
        f.write(helm_values)


def print_deployment_instructions(
    comms_json_path: str | Path, helm_values_path: str | Path
) -> None:
    """Print instructions for deploying with the generated values.

    Args:
        comms_json_path: Path to the comms export JSON
        helm_values_path: Path to the generated Helm values
    """
    print("\n" + "=" * 70)
    print("Helm Values Generated Successfully")
    print("=" * 70)
    print(f"\nGenerated from:  {comms_json_path}")
    print(f"Values written:  {helm_values_path}")
    print("\nNext Steps:")
    print("-" * 70)
    timeline_file = _timeline_filename(comms_json_path)
    print("\n1. Copy the timeline to your cluster:")
    print(f"   kubectl create configmap {TIMELINE_CONFIGMAP} \\")
    print(f"     --from-file={timeline_file}={comms_json_path}")
    print("\n2. Deploy Locust using the generated values:")
    print("   helm install modsim-loadtest ./locust/helm-chart \\")
    print(f"     -f {helm_values_path}")
    print("\n3. Access the Locust web UI:")
    print("   kubectl port-forward svc/modsim-loadtest-modsim-locust-master 8089:8089")
    print("   Open http://localhost:8089")
    print("\n4. (Optional) Run headless load test:")
    print("   helm upgrade modsim-loadtest ./locust/helm-chart \\")
    print(f"     -f {helm_values_path} \\")
    print("     --set test.headless=true")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m modsim.comm.helm_values <comms_export.json> [output.yaml]")
        print("\nExample:")
        print("  python -m modsim.comm.helm_values output/comms_test_comms.json values.yaml")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "helm-values-generated.yaml"

    try:
        generate_helm_values(input_path, output_path)
        print_deployment_instructions(input_path, output_path)
    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except (OSError, RuntimeError, TypeError, KeyError) as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

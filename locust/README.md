# Modsim Locust Load Testing

Locust-based load testing infrastructure for replaying modsim communication timelines against backend services.

## Overview

This directory contains:
- **locustfile.py** - Timeline replay script with deterministic event assignment
- **Dockerfile** - Container image for deployment
- **helm-chart/** - Kubernetes deployment manifests
- **scripts/** - Helper scripts for timeline embedding

## Quick Start

All commands below run from the repository root.

### 1. Generate Timeline and Helm Values

```bash
# One step: generate the timeline + Helm values
uv run modsim run scenarios/comms_test.yaml \
  --no-viz --export-mode comms --helm-values

# This creates:
#   - output/comms_test_comms.json         (timeline)
#   - output/comms_test_comms_values.yaml  (Helm values)
```

The generated Helm values include:
- Recommended deployment parameters (users, workers, spawn rate, run time)
- Backend configuration
- Timeline mount settings pointing at a `locust-timeline` ConfigMap

**Alternatively**, generate Helm values from an existing timeline:

```bash
uv run modsim helm-values output/comms_test_comms.json -o my-values.yaml
```

### 2. Local Testing with Docker

```bash
docker build -t modsim-locust:latest locust/

docker run -p 8089:8089 \
  -v $(pwd)/output/comms_test_comms.json:/data/timeline.json \
  -e TIMELINE_PATH=/data/timeline.json \
  modsim-locust:latest \
  locust -f /locust/locustfile.py --host=http://host.docker.internal:8080

# Open http://localhost:8089
```

The image defaults to `TIMELINE_PATH=/data/timeline.jsonl`; set `TIMELINE_PATH` to match the file you mount. The extension (`.json` or `.jsonl`) selects the parser.

### 3. Kubernetes Deployment

No prebuilt image is published yet. Build the image and make it available to your cluster first (e.g. `kind load docker-image modsim-locust:latest`, or push it to your own registry and set `image.repository`).

#### Recommended: Generated Values + ConfigMap

```bash
# 1. Create a ConfigMap holding the timeline. The key must match the file name
#    in timeline.path (timeline.json here; timeline.jsonl for a .jsonl export)
kubectl create configmap locust-timeline \
  --from-file=timeline.json=output/comms_test_comms.json

# 2. Deploy with the generated values
helm install modsim-loadtest ./locust/helm-chart \
  -f output/comms_test_comms_values.yaml \
  --set backend.host=http://my-backend-svc.default.svc:8080

# 3. Access the UI
kubectl port-forward svc/modsim-loadtest-modsim-locust-master 8089:8089
```

#### Alternative: Embed the Timeline in a Values File

For small timelines (< 1MiB) when you'd rather not manage a ConfigMap yourself:

```bash
./locust/scripts/embed_timeline.sh output/comms_test_comms.json values-custom.yaml

helm install modsim-loadtest ./locust/helm-chart \
  -f values-custom.yaml \
  --set backend.host=http://my-backend-svc.default.svc:8080
```

#### Large Timelines (> 1MiB)

ConfigMaps are capped at 1MiB. For larger timelines, mount a directory from the node with `timeline.hostPath` (suited to single-node clusters such as kind, where you control the node filesystem):

```bash
helm install modsim-loadtest ./locust/helm-chart \
  --set timeline.hostPath=/data/timelines \
  --set timeline.path=/data/timeline.jsonl \
  --set backend.host=http://my-backend-svc.default.svc:8080
```

The directory must contain a file with the same name as `timeline.path`.

## Configuration

### values.yaml Options

```yaml
# Backend target (REST transport)
backend:
  host: "http://backend-service.default.svc.cluster.local:8080"

# Worker scaling
locust:
  worker:
    replicas: 5

# Test parameters
test:
  users: 100
  spawnRate: 10
  runTime: 300s
  headless: false  # Set true for CI/CD

# Timeline source: set one of existingConfigMap, inline or hostPath
timeline:
  path: /data/timeline.jsonl
  existingConfigMap: ""
  inline: ""
  hostPath: ""
```

### Environment Variables

- `TIMELINE_PATH` - Path to the timeline inside the container (image default: `/data/timeline.jsonl`)
- `ZENOH_TRANSPORT` - `rest` (default), `native` or `udl`; see [Transport Modes](../README.md#transport-modes)

## Architecture

```
Timeline JSON → Locust Master → Workers → Backend
                     ↓
                  Web UI (8089)
```

- **Master**: Coordinates test, serves web UI
- **Workers**: Publish events using the selected transport
- **Events**: Assigned deterministically by sensor hash

## How It Works

1. **Timeline Loading**: JSON loaded at module level; JSONL streamed lazily per worker
2. **Event Assignment**: Each user gets events from specific sensors (hash-based)
3. **Timing**: Best-effort replay at original timestamps
4. **Transport**: Chosen with `ZENOH_TRANSPORT` (REST, native Zenoh or UDL)
5. **Metrics**: Per-sender/receiver stats in Locust UI

## Headless Mode

For CI/CD pipelines:

```bash
helm install modsim-loadtest ./locust/helm-chart \
  --set test.headless=true \
  --set test.users=100 \
  --set test.spawnRate=10 \
  --set test.runTime=300s
```

Export results:

```bash
# Via API
kubectl port-forward svc/modsim-loadtest-modsim-locust-master 8089:8089 &
curl http://localhost:8089/stats/requests > results.json
```

## Cleanup

```bash
# Uninstall Helm release
helm uninstall modsim-loadtest

# Delete the timeline ConfigMap if you created one
kubectl delete configmap locust-timeline
```

## Troubleshooting

### Workers Not Connecting

Check master hostname resolution:

```bash
kubectl logs -l component=worker --tail=50
```

### Timeline Not Loading

Check that the mounted file name matches `timeline.path`:

```bash
kubectl exec -it <master-pod> -- sh -c 'echo $TIMELINE_PATH; ls -l /data'
```

### Timing Drift

Check Locust stats for request latency. If backend is slow, timing will drift from simulation.

### High CPU/Memory Usage

Reduce worker replicas or adjust resource limits in values.yaml.

## Validation Checklist

- [ ] Timeline loads without errors
- [ ] Events distributed across workers
- [ ] HTTP requests succeed (check backend logs)
- [ ] Locust UI accessible and showing metrics
- [ ] RPS matches expected rate from timeline
- [ ] Resource usage within limits

## Examples

### Compare Strategies

Generate a timeline per communication strategy and deploy each as its own release. Give each release its own ConfigMap via `--set timeline.existingConfigMap=<name>`:

```bash
uv run modsim run scenarios/comms_test.yaml --no-viz --export-mode comms \
  --export output/direct.json --helm-values
kubectl create configmap locust-timeline --from-file=timeline.json=output/direct.json
helm install direct ./locust/helm-chart -f output/direct_values.yaml

# Compare RPS, latency, errors in Locust UI
```

### Long-Running Test

```bash
helm install long-test ./locust/helm-chart \
  --set test.headless=true \
  --set test.runTime=3600s \
  --set locust.worker.replicas=20
```

## Performance Tips

1. **Scale workers** based on target RPS (1 worker ≈ 100-500 RPS)
2. **Use headless mode** for automated testing
3. **Monitor backend** - Locust only shows client-side metrics
4. **Test incrementally** - start small, scale up
5. **hostPath for large timelines** - ConfigMaps are limited to 1MiB

## References

- [Locust Documentation](https://docs.locust.io/)
- [Helm Chart Guide](https://helm.sh/docs/chart_template_guide/)
- [Modsim Communication Export](../README.md#comms-export-format)

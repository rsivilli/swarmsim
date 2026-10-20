# Modsim Locust Load Testing

Locust-based load testing infrastructure for replaying modsim communication timelines against backend services.

## Overview

This directory contains:
- **locustfile.py** - Timeline replay script with deterministic event assignment
- **Dockerfile** - Container image for deployment
- **helm-chart/** - Kubernetes deployment manifests
- **scripts/** - Helper scripts for timeline embedding

## Quick Start

### 1. Generate Timeline and Helm Values

The new streamlined workflow automatically generates both the timeline and deployment configuration:

```bash
# One-step: Generate timeline + Helm values
modsim run examples/comms_test.yaml \
  --no-viz --export-mode comms --helm-values

# This creates:
#   - output/comms_test_comms.json (timeline)
#   - output/comms_test_comms_values.yaml (Helm values)
```

The generated Helm values include:
- Communication configuration (strategy, topology, bundling)
- Recommended deployment parameters (users, workers, spawn rate)
- Backend configuration
- Test duration and statistics

**Alternatively**, generate Helm values from an existing timeline:

```bash
modsim helm-values output/comms_test_comms.json -o my-values.yaml
```

### 2. Local Testing with Docker

Build and run locally:

```bash
# Build image
docker build -t modsim-locust:latest .

# Run with local timeline
docker run -p 8089:8089 \
  -v $(pwd)/../comms_timeline.json:/timeline.json \
  -e BACKEND_HOST=http://host.docker.internal:8080 \
  modsim-locust:latest \
  locust -f /locust/locustfile.py --host=http://host.docker.internal:8080

# Open browser to http://localhost:8089
```

### 3. Kubernetes Deployment

#### Recommended: Automated Workflow

Using the auto-generated values from step 1:

```bash
# 1. Create ConfigMap with timeline
kubectl create configmap locust-timeline \
  --from-file=timeline.json=output/comms_test_comms.json

# 2. Deploy with auto-generated values
helm install modsim-loadtest ./helm-chart \
  -f output/comms_test_comms_values.yaml \
  --set backend.host=http://my-backend-svc.default.svc:8080

# 3. Access UI
kubectl port-forward svc/modsim-loadtest-modsim-locust-master 8089:8089
# Open http://localhost:8089
```

The auto-generated values already include optimized worker counts, user counts, and other parameters based on your simulation.

#### Alternative: Manual ConfigMap Embedding

For small timelines (<1MB) with custom values:

```bash
# Embed timeline in values file
./scripts/embed_timeline.sh output/comms_test_comms.json values-custom.yaml

# Install Helm chart
helm install modsim-loadtest ./helm-chart \
  -f values-custom.yaml \
  --set backend.host=http://my-backend-svc.default.svc:8080

# Access UI
kubectl port-forward svc/modsim-loadtest-modsim-locust-master 8089:8089
```

#### Large Timelines (>1MB)

Use PersistentVolume:

```bash
# Create PVC
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: timeline-storage
spec:
  accessModes: [ReadOnlyMany]
  resources:
    requests:
      storage: 1Gi
EOF

# Copy timeline to PV
kubectl run -i --rm --restart=Never timeline-copy --image=busybox \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "timeline-copy",
      "image": "busybox",
      "command": ["sh"],
      "stdin": true,
      "volumeMounts": [{
        "name": "timeline",
        "mountPath": "/mnt/timeline"
      }]
    }],
    "volumes": [{
      "name": "timeline",
      "persistentVolumeClaim": {
        "claimName": "timeline-storage"
      }
    }]
  }
}' < ../comms_timeline.json > /mnt/timeline/timeline.json

# Install with PV
helm install modsim-loadtest ./helm-chart \
  --set timeline.externalPath=/mnt/timeline/timeline.json \
  --set backend.host=http://my-backend-svc.default.svc:8080
```

## Configuration

### values.yaml Options

```yaml
# Backend target
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

# Timeline storage
timeline:
  inline: ""  # Embed JSON directly
  externalPath: ""  # Or use PV mount path
```

### Environment Variables

- `TIMELINE_PATH` - Path to timeline JSON (default: `/timeline.json`)
- `BACKEND_HOST` - Backend service URL

## Architecture

```
Timeline JSON → Locust Master → Workers → Backend
                     ↓
                  Web UI (8089)
```

- **Master**: Coordinates test, serves web UI
- **Workers**: Execute HTTP requests
- **Events**: Assigned deterministically by sensor hash

## How It Works

1. **Timeline Loading**: JSON loaded at module level for efficiency
2. **Event Assignment**: Each user gets events from specific sensors (hash-based)
3. **Timing**: Best-effort replay at original timestamps
4. **Transport**: REST primary, Zenoh HTTP bridge optional
5. **Metrics**: Per-sender/receiver stats in Locust UI

## Transports

### REST (Default)

Events with `transport: "rest_post"` send to `/api/tracks` endpoint.

### Zenoh HTTP Bridge

Events with `transport: "zenoh_pubsub"` send to `/zenoh/pub/{sensor_id}` endpoint.

## Headless Mode

For CI/CD pipelines:

```bash
helm install modsim-loadtest ./helm-chart \
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

# Delete PVC if used
kubectl delete pvc timeline-storage
```

## Troubleshooting

### Workers Not Connecting

Check master hostname resolution:

```bash
kubectl logs -l component=worker --tail=50
```

### Timeline Not Loading

Verify ConfigMap or PV mount:

```bash
kubectl exec -it <master-pod> -- cat /timeline.json
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

Run multiple tests with different timelines:

```bash
# Direct REST
uv run modsim run ../examples/direct_rest.yaml --no-viz --export-mode comms --export direct.json
helm install direct ./helm-chart -f direct-values.yaml

# Bundled Relay
uv run modsim run ../examples/bundled_relay.yaml --no-viz --export-mode comms --export bundled.json
helm install bundled ./helm-chart -f bundled-values.yaml

# Compare RPS, latency, errors in Locust UI
```

### Long-Running Test

```bash
helm install long-test ./helm-chart \
  --set test.headless=true \
  --set test.runTime=3600s \
  --set locust.worker.replicas=20
```

## Performance Tips

1. **Scale workers** based on target RPS (1 worker ≈ 100-500 RPS)
2. **Use headless mode** for automated testing
3. **Monitor backend** - Locust only shows client-side metrics
4. **Test incrementally** - start small, scale up
5. **PV for large timelines** - ConfigMap has 1MB etcd limit

## References

- [Locust Documentation](https://docs.locust.io/)
- [Helm Chart Guide](https://helm.sh/docs/chart_template_guide/)
- [Modsim Communication Export](../README.md#comms-export-format)

# Base image pinned to SHA digest for air-gap reproducibility and supply chain security.
# To update: docker pull python:3.11-slim && docker inspect python:3.11-slim --format='{{index .RepoDigests 0}}'
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928

WORKDIR /locust

# Install all transport dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy locustfile
COPY locustfile.py /locust/

# Environment defaults
ENV TIMELINE_PATH=/data/timeline.jsonl
ENV ZENOH_TRANSPORT=rest

# Expose Locust ports
# 8089: Web UI
# 5557: Master-worker communication
EXPOSE 8089 5557

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --ingroup appgroup appuser
USER appuser

# Default command (master mode with UI)
CMD ["locust", "-f", "/locust/locustfile.py", "--master"]

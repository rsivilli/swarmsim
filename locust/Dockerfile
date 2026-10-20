# Base image pinned to SHA digest for air-gap reproducibility and supply chain security.
# To update: docker pull python:3.11-slim && docker inspect python:3.11-slim --format='{{index .RepoDigests 0}}'
FROM python:3.11-slim@sha256:5be45dbade29bebd6886af6b438fd7e0b4eb7b611f39ba62b430263f82de36d2

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

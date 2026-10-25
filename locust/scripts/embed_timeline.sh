#!/bin/bash
# Embed timeline JSON/JSONL into a Helm values file

TIMELINE_FILE=$1
OUTPUT_FILE=${2:-"values-with-timeline.yaml"}

if [ ! -f "$TIMELINE_FILE" ]; then
  echo "Error: Timeline file not found: $TIMELINE_FILE"
  exit 1
fi

# The mounted file name's extension tells the locustfile how to parse it
case "$TIMELINE_FILE" in
  *.jsonl) TIMELINE_NAME=timeline.jsonl ;;
  *) TIMELINE_NAME=timeline.json ;;
esac

# Create values file with embedded timeline
cat > "$OUTPUT_FILE" <<VALUES
# Auto-generated values with embedded timeline
timeline:
  path: /data/$TIMELINE_NAME
  inline: |
$(sed 's/^/    /' "$TIMELINE_FILE")
VALUES

echo "Created $OUTPUT_FILE with embedded timeline"

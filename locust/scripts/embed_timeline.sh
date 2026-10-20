#!/bin/bash
# Embed timeline JSON into Helm values file

TIMELINE_FILE=$1
OUTPUT_FILE=${2:-"values-with-timeline.yaml"}

if [ ! -f "$TIMELINE_FILE" ]; then
  echo "Error: Timeline file not found: $TIMELINE_FILE"
  exit 1
fi

# Create values file with embedded timeline
cat > "$OUTPUT_FILE" <<EOF
# Auto-generated values with embedded timeline
timeline:
  inline: |
$(cat "$TIMELINE_FILE" | sed 's/^/    /')
EOF

echo "Created $OUTPUT_FILE with embedded timeline"

#!/bin/bash
echo "=== Task 07: RTSP Check ==="
if gst-inspect-1.0 rtspsrc >/dev/null 2>&1; then echo "PASS: RTSP source available"; else echo "FAIL: RTSP source missing"; fi
if gst-inspect-1.0 rtspclientsink >/dev/null 2>&1; then echo "PASS: RTSP client sink available"; else echo "FAIL: RTSP client sink missing"; fi
echo "=== Listening ports ==="
ss -ltn | grep -E ":8554|:8555" || echo "No RTSP server listening on ports 8554/8555"

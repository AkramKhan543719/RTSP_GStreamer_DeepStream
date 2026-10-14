#!/bin/bash
set -o pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
mkdir -p outputs/deepstream/task05_video_output
deepstream-app -c "$PWD/tasks/deepstream/task04_inference/app_config.txt" 2>&1 | tee outputs/deepstream/task05_video_output/video_output.log
status=${PIPESTATUS[0]}
if [ "$status" -eq 0 ]; then echo "Inference pipeline completed; video recording is not enabled in the current sink configuration." | tee outputs/deepstream/task05_video_output/video_output_report.txt; else echo "Inference pipeline failed; see video_output.log." | tee outputs/deepstream/task05_video_output/video_output_report.txt; fi
exit "$status"

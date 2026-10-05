$gst = "$env:LOCALAPPDATA\Programs\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"

$projectRoot = Split-Path $PSScriptRoot -Parent
$output = Join-Path $projectRoot "test_videos\gstreamer_test.raw"

Write-Host "============================================"
Write-Host "GStreamer Test Source -> RAW File"
Write-Host "============================================"
Write-Host "Output:"
Write-Host $output
Write-Host ""

& $gst videotestsrc num-buffers=300 ! videoconvert ! filesink location="$output"

Write-Host ""
Write-Host "Recording completed."
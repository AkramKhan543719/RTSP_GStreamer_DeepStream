$gst = "$env:LOCALAPPDATA\Programs\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"

Write-Host "============================================"
Write-Host "GStreamer Test Source -> Display"
Write-Host "============================================"
Write-Host "Close the video window or press Ctrl+C to stop."
Write-Host ""

& $gst videotestsrc ! videoconvert ! autovideosink
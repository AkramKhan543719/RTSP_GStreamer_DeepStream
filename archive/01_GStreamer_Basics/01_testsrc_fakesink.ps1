$gst = "$env:LOCALAPPDATA\Programs\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"

Write-Host "============================================"
Write-Host "GStreamer Test Source -> Fake Sink"
Write-Host "============================================"

& $gst videotestsrc num-buffers=300 ! videoconvert ! fakesink

Write-Host ""
Write-Host "Pipeline completed."
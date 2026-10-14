#!/usr/bin/env python3
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")

from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)

server = GstRtspServer.RTSPServer()
server.set_service("8554")

factory = GstRtspServer.RTSPMediaFactory()
factory.set_launch(
    '( videotestsrc is-live=true ! '
    'video/x-raw,width=640,height=480,framerate=30/1 ! '
    'videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast ! '
    'rtph264pay name=pay0 pt=96 )'
)
factory.set_shared(True)

server.get_mount_points().add_factory("/test", factory)

if server.attach(None) == 0:
    raise RuntimeError("Could not start RTSP server on port 8554")

print("RTSP server running at rtsp://127.0.0.1:8554/test", flush=True)
GLib.MainLoop().run()

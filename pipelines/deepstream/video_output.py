from common import VIDEO, task_output, require_file

out = task_output("task05_video_output")
report = out / "video_output_preflight.txt"

messages = []

try:
    require_file(VIDEO)
    messages.append(f"PASS: Input video exists: {VIDEO}")
except FileNotFoundError as exc:
    messages.append(f"FAIL: {exc}")

messages.extend([
    "PENDING: DeepStream inference must produce valid detection metadata.",
    "PENDING: Configure the video encoder and MP4 muxer.",
    "PENDING: Verify the final annotated video by playing it.",
])

report.write_text("\n".join(messages) + "\n", encoding="utf-8")

for message in messages:
    print(message)

print(f"\nReport: {report}")

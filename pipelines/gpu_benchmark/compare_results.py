cat > pipelines/gpu_benchmark/compare_results.py <<'PY'
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs" / "gpu_benchmark"
REPORT = ROOT / "experiments" / "gpu_pt_onnx_trt_comparison.csv"

BACKENDS = ["pt", "onnx", "trt"]


def main():
    rows = []

    for backend in BACKENDS:
        result_file = OUTPUT_ROOT / backend / "results.json"

        if not result_file.exists():
            print(f"Skipping {backend}: results not found at {result_file}")
            continue

        with result_file.open() as f:
            row = json.load(f)
            rows.append(row)

    if not rows:
        raise RuntimeError("No benchmark results found. Run benchmark_backends.py first.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with REPORT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== YOLO11m BACKEND COMPARISON ==========")
    print(f"{'Backend':<10} {'Avg inference (ms)':>20} {'Inference FPS':>16} {'End-to-end FPS':>17}")

    for row in rows:
        print(
            f"{row['backend']:<10}"
            f"{row['average_inference_ms']:>20.3f}"
            f"{row['inference_fps']:>16.2f}"
            f"{row['end_to_end_fps']:>17.2f}"
        )

    print(f"\nCSV report: {REPORT}")


if __name__ == "__main__":
    main()
PY
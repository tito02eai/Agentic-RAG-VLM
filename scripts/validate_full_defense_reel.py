#!/usr/bin/env python3
"""Validate a complete defense reel and its linked evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "output" / "defense_suite" / "agentic_rag_vlm_full_defense_reel_v9.mp4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--expected-duration", type=float, default=67.0)
    parser.add_argument("--expected-frames", type=int, default=2010)
    args = parser.parse_args()
    video = args.video.resolve()
    probe = json.loads(subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration,size",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt",
        "-of", "json", str(video),
    ], check=True, capture_output=True, text=True).stdout)
    stream = next(item for item in probe["streams"] if item.get("codec_name") == "h264")
    # The corrected complete Showcase occupies [12.1, 38.7) without editorial
    # cuts.  Its 798 source frames were admitted by the visible-mesh gate.
    frame_md5 = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", "12.1", "-t", "26.6", "-i", str(video),
        "-vf", "fps=2", "-f", "framemd5", "-",
    ], check=True, capture_output=True, text=True).stdout
    hashes = [line.rsplit(",", 1)[-1].strip() for line in frame_md5.splitlines() if line and not line.startswith("#")]
    paired_md5 = subprocess.run([
        "ffmpeg", "-v", "error", "-ss", "39.0", "-t", "13.2", "-i", str(video),
        "-vf", "fps=2", "-f", "framemd5", "-",
    ], check=True, capture_output=True, text=True).stdout
    paired_hashes = [
        line.rsplit(",", 1)[-1].strip()
        for line in paired_md5.splitlines()
        if line and not line.startswith("#")
    ]
    suite_validation = json.loads((PROJECT_ROOT / "output" / "defense_suite" / "validation_report.json").read_text())
    logo_source = json.loads((PROJECT_ROOT / "video" / "guanghua_defense_reel" / "assets" / "fudan_logo_source.json").read_text())
    checks = {
        "resolution_1080p": stream["width"] == 1920 and stream["height"] == 1080,
        "fps_30": stream["r_frame_rate"] == "30/1",
        "frame_count_expected": int(stream["nb_frames"]) == args.expected_frames,
        "duration_expected": abs(float(probe["format"]["duration"]) - args.expected_duration) < 0.02,
        "pixel_format": stream["pix_fmt"] == "yuv420p",
        "showcase_motion_present": len(hashes) >= 53 and len(set(hashes)) >= 45,
        "paired_motion_present": len(paired_hashes) >= 26 and len(set(paired_hashes)) >= 20,
        "four_rollout_suite_passed": suite_validation["passed"] is True,
        "official_logo_provenance_recorded": logo_source["source_page"].startswith("https://www.fudan.edu.cn/"),
    }
    report = {
        "passed": all(checks.values()),
        "video": str(video),
        "sha256": sha256(video),
        "probe": probe,
        "showcase_sampled_frames": len(hashes),
        "showcase_unique_sampled_frames": len(set(hashes)),
        "paired_sampled_frames": len(paired_hashes),
        "paired_unique_sampled_frames": len(set(paired_hashes)),
        "checks": checks,
    }
    report_path = video.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

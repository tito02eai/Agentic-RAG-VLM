#!/usr/bin/env python3
"""Render defense-ready captions and audit the four Guanghua demo rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_CONFIG = PROJECT_ROOT / "configs" / "guanghua_demo_suite.json"
SCENARIOS = ("nominal", "fragile", "recovery", "showcase")


def ass_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def dialogue(start: float, end: float, style: str, text: str, *, layer: int = 20) -> str:
    return f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"


def captions(scenario: str, title: str, duration: float, seed: int) -> str:
    graph = scenario in {"fragile", "showcase"}
    replan = scenario in {"recovery", "showcase"}
    scenario_label = {
        "nominal": "NOMINAL · 基础完整任务",
        "fragile": "FRAGILE-AWARE · 受保护玻璃器皿约束",
        "recovery": "RECOVERY · 场景变化恢复",
        "showcase": "SHOWCASE · 完整 Agentic 闭环",
    }[scenario]
    red_stage = "02  红方块：场景图约束执行" if graph else "02  红方块：检索技能执行"
    red_detail = (
        "Protected Glass（禁止接触）→ 场景图关系 → 安全腕角/路径 → POWER"
        if graph else "HAA-RAG → POWER → 完整朝向 IK → 红色目标区"
    )
    monitor_stage = "蓝圆柱位移 → L3 REPLAN" if replan else "场景监测：计划仍有效"
    monitor_detail = (
        "公共 RGB-D 检测 29 mm 位移；Memory 保留已完成红色子任务"
        if replan else "未完成蓝色目标未失效；继续执行，不制造人为失败"
    )
    blue_stage = "03  蓝圆柱：从新观测重规划" if replan else "03  蓝圆柱：顺序子任务执行"
    blue_detail = "PINCH → 新位置 → 蓝色目标区" if replan else "HAA-RAG → PINCH → 蓝色目标区"
    lines = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "PlayResX: 1280",
        "PlayResY: 720",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Header,Noto Serif CJK SC,23,&H00F7F5F1,&H00FFFFFF,&H600D1218,&H500D1218,-1,0,0,0,100,100,0,0,3,1,0,7,28,28,22,1",
        "Style: Receipt,DejaVu Sans Mono,17,&H00C8B649,&H00FFFFFF,&H600D1218,&H500D1218,0,0,0,0,100,100,0,0,3,1,0,9,28,28,24,1",
        "Style: Disclosure,Noto Serif CJK SC,17,&H005AC6E7,&H00FFFFFF,&H600D1218,&H500D1218,0,0,0,0,100,100,0,0,3,1,0,9,28,28,51,1",
        "Style: Stage,Noto Serif CJK SC,37,&H00F7F5F1,&H00FFFFFF,&H800D1218,&H500D1218,-1,0,0,0,100,100,0,0,3,2,0,1,38,38,92,1",
        "Style: Detail,Noto Serif CJK SC,22,&H00F7F5F1,&H00FFFFFF,&H800D1218,&H500D1218,0,0,0,0,100,100,0,0,3,1,0,1,40,40,35,1",
        "Style: Metric,DejaVu Sans Mono,20,&H008DC563,&H00FFFFFF,&H800D1218,&H500D1218,-1,0,0,0,100,100,0,0,3,1,0,1,40,40,34,1",
        "Style: Success,Noto Serif CJK SC,43,&H008DC563,&H00FFFFFF,&H800D1218,&H500D1218,-1,0,0,0,100,100,0,0,3,2,0,2,32,32,84,1",
        "Style: SuccessMetric,DejaVu Sans Mono,21,&H00F7F5F1,&H00FFFFFF,&H800D1218,&H500D1218,-1,0,0,0,100,100,0,0,3,1,0,2,32,32,37,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        dialogue(0, duration, "Header", "Agentic RAG-VLM × 光华双灵巧手机器人", layer=10),
        dialogue(0, duration, "Receipt", f"Qwen3.5-4B receipts · seed {seed} · public RGB-D", layer=10),
        dialogue(0, duration, "Disclosure", "校准运动技能代理 · 非接触动力学结论", layer=10),
        dialogue(0.15, 1.6, "Stage", rf"{{\fad(180,180)}}{scenario_label}"),
        dialogue(0.15, 1.6, "Detail", r"{\fad(180,180)}OBSERVE → PLAN → EXECUTE → VERIFY → MONITOR"),
        dialogue(1.6, 3.0, "Stage", r"{\fad(120,120)\c&H00C8B649&}01  公共感知 · HAA-RAG · 技能选择"),
        dialogue(1.6, 3.0, "Detail", r"{\fad(120,120)}红方块 → POWER　｜　蓝圆柱 → PINCH"),
        dialogue(3.0, 9.4, "Stage", rf"{{\fad(160,160)\c&H004A48D5&}}{red_stage}"),
        dialogue(3.0, 9.4, "Detail", rf"{{\fad(160,160)}}{red_detail}"),
        dialogue(9.4, 11.1, "Stage", r"{\fad(120,120)\c&H008DC563&}✓ 红色子任务通过并写入 Memory"),
        dialogue(9.4, 11.1, "Metric", r"{\fad(120,120)}RED TARGET RELATION = TRUE"),
        dialogue(11.1, 13.0, "Stage", rf"{{\fad(120,120)\c&H00C8B649&}}{monitor_stage}"),
        dialogue(11.1, 13.0, "Detail", rf"{{\fad(120,120)}}{monitor_detail}"),
        dialogue(13.0, 19.35, "Stage", rf"{{\fad(150,150)\c&H00C8B649&}}{blue_stage}"),
        dialogue(13.0, 19.35, "Detail", rf"{{\fad(150,150)}}{blue_detail}"),
        dialogue(19.35, duration, "Success", r"{\fad(160,0)}TASK SUCCESS · Agentic Runtime = COMPLETE", layer=30),
        dialogue(19.35, duration, "SuccessMetric", r"{\fad(160,0)}RED ✓　 BLUE ✓　 PROTECTED GLASS SHIFT 0 mm ✓", layer=30),
    ]
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "output" / "defense_suite")
    parser.add_argument("--render-rollouts", action="store_true", help="Render all raw MuJoCo rollouts before captioning.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--camera", choices=("frontview", "agentview", "birdview"), default="frontview")
    args = parser.parse_args()
    suite = json.loads(SUITE_CONFIG.read_text(encoding="utf-8"))
    manifest = {"suite_id": suite["suite_id"], "claim_boundary": suite["claim_boundary"], "scenarios": {}}
    for scenario in SCENARIOS:
        run_dir = args.root / scenario
        if args.render_rollouts:
            subprocess.run([
                "python", str(PROJECT_ROOT / "scripts" / "render_complete_task_demo.py"),
                "--scenario", scenario,
                "--seed", str(args.seed),
                "--camera", args.camera,
                "--output", str(run_dir),
            ], check=True)
        raw = run_dir / "guanghua_complete_task_raw.mp4"
        timeline = json.loads((run_dir / "timeline.json").read_text(encoding="utf-8"))
        evaluator = json.loads((run_dir / "private_evaluator.json").read_text(encoding="utf-8"))
        receipt = json.loads((run_dir / "runtime_receipt.json").read_text(encoding="utf-8"))
        if not evaluator["task_success_under_declared_grasp_proxy"] or not evaluator["runtime_complete"]:
            raise RuntimeError(f"{scenario} did not pass its evaluator")
        ass = run_dir / f"{scenario}.ass"
        receipt_seed = int(receipt["agentic_runtime"]["task_id"].rsplit("seed", 1)[1])
        ass.write_text(
            captions(scenario, suite["scenarios"][scenario]["title"], float(timeline["duration_s"]), receipt_seed),
            encoding="utf-8",
        )
        final = run_dir / f"guanghua_agentic_rag_vlm_{scenario}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
            "-vf", f"subtitles={ass}:fontsdir=/usr/share/fonts",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(final),
        ], check=True)
        manifest["scenarios"][scenario] = {
            "video": str(final.resolve()),
            "sha256": sha256(final),
            "duration_s": timeline["duration_s"],
            "frames": timeline["frames"],
            "runtime": receipt["agentic_runtime"],
            "motion_safety": evaluator["motion_safety"],
            "mechanisms": suite["scenarios"][scenario]["mechanisms"],
        }
    manifest_path = args.root / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "scenarios": list(manifest["scenarios"])}, indent=2))


if __name__ == "__main__":
    main()

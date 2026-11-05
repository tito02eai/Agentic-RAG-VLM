"""Run every declared YAM physics case once; retain failures, never rerun seeds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/yam_physics_qualification.json")
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18070/v1/chat/completions")
    parser.add_argument("--model", default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B")
    parser.add_argument("--paper-store",type=Path)
    parser.add_argument("--clip-endpoint",default="http://127.0.0.1:18072")
    args = parser.parse_args()
    if args.paper_store:
        from agentic_rag_vlm.clip_client import CLIPFeatureClient
        from agentic_rag_vlm.experience import ExperienceStore
        encoder = CLIPFeatureClient(args.clip_endpoint)
        store = ExperienceStore.load(args.paper_store)
        store.validate_sources(args.paper_store.parent)
        if encoder.embedding_space != store.embedding_space:
            parser.error("CLIP service and knowledge encoder differ")
    if args.output.exists():
        parser.error("use a new output directory")
    args.output.mkdir(parents=True)
    frozen = args.config.read_bytes()
    (args.output/"frozen_config.json").write_bytes(frozen)
    config = json.loads(frozen)
    # Every child imports the same archived source, even if development resumes
    # in the active checkout while the suite is running.
    archive = args.output/"source.zip"
    subprocess.run([sys.executable,str(ROOT/"scripts/build_delivery_bundle.py"),"--output",str(archive)],check=True,capture_output=True)
    runner_root = args.output.resolve()/"source"
    with ZipFile(archive) as snapshot:
        snapshot.extractall(runner_root)
    source_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    rows = []
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            folder = args.output/f"{scenario['name']}_seed_{seed:02d}"
            command = [sys.executable,str(runner_root/"scripts/run_yam_physics.py"),"--output",str(folder),"--seed",str(seed),
                       "--protocol",str(args.output/"frozen_config.json")]
            if scenario["neighbor"]:
                command += ["--neighbor"]
            if scenario.get("multiple"):
                command += ["--multiple"]
            if scenario.get("disable_memory"):
                command += ["--disable-memory"]
            if args.paper_store:
                relative_store = args.paper_store.resolve().relative_to(ROOT)
                command += ["--paper-store",str(runner_root/relative_store),"--clip-endpoint",args.clip_endpoint,
                            "--memory-out",str(folder/"memory.json")]
            if args.vlm:
                command += ["--vlm","--endpoint",args.endpoint,"--model",args.model]
            if seed == config["seeds"][0]:
                command += ["--record"]
            run = subprocess.run(command,env={**os.environ,"MUJOCO_GL":"egl"},capture_output=True,text=True)
            folder.mkdir(exist_ok=True)
            (folder/"process.log").write_text(run.stdout+run.stderr)
            result = json.loads((folder/"result.json").read_text()) if (folder/"result.json").exists() else {}
            rows.append({"scenario":scenario["name"], "seed":seed, "exit_code":run.returncode,
                         "physical_success":result.get("physical_success",False),
                         "outcome":result.get("outcome","process_error"),"vlm_calls":result.get("vlm_calls",0),
                         "memory_hits":result.get("memory_hits",0),"reasoner_calls":len(result.get("reasoner_calls",[])),
                         "second_object":result.get("second_object_evaluation"),
                         "metrics":result.get("private_evaluator",{}),"directory":str(folder)})
            print(json.dumps(rows[-1]),flush=True)
            summary = {"evidence_kind":"mujoco_contact_dynamics", "uses_vlm":args.vlm,
                "config_sha256":hashlib.sha256(frozen).hexdigest(),"expected_cases":len(config["seeds"])*len(config["scenarios"]),
                "source_archive_sha256":source_sha,
                "completed_cases":len(rows),"physical_successes":sum(r["physical_success"] for r in rows),
                "vlm_calls":sum(r["vlm_calls"] for r in rows), "cases":rows}
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            if not result:
                print("process failed before a task result; aborting remaining cases",flush=True)
                return 1
    return 0 if all(r["physical_success"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

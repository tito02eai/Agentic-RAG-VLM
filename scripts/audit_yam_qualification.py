"""Check frozen-source provenance, case completeness and raw physics evidence."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from zipfile import ZipFile


def audit(root):
    summary = json.loads((root/"summary.json").read_text())
    frozen = (root/"frozen_config.json").read_bytes()
    config = json.loads(frozen)
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(hashlib.sha256(frozen).hexdigest() == summary["config_sha256"],"config hash mismatch")
    archive = (root/"source.zip").read_bytes()
    check(hashlib.sha256(archive).hexdigest() == summary["source_archive_sha256"],"source archive hash mismatch")
    with ZipFile(root/"source.zip") as z:
        manifest = json.loads(z.read("SOURCE_SHA256.json"))
        for file, digest in manifest.items():
            check(hashlib.sha256(z.read(file)).hexdigest() == digest,f"source file hash mismatch: {file}")
    expected = set(itertools.product([s["name"] for s in config["scenarios"]],config["seeds"]))
    scenarios = {s["name"]:s for s in config["scenarios"]}
    observed = [(c["scenario"],c["seed"]) for c in summary["cases"]]
    check(set(observed) == expected and len(observed) == len(expected),"missing/duplicate/unexpected cases")
    successes, calls = 0, 0
    for case in summary["cases"]:
        folder = root/Path(case["directory"]).name
        result = json.loads((folder/"result.json").read_text())
        scenario = scenarios[case["scenario"]]
        label = f"{case['scenario']}/{case['seed']}"
        check(result["seed"] == case["seed"],f"{label}: seed mismatch")
        check(result["object_attachment_used"] is False and result["object_pose_reset_after_initialization"] is False,
              f"{label}: forbidden object manipulation flag")
        check(result["physical_success"] == all(result["qualification_checks"].values()),f"{label}: verdict mismatch")
        check(result["physical_success"] == case["physical_success"],f"{label}: summary mismatch")
        successes += result["physical_success"]
        traces = [json.loads(line) for line in (folder/"private_evaluator.jsonl").read_text().splitlines()]
        final = traces[-1]["target_xyz"]
        recorded = result["private_evaluator"]["final_target_xyz"]
        check(max(abs(a-b) for a,b in zip(final,recorded)) < 1e-5,f"{label}: final pose/trace mismatch")
        if result["physical_success"]:
            check(max(t["target_xyz"][2] for t in traces) > .10,f"{label}: no raw lift evidence")
            contacts = traces[-1]["contacts"]
            check(any(set(c[:2]) == {"target_geom","table"} for c in contacts),f"{label}: no raw support evidence")
            check(not any("target_geom" in c[:2] and any(n and n.endswith('collision') for n in c[:2])
                          for c in contacts),f"{label}: raw final finger contact")
            if scenario.get("multiple"):
                second = traces[-1]["objects"]["target_cube2"]
                destination = next(r["payload"]["sent"]["parameters"]["destination_xyz_m"] for r in result["receipts"]
                    if r["kind"] == "command" and r["payload"]["sent"]["command_type"] == "place"
                    and r["payload"]["sent"]["target"] == "cube2")
                import math
                check(math.dist(second,destination) < config["criteria"]["destination_error_m_lt"],f"{label}: raw second placement mismatch")
                check(any(set(c[:2]) == {"target_geom2","table"} for c in contacts),f"{label}: raw second support missing")
                check(result["second_object_evaluation"]["bilateral_lift_seconds"]>.5,f"{label}: second lift missing")
                plans=[e for e in result["trace"] if e["event"] == "subgoal_planned"]
                hits=sum(e["payload"]["action"]["source"] == "episodic_memory" for e in plans)
                check(hits == (0 if scenario.get("disable_memory") else 1),f"{label}: unexpected memory reuse")
                check(hits == result["memory_hits"],f"{label}: memory metric mismatch")
                check(len(result["reasoner_calls"])>=2,f"{label}: missing VLM action decisions")
                if scenario.get("disable_memory"):
                    check(not result["memory"],f"{label}: disabled memory was written")
        model_receipts = [r["payload"]["vlm_result"] for r in result["receipts"]
                          if r["kind"] == "perception" and r["payload"].get("vlm_result") is not None]
        calls += len(model_receipts)
        saved = sorted(folder.glob("observation_*/vlm_response.json"))
        check(len(saved) == len(model_receipts),f"{label}: missing raw model responses")
        for path, receipt in zip(saved,model_receipts):
            response = json.loads(path.read_text())
            check(response["raw_text"] == receipt["raw_text"],f"{label}: model receipt mismatch")
    check(successes == summary["physical_successes"],"aggregate success mismatch")
    check(calls == summary["vlm_calls"],"aggregate model call mismatch")
    return {"passed":not errors,"checked_cases":len(observed),"physical_successes":successes,
            "model_calls":calls,"source_files":len(manifest),"errors":errors,
            "scope":"artifact integrity and sampled raw evidence; not independent physics certification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.directory)
    except (OSError,ValueError,KeyError) as error:
        result = {"passed":False,"errors":[str(error)]}
    print(json.dumps(result,indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

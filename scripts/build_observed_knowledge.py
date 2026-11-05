"""Build a new labelled CLIP store from saved successful contact episodes.

This imports measured reconstructed runs, not the unavailable paper-era 116
entries. It never assigns outcome=1 to a hand-authored recommendation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.clip_client import CLIPFeatureClient
from agentic_rag_vlm.experience import ExperienceStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--clip-endpoint",default="http://127.0.0.1:18072")
    parser.add_argument("--domain",default="mjlab_yam_top_grasp_v1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new store file")
    encoder = CLIPFeatureClient(args.clip_endpoint)
    evidence_dir = args.output.parent/(args.output.stem+"_sources")
    evidence_dir.mkdir(parents=True,exist_ok=False)
    store = ExperienceStore(embedding_space=encoder.embedding_space,domain=args.domain)
    for folder in sorted(args.episodes.glob("single_seed_*")):
        path = folder/"result.json"
        data = json.loads(path.read_text())
        if not data.get("physical_success"):
            continue
        plan = next(e["payload"] for e in data["trace"] if e["event"] == "subgoal_planned")
        action = plan["action"]
        category,affordance,material,fragility,region = action["strategy"]["_semantic_signature"]
        response = json.loads((folder/"observation_001/vlm_response.json").read_text())
        candidate = next(name for name,label in response["parsed"].items() if label == action["target"])
        image = folder/"observation_001"/f"{candidate}.png"
        evidence = evidence_dir/folder.name
        evidence.mkdir()
        shutil.copy2(image,evidence/"initial.png")
        shutil.copy2(path,evidence/"result.json")
        shutil.copy2(folder/"public_trace.jsonl",evidence/"public_trace.jsonl")
        vector = encoder.encode([image])[0]
        commands = [json.loads(line) for line in (folder/"public_trace.jsonl").read_text().splitlines()]
        grip = next(e for e in commands if e["event"] == "gripper_parameters")
        strategy = {k:v for k,v in action["strategy"].items() if not k.startswith('_')}
        strategy["force_n"] = grip["applied_motor_force_cap_n"]
        strategy["base_force_n"] = grip["applied_motor_force_cap_n"]
        observed_node = next(n for n in plan["evidence"]["scene_graph"]["nodes"] if n["name"] == action["target"])
        store.add({"id":f"observed_{folder.name}","category":category,"superclass":observed_node["superclass"],
            "affordance_type":affordance,"material":material,"fragility":fragility,"region":region,
            "strategy":strategy,"outcome":1,"visual_embedding":vector.tolist(),
            "provenance":{"domain":store.domain,"source_episode":folder.name,"evidence_kind":"mujoco_contact_dynamics",
                "source_result_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "image_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
                "observation_image":str((evidence/"initial.png").relative_to(args.output.parent)),
                "source_result_path":str((evidence/"result.json").relative_to(args.output.parent)),
                "label_scope":"verified_complete_grasp_place",
                "force_semantics":"simulated actuator force cap; not calibrated fingertip force",
                "original_paper_experience":False}})
    if not store.records:
        raise ValueError("no verified observed episodes imported")
    store.save(args.output)
    print(json.dumps({"records":len(store.records),"embedding_space":store.embedding_space,"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()

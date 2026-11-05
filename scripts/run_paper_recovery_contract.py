"""Synthetic analytical contract with real VLM reasoning, NOT physical evidence.

Known actuator-demand fixture: 12 N fails a declared 14 N minimum; L1 raises
force to 14.4 N; the next same-category task reuses that corrected strategy.
This fixture tests software mechanisms and must not be counted as robot trials.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.experience import ExperienceStore,atomic_json
from agentic_rag_vlm.retrieval import HAARetriever
from agentic_rag_vlm.pipeline import AgenticPipeline
from agentic_rag_vlm.reasoning import VLMReasoner
from agentic_rag_vlm.qwen_adapter import OpenAICompatibleQwenVL
from agentic_rag_vlm.analytical import GraspMeasurements,analytical_factors
from dataclasses import asdict
from scripts.agentic_framework import PublicObject


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--endpoint",default="http://127.0.0.1:18070/v1/chat/completions")
    p.add_argument("--model",default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B")
    args=p.parse_args()
    if args.output.exists():p.error("use a new output directory")
    store=ExperienceStore.load(args.store)
    r=store.records[0]
    target=PublicObject("first",(.3,-.12,.018),r["category"],r["superclass"],r["affordance_type"],r["material"],r["fragility"],r["region"],1.)
    reasoner=VLMReasoner(OpenAICompatibleQwenVL(args.endpoint,args.model,max_tokens=250))
    reasoner.images=[args.store.parent/r["provenance"]["observation_image"]]
    agent=AgenticPipeline([],retriever=HAARetriever(store),reasoner=reasoner)
    agent.visual_features={"first":r["visual_embedding"],"second":r["visual_embedding"]}
    agent.embedding_space=store.embedding_space
    def execute(action):
        force=action.strategy["force_n"]
        measurements=GraspMeasurements(0,.036,.036,force,14,20,1,.04,.1,1,force>=14)
        return {"quality_factors":asdict(analytical_factors(measurements)),"success":force>=14,
            "failure_type":"slip" if force<14 else None,
            "public_feedback":{"fixture_minimum_secure_force_n":14,"commanded_force_n":force}}
    first=agent.run("fixture_first",target,{target.name:target},execute)
    second_target=replace(target,name="second",center_xyz_m=(.2,-.04,.018))
    second=agent.run("fixture_second",second_target,{second_target.name:second_target},execute)
    passed=first["success"] and first["attempts"]==2 and second["success"] and second["attempts"]==1
    atomic_json(args.output,{"passed":passed,"evidence_kind":"synthetic_analytical_contract_with_real_vlm",
        "physical_robot_trials":0,"first":first,"second":second,"model_calls":reasoner.calls})
    print(json.dumps({"passed":passed,"first_attempts":first["attempts"],"second_attempts":second["attempts"],"real_vlm_calls":len(reasoner.calls)}))
    return 0 if passed else 1

if __name__=="__main__":raise SystemExit(main())

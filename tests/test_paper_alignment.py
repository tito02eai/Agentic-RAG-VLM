"""Independent counterexamples to the paper-aligned mechanisms.

Two-dimensional vectors here are test fixtures, not CLIP or experiment evidence.
"""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from agentic_rag_vlm.experience import ExperienceStore
from agentic_rag_vlm.retrieval import HAARetriever,NoCompatibleExperience
from agentic_rag_vlm.memory import EpisodicMemory
from agentic_rag_vlm.pipeline import AgenticPipeline
from agentic_rag_vlm.quality import QualityFactors,evaluate_quality
from agentic_rag_vlm.recovery import FailureType,classify_failure,recover
from agentic_rag_vlm.runtime import TaskSpec,Subgoal
from agentic_rag_vlm.scene import schedule_dependencies,build_scene_graph,RELATIONS
from agentic_rag_vlm.constraints import compile_constraints
from scripts.agentic_framework import PublicObject

ROOT=Path(__file__).resolve().parents[1]

def query():
    return PublicObject("cup1",(.3,0,.1),"cup","container","graspable_body","ceramic",.6,"body",1.)

def record(name="good", **overrides):
    q=query()
    item={"id":name,"category":q.category,"superclass":q.superclass,"affordance_type":q.affordance_type,
        "material":q.material,"fragility":q.fragility,"region":q.region,
        "strategy":{"synergy":"power","force_n":8.,"aperture_m":.06,"approach":"top"},
        "outcome":1,"visual_embedding":[1.,0.],
        "provenance":{"domain":"test","source_episode":"fixture","evidence_kind":"test_fixture","image_sha256":"test"}}
    item.update(overrides)
    return item

def retriever(*records):
    store=ExperienceStore(embedding_space="test_vectors",dimension=2,domain="test")
    for item in records:store.add(item)
    return HAARetriever(store)

def test_exact_equation_and_failed_experience_retained():
    r=retriever(record("ok"),record("failed",outcome=0))
    result=r.retrieve(query(),[1.,0.],embedding_space="test_vectors")
    assert result.selected_card_id=="ok"
    scores={e["card_id"]:e["final_score"] for e in result.ranked_cards}
    assert scores==pytest.approx({"ok":1.,"failed":.7})
    assert result.stages["negative_examples"][0]["card_id"]=="failed"

def test_affordance_rejection_and_no_unrelated_fallback():
    bad=record("bad",affordance_type="tool_handle",material="metal",fragility=0.,region="handle")
    with pytest.raises(NoCompatibleExperience):retriever(bad).retrieve(query(),[1,0],embedding_space="test_vectors")
    with pytest.raises(NoCompatibleExperience):
        retriever(record(category="screwdriver",superclass="tool")).retrieve(query(),[1,0],embedding_space="test_vectors")

def test_stage1_keeps_exact_and_fuzzy_but_caps_thirty():
    r=retriever(*(record(f"{i:02d}",category="cup" if i<20 else "mug") for i in range(40)))
    got=r.retrieve(query(),[1,0],embedding_space="test_vectors")
    assert got.stages["category_matches"]==40 and got.stages["category_top30"]==30
    assert len(got.ranked_cards)==3

def test_real_cosine_and_weighting_not_aspect_proxy():
    r=retriever(record("orthogonal",visual_embedding=[0,1]),record("aligned"))
    got=r.retrieve(query(),[1,0],embedding_space="test_vectors")
    scores={x["card_id"]:x for x in got.ranked_cards}
    assert scores["orthogonal"]["visual_score"]==0
    assert scores["orthogonal"]["final_score"]==pytest.approx(.6)
    with pytest.raises(ValueError):r.retrieve(query(),[1,0],embedding_space="wrong")

def test_store_roundtrip_failure_and_provenance(tmp_path):
    r=retriever(record(outcome=0));p=tmp_path/"store.json";r.store.save(p)
    assert ExperienceStore.load(p).records[0]["outcome"]==0
    with pytest.raises(ValueError):r.store.add(record("invented",provenance={}))
    with pytest.raises(ValueError):r.store.add(record("nan",visual_embedding=[float('nan'),0]))

def test_session_memory_reapplies_current_constraints_and_current_pose(tmp_path):
    cards=json.loads((ROOT/"knowledge_base/blupe_affordance_cards.json").read_text())["cards"]
    p=AgenticPipeline(cards)
    first=query();vase=replace(first,name="vase",center_xyz_m=(.375,0,.1),fragility=1.)
    action,_=p.plan(first,{first.name:first,"vase":vase})
    p.remember("cup",action,.9,"first_verified")
    p.memory.save(tmp_path/"session/memory.json")
    restored=AgenticPipeline(cards);restored.memory=EpisodicMemory.load(tmp_path/"session/memory.json")
    second=replace(first,name="cup2",center_xyz_m=(.2,.2,.1))
    reused,evidence=restored.plan(second,{second.name:second})
    assert reused.source=="episodic_memory" and reused.position_xyz_m==second.center_xyz_m
    assert reused.strategy["force_n"]==pytest.approx(8.)
    assert reused.strategy["approach_height_delta_m"]==0
    assert evidence["memory"]["source_episode"]=="first_verified"
    incompatible=replace(second,material="glass")
    assert restored.plan(incompatible,{incompatible.name:incompatible})[0].source=="haa_rag"
    external=restored.memory.get("cup");external.strategy["force_n"]=999
    assert restored.memory.get("cup").strategy["force_n"]!=999

def test_support_prerequisite_insertion_and_cycle_protection():
    task=TaskSpec("support","remove cup",(Subgoal("base","zone"),))
    context={"relations":[{"source":"base","target":"top","relation":"supports"}]}
    scheduled=schedule_dependencies(task,context,prerequisite_destinations={"top":"safe"})
    assert [s.target for s in scheduled.subgoals]==["top","base"]
    with pytest.raises(ValueError):schedule_dependencies(replace(task,protected_objects=("top",)),context,prerequisite_destinations={"top":"safe"})
    context["relations"].append({"source":"top","target":"base","relation":"supports"})
    with pytest.raises(ValueError):schedule_dependencies(task,context,prerequisite_destinations={"top":"safe"})

def test_content_and_occlusion_require_validated_physical_path():
    q=query();block=replace(q,name="block",center_xyz_m=(.34,0,.1))
    objects={q.name:q,"block":block}
    context={"states":{q.name:"filled"},"relations":[{"source":"block","target":q.name,"relation":"occludes"}]}
    blocked=compile_constraints(q,objects,context)
    assert blocked["blocked"]
    validated=compile_constraints(q,objects,{"relations":context["relations"]},path_validator=lambda p,o: True)
    assert not validated["blocked"] and validated["required_approach"]=="lateral"
    assert len(RELATIONS)==16
    graph=build_scene_graph(objects,context)
    assert graph["analysis_radius_m"]==.15 and any(n["state"]=="filled" for n in graph["nodes"])

@pytest.mark.parametrize("failure",list(FailureType))
def test_all_fourteen_failures_are_classified_and_recovery_bounded(failure):
    assert classify_failure({"failure_type":failure.value})==failure
    strategy={"synergy":"power","force_n":10.,"aperture_m":.05}
    assert recover(strategy,failure,1).level==1
    l2=recover(strategy,failure,2)
    assert l2.corrected_strategy["synergy"]=="pinch" and l2.corrected_strategy["force_n"]==9
    one=recover(strategy,failure,3,rng=np.random.default_rng(5)).corrected_strategy["position_perturbation_m"]
    two=recover(strategy,failure,3,rng=np.random.default_rng(5)).corrected_strategy["position_perturbation_m"]
    assert one==two and np.max(np.abs(one))<=.01
    with pytest.raises(ValueError):recover(strategy,failure,4)

def test_nonfinite_quality_and_explicit_physical_failure_cannot_succeed():
    with pytest.raises(ValueError):evaluate_quality(QualityFactors(float('nan'),1,1,1,1,1,1))
    assert not evaluate_quality(QualityFactors(1,1,1,1,1,1,1),observed_success=False).success
    assert not evaluate_quality(QualityFactors(1,1,1,1,1,1,1),observed_success=np.bool_(False)).success
    with pytest.raises(ValueError):evaluate_quality(QualityFactors(1,1,1,1,1,1,1),observed_success="false")


def test_corrected_strategy_reduces_next_session_subgoal_attempts():
    cards=json.loads((ROOT/"knowledge_base/blupe_affordance_cards.json").read_text())["cards"]
    p=AgenticPipeline(cards)
    def executor(action):
        # Synthetic mechanism fixture, not a physical grasp success metric.
        return {"quality_factors":{key:1. for key in ("position_accuracy","width_compatibility","force_appropriateness",
            "grasp_type_match","approach_clearance","object_difficulty","grip_security")},
            "success":action.strategy["force_n"]>=9,"failure_type":"slip"}
    first=p.run("first",query(),{query().name:query()},executor)
    other=replace(query(),name="cup2",center_xyz_m=(.1,.2,.1))
    second=p.run("second",other,{other.name:other},executor)
    assert first["attempts"]==2 and second["attempts"]==1
    assert second["trace"][0]["action"]["source"]=="episodic_memory"
    assert second["trace"][0]["action"]["strategy"]["force_n"]==pytest.approx(9.6)


def test_vlm_thought_precedes_executor_and_rejection_cannot_move():
    calls=[]
    class Reasoner:
        negative_experiences=[]
        def think(self,*args):calls.append("think");return {"approved":True}
    cards=json.loads((ROOT/"knowledge_base/blupe_affordance_cards.json").read_text())["cards"]
    p=AgenticPipeline(cards,reasoner=Reasoner())
    def execute(action):
        calls.append("execute")
        return {"quality_factors":{name:1 for name in QualityFactors.__dataclass_fields__}}
    p.run("order",query(),{query().name:query()},execute)
    assert calls==["think","execute"]
    def reject(*args):raise ValueError("model veto")
    p.reasoner.think=reject
    with pytest.raises(ValueError):p.run("reject",query(),{query().name:query()},execute)
    assert calls==["think","execute"]


def test_bundled_observed_clip_store_has_matching_source_evidence():
    path=ROOT/"knowledge_base/yam_observed_clip.json"
    store=ExperienceStore.load(path)
    assert store.dimension==768 and len(store.records)==10
    assert store.embedding_space.startswith("openai/clip-vit-large-patch14@sha256:")
    store.validate_sources(path.parent)


def test_filled_container_cannot_be_overridden_by_lateral_occlusion_rule():
    q=query();block=replace(q,name="block",center_xyz_m=(.34,0,.1))
    constraints=compile_constraints(q,{q.name:q,"block":block},
        {"states":{q.name:"filled"},"relations":[{"source":"block","target":q.name,"relation":"occludes"}]},
        path_validator=lambda *args:True)
    assert constraints["blocked"] and constraints["required_approach"]=="top"


def test_swept_envelope_accepts_clear_route_and_rejects_obstacle():
    from agentic_rag_vlm.paths import EnvelopePathValidator
    q=query();obstacle=replace(q,name="block",center_xyz_m=(.35,0,.1))
    validate=EnvelopePathValidator({q.name:[.01]*3,"block":[.01]*3},gripper_half_extents=[.01]*3,
        workspace_bounds=[[0,.6],[-.4,.4],[.02,.4]],table_z=0)
    proposal={"target":q.name,"approach_offset_xy_m":[-.05,0]}
    assert validate(proposal,{q.name:q,"block":obstacle})
    assert len(proposal["waypoints_xyz_m"])==3
    assert not validate(proposal,{q.name:q,"block":replace(obstacle,center_xyz_m=(.305,0,.1))})


def test_session_executes_support_removal_before_original_target():
    from agentic_rag_vlm.runtime import AgenticRuntime
    from agentic_rag_vlm.platform import BluPeSession,MockPlatformTransport,RegistryMetadataPerception,GraspActionMapper,SceneChangeVerifier
    from scripts.agentic_framework import load_semantic_registry
    cards=json.loads((ROOT/"knowledge_base/blupe_affordance_cards.json").read_text())["cards"]
    registry=load_semantic_registry(ROOT/"knowledge_base/blupe_semantic_registry.json")
    class ObservedRelations(RegistryMetadataPerception):
        def perceive(self,observation):
            objects,receipt=super().perceive(observation)
            supported=objects["cube"].center_xyz_m[1]<.1
            relations=[{"source":"cup","target":"cube","relation":"supports"}] if supported else []
            return objects,replace(receipt,vlm_result={"scene_context":{"relations":relations}})
    transport=MockPlatformTransport({"cup":[.3,0,.1],"cube":[.3,0,.14]})
    session=BluPeSession(AgenticRuntime(AgenticPipeline(cards)),transport,ObservedRelations(registry),
        GraspActionMapper(),SceneChangeVerifier(),destinations={"zone":[.1,.2,.1],"safe":[.4,.2,.1]},
        prerequisite_destinations={"cube":"safe"})
    outcome=session.run(TaskSpec("support","remove cup",(Subgoal("cup","zone"),)))
    assert outcome["outcome"]=="complete"
    targets=[r["payload"]["sent"]["target"] for r in outcome["receipts"] if r["kind"]=="command" and r["payload"]["sent"]["command_type"]=="pick"]
    assert targets==["cube","cup"]


def test_strict_l3_refreshes_clip_features_and_resets_failed_parameters():
    from agentic_rag_vlm.observation import PlanningObservation
    q=query();r=retriever(record())
    agent=AgenticPipeline([],retriever=r,seed=123)
    agent.visual_features={q.name:[1.,0.]};agent.embedding_space="test_vectors"
    moved=replace(q,center_xyz_m=(.2,.1,.1))
    def execute(action):
        ok=action.source=="recovery_l3"
        return {"quality_factors":{name:float(ok) for name in QualityFactors.__dataclass_fields__},
            "success":ok,"failure_type":"orientation"}
    result=agent.run("strict_l3",q,{q.name:q},execute,observer=lambda:PlanningObservation(
        {q.name:moved},{q.name:[0.,1.]},"test_vectors","fresh_frame"))
    assert result["success"] and result["attempts"]==4
    replan=result["trace"][2]["replan_evidence"]
    assert replan["retrieval"]["ranked_cards"][0]["visual_score"]==0
    action=result["trace"][-1]["action"]
    assert action["strategy"]["force_n"]==8 and action["strategy"]["synergy"]=="power"
    assert "wrist_yaw_delta_rad" not in action["strategy"]
    assert np.max(np.abs(np.asarray(action["position_xyz_m"])-moved.center_xyz_m))<=.01


def test_language_task_planner_cannot_invent_objects_or_coordinates():
    from agentic_rag_vlm.task_planner import VLMTaskPlanner
    from agentic_rag_vlm.qwen_adapter import VLMResult
    class Client:
        parsed={"subgoals":[{"target":"cup1","destination":"zone"}]}
        def infer(self,*args):return VLMResult(self.parsed,json.dumps(self.parsed),1,"test","test",{})
    client=Client();planner=VLMTaskPlanner(client)
    task,_=planner.plan("nl","move the cup",{"cup1":query()},["zone"])
    assert task.subgoals==(Subgoal("cup1","zone"),)
    for goal in ({"target":"invented","destination":"zone"},
                 {"target":"cup1","destination":"unknown"},
                 {"target":"cup1","destination":"zone","xyz":[999,999,999]}):
        client.parsed={"subgoals":[goal]}
        with pytest.raises(ValueError):planner.plan("nl","move cup",{"cup1":query()},["zone"])

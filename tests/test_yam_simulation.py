"""Independent identity-contract counterexamples and contact-dynamics smoke tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agentic_rag_vlm.platform.perception import PerceptionError
from agentic_rag_vlm.platform.transport import PlatformObservation
from agentic_rag_vlm.qwen_adapter import VLMResult
from agentic_rag_vlm.simulation.perception import CandidateRGBDVLMPerception
from scripts.agentic_framework import load_semantic_registry

ROOT = Path(__file__).resolve().parents[1]


def candidate_observation(tmp_path):
    return PlatformObservation("frame", (), receipt={"rgbd_candidates":[
        {"candidate_id":"candidate_0","top_xyz_m":[.3,-.12,.036],"crop_path":str(tmp_path/"crop.png")},
        {"candidate_id":"candidate_1","top_xyz_m":[.375,-.12,.08],"crop_path":str(tmp_path/"crop2.png")}]})


def classifier(labels):
    class Client:
        def infer(self, images, prompt):
            return VLMResult(labels,json.dumps(labels),1,"test_model","test_model",{})
    registry = load_semantic_registry(ROOT/"knowledge_base/blupe_semantic_registry.json")
    return CandidateRGBDVLMPerception(Client(),registry,{"cube":.018,"vase":.04})


def test_vlm_identity_joins_depth_and_known_dimensions(tmp_path):
    objects, receipt = classifier({"candidate_0":"cube","candidate_1":"vase"}).perceive(candidate_observation(tmp_path))
    assert objects["cube"].center_xyz_m == pytest.approx([.3,-.12,.018])
    assert objects["vase"].center_xyz_m == pytest.approx([.375,-.12,.04])
    assert receipt.mode == "vlm_classification_color_rgbd_proposals"
    assert objects["vase"].fragility == 1


@pytest.mark.parametrize("labels", [None,{}, {"candidate_0":"cube"},
    {"candidate_0":"cube","candidate_1":"cube"},
    {"candidate_0":"cube","candidate_1":"bottle"},
    {"candidate_0":"cube","candidate_1":"vase","metric_xyz":[999,999,999]}])
def test_bad_model_identity_never_falls_back_to_color(tmp_path, labels):
    with pytest.raises(PerceptionError):
        classifier(labels).perceive(candidate_observation(tmp_path))
    assert (tmp_path/"vlm_response.json").exists()


def test_unknown_is_omitted_not_invented(tmp_path):
    objects, _ = classifier({"candidate_0":"unknown","candidate_1":"vase"}).perceive(candidate_observation(tmp_path))
    assert "cube" not in objects


@pytest.mark.parametrize("neighbor", [False,True])
def test_real_free_object_grasp_release_and_contact_checks(tmp_path, neighbor):
    pytest.importorskip("mujoco")
    command = [sys.executable,str(ROOT/"scripts/run_yam_physics.py"),"--output",str(tmp_path/"run")]
    if neighbor:
        command += ["--neighbor"]
    run = subprocess.run(command,env={**os.environ,"MUJOCO_GL":"egl"},capture_output=True,text=True,timeout=90)
    assert run.returncode == 0, run.stdout+run.stderr
    result = json.loads((tmp_path/"run/result.json").read_text())
    assert result["physical_success"] and result["outcome"] == "complete"
    assert result["object_attachment_used"] is False
    assert result["object_pose_reset_after_initialization"] is False
    assert result["private_evaluator"]["bilateral_contact_above_0_10m_seconds"] > .5
    assert result["private_evaluator"]["protected_robot_contacts"] == 0


def test_unreachable_destination_stops_before_gripper_motion(tmp_path):
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL","egl")
    from agentic_rag_vlm.simulation.yam import YamPhysics
    from scripts.run_yam_physics import PhysicsTransport
    from agentic_rag_vlm.platform.commands import PlatformCommand
    sim = YamPhysics(tmp_path/"world")
    try:
        sim.setup["destination"] = [2,2,.018]
        time_before = sim.data.time
        with pytest.raises(RuntimeError,match="IK failed"):
            PhysicsTransport(sim).send_command(PlatformCommand("bad","pick","cube","test",
                {"approach":"top","position_xyz_m":[.30,-.12,.018]}))
        assert sim.data.time == time_before
        assert not any(row["event"] == "motion" for row in sim.public_trace)
    finally:
        sim.renderer.close()


def test_table_collision_stops_actuator_rollout(tmp_path):
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL","egl")
    from agentic_rag_vlm.simulation.yam import YamPhysics, PhysicsSafetyError
    sim = YamPhysics(tmp_path/"world")
    try:
        # An intentionally invalid empty-gripper descent is a guard test,
        # not a task episode and not counted in the qualification success rate.
        with pytest.raises(PhysicsSafetyError):
            sim.move([.26,.12,-.015],duration=.5,stage="invalid_table_descent")
        assert sim.forbidden_contacts
    finally:
        sim.renderer.close()

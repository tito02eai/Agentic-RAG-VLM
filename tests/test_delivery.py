"""First-delivery geometry, native model and admission counterexamples."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
import time

from agentic_rag_vlm.platform.geometry import PlaneProjector, GeometryError
from agentic_rag_vlm.platform.native_kinematics import NativeYamKinematics
from agentic_rag_vlm.platform.deployment import validate_profile, preflight
from agentic_rag_vlm.platform.perception import GroundedVLMPerception, PerceptionError
from agentic_rag_vlm.platform.transport import PlatformObservation
from agentic_rag_vlm.platform.transport import RemotePlatformTransport, PlatformUnavailableError
from agentic_rag_vlm.platform.config import load_platform_config
from agentic_rag_vlm.pipeline import AgenticPipeline
from agentic_rag_vlm.pipeline import PipelineConfig
from agentic_rag_vlm.quality import WEIGHTS
from agentic_rag_vlm.constraints import compile_constraints
from scripts.agentic_framework import load_semantic_registry, PublicObject

ROOT = Path(__file__).resolve().parents[1]


def calibration():
    return {"calibration_id": "test-cal", "cameras": {"top": {
        "K": [[500, 0, 320], [0, 500, 180], [0, 0, 1]], "image_size": [640, 360],
        "distortion_model": "none", "distortion": [0]*5,
        "T_base_camera": {"right": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]]},
    }}}


def projector():
    return PlaneProjector(calibration(), "right", plane_normal_base=[0, 0, 1], plane_offset_m=0)


def test_pixel_ray_intersects_measured_surface():
    p = projector()
    assert p.project([370, 180]) == pytest.approx([0.1, 0, 1])
    assert p.project_base([370, 180], height_m=0.1) == pytest.approx([0.09, 0, 0.1])


@pytest.mark.parametrize("pixel", [[-1, 2], [640, 0], [0, 360], [float("nan"), 0], [1, 2, 3]])
def test_grounding_rejects_invalid_pixels(pixel):
    with pytest.raises(GeometryError):
        projector().project(pixel)


def test_distortion_roundtrip():
    c = calibration()
    c["cameras"]["top"].update(distortion_model="opencv_brown_conrady", distortion=[.15, -.3, .002, -.001, .2])
    p = PlaneProjector(c, "right", plane_normal_base=[0, 0, 1], plane_offset_m=0)
    xy = np.array([.3, .15])
    pixel = (p.K @ np.r_[p._distort(xy), 1])[:2]
    assert p.project(pixel) == pytest.approx([.3, .15, 1], abs=1e-6)


def test_plane_behind_camera_rejected():
    p = PlaneProjector(calibration(), "right", plane_normal_base=[0, 0, 1], plane_offset_m=-2)
    with pytest.raises(GeometryError):
        p.project([320, 180])


def test_grounded_perception_ignores_vlm_metric_guess(tmp_path):
    registry = load_semantic_registry(ROOT / "knowledge_base/blupe_semantic_registry.json")
    class VLM:
        def infer(self, images, prompt):
            assert len(images) == 1
            return SimpleNamespace(parsed={"objects": [{"name": "cup", "pixel_xy": [370, 180],
                                    "center_xyz_m": [999, 999, 999]}]}, raw_text="saved response")
    perception = GroundedVLMPerception(VLM(), registry, projector(), {"cup": .1})
    from PIL import Image
    frame = tmp_path / "top.jpg"
    Image.new("RGB", (640, 360)).save(frame)
    obs = PlatformObservation("sample1", (), (str(frame),), {"calibration_id": "test-cal", "camera_role": "top"})
    objects, receipt = perception.perceive(obs)
    assert objects["cup"].center_xyz_m == pytest.approx([.09, 0, .1])
    assert receipt.mode == "vlm_measured_plane"


@pytest.mark.parametrize("receipt", [{}, {"camera_role": "left", "calibration_id": "test-cal"},
                                    {"camera_role": "top", "calibration_id": "different"}])
def test_grounded_perception_rejects_wrong_camera_or_calibration(receipt):
    registry = load_semantic_registry(ROOT / "knowledge_base/blupe_semantic_registry.json")
    perception = GroundedVLMPerception(None, registry, projector(), {"cup": .1})
    with pytest.raises(PerceptionError):
        perception.perceive(PlatformObservation("sample", (), ("top.jpg",), receipt))


def native_fixture():
    return {"calibration_id": "synthetic-native", "kinematics": {"joint_units": "degrees", "right": {
        "joints": [{"axis": [0, 0, 1], "position_m": [i*.02, 0, 0]} for i in range(6)],
        "T_base_grasp_zero": [[1, 0, 0, .3], [0, 1, 0, 0], [0, 0, 1, .2], [0, 0, 0, 1]],
    }}}


def test_native_home_and_single_axis_rotation():
    kin = NativeYamKinematics(native_fixture(), "right", [[-180, 180]]*6)
    assert kin.fk(np.zeros(6))[:3, 3] == pytest.approx([.3, 0, .2])
    assert kin.fk(np.array([np.pi/2, 0, 0, 0, 0, 0]))[:3, 3] == pytest.approx([0, .3, .2])
    # No URDF TCP extension or hidden native joint shift is added.
    assert kin.tcp_offset_m == 0


def test_published_native_kinematics_fk_ik_roundtrip():
    data = json.loads((ROOT / "tests/fixtures/native_yam_model.json").read_text())
    # Wide synthetic limits isolate FK/IK mathematics; not station admission.
    kin = NativeYamKinematics(data, "right", [[-180, 180]]*6)
    zero = kin.fk(np.zeros(6))
    assert np.allclose(zero, data["kinematics"]["right"]["T_base_grasp_zero"])
    q = np.radians([5, 15, -10, 8, 5, -3])
    target = kin.fk(q)
    result = kin.ik(target[:3, 3], target[:3, :3], q_init_rad=q + .01)
    assert result.success and result.within_limits
    assert np.linalg.norm(kin.fk(result.joints_rad)[:3, 3] - target[:3, 3]) <= kin.ik_position_tol_m


def test_profile_template_cannot_command_hardware():
    template = json.loads((ROOT / "configs/blupe_deployment.template.json").read_text())
    with pytest.raises(GeometryError):
        validate_profile(template, calibration(), robot_id="yam-1", arm="right")


def measured_test_profile():
    return {"robot_id": "yam-1", "arm": "right", "calibration_id": "test-cal", "coordinate_frame": "arm_base",
            "physical_validation": {"passed": True, "record": "SYNTHETIC TEST ONLY"},
            "support_plane": {"normal_base": [0, 0, 1], "offset_m": 0},
            "joint_limits_deg": [[-180, 180]]*6, "workspace_bounds_m": [[-.5, .5], [-.5, .5], [.01, 1.]],
            "feature_heights_m": {"cup": .1}, "grasp_offsets_base_m": {"cup": [0, 0, -.04]},
            "gripper": {"max_aperture_m": .085}, "approach": "top", "max_observation_age_s": 2}


def test_runner_and_cli_planner_share_measured_motion_configuration():
    from agentic_rag_vlm.platform.motion import YamMotionPlanner
    profile = measured_test_profile()
    profile.update(max_joint_delta_deg_per_step=1.25, lift_height_m=.11,
                   observation_park_xyz_m=[.2, -.1, .3])
    planner = YamMotionPlanner.from_deployment(native_fixture(), profile)
    assert isinstance(planner.kinematics, NativeYamKinematics)
    assert planner.max_joint_delta_deg_per_step == 1.25
    assert planner.lift_height_m == .11
    assert planner.observation_park_xyz_m == [.2, -.1, .3]
    assert planner.cartesian_step_m == .01
    assert planner.grasp_offset_m == planner.place_offset_m == 0.
    assert planner.kinematics.max_aperture_m == .085


def test_valid_profile_and_native_joint_limits_are_explicit():
    c = calibration()
    c["kinematics"] = native_fixture()["kinematics"]
    validate_profile(measured_test_profile(), c, robot_id="yam-1", arm="right")
    bad = measured_test_profile()
    bad["grasp_offsets_base_m"] = {}
    with pytest.raises(GeometryError, match="offset required"):
        validate_profile(bad, c, robot_id="yam-1", arm="right")


def test_remote_grounding_preserves_role_timestamp_and_unique_paths(tmp_path):
    class Client:
        image_metadata = {}
        def get_station_observation(self, robot):
            return {"observed_at": time.time(), "source": "hardware", "settled": True}
        def get_calibration(self, robot):
            return calibration()
        def download_observation_images(self, station, directory):
            result = {role: str(directory / f"{role}.jpg") for role in ("left", "top", "right")}
            self.image_metadata.update({path: {"X-Captured-At": str(time.time())} for path in result.values()})
            return result
    transport = RemotePlatformTransport(load_platform_config(ROOT / "configs/blupe_platform.example.json"),
        client=Client(), deployment_profile=measured_test_profile(), image_dir=tmp_path)
    first, second = transport.get_observation(), transport.get_observation()
    assert first.image_paths[0].endswith("top.jpg") and len(first.image_paths) == 1
    assert first.image_paths != second.image_paths
    assert first.receipt["calibration_id"] == "test-cal" and first.receipt["camera_role"] == "top"
    assert transport._camera_to_base([.1, .2, .3]) == (.1, .2, .3)  # already grounded; no second transform
    with pytest.raises(PlatformUnavailableError):
        transport._check_age(time.time() - 10, "test")
    with pytest.raises(PlatformUnavailableError):
        transport._check_age(None, "test")


def test_preflight_only_reads_and_reports_disabled_station():
    calls = []
    class Client:
        def list_robots(self):
            calls.append("GET robots")
            return [{"robot_id": "yam-1"}]
        def get_queue(self, robot):
            calls.append("GET queue")
            return {}
        def get_station_observation(self, robot):
            calls.append("GET observation")
            return {"mode": "DISABLED", "safety": {"ok": False}}
        def get_calibration(self, robot):
            calls.append("GET calibration")
            return calibration()
    result = preflight(Client(), robot_id="yam-1", arm="right")
    assert not result["ready_for_session"]
    assert "station_safety_not_ready" in result["blockers"]
    assert len(calls) == 4 and all(call.startswith("GET") for call in calls)


def test_scene_constraint_preserves_grasp_point_and_memory_force():
    registry = load_semantic_registry(ROOT / "knowledge_base/blupe_semantic_registry.json")
    cards = json.loads((ROOT / "knowledge_base/blupe_affordance_cards.json").read_text())["cards"]
    def obj(name, xyz):
        entry = registry[name]
        return PublicObject(name, xyz, **{k: entry[k] for k in (
            "category", "superclass", "affordance_type", "material", "fragility", "region")}, visual_aspect=1)
    cup = obj("cup", (.3, 0, .8))
    vase = obj("vase", (.34, .03, .8))
    pipeline = AgenticPipeline(cards)
    action, _ = pipeline.plan(cup, {"cup": cup, "vase": vase})
    assert action.constraint["active"]
    assert action.position_xyz_m == cup.center_xyz_m
    pipeline.memory.put(cup.category, action.strategy, 1, "first")
    second, _ = pipeline.plan(cup, {"cup": cup, "vase": vase})
    assert second.strategy["force_n"] == action.strategy["force_n"]
    new_material = replace(cup, material="glass", fragility=1.0)
    incompatible, _ = pipeline.plan(new_material, {"cup": new_material})
    assert incompatible.source == "haa_rag"
    from dataclasses import replace as change
    corrected = change(action, strategy={**action.strategy, "force_n": action.strategy["force_n"] * 1.2})
    pipeline.remember(cup.category, corrected, 1, "recovered")
    reused, _ = pipeline.plan(cup, {"cup": cup, "vase": vase})
    assert reused.strategy["force_n"] == pytest.approx(corrected.strategy["force_n"])


def test_support_and_occlusion_constraints_block_unvalidated_paths():
    cup = PublicObject("cup", (.3, 0, .1), "cup", "container", "graspable_body", "ceramic", .6, "body", 1)
    block = replace(cup, name="cube", category="cube", fragility=.1)
    objects = {"cup": cup, "cube": block}
    support = compile_constraints(cup, objects, {"relations": [{"source": "cup", "target": "cube", "relation": "supports"}]})
    assert support["blocked"] and support["prerequisites"] == ["cube"]
    occluded = compile_constraints(cup, objects, {"relations": [{"source": "cube", "target": "cup", "relation": "occludes"}]})
    assert occluded["blocked"]
    filled = compile_constraints(cup, objects, {"states": {"cup": "filled"}})
    assert filled["required_approach"] == "top" and filled["velocity_scale"] == .5


def test_l3_replanning_requires_and_uses_fresh_observation():
    cards = json.loads((ROOT / "knowledge_base/blupe_affordance_cards.json").read_text())["cards"]
    cup = PublicObject("cup", (.3, 0, .1), "cup", "container", "graspable_body", "ceramic", .6, "body", 1)
    moved = replace(cup, center_xyz_m=(.2, .1, .1))
    actions = []
    def executor(action):
        actions.append(action)
        ok = action.source == "recovery_l3"
        return {"quality_factors": {key: 1.0 if ok else 0.0 for key in WEIGHTS}, "failure_type": "orientation"}
    pipe = AgenticPipeline(cards, PipelineConfig(max_retries=3))
    observed = []
    def observe():
        observed.append(True)
        return {"cup": moved}
    result = pipe.run("l3", cup, {"cup": cup}, executor, observer=observe)
    assert result["success"] and len(actions) == 4 and len(observed) == 1
    # Paper L3 perturbs a freshly observed pose, never the stale original pose.
    perturbation = np.asarray(actions[-1].strategy["position_perturbation_m"])
    assert np.all(np.abs(perturbation) <= .01)
    assert actions[-1].position_xyz_m == pytest.approx(np.asarray(moved.center_xyz_m)+perturbation)
    actions.clear()
    result = AgenticPipeline(cards).run("no-observer", cup, {"cup": cup}, executor)
    assert not result["success"] and len(actions) == 3
    assert result["trace"][-1]["recovery_stop"] == "fresh_observation_required"

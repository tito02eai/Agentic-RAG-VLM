from __future__ import annotations

from pathlib import Path
import json
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from run_guanghua_env import (  # noqa: E402
    apply_declared_perturbation,
    configure_scene,
    initialize_position_targets,
)
from guanghua_control import (  # noqa: E402
    HAND_SYNERGIES,
    RIGHT_HAND_JOINTS,
    RightArmController,
    command_hand_synergy,
)
from grasp_planning import (  # noqa: E402
    calibrate_grasp_frame,
    evaluate_top_down_candidate,
    hand_visual_geoms,
    minimum_geom_clearance,
    top_down_rotation,
)
from guanghua_perception import (  # noqa: E402
    _color_masks,
    _largest_connected_component,
    _object_surface_mask,
)


ASSET_ROOT = PROJECT_ROOT / "assets" / "guanghua_hand_env"
MJCF = ASSET_ROOT / "mjcf" / "guanghua_hand_env.xml"
URDF = ASSET_ROOT / "urdf" / "guanghua_robot_with_hands.urdf"
PROTOCOL = PROJECT_ROOT / "configs" / "guanghua_experiment_protocol.json"
DEMO_CONFIG = PROJECT_ROOT / "configs" / "complete_task_demo_v2.json"


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, name
    return object_id


def test_canonical_mjcf_loads_with_table_and_actuators() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    mujoco.mj_forward(model, data)

    assert model.nu == 41
    table_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision")
    assert np.allclose(model.geom_size[table_id], [0.4, 0.4, 0.025])
    for name in (
        "table_leg_front_left",
        "table_leg_rear_left",
        "table_leg_rear_right",
        "table_leg_front_right",
    ):
        geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[geom_id] == 1
        assert model.geom_conaffinity[geom_id] == 1

    actuator_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    }
    assert "act_arm_r1_joint" in actuator_names
    assert "act_arm_r7_joint" in actuator_names
    assert "act_if_proximal_link" in actuator_names
    hand_actuator_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_if_proximal_link")
    assert np.isclose(model.actuator_gainprm[hand_actuator_id, 0], 30.0)
    assert np.allclose(model.actuator_forcerange[hand_actuator_id], [-8.0, 8.0])
    for name in (
        "blue_cylinder",
        "fragile_proxy",
        "red_target_zone",
        "blue_target_zone",
        "palm_col_r",
        "palm_col_l",
        "right_wrist",
    ):
        object_type = mujoco.mjtObj.mjOBJ_CAMERA if name == "right_wrist" else (
            mujoco.mjtObj.mjOBJ_BODY
            if name in {"blue_cylinder", "fragile_proxy"}
            else mujoco.mjtObj.mjOBJ_GEOM
        )
        _id(model, object_type, name)


def test_initial_state_has_no_penetration_and_remains_finite() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    mujoco.mj_forward(model, data)
    assert all(contact.dist >= -1e-8 for contact in data.contact)

    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    cube_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    assert np.isclose(data.xpos[cube_id, 2], 0.834, atol=0.003)


def test_right_arm_position_actuator_tracks_a_small_command() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    actuator_id = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_arm_r1_joint")
    joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_r1_joint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    data.ctrl[actuator_id] = 0.2
    for _ in range(round(1.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.isclose(data.qpos[qpos_address], 0.2, atol=0.025)


def test_power_synergy_closes_fingertips_toward_thumb() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    mujoco.mj_forward(model, data)
    finger_ids = [
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("if_tip_col", "mf_tip_col", "rf_tip_col", "lf_tip_col")
    ]
    thumb_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "th_tip_col")

    def mean_distance() -> float:
        return float(
            np.mean(
                [
                    np.linalg.norm(data.geom_xpos[finger_id] - data.geom_xpos[thumb_id])
                    for finger_id in finger_ids
                ]
            )
        )

    open_distance = mean_distance()
    command_hand_synergy(model, data, "power", duration_s=1.0)
    for _ in range(round(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert mean_distance() < 0.65 * open_distance


def test_color_grounding_rejects_small_same_color_robot_highlight() -> None:
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:50, 30:65] = True
    mask[2:7, 4:10] = True
    selected = _largest_connected_component(mask)
    assert int(selected.sum()) == 30 * 35
    assert selected[25, 40]
    assert not selected[3, 5]


def test_red_grounding_does_not_alias_protected_orange_rim() -> None:
    rgb = np.asarray([[[225, 35, 30], [255, 140, 0], [230, 180, 20]]], dtype=np.uint8)
    masks = _color_masks(rgb)
    assert masks["red_cube"][0].tolist() == [True, False, False]
    assert masks["fragile_proxy"][0].tolist() == [False, True, True]


def test_red_surface_envelope_rejects_high_robot_highlight() -> None:
    color = np.ones((1, 4), dtype=bool)
    world_z = np.asarray([[0.802, 0.837, 0.870, 0.940]])
    selected = _object_surface_mask(
        color, world_z, name="red_cube", support_z=0.819
    )
    assert selected[0].tolist() == [False, True, True, False]


def test_portable_urdf_loads_without_absolute_mesh_paths() -> None:
    text = URDF.read_text(encoding="utf-8")
    assert 'filename="/' not in text
    assert "file:///" not in text
    model = mujoco.MjModel.from_xml_path(str(URDF))
    assert model.nbody > 50


def test_frozen_protocol_and_scene_perturbation_are_reproducible() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert set(protocol["scene_families"]) == {"G0", "G1", "G2", "G3", "G4"}
    assert len(protocol["seeds"]["pilot"]) == 5
    assert len(protocol["seeds"]["main"]) == 20
    assert len(protocol["seeds"]["headline_confirmation"]) == 30
    assert set(protocol["observation_contract"]["public"]).isdisjoint(
        protocol["observation_contract"]["evaluator_only"]
    )

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    first = mujoco.MjData(model)
    second = mujoco.MjData(model)
    first_spec = configure_scene(model, first, protocol_path=PROTOCOL, scene_id="G3", seed=13)
    second_spec = configure_scene(model, second, protocol_path=PROTOCOL, scene_id="G3", seed=13)
    assert first_spec == second_spec

    body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cylinder")
    before_xy = first.xpos[body_id, :2].copy()
    event = apply_declared_perturbation(
        model, first, protocol_path=PROTOCOL, scene_id="G3"
    )
    assert event is not None and event["role"] == "evaluator_only"
    assert np.allclose(first.xpos[body_id, :2] - before_xy, [0.025, 0.015])


def test_v2_task_ready_pose_has_full_orientation_and_table_clearance() -> None:
    demo = json.loads(DEMO_CONFIG.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    configure_scene(model, data, protocol_path=PROTOCOL, scene_id="G2", seed=101)
    controller = RightArmController(model, data)
    desired_x = np.asarray(demo["tool_orientation_world"]["local_x"], dtype=float)
    desired_y = np.asarray(demo["tool_orientation_world"]["local_y"], dtype=float)
    rotation = np.column_stack((desired_x, desired_y, np.cross(desired_x, desired_y)))
    power_grasp = np.asarray(demo["grasp_center_local_m"]["power"], dtype=float)
    cube_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    mujoco.mj_forward(model, data)
    target = data.xpos[cube_id] - rotation @ power_grasp
    target[2] += float(demo["transport_clearance_m"])
    ik = demo["ik"]
    solution = controller.solve_position(
        target,
        tolerance_m=float(ik["position_tolerance_m"]),
        local_x_world=desired_x,
        local_y_world=desired_y,
        axis_tolerance=float(ik["axis_tolerance"]),
        position_weight=float(ik["position_weight"]),
        continuity_weight=float(ik["continuity_weight"]),
    )
    data.qpos[controller.qpos_addresses] = solution.joint_positions
    for name in RIGHT_HAND_JOINTS:
        joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = HAND_SYNERGIES["open"][name]
    mujoco.mj_forward(model, data)
    assert solution.position_error_m <= 0.005
    assert solution.local_x_axis_error is not None and solution.local_x_axis_error <= 0.04
    assert solution.local_y_axis_error is not None and solution.local_y_axis_error <= 0.04
    fingertip_ids = [
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("if_tip_col", "mf_tip_col", "rf_tip_col", "lf_tip_col", "th_tip_col")
    ]
    minimum_clearance = min(
        float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0] - 0.8)
        for geom_id in fingertip_ids
    )
    assert minimum_clearance >= -0.002


def test_mesh_clearance_calibration_replaces_closed_fingertip_midpoint() -> None:
    demo = json.loads(DEMO_CONFIG.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    controller = RightArmController(model, data)
    for synergy in ("power", "pinch"):
        frame = calibrate_grasp_frame(model, data, controller, synergy)
        assert np.allclose(frame.center_local_m, demo["grasp_center_local_m"][synergy], atol=1e-5)


def test_calibrated_holds_do_not_visibly_penetrate_objects_or_table() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    configure_scene(model, data, protocol_path=PROTOCOL, scene_id="G2", seed=101)
    controller = RightArmController(model, data)
    hand_addresses = np.asarray([
        model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in RIGHT_HAND_JOINTS
    ])
    hand_meshes = hand_visual_geoms(model)
    table_visual = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_visual")
    rotation = top_down_rotation(90.0)
    for object_geom, visual_geom, synergy in (
        ("cube_col", "cube_visual", "power"),
        ("blue_cylinder_col", "blue_cylinder_visual", "pinch"),
    ):
        collision_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, object_geom)
        visual_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, visual_geom)
        mujoco.mj_forward(model, data)
        object_center = data.geom_xpos[collision_id].copy()
        frame = calibrate_grasp_frame(model, data, controller, synergy)
        solution = controller.solve_position(
            object_center - rotation @ np.asarray(frame.center_local_m),
            local_x_world=rotation[:, 0],
            local_y_world=rotation[:, 1],
            position_weight=20.0,
            continuity_weight=0.005,
            posture_weight=0.025,
            joint_center_weight=0.005,
        )
        data.qpos[controller.qpos_addresses] = solution.joint_positions
        data.qpos[hand_addresses] = [HAND_SYNERGIES[synergy][name] for name in RIGHT_HAND_JOINTS]
        mujoco.mj_forward(model, data)
        object_clearance = minimum_geom_clearance(model, data, hand_meshes, (visual_id,))
        table_clearance = minimum_geom_clearance(model, data, hand_meshes, (table_visual,))
        assert object_clearance >= -0.0005
        assert table_clearance >= -0.0005


def test_relation_conditioned_wrist_yaw_improves_protected_clearance() -> None:
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    initialize_position_targets(model, data)
    controller = RightArmController(model, data)
    red = np.asarray([-0.1729, -0.2022, 0.834])
    fragile = red + np.asarray([-0.04, -0.04, 0.042])
    red_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    fragile_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "fragile_proxy_free")
    data.qpos[model.jnt_qposadr[red_joint]:model.jnt_qposadr[red_joint] + 3] = red
    data.qpos[model.jnt_qposadr[fragile_joint]:model.jnt_qposadr[fragile_joint] + 3] = fragile
    mujoco.mj_forward(model, data)
    frame = calibrate_grasp_frame(model, data, controller, "power")
    relation_conditioned = evaluate_top_down_candidate(
        model, data, controller, red, frame, 60.0, protected_geom_names=("fragile_proxy_col",)
    )
    legacy_fixed = evaluate_top_down_candidate(
        model, data, controller, red, frame, 120.0, protected_geom_names=("fragile_proxy_col",)
    )
    assert relation_conditioned.protected_clearance_m is not None
    assert legacy_fixed.protected_clearance_m is not None
    assert relation_conditioned.protected_clearance_m >= 0.02
    assert legacy_fixed.protected_clearance_m < 0.0

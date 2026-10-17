#!/usr/bin/env python3
"""Prepare portable Guanghua robot MJCF and URDF files from the school bundle."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets" / "guanghua_hand_env"
MJCF_DIR = ASSET_ROOT / "mjcf"
URDF_DIR = ASSET_ROOT / "urdf"

SOURCE_MJCF = MJCF_DIR / "guanghua_hand_env.bundle.xml"
CANONICAL_MJCF = MJCF_DIR / "guanghua_hand_env.xml"
SOURCE_URDF = URDF_DIR / "fdr3_robot_with_hands_from_mjcf.urdf"
CANONICAL_URDF = URDF_DIR / "guanghua_robot_with_hands.urdf"


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _ensure_material(asset: ET.Element, name: str, rgba: str) -> None:
    if asset.find(f"material[@name='{name}']") is None:
        ET.SubElement(
            asset,
            "material",
            {"name": name, "rgba": rgba, "specular": "0.25", "shininess": "0.15"},
        )


def _add_task_objects(root: ET.Element) -> list[str]:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise RuntimeError("bundle MJCF is missing asset or worldbody")
    _ensure_material(asset, "obj_blue", "0.12 0.35 0.85 1")
    _ensure_material(asset, "obj_yellow", "0.95 0.75 0.08 1")

    # Thin visual-only target regions. Placement success is evaluated from the
    # object pose after a rollout; these markers provide no physical funnel.
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "red_target_zone",
            "type": "box",
            "pos": "-0.03 -0.23 0.802",
            "size": "0.055 0.055 0.002",
            "rgba": "0.85 0.15 0.15 0.35",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "blue_target_zone",
            "type": "box",
            "pos": "-0.03 -0.05 0.802",
            "size": "0.055 0.055 0.002",
            "rgba": "0.12 0.35 0.85 0.35",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "staging_zone",
            "type": "box",
            "pos": "-0.18 0.13 0.802",
            "size": "0.07 0.07 0.002",
            "rgba": "0.25 0.75 0.35 0.25",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )

    blue = ET.SubElement(worldbody, "body", {"name": "blue_cylinder", "pos": "-0.18 -0.05 0.861"})
    ET.SubElement(blue, "joint", {"name": "blue_cylinder_free", "type": "free"})
    ET.SubElement(
        blue,
        "geom",
        {
            "name": "blue_cylinder_visual",
            "type": "cylinder",
            "size": "0.028 0.06",
            "material": "obj_blue",
            "density": "0",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        blue,
        "geom",
        {
            "name": "blue_cylinder_col",
            "type": "box",
            "size": "0.026 0.026 0.06",
            "rgba": "0 0 0 0",
            "density": "650",
            "friction": "0.8 0.01 0.001",
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "margin": "0.001",
            "group": "0",
        },
    )
    ET.SubElement(blue, "site", {"name": "blue_cylinder_center", "size": "0.004", "rgba": "0 0 0 0"})

    fragile = ET.SubElement(worldbody, "body", {"name": "fragile_proxy", "pos": "0.05 0.10 0.876"})
    ET.SubElement(fragile, "joint", {"name": "fragile_proxy_free", "type": "free"})
    ET.SubElement(
        fragile,
        "geom",
        {
            "name": "fragile_proxy_visual",
            "type": "cylinder",
            "size": "0.035 0.055",
            "pos": "0 0 0.015",
            "rgba": "1 0.72 0.08 0.34",
            "density": "0",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    for name, radius, half_height, z, rgba in (
        ("fragile_proxy_rim", "0.039", "0.004", "0.071", "1 0.55 0.02 0.95"),
        ("fragile_proxy_base", "0.040", "0.004", "-0.071", "1 0.55 0.02 0.95"),
        ("fragile_proxy_keepout_halo", "0.055", "0.001", "-0.074", "1 0.75 0.05 0.24"),
    ):
        ET.SubElement(
            fragile,
            "geom",
            {
                "name": name,
                "type": "cylinder",
                "size": f"{radius} {half_height}",
                "pos": f"0 0 {z}",
                "rgba": rgba,
                "density": "0",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
    ET.SubElement(
        fragile,
        "geom",
        {
            "name": "fragile_proxy_col",
            "type": "box",
            "size": "0.032 0.032 0.075",
            "rgba": "0 0 0 0",
            "density": "450",
            "friction": "0.7 0.01 0.001",
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "margin": "0.001",
            "group": "0",
        },
    )
    ET.SubElement(fragile, "site", {"name": "fragile_proxy_center", "size": "0.004", "rgba": "0 0 0 0"})
    return ["cube", "blue_cylinder", "fragile_proxy"]


def _add_robot_collision_proxies(root: ET.Element) -> list[str]:
    generated: list[str] = []

    for hand_name, suffix in (("hand_root", "r"), ("hand_root_l", "l")):
        hand = root.find(f".//body[@name='{hand_name}']")
        if hand is None:
            raise RuntimeError(f"missing hand body {hand_name}")
        name = f"palm_col_{suffix}"
        ET.SubElement(
            hand,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": "-0.038 0 0",
                "size": "0.038 0.046 0.014",
                "rgba": "0 0 0 0",
                "density": "0",
                "contype": "2",
                "conaffinity": "1",
                "condim": "4",
                "friction": "1.0 0.02 0.002",
                "group": "0",
            },
        )
        generated.append(name)
        if suffix == "r":
            ET.SubElement(
                hand,
                "camera",
                {
                    "name": "right_wrist",
                    "pos": "-0.03 0 0.04",
                    "xyaxes": "0 1 0 0 0 1",
                    "fovy": "70",
                },
            )

    finger_stems = ("if", "mf", "rf", "lf", "th")
    for suffix in ("", "_l"):
        side = "r" if not suffix else "l"
        for stem in finger_stems:
            for segment, length, radius in (("proximal", 0.026, 0.0065), ("distal", 0.026, 0.0055)):
                body = root.find(f".//body[@name='{stem}_{segment}_link{suffix}']")
                if body is None:
                    raise RuntimeError(f"missing finger body {stem}_{segment}_link{suffix}")
                name = f"{stem}_{segment}_col_{side}"
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": name,
                        "type": "capsule",
                        "fromto": f"0 0 0 {-length} 0 0",
                        "size": str(radius),
                        "rgba": "0 0 0 0",
                        "density": "0",
                        "contype": "2",
                        "conaffinity": "1",
                        "condim": "4",
                        "friction": "1.1 0.02 0.002",
                        "margin": "0.003",
                        "gap": "0.001",
                        "group": "0",
                    },
                )
                generated.append(name)

    for tip in root.findall(".//geom"):
        if tip.get("name", "").endswith(("_tip_col", "_tip_col_l")):
            tip.set("contype", "2")
            tip.set("conaffinity", "1")
            tip.set("margin", "0.003")
            tip.set("gap", "0.001")

    return generated


def prepare_mjcf() -> dict[str, object]:
    tree = ET.parse(SOURCE_MJCF)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("bundle MJCF has no compiler element")
    compiler.set("meshdir", "..")
    compiler.set("texturedir", "../textures")

    for mesh in root.findall("./asset/mesh"):
        filename = mesh.get("file", "")
        if filename.startswith("meshes_fdr3/"):
            mesh.set("file", f"urdf/{filename.removeprefix('meshes_fdr3/')}")

    table = root.find("./worldbody/body[@name='table']")
    if table is None:
        raise RuntimeError("bundle MJCF has no table body")
    table_collision = table.find("geom[@name='table_collision']")
    if table_collision is None:
        raise RuntimeError("bundle MJCF has no table collision geom")
    table_collision.set("rgba", "0 0 0 0")

    leg_names = ("front_left", "rear_left", "rear_right", "front_right")
    legs = [geom for geom in table.findall("geom") if geom.get("type") == "cylinder"]
    if len(legs) != 4:
        raise RuntimeError(f"expected four table legs, found {len(legs)}")
    for leg, suffix in zip(legs, leg_names, strict=True):
        leg.set("name", f"table_leg_{suffix}")
        leg.set("contype", "1")
        leg.set("conaffinity", "1")
        leg.set("condim", "3")
        leg.set("friction", "0.8 0.05 0.005")

    cube = root.find("./worldbody/body[@name='cube']")
    if cube is None:
        raise RuntimeError("bundle MJCF has no cube body")
    # Table top is z=0.800 m. The 0.033 m half-size plus 1 mm contact
    # margin puts the non-penetrating cube center at z=0.834 m.
    cube.set("pos", "-0.18 -0.23 0.834")
    cube_collision = cube.find("geom[@name='cube_col']")
    if cube_collision is None:
        raise RuntimeError("bundle MJCF has no cube collision geom")
    cube_collision.set("rgba", "0 0 0 0")
    if cube.find("site[@name='red_cube_center']") is None:
        ET.SubElement(cube, "site", {"name": "red_cube_center", "size": "0.004", "rgba": "0 0 0 0"})

    task_objects = _add_task_objects(root)
    collision_proxies = _add_robot_collision_proxies(root)

    robot_root = root.find("./worldbody/body[@name='robot_root']")
    if robot_root is None:
        raise RuntimeError("bundle MJCF has no robot_root body")
    # The source upper-body base places the shoulders only 0.217 m above the
    # tabletop, leaving no vertical workspace for a downward dexterous grasp.
    # Mount the fixed torso 0.25 m higher, analogous to a pedestal adjustment.
    robot_root.set("pos", "-0.52 0 1.05")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.Element("global")
        visual.insert(0, global_visual)
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "960")

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    existing_joints = {
        element.get("joint")
        for element in actuator
        if element.get("joint") is not None
    }
    compliant_hand_actuators = []
    for position in actuator.findall("position"):
        joint_name = position.get("joint", "")
        if joint_name and not joint_name.startswith(("arm_", "waist_", "head_")):
            position.set("kp", "30")
            position.set("dampratio", "1")
            position.set("forcerange", "-8 8")
            joint_element = root.find(f".//joint[@name='{joint_name}']")
            if joint_element is not None:
                joint_element.set("actuatorfrcrange", "-8 8")
            compliant_hand_actuators.append(joint_name)
    generated = []
    insert_at = 0
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if not name or name == "cube_free" or name in existing_joints:
            continue
        if name.startswith("arm_"):
            kp, force = "800", "-100 100"
            joint.set("actuatorfrcrange", force)
        elif name.startswith("waist_"):
            kp, force = "300", "-80 80"
            joint.set("actuatorfrcrange", force)
        elif name.startswith("head_"):
            kp, force = "100", "-25 25"
            joint.set("actuatorfrcrange", force)
        else:
            continue
        position = ET.Element(
            "position",
            {
                "name": f"act_{name}",
                "joint": name,
                "kp": kp,
                "dampratio": "1",
                "forcerange": force,
            },
        )
        actuator.insert(insert_at, position)
        insert_at += 1
        generated.append(name)

    _write_xml(tree, CANONICAL_MJCF)
    return {
        "generated_hold_actuators": generated,
        "compliant_hand_actuators": compliant_hand_actuators,
        "table_leg_collisions": [f"table_leg_{name}" for name in leg_names],
        "task_objects": task_objects,
        "robot_collision_proxies": collision_proxies,
    }


def _portable_mesh_path(filename: str) -> str:
    normalized = filename.removeprefix("file://")
    if "/urdf/" in normalized:
        return normalized.split("/urdf/", 1)[1]
    if "/meshes_fdr3/" in normalized:
        return Path(normalized).name
    if Path(normalized).is_absolute():
        return Path(normalized).name
    return normalized


def prepare_urdf() -> dict[str, object]:
    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()
    rewritten = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        portable = _portable_mesh_path(filename)
        mesh.set("filename", portable)
        rewritten.append(portable)
        if not (URDF_DIR / portable).is_file():
            raise FileNotFoundError(f"URDF mesh is missing: {portable}")

    mujoco_extension = root.find("mujoco")
    if mujoco_extension is None:
        mujoco_extension = ET.Element("mujoco")
        mujoco_extension.append(
            ET.Element(
                "compiler",
                {
                    "meshdir": ".",
                    "balanceinertia": "true",
                    "discardvisual": "false",
                    "fusestatic": "false",
                },
            )
        )
        root.insert(0, mujoco_extension)

    _write_xml(tree, CANONICAL_URDF)
    return {"mesh_references": len(rewritten), "unique_meshes": len(set(rewritten))}


def main() -> None:
    mjcf_summary = prepare_mjcf()
    urdf_summary = prepare_urdf()
    manifest = {
        "canonical_mjcf": str(CANONICAL_MJCF.relative_to(PROJECT_ROOT)),
        "canonical_urdf": str(CANONICAL_URDF.relative_to(PROJECT_ROOT)),
        "source_mjcf": str(SOURCE_MJCF.relative_to(PROJECT_ROOT)),
        "source_urdf": str(SOURCE_URDF.relative_to(PROJECT_ROOT)),
        "table": {
            "top_size_m": [0.8, 0.8],
            "top_thickness_m": 0.05,
            "surface_height_m": 0.8,
        },
        "robot_root_position_m": [-0.52, 0.0, 1.05],
        "task_objects": {
            "red_cube": {"body": "cube", "side_length_m": 0.066, "rest_z_m": 0.834},
            "blue_cylinder": {"radius_m": 0.028, "half_height_m": 0.06, "rest_z_m": 0.861},
            "fragile_proxy": {"radius_m": 0.035, "half_height_m": 0.075, "rest_z_m": 0.876},
        },
        "mjcf_changes": mjcf_summary,
        "urdf_changes": urdf_summary,
    }
    manifest_path = ASSET_ROOT / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Object-independent constraints compiled from observed scene relations."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from scripts.agentic_framework import PublicObject


def compile_constraints(target: PublicObject, objects: Mapping[str, PublicObject],
                        context: Mapping[str, Any] | None = None, *, enabled: bool = True,
                        path_validator=None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "active": False, "reason": "no_active_constraint", "approach_offset_xy_m": [0., 0.],
        "approach_height_delta_m": 0., "force_scale": 1., "constraints": [],
        "prerequisites": [], "blocked": False,
    }
    if not enabled:
        result["reason"] = "scene_graph_disabled"
        return result
    context = context or {}
    neighbors = []
    for name, obj in objects.items():
        if name == target.name or float(obj.fragility) < .7:
            continue
        away = np.asarray(target.center_xyz_m[:2]) - np.asarray(obj.center_xyz_m[:2])
        distance = float(np.linalg.norm(away))
        if distance <= .10:
            neighbors.append((distance, name, away))
    if neighbors:
        neighbors.sort(key=lambda item: (item[0], item[1]))
        _, _, away = neighbors[0]
        norm = np.linalg.norm(away)
        if norm < 1e-6:
            result["blocked"] = True
        else:
            result["approach_offset_xy_m"] = (away / norm * .05).tolist()
        result.update(active=True, reason="fragile_neighbor_within_0.10_m",
                      approach_height_delta_m=.03, force_scale=.8,
                      fragile_neighbors=[name for _, name, _ in neighbors])
        result["constraints"].append("collision_avoidance")
    if context.get("states", {}).get(target.name) == "filled":
        result.update(active=True, required_approach="top", velocity_scale=.5)
        result["constraints"].append("content_preservation")
    for edge in context.get("relations", []):
        source, dest, relation = edge["source"], edge["target"], edge["relation"]
        if source not in objects or dest not in objects:
            raise ValueError("scene relation references unobserved object")
        if np.linalg.norm(np.asarray(objects[source].center_xyz_m)-objects[dest].center_xyz_m) > .15:
            continue
        if relation == "on_top_of":
            source,dest,relation = dest,source,"supports"
        if relation == "supports" and source == target.name and dest in objects:
            result["prerequisites"].append(dest)
            result["constraints"].append("support_dependency")
            result["blocked"] = True
        if relation == "occludes" and dest == target.name:
            result["constraints"].append("occlusion_handling")
            away = np.asarray(target.center_xyz_m[:2])-np.asarray(objects[source].center_xyz_m[:2])
            norm = np.linalg.norm(away)
            offset = away/norm*.05 if norm > 1e-6 else np.array([.05,0])
            proposal = {"target":target.name,"approach":"lateral","approach_offset_xy_m":offset.tolist(),
                        "obstacle":source,"analysis_radius_m":.15,"active_constraints":list(result["constraints"])}
            result["lateral_path_proposal"] = proposal
            vertical_required = result.get("required_approach") == "top"
            if not vertical_required and path_validator is not None and path_validator(proposal,objects) is True:
                bias = result["approach_offset_xy_m"] if "collision_avoidance" in result["constraints"] else offset.tolist()
                result.update(required_approach="lateral",approach_offset_xy_m=bias,path_validated=True)
            else:
                result["blocked"] = True
                result["path_validated"] = False
    result["active"] = bool(result["constraints"])
    return result

"""Directed observed scene graph and dependency scheduling.

The paper names six of sixteen relations. The other ten registry names below
are documented reconstruction choices, not claimed recovered definitions.
"""
from dataclasses import replace
from math import dist, isfinite
from scripts.agentic_framework import build_scene_graph as geometric_graph

PAPER_NAMED_RELATIONS = {"on_top_of","contains","occludes","supports","adjacent_to","behind"}
ADDITIONAL_RELATIONS = {"in_front_of","left_of","right_of","above","below","inside","touching","near","overlaps","between"}
RELATIONS = PAPER_NAMED_RELATIONS | ADDITIONAL_RELATIONS
STATES = {"empty","filled","stacked","unknown"}


def build_scene_graph(objects, context=None):
    context = context or {}
    graph = geometric_graph(objects)
    graph["edges"] += [{**edge,"source":edge["target"],"target":edge["source"]} for edge in list(graph["edges"])]
    states = context.get("states",{})
    for name,state in states.items():
        if name not in objects or state not in STATES:
            raise ValueError("invalid observed object state")
    for node in graph["nodes"]:
        node["state"] = states.get(node["name"],"unknown")
    for raw in context.get("relations",[]):
        edge = dict(raw)
        if edge.get("source") not in objects or edge.get("target") not in objects or edge.get("relation") not in RELATIONS:
            raise ValueError("invalid observed scene relation")
        if edge["source"] == edge["target"]:
            raise ValueError("self scene relation")
        if edge not in graph["edges"]:
            graph["edges"].append(edge)
    graph["analysis_radius_m"] = .15
    return graph


def schedule_dependencies(task, context, *, prerequisite_destinations):
    """Insert support-removal subgoals; reject protected objects and cycles."""
    from .runtime import Subgoal
    dependencies = {}
    for edge in context.get("relations",[]):
        if edge["relation"] == "supports":
            dependencies.setdefault(edge["source"],[]).append(edge["target"])
        elif edge["relation"] == "on_top_of":
            dependencies.setdefault(edge["target"],[]).append(edge["source"])
    ordered, visiting, done = [],set(),set()
    original = {g.target:g for g in task.subgoals}
    def visit(name):
        if name in visiting:
            raise ValueError("cyclic support dependency")
        if name in done:
            return
        if name in task.protected_objects:
            raise ValueError("support prerequisite is protected")
        visiting.add(name)
        for other in dependencies.get(name,[]):
            visit(other)
        if name in original:
            goal = original[name]
        elif name in prerequisite_destinations:
            goal = Subgoal(name,prerequisite_destinations[name])
        else:
            raise ValueError(f"no verified removal destination for supported object: {name}")
        ordered.append(goal)
        visiting.remove(name)
        done.add(name)
    for goal in task.subgoals:
        visit(goal.target)
    return replace(task,subgoals=tuple(ordered))


def schedule_visible_support_removal(task, objects, context, completed, *, observation_id):
    """Defer unresolved goals; never name an anonymous supporter by elimination.

    Only a reorderable task and a fresh, metric support witness can authorize
    moving a visible task target first. This grants no authority to grasp the
    unidentified support, skip it, or move any protected object.
    """
    from .constraints import compile_constraints
    pending=[goal for goal in task.subgoals if goal.target not in completed]
    missing=[goal.target for goal in pending if goal.target not in objects]
    if (not task.reorderable or not missing or not observation_id
            or set(task.protected_objects)-set(objects)):
        return None
    witnesses=context.get('unresolved_supports',[])
    for goal in pending:
        if goal.target not in objects or goal.target in task.protected_objects:
            continue
        upper=objects[goal.target]
        if not isfinite(upper.fragility) or not 0<=upper.fragility<=.25:
            continue
        if compile_constraints(upper,objects,context)['blocked']:
            continue
        for witness in witnesses:
            if (witness.get('upper_target')!=goal.target or witness.get('observation_id')!=observation_id
                    or witness.get('support_identity')!='unresolved'
                    or witness.get('source')!='public_rgbd_surface_plus_known_upper_dimensions'):
                continue
            xyz=witness.get('support_top_xyz_m',[]);half=witness.get('upper_half_height_m',0)
            if (len(xyz)!=3 or not all(isfinite(float(x)) for x in xyz)
                    or not isfinite(half) or not 0<half<.2
                    or not all(isfinite(x) for x in upper.center_xyz_m)):
                continue
            gap=upper.center_xyz_m[2]-half-xyz[2]
            if abs(gap)>=.005 or any(abs(a-b)>=.026 for a,b in zip(upper.center_xyz_m[:2],xyz[:2])):
                continue
            a,b=witness.get('upper_bbox_xyxy',[]),witness.get('support_bbox_xyxy',[])
            if (len(a)!=4 or len(b)!=4 or not all(isfinite(float(x)) for x in a+b)
                    or min(a[2],b[2])<=max(a[0],b[0]) or min(a[3],b[3])<=max(a[1],b[1])):
                continue
            done=[g for g in task.subgoals if g.target in completed]
            reordered=done+[goal]+[g for g in pending if g.target!=goal.target]
            return replace(task,subgoals=tuple(reordered)),{
                'selected_target':goal.target,'deferred_targets':missing,
                'support_identity':'unresolved','witness':dict(witness),
                'preserved_completed_targets':list(completed),
                'policy':'move_verified_visible_upper_target_then_reobserve'}
    return None


def schedule_visible_goal_progress(task, objects, context, completed, *, observation_id):
    """Choose a stable, identified elevated task goal without claiming that it
    supports/occludes an unidentified object. All original goals remain due.
    """
    from .constraints import compile_constraints
    pending=[g for g in task.subgoals if g.target not in completed]
    missing=[g.target for g in pending if g.target not in objects]
    pair=context.get('visible_goal_stability',{})
    plane=context.get('support_plane',{})
    old_time=pair.get('previous_capture_time');new_time=pair.get('current_capture_time')
    if (not task.reorderable or not missing or not observation_id
            or set(task.protected_objects)-set(objects)
            or pair.get('current_observation_id')!=observation_id
            or not pair.get('previous_observation_id')
            or pair['previous_observation_id']==observation_id
            or not isinstance(old_time,(int,float)) or not isinstance(new_time,(int,float))
            or not isfinite(old_time) or not isfinite(new_time) or new_time<=old_time
            or pair.get('position_tolerance_m')!=.004
            or plane.get('source')!='calibrated_testbed_table'
            or not isfinite(plane.get('z_m',float('nan')))):
        return None
    eligible=[]
    stable=set(pair.get('stable_visible_targets',[]))
    if not set(task.protected_objects)<=stable:return None
    for goal in pending:
        if goal.target not in objects or goal.target not in stable or goal.target in task.protected_objects:continue
        item=objects[goal.target]
        if not isfinite(item.fragility) or not 0<=item.fragility<=.25:continue
        if compile_constraints(item,objects,context)['blocked']:continue
        for geometry in context.get('visible_geometry',[]):
            xyz=geometry.get('center_xyz_m',[]);half=geometry.get('known_half_height_m',0)
            if (geometry.get('target')!=goal.target or geometry.get('observation_id')!=observation_id
                    or geometry.get('source')!='public_rgbd_and_vlm_identified_target'
                    or len(xyz)!=3 or not all(isfinite(v) for v in xyz)
                    or not isfinite(half) or not 0<half<.2 or dist(xyz,item.center_xyz_m)>.000001):continue
            if xyz[2]-half-plane['z_m']>.01:
                eligible.append((xyz[2],goal,geometry))
    if not eligible:return None
    _,selected,geometry=max(eligible,key=lambda entry:entry[0])
    done=[g for g in task.subgoals if g.target in completed]
    reordered=done+[selected]+[g for g in pending if g.target!=selected.target]
    return replace(task,subgoals=tuple(reordered)),{
        'selected_target':selected.target,'deferred_targets':missing,
        'support_identity':'not_inferred','witness':dict(geometry),
        'observation_pair':dict(pair),'support_plane':dict(plane),
        'preserved_completed_targets':list(completed),
        'policy':'visible_target_progress_then_reobserve'}

"""YAM MuJoCo contact testbed. Object truth is used only by the evaluator.

The controller consumes RGB-D detections, joints and command parameters. It
never attaches or teleports a manipulated object. qpos initialization happens
once at reset; all subsequent motion uses actuator targets and mj_step.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any
import numpy as np
import mujoco

from ..platform.kinematics import _rotation_log

ROOT = Path(__file__).resolve().parents[2]


class PhysicsSafetyError(RuntimeError):
    """Stop the rollout on an observed unexpected contact or penetration."""


def build_scene(path: Path, *, seed: int = 0, neighbor: bool = False, multiple: bool = False) -> dict[str, Any]:
    source = ROOT / "assets/yam_mjlab/yam.xml"
    root = ET.parse(source).getroot()
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", iterations="80")
    ET.SubElement(root, "visual").append(ET.Element("global", offwidth="960", offheight="720"))
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0.2 -0.5 1.4", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "geom", name="table", type="box", pos=".3 0 -.035", size=".55 .5 .035", rgba=".55 .43 .30 1", friction=".8 .01 .001")
    ET.SubElement(world, "geom", name="floor", type="plane", pos="0 0 -.75", size="2 2 .1", rgba=".18 .20 .23 1")
    ET.SubElement(world, "camera", name="overview", pos=".85 -.85 .65", xyaxes=".71 .71 0 -.35 .35 .87", fovy="48")
    ET.SubElement(world, "camera", name="top", pos=".30 0 1.10", quat="1 0 0 0", fovy="45")
    ET.SubElement(world, "geom", name="destination_visual", type="box", pos=".26 .12 .001", size=".055 .055 .001", rgba=".18 .45 .65 1", contype="0", conaffinity="0")
    # Full original robot collision geometry remains active. Compensation is
    # ideal link gravity compensation, disclosed separately from motor limits.
    for body in world.findall(".//body"):
        body.set("gravcomp", "1")
    defaults = root.find("default")
    ET.SubElement(defaults, "joint", damping="1.0", armature=".01")
    for geom in root.findall(".//geom"):
        if geom.get("class") in {"collision", "sphere_collision"}:
            geom.set("friction", ".9 .005 .0005")
            geom.set("solref", ".01 1")
    rng = np.random.default_rng(seed)
    x = float(.30 + rng.uniform(-.012, .012))
    y = float(-.12 + rng.uniform(-.012, .012))
    cube = ET.SubElement(world, "body", name="target_cube", pos=f"{x} {y} .019")
    ET.SubElement(cube, "freejoint", name="target_free")
    ET.SubElement(cube, "geom", name="target_geom", type="box", size=".018 .018 .018", mass=".06", rgba=".85 .08 .06 1", friction="1 .01 .001", condim="4", solref=".01 1")
    if multiple:
        second = ET.SubElement(world,"body",name="target_cube2",pos=f"{x-.08} {y+.08} .019")
        ET.SubElement(second,"freejoint",name="target2_free")
        ET.SubElement(second,"geom",name="target_geom2",type="box",size=".018 .018 .018",mass=".06",
            rgba=".90 .75 .06 1",friction="1 .01 .001",condim="4",solref=".01 1")
        ET.SubElement(world,"geom",name="destination2_visual",type="box",pos=".20 .17 .001",
            size=".025 .025 .001",rgba=".18 .45 .65 1",contype="0",conaffinity="0")
    if neighbor:
        vessel = ET.SubElement(world, "body", name="protected_vase", pos=f"{x+.075} {y} .04")
        ET.SubElement(vessel, "freejoint", name="protected_free")
        ET.SubElement(vessel, "geom", name="protected_geom", type="cylinder", size=".018 .04", mass=".15", rgba=".12 .65 .25 1", friction=".8 .01 .001")
    actuators = ET.SubElement(root, "actuator")
    for i, (kp, kv, force) in enumerate(zip([300,300,250,120,80,60], [30,30,25,12,8,6], [28,28,28,10,10,10]), 1):
        ET.SubElement(actuators, "position", name=f"act{i}", joint=f"joint{i}", kp=str(kp), kv=str(kv), forcelimited="true", forcerange=f"-{force} {force}")
    ET.SubElement(actuators, "position", name="grip", joint="left_finger", kp="180", kv="8", forcelimited="true", forcerange="-12 12", ctrllimited="true", ctrlrange="0 .0375")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="unicode")
    return {"target_initial": [x, y, .019], "destination": [.26, .12, .018], "destination2":[.20,.17,.018],
        "neighbor": neighbor,"multiple":multiple}


class YamPhysics:
    def __init__(self, output: Path, *, seed: int = 0, neighbor: bool = False, record: bool = False, multiple: bool = False,
                 scene_builder=None) -> None:
        self.output = output
        builder = build_scene if scene_builder is None else scene_builder
        self.setup = builder(output / "scene.xml", seed=seed, neighbor=neighbor,multiple=multiple)
        self.model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(f"joint{i}").id for i in range(1,7)]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.dadr = self.model.jnt_dofadr[self.joints]
        self.site = self.model.site("grasp_site").id
        self.data.qpos[self.qadr] = [0, 1.047, 1.05, -.9, 0, 0]
        self.data.qpos[self.model.joint("left_finger").qposadr] = .0375
        self.data.qpos[self.model.joint("right_finger").qposadr] = -.0375
        self.data.ctrl[:6] = self.data.qpos[self.qadr]
        self.data.ctrl[6] = .0375
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.record = record
        self.frames = []
        self.public_trace = []
        self.private_trace = []
        self.step_count = 0
        self.stage = "settle"
        self.max_lift = 0.0
        self.forbidden_contacts = []
        self.protected_contacts = []
        self.self_contacts = []
        self.max_protected_displacement = 0.0
        self.bilateral_lift_seconds = 0.0
        self.max_target_penetration = 0.0
        self.target_geoms = self.setup.get("target_geoms",{"target_cube":"target_geom",**({"target_cube2":"target_geom2"} if multiple else {})})
        self.lift_seconds_by_object = {body:0. for body in self.target_geoms}
        self.initial_neighbor = None
        self.step(500)
        self.initial_target = self.data.body("target_cube").xpos.copy()
        self.initial_neighbor = self.data.body("protected_vase").xpos.copy() if neighbor else None

    def rgbd(self, camera="top"):
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def proposals(self, rgb, depth):
        """Scene-specific color proposals; no identities or body pose lookup.

        This deliberately restricted front end is for executor qualification,
        not a claim of open-world instance segmentation.
        """
        masks = [(rgb[:,:,0] > 100) & (rgb[:,:,0] > rgb[:,:,1]*1.8) & (rgb[:,:,0] > rgb[:,:,2]*1.8),
                 (rgb[:,:,1] > 70) & (rgb[:,:,1] > rgb[:,:,0]*1.7) & (rgb[:,:,1] > rgb[:,:,2]*1.4),
                 (rgb[:,:,0] > 100) & (rgb[:,:,1] > 90) & (rgb[:,:,2] < np.minimum(rgb[:,:,0],rgb[:,:,1])*.55)]
        result = []
        f = 480 / (2*np.tan(np.deg2rad(45)/2))
        camera = self.data.cam("top")
        rotation = camera.xmat.reshape(3,3)
        for color_index, mask in enumerate(masks):
            rows, cols = np.where(mask & np.isfinite(depth) & (depth > .05) & (depth < 2))
            if len(rows) < 8:
                continue
            z = depth[rows, cols]
            points = np.column_stack(((cols + .5 - 320)*z/f, -(rows+.5-240)*z/f, -z))
            world = points @ rotation.T + camera.xpos
            top = np.median(world, axis=0)
            result.append({"color_index":color_index, "top_xyz_m":top.tolist(),
                "visible_pixel_count":int(len(rows)),
                "bbox_xyxy":[int(cols.min()),int(rows.min()),int(cols.max())+1,int(rows.max())+1]})
        return result

    def detect(self) -> dict[str, tuple[float,float,float]]:
        """Color-labelled RGB-D diagnostic; no VLM and no body pose lookup."""
        rgb, depth = self.rgbd()
        result = {}
        for item in self.proposals(rgb,depth):
            name, half_height = (("cube",.018),("vase",.04),("cube2",.018))[item["color_index"]]
            top = np.array(item["top_xyz_m"])
            top[2] -= half_height
            result[name] = tuple(float(v) for v in top)
        self.public_trace.append({"event":"rgbd_observation", "stage":self.stage, "time":self.data.time, "objects":result, "source":"color_labelled_rgbd_diagnostic"})
        return result

    def tactile(self):
        """Public finger-contact telemetry; no object identity or pose output."""
        normal = {"left":0.,"right":0.}
        vertical_force = 0.
        for index,contact in enumerate(self.data.contact):
            a,b = self.model.geom(contact.geom1).name,self.model.geom(contact.geom2).name
            for name,other in ((a,b),(b,a)):
                side = "left" if name and name.startswith("lf_") else "right" if name and name.startswith("rf_") else None
                if side and other and not other.endswith("collision") and other not in {"table","floor"}:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model,self.data,index,wrench)
                    normal[side] += max(0.,float(wrench[0]))
                    world_force=contact.frame.reshape(3,3).T@wrench[:3]
                    vertical_force += float((-1 if name==a else 1)*world_force[2])
        return {"normal_force_n":normal,"bilateral":all(force>.01 for force in normal.values()),
                "left_finger_q_m":float(self.data.qpos[self.model.joint("left_finger").qposadr][0]),
                "actuator_force_n":float(self.data.actuator_force[self.model.actuator('grip').id]),
                "downward_load_n":-vertical_force}

    def solve(self, position, *, rotation=None, seed=None):
        rotation = np.diag([-1., 1., -1.]) if rotation is None else np.asarray(rotation)
        initial = self.data.qpos[self.qadr].copy() if seed is None else np.asarray(seed).copy()
        ranges = self.model.jnt_range[self.joints]
        best = None
        for start in (initial, np.array([0,1.2,1.6,-.6,0,0]), ranges.mean(axis=1)):
            q = start.copy()
            for _ in range(250):
                self.ik_data.qpos[self.qadr] = q
                mujoco.mj_forward(self.model, self.ik_data)
                site = self.ik_data.site(self.site)
                ep = np.asarray(position) - site.xpos
                er = _rotation_log(rotation @ site.xmat.reshape(3,3).T)
                score = np.linalg.norm(ep) + .1*np.linalg.norm(er)
                if best is None or score < best[0]:
                    best = (score, q.copy(), float(np.linalg.norm(ep)), float(np.linalg.norm(er)))
                if np.linalg.norm(ep) < .0005 and np.linalg.norm(er) < .015:
                    return q
                jp, jr = np.zeros((3,self.model.nv)), np.zeros((3,self.model.nv))
                mujoco.mj_jacSite(self.model, self.ik_data, jp, jr, self.site)
                jac = np.vstack((jp[:,self.dadr], .2*jr[:,self.dadr]))
                error = np.r_[ep, .2*er]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*1e-5, error)
                q = np.clip(q + np.clip(dq,-.12,.12), ranges[:,0]+.001, ranges[:,1]-.001)
        raise RuntimeError(f"IK failed at {position}: pos={best[2]:.4f} ori={best[3]:.4f}")

    def move(self, position, *, duration=1.5, grip=None, stage="move",rotation=None, step_monitor=None):
        self.stage = stage
        start = self.data.qpos[self.qadr].copy()
        goal = self.solve(position,rotation=rotation)
        self.public_trace.append({"event":"motion", "stage":stage, "target_xyz":list(position), "joints":goal.tolist()})
        count = round(duration/self.model.opt.timestep)
        for i in range(count):
            t = (i+1)/count
            blend = t*t*(3-2*t)
            self.data.ctrl[:6] = start + (goal-start)*blend
            if grip is not None:
                self.data.ctrl[6] = grip
            self.step(1)
            if step_monitor is not None:
                step_monitor()
        if step_monitor is None:
            self.step(100)
        else:
            for _ in range(100):
                self.step(1)
                step_monitor()
        error = float(np.linalg.norm(self.data.site(self.site).xpos - position))
        self.public_trace.append({"event":"motion_settled", "stage":stage, "tcp_error_m":error})
        if error > .008:
            raise RuntimeError(f"actuator tracking failed during {stage}: {error:.4f} m")
        return error

    def step(self, count):
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            p = self.data.body("target_cube").xpos.copy()
            self.max_lift = max(self.max_lift, float(p[2]))
            if self.initial_neighbor is not None:
                displacement = np.linalg.norm(self.data.body("protected_vase").xpos - self.initial_neighbor)
                self.max_protected_displacement = max(self.max_protected_displacement, float(displacement))
            left, right = False, False
            contacts = []
            for contact in self.data.contact:
                a,b = self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name
                contacts.append([a,b,float(contact.dist)])
                robot = [name for name in (a,b) if name and name.endswith("collision")]
                violation = {"time":float(self.data.time),"stage":self.stage,"geoms":[a,b],"distance":float(contact.dist)}
                if "table" in (a,b) and any(name != "base_collision" for name in robot) and contact.dist < -.001:
                    self.forbidden_contacts.append(violation)
                if "protected_geom" in (a,b) and robot:
                    self.protected_contacts.append(violation)
                if len(robot) == 2 and contact.dist < -.001:
                    self.self_contacts.append(violation)
                if any(name and name.startswith("target_geom") for name in (a,b)) and robot:
                    left |= any(name.startswith("lf_") for name in robot)
                    right |= any(name.startswith("rf_") for name in robot)
                    self.max_target_penetration = max(self.max_target_penetration, -float(contact.dist))
            if left and right and p[2] > .10:
                self.bilateral_lift_seconds += self.model.opt.timestep
            for body,geom in self.target_geoms.items():
                if body not in self.lift_seconds_by_object:
                    continue
                sides = set()
                for a,b,_ in contacts:
                    if geom in (a,b):
                        for name in (a,b):
                            if name and name.startswith("lf_"):
                                sides.add("left")
                            if name and name.startswith("rf_"):
                                sides.add("right")
                if len(sides) == 2 and self.data.body(body).xpos[2] > .10:
                    self.lift_seconds_by_object[body] += self.model.opt.timestep
            unsafe = (self.forbidden_contacts or self.protected_contacts or self.self_contacts
                      or self.max_target_penetration > .003 or self.max_protected_displacement > .015)
            if self.step_count % 10 == 0 or unsafe:
                self.private_trace.append({"time":float(self.data.time),"stage":self.stage,"target_xyz":p.tolist(),
                    "joints":self.data.qpos[self.qadr].tolist(), "contacts":contacts,
                    "objects":{name:self.data.body(name).xpos.copy().tolist() for name in self.lift_seconds_by_object}})
            if unsafe:
                raise PhysicsSafetyError(f"unexpected physical contact/displacement in {self.stage}; see private_evaluator.jsonl")
            if self.record and self.step_count % 20 == 0:
                self.renderer.update_scene(self.data,camera="overview")
                frame=self.renderer.render().copy()
                self.frames.append(frame)
                observer=getattr(self,'observer',None)
                if observer is not None:observer.frame(frame,self.stage,float(self.data.time))

    def save(self, result):
        self.output.mkdir(parents=True, exist_ok=True)
        # Observation probes can leave the final time between 20-ms trace
        # samples. Flush the actual last state without advancing physics.
        if not self.private_trace or abs(self.private_trace[-1]['time']-float(self.data.time))>1e-9:
            self.private_trace.append({'time':float(self.data.time),'stage':self.stage,
                'target_xyz':self.data.body('target_cube').xpos.copy().tolist(),
                'joints':self.data.qpos[self.qadr].tolist(),
                'contacts':[[self.model.geom(c.geom1).name,self.model.geom(c.geom2).name,float(c.dist)] for c in self.data.contact],
                'objects':{name:self.data.body(name).xpos.copy().tolist() for name in self.lift_seconds_by_object},
                'final_snapshot':True})
        (self.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        for name, events in (("public_trace", self.public_trace), ("private_evaluator", self.private_trace)):
            (self.output/f"{name}.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
        if self.frames:
            import mediapy
            mediapy.write_video(self.output/"rollout.mp4",self.frames,fps=25)
        self.renderer.close()

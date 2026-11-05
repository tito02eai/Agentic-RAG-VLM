"""Physical YAM grasp/place qualification through the Agentic session.

Color-labelled RGB-D is the explicitly labelled diagnostic perception here;
this run does not claim to use a VLM. Every object is free and contact-driven.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentic_rag_vlm.simulation.yam import YamPhysics
from agentic_rag_vlm.platform.transport import PlatformObservation, ObjectMetadata
from agentic_rag_vlm.platform.session import BluPeSession
from agentic_rag_vlm.platform.perception import RegistryMetadataPerception
from agentic_rag_vlm.platform.commands import GraspActionMapper
from agentic_rag_vlm.platform.verifier import SceneChangeVerifier
from agentic_rag_vlm.pipeline import AgenticPipeline
from agentic_rag_vlm.runtime import AgenticRuntime, TaskSpec, Subgoal
from scripts.agentic_framework import load_semantic_registry


class PhysicsTransport:
    def __init__(self, sim, *, use_vlm=False, destinations_by_target=None):
        self.sim = sim
        self.counter = 0
        self.use_vlm = use_vlm
        self.destinations_by_target = destinations_by_target or {}
        self.grasp_offset = np.array([.014,0,.015])
        self.rotation = np.diag([-1.,1.,-1.])

    def get_observation(self):
        self.sim.step(100)
        self.counter += 1
        receipt = {"transport":"mujoco_contact_dynamics", "time":float(self.sim.data.time),
                     "captured_at":float(self.sim.data.time),
                     "release_state":{"gripper_open":bool(self.sim.data.qpos[self.sim.model.joint('left_finger').qposadr][0] > .03),
                                      "retracted":self.sim.stage == "settle_final"},
                     "perception":"color_labelled_rgbd_diagnostic"}
        if not self.use_vlm:
            objects = self.sim.detect()
            return PlatformObservation(f"physics_{self.counter}", tuple(ObjectMetadata(name,xyz) for name,xyz in objects.items()), receipt=receipt)
        from PIL import Image
        folder = self.sim.output/f"observation_{self.counter:03d}"
        folder.mkdir()
        rgb,depth = self.sim.rgbd()
        Image.fromarray(rgb).save(folder/"top.png")
        np.save(folder/"depth_m.npy",depth)
        proposals = self.sim.proposals(rgb,depth)
        # Rotate proposal order so candidate IDs do not encode class identity.
        if self.counter % 2 == 0:
            proposals.reverse()
        candidates = []
        for index, item in enumerate(proposals):
            x0,y0,x1,y1 = item["bbox_xyxy"]
            path = folder/f"candidate_{index}.png"
            Image.fromarray(rgb).crop((max(0,x0-8),max(0,y0-8),min(640,x1+8),min(480,y1+8))).resize((192,192)).save(path)
            candidates.append({"candidate_id":f"candidate_{index}","top_xyz_m":item["top_xyz_m"],
                "visible_pixel_count":item.get('visible_pixel_count'),
                "bbox_xyxy":item["bbox_xyxy"],"crop_path":str(path.resolve())})
        receipt.update(perception="vlm_classification_color_rgbd_proposals",rgbd_candidates=candidates)
        return PlatformObservation(f"physics_{self.counter}", (), image_paths=(str((folder/"top.png").resolve()),),receipt=receipt)

    def send_command(self, command):
        params = command.parameters
        velocity = float(params.get("velocity_scale",1))
        if not 0 < velocity <= 1:
            return {"accepted":False, "execution_started":False, "reason":"invalid_velocity_scale"}
        def move(point, *, duration=1.5, **kwargs):
            return self.sim.move(point, duration=duration/velocity,rotation=self.rotation, **kwargs)
        if command.command_type == "pick":
            if params.get("approach") != "top":
                return {"accepted":False, "execution_started":False, "reason":"unsupported_approach"}
            target = np.array(params["position_xyz_m"])
            yaw = float(params.get("approach_yaw_delta_rad",0))+float(params.get("wrist_yaw_delta_rad",0))
            if not np.isfinite(yaw) or abs(yaw) > np.pi/2:
                return {"accepted":False,"execution_started":False,"reason":"unsupported_yaw"}
            c,s = np.cos(yaw),np.sin(yaw)
            turn = np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            self.rotation = turn @ np.diag([-1.,1.,-1.])
            self.grasp_offset = turn @ np.array([.014,0,.015])
            grasp = target + self.grasp_offset
            height = float(params.get("approach_height_delta_m",0))
            offset = np.r_[params.get("approach_offset_xy_m",[0,0]),0]
            # Reject impossible future placement before closing the fingers.
            destination = np.array(self.destinations_by_target.get(command.target,self.sim.setup["destination"])) + self.grasp_offset
            pregrasp = grasp + [0,0,.10+height] + offset
            for point in (pregrasp, grasp, grasp+[0,0,.12], destination+[0,0,.12], destination, [.20,-.20,.15]):
                self.sim.solve(point,rotation=self.rotation)
            opening = min(.0375, max(0., float(params["aperture_m"])/2))
            # Requested force is a motor-force cap, not a calibrated contact-force controller.
            force_cap = min(12., max(.1, float(params["force_n"])))
            self.sim.model.actuator_forcerange[6] = [-force_cap, force_cap]
            self.sim.public_trace.append({"event":"gripper_parameters", "requested_force_n":params["force_n"],
                "applied_motor_force_cap_n":force_cap, "opening_joint_m":opening,
                "synergy_mapping":"parallel_jaw_only"})
            move(pregrasp, grip=opening, stage="pregrasp")
            move(grasp, grip=opening, duration=2, stage="descend")
            self.sim.stage = "close"
            self.sim.data.ctrl[6] = .0
            self.sim.step(600)
            move(grasp + [0,0,.12], grip=.0, duration=2, stage="lift")
        elif command.command_type == "place":
            target = np.array(params["destination_xyz_m"]) + self.grasp_offset
            move(target+[0,0,.12], grip=.0, duration=2, stage="transport")
            move(target, grip=.0, duration=2, stage="place")
            self.sim.stage = "release"
            self.sim.data.ctrl[6] = .0375
            self.sim.step(600)
            move(target+[0,0,.12], grip=.0375, duration=2, stage="retract")
            # Clear the destination camera view before judging placement.
            move([.20,-.20,.15], grip=.0375, duration=2, stage="park")
            self.sim.stage = "settle_final"
            self.sim.step(750)
        else:
            raise ValueError("unsupported physical skill")
        return {"accepted":True, "command_id":command.command_id, "execution":"actuators_and_mj_step"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbor", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--vlm", action="store_true", help="Use real VLM crop classification, no metadata identity fallback")
    parser.add_argument("--endpoint", default="http://127.0.0.1:18070/v1/chat/completions")
    parser.add_argument("--model", default="/home/admin1/g2_multimodal_agent/models/Qwen3.5-4B")
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/yam_physics_qualification.json")
    parser.add_argument("--multiple",action="store_true")
    parser.add_argument("--paper-store",type=Path,help="Strict paper HAA-RAG with actual CLIP vectors and VLM decision summaries")
    parser.add_argument("--clip-endpoint",default="http://127.0.0.1:18072")
    parser.add_argument("--memory-in",type=Path)
    parser.add_argument("--memory-out",type=Path)
    parser.add_argument("--disable-memory",action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a new output directory")
    if args.paper_store and not args.vlm:
        parser.error("paper profile requires --vlm")
    paper_store = None
    if args.paper_store:
        from agentic_rag_vlm.experience import ExperienceStore
        paper_store = ExperienceStore.load(args.paper_store)
        paper_store.validate_sources(args.paper_store.parent)
    sim = YamPhysics(args.output,seed=args.seed,neighbor=args.neighbor,record=args.record,multiple=args.multiple)
    if args.multiple:
        # The home wrist occludes the second object's top view. Move the empty
        # hand before the first observation; do not invent a hidden detection.
        sim.move([.20,-.20,.15],grip=.0375,stage="initial_view_clearance")
    cards = json.loads((ROOT/"knowledge_base/blupe_affordance_cards.json").read_text())["cards"]
    from agentic_rag_vlm.pipeline import PipelineConfig
    from agentic_rag_vlm.runtime import RuntimeConfig
    retriever,reasoner,encoder = None,None,None
    if args.paper_store:
        from agentic_rag_vlm.experience import ExperienceStore
        from agentic_rag_vlm.retrieval import HAARetriever
        from agentic_rag_vlm.clip_client import CLIPFeatureClient
        from agentic_rag_vlm.reasoning import VLMReasoner
        from agentic_rag_vlm.qwen_adapter import OpenAICompatibleQwenVL
        retriever = HAARetriever(paper_store)
        if retriever.store.domain != "mjlab_yam_top_grasp_v1":
            parser.error("knowledge store belongs to a different robot/executor domain")
        encoder = CLIPFeatureClient(args.clip_endpoint)
        if encoder.embedding_space != retriever.store.embedding_space:
            parser.error("knowledge and query encoder weights differ")
        reasoner = VLMReasoner(OpenAICompatibleQwenVL(args.endpoint,args.model,max_tokens=250))
    config = PipelineConfig(memory_enabled=not args.disable_memory)
    if args.paper_store:
        from agentic_rag_vlm.paper import PaperAgent
        pipeline = PaperAgent(args.paper_store,reasoner.client,seed=args.seed,config=config)
        reasoner = pipeline.reasoner
    else:
        pipeline = AgenticPipeline(cards,config,seed=args.seed)
    if args.memory_in:
        from agentic_rag_vlm.memory import EpisodicMemory
        pipeline.memory = EpisodicMemory.load(args.memory_in)
    runtime = AgenticRuntime(pipeline,RuntimeConfig(memory_enabled=not args.disable_memory))
    registry = load_semantic_registry(ROOT/"knowledge_base/blupe_semantic_registry.json")
    if args.multiple:
        registry["cube2"] = dict(registry["cube"])
    perception = RegistryMetadataPerception(registry)
    if args.vlm:
        from agentic_rag_vlm.qwen_adapter import OpenAICompatibleQwenVL
        from agentic_rag_vlm.simulation.perception import CandidateRGBDVLMPerception
        perception = CandidateRGBDVLMPerception(OpenAICompatibleQwenVL(args.endpoint,args.model),registry,
            {"cube":.018,"vase":.04,**({"cube2":.018} if args.multiple else {})},visual_encoder=encoder,
            catalogue="cube (red cube), cube2 (yellow cube), vase (green cylindrical vessel proxy), unknown" if args.multiple else None)
    destinations = {"tray":sim.setup["destination"],"tray2":sim.setup["destination2"]}
    target_destinations = {"cube":destinations["tray"],"cube2":destinations["tray2"]}
    session = BluPeSession(runtime,PhysicsTransport(sim,use_vlm=args.vlm,destinations_by_target=target_destinations),perception,
        GraspActionMapper(),SceneChangeVerifier(destination_tolerance_m=.04), destinations=destinations,
        require_release_confirmation=True)
    task = TaskSpec(f"yam_physics_seed_{args.seed}", "Place the cubes in their designated blue zones",
        (Subgoal("cube","tray"),)+( (Subgoal("cube2","tray2"),) if args.multiple else ()), ("vase",) if args.neighbor else ())
    result = session.run(task)
    target = sim.data.body("target_cube").xpos.copy()
    error = float(np.linalg.norm(target - sim.setup["destination"]))
    neighbor_error = float(np.linalg.norm(sim.data.body("protected_vase").xpos - sim.initial_neighbor)) if args.neighbor else 0
    final_contacts = [item for item in sim.private_trace[-1]["contacts"] if "target_geom" in item[:2]]
    target_on_table = any("table" in item[:2] for item in final_contacts)
    finger_contact = any("collision" in (item[0] or "") or "collision" in (item[1] or "") for item in final_contacts)
    criteria = json.loads(args.protocol.read_text())["criteria"]
    checks = {
        "framework_complete":result["outcome"] == criteria["framework_outcome"],
        "destination":error < criteria["destination_error_m_lt"],
        "bilateral_lift":sim.bilateral_lift_seconds > criteria["bilateral_contact_above_0_10m_seconds_gt"],
        "table_support":target_on_table == criteria["target_supported_after_release"],
        "released":finger_contact == criteria["finger_contact_at_end"],
        "protected_displacement":sim.max_protected_displacement < criteria["max_protected_displacement_m_lt"],
        "arm_table_contact":len(sim.forbidden_contacts) == criteria["arm_table_penetration_beyond_1mm_contacts"],
        "protected_contact":len(sim.protected_contacts) == criteria["protected_robot_contacts"],
        "self_contact":len(sim.self_contacts) == criteria["self_penetration_beyond_1mm_contacts"],
        "target_penetration":sim.max_target_penetration < criteria["max_target_robot_penetration_m_lt"],
    }
    success = all(checks.values())
    if args.multiple:
        second = sim.data.body("target_cube2").xpos.copy()
        second_error = float(np.linalg.norm(second-sim.setup["destination2"]))
        contacts = sim.private_trace[-1]["contacts"]
        checks.update(second_destination=second_error < criteria["destination_error_m_lt"],
            second_bilateral_lift=sim.lift_seconds_by_object["target_cube2"] > .5,
            second_supported=any(set(c[:2]) == {"target_geom2","table"} for c in contacts),
            second_released=not any("target_geom2" in c[:2] and any(n and n.endswith("collision") for n in c[:2]) for c in contacts))
        success = all(checks.values())
    result.update(physical_success=bool(success), evidence_kind="mujoco_contact_dynamics",
        controller_perception="vlm_classification_color_rgbd_proposals" if args.vlm else "color_labelled_rgbd_diagnostic", uses_vlm=args.vlm,
        seed=args.seed, neighbor=args.neighbor,
        multiple=args.multiple,paper_profile=bool(args.paper_store),
        memory=pipeline.memory.to_dict(),
        memory_hits=sum(e["event"] == "subgoal_planned" and e["payload"]["action"]["source"] == "episodic_memory" for e in result["trace"]),
        reasoner_calls=reasoner.calls if reasoner else [],
        second_object_evaluation={"destination_error_m":second_error,"bilateral_lift_seconds":sim.lift_seconds_by_object["target_cube2"]} if args.multiple else None,
        qualification_checks=checks, qualification_criteria=criteria,
        simulator={"engine":"MuJoCo", "version":__import__("mujoco").__version__,
                   "timestep_s":float(sim.model.opt.timestep), "elapsed_simulation_s":float(sim.data.time)},
        vlm_calls=sum(r["kind"] == "perception" and r["payload"].get("vlm_result") is not None for r in result["receipts"]),
        private_evaluator={"final_target_xyz":target.tolist(),"destination_error_m":error,
        "max_target_height_m":sim.max_lift,"protected_displacement_m":neighbor_error,
        "max_protected_displacement_m":sim.max_protected_displacement,
        "bilateral_contact_above_0_10m_seconds":sim.bilateral_lift_seconds,
        "max_target_robot_penetration_m":sim.max_target_penetration,
        "protected_robot_contacts":len(sim.protected_contacts), "self_penetration_contacts":len(sim.self_contacts),
        "target_supported":target_on_table,"finger_contact_at_end":finger_contact,
        "forbidden_arm_table_contacts":len(sim.forbidden_contacts)},
        robot_variant="mjlab YAM; not a calibrated BluPe twin",
        object_attachment_used=False, object_pose_reset_after_initialization=False,
        ideal_robot_gravity_compensation=True)
    if args.memory_out:
        pipeline.memory.save(args.memory_out)
    sim.save(result)
    print(json.dumps({k:result[k] for k in ("outcome","physical_success","private_evaluator")},indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

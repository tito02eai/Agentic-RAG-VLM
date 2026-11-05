"""Contact-driven unstack/sort and sensed failure recovery; no hardware API."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.simulation.extended import ExtendedPhysics,RetentionTransport,SupportRGBDPerception,ObservedReleaseRecovery,HALVES
from agentic_rag_vlm.paper import PaperAgent
from agentic_rag_vlm.pipeline import AgenticPipeline,PipelineConfig
from agentic_rag_vlm.runtime import AgenticRuntime,RuntimeConfig,TaskSpec,Subgoal
from agentic_rag_vlm.platform.session import BluPeSession
from agentic_rag_vlm.platform.commands import GraspActionMapper
from agentic_rag_vlm.platform.verifier import SceneChangeVerifier
from agentic_rag_vlm.qwen_adapter import OpenAICompatibleQwenVL
from agentic_rag_vlm.clip_client import CLIPFeatureClient
from scripts.agentic_framework import load_semantic_registry
from agentic_rag_vlm.services import vlm_endpoint,clip_endpoint,resolve_model

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--mode',choices=['calibration','single_shift','flat_sort','stacked_sort'],default='stacked_sort')
    p.add_argument('--friction',type=float,default=.7);p.add_argument('--initial-force',type=float,default=.8)
    p.add_argument('--store',type=Path);p.add_argument('--disable-memory',action='store_true');p.add_argument('--disable-recovery',action='store_true')
    p.add_argument('--disable-scene-graph',action='store_true');p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--record',action='store_true');p.add_argument('--endpoint',default=vlm_endpoint())
    p.add_argument('--live-output',type=Path,help='read-only UI frames/runtime events; requires --record')
    p.add_argument('--controller-profile',choices=['legacy','robust','guarded'],default='legacy')
    p.add_argument('--scene-variation',choices=['fixed','independent','mixed_contact'],default='fixed')
    p.add_argument('--recovery-policy',choices=['paper','force_ladder'],default='paper')
    p.add_argument('--mixed-shapes',action='store_true')
    p.add_argument('--grounding-repair',action='store_true')
    p.add_argument('--multiccd',action='store_true',help='uniform MuJoCo multi-contact convex collision; no material changes')
    p.add_argument('--visibility-replanning',action='store_true',help='reorder this independent sorting task to clear observed occlusion')
    p.add_argument('--visible-goal-fallback',action='store_true',help='two-observation, VLM-reviewed progress on an identified elevated goal')
    p.add_argument('--upper-half-width',type=float,help='physical upper-block half-width in metres; stacked scenes only')
    p.add_argument('--retrieval-profile',choices=['haa','fixed'],default='haa')
    p.add_argument('--planning-policy',choices=['agentic','geometric'],default='agentic')
    p.add_argument('--model',help='served model; otherwise AGENTIC_VLM_MODEL or /health identity')
    p.add_argument('--clip-endpoint',default=clip_endpoint());args=p.parse_args()
    if args.output.exists():p.error('new output directory required')
    if args.live_output and not args.record:p.error('--live-output requires --record')
    if args.diagnostic and args.store:p.error('diagnostic identity mode is separate from the real-VLM paper profile')
    if args.mixed_shapes and args.scene_variation=='fixed':p.error('mixed shapes require an explicit varied scene')
    if args.grounding_repair and args.diagnostic:p.error('identity reinspection requires real VLM mode')
    if args.upper_half_width is not None and (args.mode!='stacked_sort' or args.scene_variation=='fixed'
            or not np.isfinite(args.upper_half_width) or not .012<=args.upper_half_width<=.017):
        p.error('upper footprint variation requires a varied stacked scene and half-width in [0.012,0.017]')
    if args.retrieval_profile=='fixed' and not args.store:p.error('fixed nominal comparison requires a provenance-checked executor store')
    if args.visible_goal_fallback and (not args.visibility_replanning or args.planning_policy!='agentic'
            or args.controller_profile!='guarded' or args.disable_scene_graph or args.diagnostic):
        p.error('visible goal fallback requires real VLM, agentic visibility replanning, scene graph and guarded execution')
    if args.planning_policy=='geometric' and (args.retrieval_profile!='fixed' or not args.disable_memory
            or not args.disable_scene_graph or args.visibility_replanning):
        p.error('geometric control must explicitly disable memory and scene graph, and use fixed skill retrieval')
    if not args.diagnostic:
        try:args.model=resolve_model(args.endpoint,args.model)
        except ValueError as error:p.error(str(error))
    client=OpenAICompatibleQwenVL(args.endpoint,args.model or 'diagnostic_not_called',max_tokens=300)
    encoder=CLIPFeatureClient(args.clip_endpoint) if args.store else None
    config=PipelineConfig(memory_enabled=not args.disable_memory,scene_graph_enabled=not args.disable_scene_graph,
                          recovery_enabled=not args.disable_recovery)
    if args.store:
        agent=PaperAgent(args.store,client,config=config,seed=args.seed)
        if agent.retriever.store.domain!='mjlab_yam_retention_v1':p.error('wrong executor domain')
        if encoder.embedding_space!=agent.retriever.store.embedding_space:p.error('wrong encoder')
        if args.retrieval_profile=='fixed':
            from agentic_rag_vlm.retrieval import FixedSkillRetriever
            agent.retriever=FixedSkillRetriever(agent.retriever.store)
        if args.planning_policy=='geometric':agent.reasoner=None
    else:
        cards=json.loads((ROOT/'knowledge_base/blupe_affordance_cards.json').read_text())['cards']
        for card in cards:
            if card['category']=='cube':card['strategy']['force_n']=args.initial_force
        agent=AgenticPipeline(cards,config,seed=args.seed)
    Physics,Transport=ExtendedPhysics,RetentionTransport
    if args.controller_profile in {'robust','guarded'}:
        from agentic_rag_vlm.simulation.robust import RobustPhysics,RobustRetentionTransport
        Physics,Transport=RobustPhysics,RobustRetentionTransport
    if args.controller_profile=='guarded':
        from agentic_rag_vlm.simulation.guarded import GuardedRetentionTransport
        Transport=GuardedRetentionTransport
    scene_builder=None
    if args.scene_variation!='fixed':
        from functools import partial
        from agentic_rag_vlm.simulation.robust import varied_scene
        scene_builder=partial(varied_scene,variation=args.scene_variation,mixed_shapes=args.mixed_shapes,
                              upper_half_width=args.upper_half_width)
    if args.multiccd:
        from functools import partial
        from agentic_rag_vlm.simulation.guarded import multicontact_scene
        scene_builder=partial(multicontact_scene,builder=scene_builder)
    sim=Physics(args.output,seed=args.seed,mode=args.mode,friction=args.friction,record=args.record,scene_builder=scene_builder)
    observer=None
    if args.live_output:
        from agentic_rag_vlm.simulation.live import LiveSimulationObserver
        observer=LiveSimulationObserver(args.live_output);sim.observer=observer
    try:
        sim.move([.2,-.2,.15],grip=.0375,stage='initial_view_clearance')
        names=list(sim.setup['destinations']);destinations={f'zone_{n}':xyz for n,xyz in sim.setup['destinations'].items()}
        transport=Transport(sim,use_vlm=not args.diagnostic,destinations_by_target=sim.setup['destinations'])
        registry=load_semantic_registry(ROOT/'knowledge_base/blupe_semantic_registry.json')
        for name in ('cube2','cube3'):registry[name]=dict(registry['cube'])
        if args.mixed_shapes:registry['cube3']['category']='cylinder'
        half_heights={n:HALVES[n] for n in names+(['vase'] if sim.setup['neighbor'] else [])}
        catalogue='cube (red block; may be partly under yellow), cube2 (yellow block), cube3 (magenta '+('cylinder' if args.mixed_shapes else 'block')+'), vase (green vessel), unknown'
        reviewer=None
        if args.grounding_repair:
            from agentic_rag_vlm.grounding import IdentityConflictReviewer
            reviewer=IdentityConflictReviewer(client,half_heights,catalogue)
        perception=SupportRGBDPerception(client,registry,half_heights,
            visual_encoder=encoder,catalogue=catalogue,identity_reviewer=reviewer,
            cylinder_models=({'vase':{'half_height_m':.04,'radius_m':.018},
                **({'cube3':{'half_height_m':.018,'radius_m':.018}} if args.mixed_shapes else {})}
                if args.controller_profile=='guarded' and sim.setup['neighbor'] else None))
        if args.diagnostic:
            from agentic_rag_vlm.platform.perception import RegistryMetadataPerception,PerceptionReceipt
            class Diagnostic(RegistryMetadataPerception):
                def perceive(self,obs):
                    objects,receipt=super().perceive(obs)
                    relations=[]
                    if 'cube2' in objects and abs(objects['cube2'].center_xyz_m[2]-.05)<.005:
                        relations=[{'source':'cube','target':'cube2','relation':'supports'}]
                    return objects,PerceptionReceipt('diagnostic_color_geometry',len(objects),(),{'scene_context':{'relations':relations}})
            perception=Diagnostic(registry)
        runtime=AgenticRuntime(agent,RuntimeConfig(memory_enabled=not args.disable_memory,maximum_replans=3,
            visibility_replanning=args.visibility_replanning,visible_goal_fallback=args.visible_goal_fallback,planning_policy=args.planning_policy,
            replanning_enabled=args.planning_policy!='geometric'),event_sink=observer.event if observer else None)
        decision_policy=None
        if args.recovery_policy=='force_ladder':
            from agentic_rag_vlm.simulation.robust import feedback_force_recovery
            decision_policy=feedback_force_recovery
        session=BluPeSession(runtime,transport,perception,
            GraspActionMapper(retention_force_adjustment=args.controller_profile=='guarded'),SceneChangeVerifier(destination_tolerance_m=.025),
            destinations=destinations,require_release_confirmation=True,
            contact_recovery=None if args.disable_recovery else ObservedReleaseRecovery(decision_policy))
        task=TaskSpec(f'extended_{args.seed}', 'Sort all three objects individually; remove anything resting on an object before its base; protect the green vessel.',
            tuple(Subgoal(n,f'zone_{n}') for n in names),('vase',) if sim.setup['neighbor'] else (),
            reorderable=True)
        result=session.run(task)
        final={};checks={}
        for name,body in zip(names,sim.target_geoms):
            xyz=sim.data.body(body).xpos.copy();geom=sim.target_geoms[body];contacts=sim.private_trace[-1]['contacts']
            error=float(np.linalg.norm(xyz-sim.setup['destinations'][name]))
            supported=any(set(c[:2])=={'table',geom} for c in contacts)
            final[name]={'xyz':xyz.tolist(),'destination_error_m':error,'bilateral_lift_seconds':sim.lift_seconds_by_object[body],
                'supported':supported}
            checks[name]=error<.025 and supported and sim.lift_seconds_by_object[body]>.5
        checks.update(framework=result['outcome']=='complete',no_protected_contact=not sim.protected_contacts,
            no_arm_table_contact=not sim.forbidden_contacts,no_self_contact=not sim.self_contacts,
            protected_stable=sim.max_protected_displacement<.015,target_penetration=sim.max_target_penetration<.003)
        sim.evaluator_end_attempt()
        checks['non_target_stable']=all(e['non_target_max_m']<.015 for e in sim.private_attempt_effects)
        checks['target_table_penetration']=sim.private_max_table_penetration<.003
        plan_order=[e['payload']['target'] for e in result['trace'] if e['event']=='subgoal_planned']
        if args.mode=='stacked_sort':
            checks['support_order']='cube' not in plan_order or ('cube2' in plan_order and plan_order.index('cube2')<plan_order.index('cube'))
        result.update(physical_success=bool(all(checks.values())),checks=checks,final_objects=final,
            tactile_telemetry_version=2,
            evidence_kind='mujoco_contact_dynamics',seed=args.seed,mode=args.mode,friction=args.friction,
            memory=agent.memory.to_dict(),attempts=transport.attempts,
            attempts_started=sum(e['event']=='gripper_parameters' for e in sim.public_trace),
            private_attempt_effects=sim.private_attempt_effects,
            memory_hits=sum(e['event']=='subgoal_planned' and e['payload']['action']['source']=='episodic_memory' for e in result['trace']),
            reasoner_calls=agent.reasoner.calls if agent.reasoner else [],disable_recovery=args.disable_recovery,
            disable_memory=args.disable_memory,uses_vlm=not args.diagnostic,controller_profile=args.controller_profile,
            recovery_policy=args.recovery_policy,scene_variation=args.scene_variation,
            private_sampled_contact=sim.setup.get('private_sampled_contact'),mixed_shapes=args.mixed_shapes,
            destinations=sim.setup['destinations'],
            grounding_repair=args.grounding_repair,
            multiccd=args.multiccd,
            visibility_replanning=args.visibility_replanning,
            visible_goal_fallback=args.visible_goal_fallback,
            upper_half_width_m=args.upper_half_width,retrieval_profile=args.retrieval_profile,
            planning_policy=args.planning_policy,disable_scene_graph=args.disable_scene_graph,
            original_paper_trials=False,object_attachment_used=False,object_pose_reset_after_initialization=False,
            max_protected_displacement_m=sim.max_protected_displacement,simulated_time_s=float(sim.data.time))
        result['max_target_table_penetration_m']=sim.private_max_table_penetration
        agent.memory.save(args.output/'memory.json');sim.save(result)
        print(json.dumps({'outcome':result['outcome'],'physical_success':result['physical_success'],'attempts':len(transport.attempts),
                          'retained':sum(a['retained'] for a in transport.attempts),'memory_hits':result['memory_hits'],'objects':final},indent=2))
        return 0 if result['physical_success'] else 1
    except Exception:
        sim.renderer.close();raise

if __name__=='__main__':raise SystemExit(main())

"""Extended tabletop scenes and sensed pick skills for physical recovery tests."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .yam import YamPhysics,build_scene
from .perception import CandidateRGBDVLMPerception

HALVES={"cube":.018,"cube2":.014,"cube3":.018,"vase":.04}


def extended_scene(path, *, seed=0, mode="stacked_sort", friction=.7, **ignored):
    many=mode in {"stacked_sort","flat_sort"}
    setup=build_scene(path,seed=seed,neighbor=many,multiple=many)
    tree=ET.parse(path);root=tree.getroot();world=root.find('worldbody')
    # Uniform solver setting, never toggled by retry/outcome. MuJoCo's NoSlip
    # postprocessor suppresses regularization drift inside the friction cone.
    root.find('option').set('noslip_iterations','10')
    root.find('option').set('cone','elliptic')
    root.find('option').set('tolerance','1e-10')
    rng=np.random.default_rng(seed)
    jitter=rng.uniform(-.004,.004,2)
    first=np.array([.26,-.05]) + jitter if mode=='stacked_sort' else np.array([.28,-.16])+jitter if many else np.array([.30,-.12])+jitter
    root.find('.//body[@name="target_cube"]').set('pos',f'{first[0]} {first[1]} .019')
    for geom in root.findall('.//geom'):
        if geom.get('name','').startswith(('lf_','rf_')):geom.set('friction','.7 .005 .0005')
        if geom.get('name','').startswith('target_geom'):
            geom.set('friction',f'{friction} .01 .001');geom.set('mass','.06')
    destinations={'cube':[.28,.16,.018] if many else [.26,.12,.018]}
    target_geoms={'target_cube':'target_geom'}
    if many:
        second=first if mode=='stacked_sort' else np.array([.20,-.04])+jitter
        z=.051 if mode=='stacked_sort' else .015
        root.find('.//body[@name="target_cube2"]').set('pos',f'{second[0]} {second[1]} {z}')
        root.find('.//geom[@name="target_geom2"]').set('size','.014 .014 .014')
        third=ET.SubElement(world,'body',name='target_cube3',pos=f'{.32+jitter[0]} {-.17+jitter[1] if mode=="stacked_sort" else -.04+jitter[1]} .019')
        ET.SubElement(third,'freejoint',name='target3_free')
        ET.SubElement(third,'geom',name='target_geom3',type='box',size='.018 .018 .018',mass='.06',
            rgba='.8 .06 .75 1',friction=f'{friction} .01 .001',condim='4',solref='.01 1')
        root.find('.//body[@name="protected_vase"]').set('pos','.43 -.10 .04')
        destinations.update(cube2=[.19,.12,.014],cube3=[.32,.04,.018])
        target_geoms.update(target_cube2='target_geom2',target_cube3='target_geom3')
    # Each destination is a non-colliding visual marker, not a physical tray.
    for geom in list(world.findall('geom')):
        if geom.get('name','').startswith('destination'):world.remove(geom)
    for index,(name,xyz) in enumerate(destinations.items()):
        ET.SubElement(world,'geom',name=f'destination_{index}',type='box',pos=f'{xyz[0]} {xyz[1]} .001',
            size='.026 .026 .001',rgba='.18 .45 .65 1',contype='0',conaffinity='0')
    tree.write(path,encoding='unicode')
    setup.update(mode=mode,multiple=many,neighbor=many,destinations=destinations,
        destination=destinations['cube'],target_geoms=target_geoms,controlled_friction=friction)
    return setup


class ExtendedPhysics(YamPhysics):
    def __init__(self,output,*,seed=0,mode='stacked_sort',friction=.7,record=False,scene_builder=None):
        self.private_max_table_penetration=0.
        def builder(path,**kw):return (scene_builder or extended_scene)(path,mode=mode,friction=friction,**kw)
        super().__init__(output,seed=seed,neighbor=mode in {'stacked_sort','flat_sort'},
            multiple=mode in {'stacked_sort','flat_sort'},record=record,scene_builder=builder)
        self._evaluation_attempt=None
        self.private_attempt_effects=[]

    def evaluator_start_attempt(self,target):
        body={'cube':'target_cube','cube2':'target_cube2','cube3':'target_cube3'}[target]
        self._evaluation_attempt={'target':target,'body':body,'initial':{n:self.data.body(n).xpos.copy() for n in self.target_geoms},
            'non_target_max_m':0.}

    def evaluator_end_attempt(self):
        if self._evaluation_attempt:
            self.private_attempt_effects.append({k:v for k,v in self._evaluation_attempt.items() if k not in {'initial','body'}})
            self._evaluation_attempt=None

    def step(self,count):
        # The evaluator is never queried by the planning/recovery policy.
        for _ in range(count):
            super().step(1)
            for c in self.data.contact:
                names=(self.model.geom(c.geom1).name,self.model.geom(c.geom2).name)
                if 'table' in names and any(n and n.startswith('target_geom') for n in names):
                    self.private_max_table_penetration=max(self.private_max_table_penetration,-float(c.dist))
            attempt=getattr(self,'_evaluation_attempt',None)
            if attempt:
                for body,xyz in attempt['initial'].items():
                    if body!=attempt['body']:
                        attempt['non_target_max_m']=max(attempt['non_target_max_m'],float(np.linalg.norm(self.data.body(body).xpos-xyz)))

    def proposals(self,rgb,depth):
        result=super().proposals(rgb,depth)
        mask=(rgb[:,:,0]>90)&(rgb[:,:,2]>80)&(rgb[:,:,1]<np.minimum(rgb[:,:,0],rgb[:,:,2])*.5)
        rows,cols=np.where(mask & np.isfinite(depth) & (depth>.05)&(depth<2))
        if len(rows)>=8:
            z=depth[rows,cols];f=480/(2*np.tan(np.deg2rad(45)/2));camera=self.data.cam('top')
            points=np.column_stack(((cols+.5-320)*z/f,-(rows+.5-240)*z/f,-z))
            world=points@camera.xmat.reshape(3,3).T+camera.xpos
            result.append({'color_index':3,'top_xyz_m':np.median(world,axis=0).tolist(),
                'visible_pixel_count':int(len(rows)),
                'bbox_xyxy':[int(cols.min()),int(rows.min()),int(cols.max())+1,int(rows.max())+1]})
        return result

    def detect(self):
        rgb,depth=self.rgbd();out={}
        for item in self.proposals(rgb,depth):
            name=('cube','vase','cube2','cube3')[item['color_index']]
            xyz=np.array(item['top_xyz_m']);xyz[2]-=HALVES[name];out[name]=tuple(xyz)
        return out


class SupportRGBDPerception(CandidateRGBDVLMPerception):
    def perceive(self,observation):
        objects,receipt=super().perceive(observation)
        relations=[];states={};unresolved_supports=[]
        for lower,a in objects.items():
            if lower=='vase':continue
            for upper,b in objects.items():
                if lower==upper or upper=='vase':continue
                xy=np.abs(np.asarray(a.center_xyz_m[:2])-b.center_xyz_m[:2])
                gap=(b.center_xyz_m[2]-HALVES[upper])-(a.center_xyz_m[2]+HALVES[lower])
                overlap_limit=HALVES[lower]+HALVES[upper]-.006
                if np.all(xy<overlap_limit) and abs(gap)<.005 and b.center_xyz_m[2]>a.center_xyz_m[2]:
                    relations.append({'source':lower,'target':upper,'relation':'supports'})
                    states[lower]='stacked'
        grounding=(receipt.vlm_result or {}).get('grounding',[])
        assigned={item['candidate_id'] for item in grounding}
        proposals=observation.receipt.get('rgbd_candidates',[])
        by_id={item['candidate_id']:item for item in proposals}
        for upper,item in objects.items():
            if upper=='vase':continue
            upper_crop=next((g['candidate_id'] for g in grounding if g['name']==upper),None)
            if upper_crop not in by_id:continue
            box=by_id[upper_crop]['bbox_xyxy']
            for candidate in proposals:
                if candidate['candidate_id'] in assigned:continue
                xyz=np.asarray(candidate['top_xyz_m'],dtype=float)
                gap=float(item.center_xyz_m[2]-HALVES[upper]-xyz[2])
                xy=np.abs(np.asarray(item.center_xyz_m[:2])-xyz[:2])
                other=candidate['bbox_xyxy']
                intersects=min(box[2],other[2])>max(box[0],other[0]) and min(box[3],other[3])>max(box[1],other[1])
                if intersects and np.all(xy<.026) and abs(gap)<.005:
                    unresolved_supports.append({'upper_target':upper,
                        'support_candidate_id':candidate['candidate_id'], 'support_top_xyz_m':xyz.tolist(),
                        'upper_half_height_m':HALVES[upper], 'upper_bbox_xyxy':box,
                        'support_bbox_xyxy':other,'observation_id':observation.observation_id,
                        'source':'public_rgbd_surface_plus_known_upper_dimensions',
                        'support_identity':'unresolved'})
        visible_geometry=[{'target':name,'center_xyz_m':list(item.center_xyz_m),
            'known_half_height_m':HALVES[name],'observation_id':observation.observation_id,
            'source':'public_rgbd_and_vlm_identified_target'} for name,item in objects.items()]
        payload={**receipt.vlm_result,'scene_context':{'states':states,'relations':relations,
                 'unresolved_supports':unresolved_supports,'visible_geometry':visible_geometry,
                 'support_plane':{'z_m':0.,'source':'calibrated_testbed_table'}},
                 'relation_source':'observed_rgbd_plus_known_dimensions; not simulator body poses'}
        return objects,replace(receipt,vlm_result=payload)


from scripts.run_yam_physics import PhysicsTransport


class RetentionTransport(PhysicsTransport):
    """Pick acceptance needs sensed retention, not just completed joint motion."""
    def __init__(self,sim,**kwargs):
        super().__init__(sim,**kwargs)
        self.attempts=[]

    def get_observation(self):
        observation=super().get_observation()
        if not self.use_vlm:return observation
        from PIL import Image
        folder=Path(observation.image_paths[0]).parent
        rgb=np.asarray(Image.open(folder/'top.png')).copy();depth=np.load(folder/'depth_m.npy')
        masks=[(rgb[:,:,0]>100)&(rgb[:,:,0]>rgb[:,:,1]*1.8)&(rgb[:,:,0]>rgb[:,:,2]*1.8),
            (rgb[:,:,1]>70)&(rgb[:,:,1]>rgb[:,:,0]*1.7)&(rgb[:,:,1]>rgb[:,:,2]*1.4),
            (rgb[:,:,0]>100)&(rgb[:,:,1]>90)&(rgb[:,:,2]<np.minimum(rgb[:,:,0],rgb[:,:,1])*.55),
            (rgb[:,:,0]>90)&(rgb[:,:,2]>80)&(rgb[:,:,1]<np.minimum(rgb[:,:,0],rgb[:,:,2])*.5)]
        proposals=self.sim.proposals(rgb,depth)
        for candidate in observation.receipt['rgbd_candidates']:
            proposal=next(item for item in proposals if item['bbox_xyxy']==candidate['bbox_xyxy'])
            x0,y0,x1,y1=candidate['bbox_xyxy']
            masked=np.full_like(rgb,80);mask=masks[proposal['color_index']];masked[mask]=rgb[mask]
            Image.fromarray(masked).crop((max(0,x0-8),max(0,y0-8),min(640,x1+8),min(480,y1+8))).resize((192,192)).save(candidate['crop_path'])
        observation.receipt['candidate_preprocessing']='visible color-proposal mask with neutral background; no semantic identity supplied'
        return observation

    def send_command(self,command):
        if command.command_type=='place':return self.place(command)
        if command.command_type!='pick':return super().send_command(command)
        p=command.parameters
        if p.get('approach')!='top':
            return {'accepted':False,'execution_started':False,'reason':'unsupported_approach'}
        if p.get('synergy','power')!='power':
            return {'accepted':False,'execution_started':False,'reason':'unvalidated_grasp_method'}
        # Put the lowest collision pad 2.2 mm above the observed supporting
        # surface, rather than pinching only the top edge of a small block.
        self.grasp_offset=np.array([.014,0,.024-HALVES[command.target]])
        target=np.asarray(p['position_xyz_m']);grasp=target+self.grasp_offset
        destination=np.asarray(self.destinations_by_target[command.target])+self.grasp_offset
        offset=np.r_[p.get('approach_offset_xy_m',[0,0]),0]
        pre=grasp+offset;pre[2]=max(.13,grasp[2]+.07)+float(p.get('approach_height_delta_m',0))
        lifted=grasp.copy();lifted[2]=.15
        over=destination.copy();over[2]=.15
        for point in (pre,grasp,lifted,over,destination,[.20,-.20,.15]):self.sim.solve(point)
        self.sim.evaluator_start_attempt(command.target)
        started=float(self.sim.data.time)
        force=min(12.,max(.1,float(p['force_n'])))
        self.sim.model.actuator_forcerange[6]=[-force,force]
        opening=min(.0375,float(p['aperture_m'])/2)
        self.sim.public_trace.append({'event':'gripper_parameters','target':command.target,'time':float(self.sim.data.time),
            'requested_force_n':p['force_n'],'applied_motor_force_cap_n':force,'opening_joint_m':opening,
            'synergy_mapping':'parallel_jaw_only'})
        self.sim.move(pre,grip=opening,stage='pregrasp')
        self.sim.move(grasp,grip=opening,duration=2,stage='descend')
        self.sim.stage='close';self.sim.data.ctrl[6]=0;self.sim.step(600)
        before=self.sim.tactile()
        initial_tactile_time=float(self.sim.data.time)
        held,samples,lift_evidence=self._lift_and_test(grasp,lifted,before)
        attempt={'target':command.target,'command_id':command.command_id,'motor_force_cap_n':force,
            'start_time_s':started,'initial_tactile_time_s':initial_tactile_time,'end_time_s':float(self.sim.data.time),
            'initial_tactile':before,'retention_samples':samples,'retained':held,'source':command.origin,
            **lift_evidence}
        self.attempts.append(attempt)
        self.sim.public_trace.append({'event':'sensed_retention',**attempt})
        if held:
            return {'accepted':True,'execution_started':True,'command_id':command.command_id,'retention':attempt,
                **({'execution_feedback': {'force_n': attempt['motor_force_cap_n'],
                    'source': 'public_tactile_low_height_force_adaptation'}} if attempt.get('force_adjustments') else {})}
        # No speculative move with an uncertain held object. Only after the
        # tactile window reports no retained grasp, open and clear the view.
        self.sim.stage='release_failed_grasp';self.sim.data.ctrl[6]=.0375;self.sim.step(600)
        self.sim.move([.20,-.20,.15],grip=.0375,duration=2,stage='recovery_retreat')
        self.sim.stage='settle_final';self.sim.step(750)
        self.sim.evaluator_end_attempt()
        touch=self.sim.tactile()
        return {'accepted':False,'execution_started':True,'reason':'grasp_not_retained',
            'failure_type':lift_evidence.get('failure_type','slip' if before['bilateral'] else 'position_error'),
            'release_confirmed':all(v<.01 for v in touch['normal_force_n'].values()) and touch['left_finger_q_m']>.03,
            'retention':attempt}

    def _lift_and_test(self,grasp,lifted,before):
        """Legacy contract, overridden only by explicit guarded controller."""
        self.sim.move(lifted,grip=0,duration=2,stage='lift')
        samples=[]
        for _ in range(10):
            self.sim.step(20);samples.append({'time':float(self.sim.data.time),**self.sim.tactile()})
        held=all(x['bilateral'] and x['left_finger_q_m']>.004 for x in samples)
        return held,samples,{}

    def place(self,command):
        """Absolute clearance also accommodates the smaller stacked target."""
        goal=np.asarray(command.parameters['destination_xyz_m'])+self.grasp_offset
        over=goal.copy();over[2]=.15
        self.sim.move(over,grip=0,duration=4,stage='transport')
        self.sim.step(150)
        baseline=self.sim.tactile()['downward_load_n']
        if baseline<.03:raise RuntimeError('object load absent before placement')
        supported=False
        for height in np.arange(.144,goal[2]-.006,-.006):
            point=goal.copy();point[2]=max(float(height),goal[2])
            self.sim.move(point,grip=0,duration=.2,stage='place_compliant')
            loads=[]
            for _ in range(3):
                self.sim.step(20);loads.append(self.sim.tactile()['downward_load_n'])
            if min(loads)<baseline*.75:
                for _ in range(10):
                    if all(load<baseline*.5 for load in loads[-3:]):break
                    self.sim.step(20);loads.append(self.sim.tactile()['downward_load_n'])
            self.sim.public_trace.append({'event':'placement_load','target':command.target,'time':float(self.sim.data.time),
                'tcp_z_m':point[2],'reference_load_n':baseline,'observed_loads_n':loads})
            if all(load<baseline*.5 for load in loads[-3:]):supported=True;break
        if not supported:raise RuntimeError('placement support not sensed; refusing blind release')
        self.sim.stage='release';self.sim.data.ctrl[6]=.0375;self.sim.step(600)
        self.sim.move(over,grip=.0375,duration=2,stage='retract')
        self.sim.move([.2,-.2,.15],grip=.0375,duration=2,stage='park')
        self.sim.stage='settle_final';self.sim.step(750)
        self.sim.evaluator_end_attempt()
        return {'accepted':True,'execution_started':True,'command_id':command.command_id}


class ObservedReleaseRecovery:
    def __init__(self, decision_policy=None):
        self.decision_policy=decision_policy

    def __call__(self,session,action,receipt,before,task,attempt):
        if (receipt.get('reason')!='grasp_not_retained' or receipt.get('execution_started') is not True
                or receipt.get('release_confirmed') is not True):return None
        from math import dist
        from ..recovery import FailureType,recover
        from ..constraints import compile_constraints
        first=session._perceive();first_capture=session._last_observation.receipt.get('captured_at')
        second=session._perceive();last=session._last_observation
        names=set(task.protected_objects)|set(session.runtime.completed_targets)|{action.target}
        if names-set(first) or names-set(second):return None
        if any(dist(first[n].center_xyz_m,second[n].center_xyz_m)>.004 for n in names):return None
        for name in names-{action.target}:
            if name not in before or dist(before[name].center_xyz_m,second[name].center_xyz_m)>.015:return None
        release=last.receipt.get('release_state',{})
        if (first_capture==last.receipt.get('captured_at') or release.get('gripper_open') is not True
                or release.get('retracted') is not True):return None
        xyz=second[action.target].center_xyz_m
        supported=abs(xyz[2]-HALVES[action.target])<.005 or any(
            e['relation']=='supports' and e['target']==action.target for e in session.runtime.pipeline.scene_context.get('relations',[]))
        if not supported:
            # A partially visible supporter may lack a semantic ID. Its RGB-D
            # top surface can still confirm support without inventing that ID.
            support_z=xyz[2]-HALVES[action.target]
            supported=any(abs(c['top_xyz_m'][2]-support_z)<.005
                and np.linalg.norm(np.asarray(c['top_xyz_m'][:2])-xyz[:2])<.04
                for c in last.receipt.get('rgbd_candidates',[]))
        if not supported:return None
        failure=FailureType(receipt['failure_type'])
        decision=(self.decision_policy or recover)(action.strategy,failure,attempt,rng=session.runtime.pipeline.rng)
        if decision.safe_stop:return None
        pipe=session.runtime.pipeline
        invalidated=receipt.get('memory_invalidated')
        if action.source=='episodic_memory':
            old=pipe.memory.invalidate(second[action.target].category)
            if old is not None:
                invalidated={'category':old.category,'source_episode':old.source_episode,
                    'failed_force_n':action.strategy['force_n'],'reason':'observed_retention_failure'}
        constraint=compile_constraints(second[action.target],second,pipe.scene_context,enabled=pipe.config.scene_graph_enabled)
        if constraint['blocked']:return None
        if decision.action=='full_replan':
            fresh,evidence=pipe.plan(second[action.target],second,use_memory=False)
            delta=decision.corrected_strategy['position_perturbation_m']
            recovered=replace(fresh,position_xyz_m=tuple(a+b for a,b in zip(fresh.position_xyz_m,delta)),source='recovery_l3')
        else:
            recovered=replace(action,position_xyz_m=tuple(xyz),strategy=decision.corrected_strategy,
                              constraint=constraint,source=(f'recovery_force_ladder_{attempt}'
                                  if decision.action=='bounded_force_escalation' else f'recovery_l{attempt}'))
        evidence={**decision.to_dict(),'release_confirmed':True,'scene_stable':True,
            'public_observation_ids':[first_capture,last.receipt.get('captured_at')],
            'target_on_observed_support':True,'retained_subgoals':list(session.runtime.completed_targets),
            'memory_invalidated':invalidated}
        if pipe.reasoner is not None:
            session._record('vlm_reflection',pipe.reasoner.reflect(action,evidence,session.receipts))
            session._record('recovery_thought',pipe.reasoner.think(recovered,{'constraint':constraint},[],attempt))
        return recovered,second,evidence

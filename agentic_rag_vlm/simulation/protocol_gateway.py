"""Local Session API + WebSocket gateway backed by actual MuJoCo actuators.

Only the simulator worker touches MuJoCo state/rendering. HTTP/WS clients use
the same leased joint-trajectory contract as the public station. The dormant
second arm is a virtual held channel, explicitly not a bimanual digital twin.
"""
import asyncio
import base64
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import queue
import re
import threading
import time
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from ..experience import atomic_json
from ..platform.local_runner import import_official_runner


def color_region_candidates(observation):
    """RGB-only saturated connected regions; no object ID/pose/segmentation truth."""
    from scipy.ndimage import label,binary_closing
    rgb=np.asarray(Image.open(observation.image_paths[0]).convert('RGB')).astype(float)
    maximum=rgb.max(2);minimum=rgb.min(2)
    saturated=(maximum>80)&((maximum-minimum)>maximum*.5)
    regions=[]
    # This registered task uses a blue painted destination, not blue task
    # objects. Exclude that known background color before proposing regions.
    # Red/green regions still have no supplied object identity.
    dominant=rgb.argmax(2)
    # Yellow surfaces can alternate between red/green dominant channels under
    # lighting/JPEG noise. Keep that hue band connected without assigning IDs.
    yellow=(rgb[:,:,0]>.6*maximum)&(rgb[:,:,1]>.6*maximum)
    masks=[saturated&(dominant==channel)&~yellow for channel in (0,1)]
    masks.append(saturated&yellow)
    for mask in masks:
        # JPEG edge pixels can disconnect the top/side faces of one object.
        # Join only a one-pixel image gap; retain opaque region identities.
        connected=binary_closing(mask,structure=np.ones((3,3),dtype=bool))
        labels,count=label(connected,structure=np.ones((3,3),dtype=bool))
        for number in range(1,count+1):
            y,x=np.where((labels==number)&mask)
            if len(x)<15:continue
            box=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
            regions.append({'bbox_xyxy':box,'pixel_xy':[float(np.median(x)),float(np.median(y))]})
    regions.sort(key=lambda r:(r['bbox_xyxy'][1],r['bbox_xyxy'][0]))
    if len(regions)>8:raise ValueError('too many unclassified image regions')
    return [dict(candidate_id=f'candidate_{i}',**r) for i,r in enumerate(regions)]


class PhysicsWorker:
    def __init__(self,output,*,seed=1000,multiple=False):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        self.seed=seed;self.multiple=multiple;self.jobs=queue.Queue();self.ready=concurrent.futures.Future()
        self.cache={};self.cache_lock=threading.RLock();self.cancel=threading.Event();self.finalized=False
        self.thread=threading.Thread(target=self._loop,name='mujoco-protocol-worker',daemon=True)
        self.thread.start();self.ready.result(timeout=40)
    def _loop(self):
        try:
            from .yam import YamPhysics
            self.sim=YamPhysics(self.output/'physics',seed=self.seed,neighbor=True,record=True,multiple=self.multiple)
            self.sim.model.actuator_forcerange[6]=[-1.6,1.6]
            from .live import LiveSimulationObserver
            self.sim.observer=LiveSimulationObserver(self.output/'live')
            self.sim.move([.2,-.2,.15],grip=.0375,stage='initial_view_clearance')
            self._make_calibration();self._capture();self.ready.set_result(True)
        except Exception as error:self.ready.set_exception(error);return
        while True:
            function,future=self.jobs.get()
            if function is None:break
            try:future.set_result(function())
            except Exception as error:future.set_exception(error)
        self.sim.renderer.close()
    def rpc(self,function,timeout=60):
        future=concurrent.futures.Future();self.jobs.put((function,future));return future.result(timeout=timeout)
    def submit(self,function):
        future=concurrent.futures.Future();self.jobs.put((function,future));return future
    def _make_calibration(self):
        import mujoco
        sim=self.sim;zero=mujoco.MjData(sim.model);zero.qpos[:]=sim.data.qpos
        zero.qpos[sim.qadr]=0;mujoco.mj_forward(sim.model,zero)
        home=np.eye(4);home[:3,:3]=zero.site_xmat[sim.site].reshape(3,3);home[:3,3]=zero.site_xpos[sim.site]
        joints=[{'axis':zero.xaxis[j].tolist(),'position_m':zero.xanchor[j].tolist()} for j in sim.joints]
        camera=sim.data.cam('top');transform=np.eye(4)
        transform[:3,:3]=camera.xmat.reshape(3,3)@np.diag([1.,-1.,-1.]);transform[:3,3]=camera.xpos
        f=480/(2*np.tan(np.deg2rad(float(sim.model.cam_fovy[sim.model.camera('top').id]))/2))
        identity='local-mujoco-'+hashlib.sha256((self.output/'physics/scene.xml').read_bytes()).hexdigest()[:16]
        self.calibration={'schema_version':1,'calibration_id':identity,'execution_environment':'simulation',
            'cameras':{'top':{'K':[[f,0,319.5],[0,f,239.5],[0,0,1]],'image_size':[640,480],
                'distortion_model':'none','distortion':[0]*5,'T_base_camera':{'right':transform.tolist()}}},
            'kinematics':{'joint_units':'degrees','right':{'joints':joints,'T_base_grasp_zero':home.tolist()}},
            'quality':{'status':'simulator_geometry','failures':[],'physical_accuracy_verified':False}}
        limits=np.degrees(sim.model.jnt_range[sim.joints]).tolist()
        self.profile={'robot_id':'yam-local-1','arm':'right','coordinate_frame':'arm_base','calibration_id':identity,
            'execution_environment':'simulation','physical_validation':{'passed':False},
            'simulation_validation':{'passed':True,'record':'MuJoCo model-derived FK and camera geometry; not physical calibration'},
            'support_plane':{'normal_base':[0,0,1],'offset_m':0.},'joint_limits_deg':limits,
            'workspace_bounds_m':[[-.15,.65],[-.45,.4],[.005,.7]],'feature_heights_m':{'cube':.036,'vase':.08},
            'grasp_offsets_base_m':{'cube':[.014,0,-.012],'vase':[0,0,-.04]},
            'gripper':{'max_aperture_m':.075},'max_observation_age_s':2.,'approach':'top',
            'observation_park_xyz_m':[.2,-.2,.15],'lift_height_m':.12,
            'retention_observation':{'minimum_blocked_aperture':.1,'maximum_blocked_aperture':.85,'minimum_tcp_lift_m':.05}}
        from ..platform.native_kinematics import NativeYamKinematics
        kin=NativeYamKinematics(self.calibration,'right',limits,max_aperture_m=.075)
        # mj_step leaves derived transforms at its pre-integration stage.
        # Refresh the same physical state before comparing analytic FK.
        mujoco.mj_forward(sim.model,sim.data)
        fk=kin.fk(sim.data.qpos[sim.qadr]);actual=sim.data.site_xpos[sim.site]
        if not np.allclose(fk[:3,3],actual,atol=1e-7):raise ValueError('model-derived native FK does not match simulator: '+str(float(np.linalg.norm(fk[:3,3]-actual))))
        self.geometry_check={'fk_position_error_m':float(np.linalg.norm(fk[:3,3]-actual)),
            'tool_rotation_for_top':kin.orientation_for_approach(np.array([0,0,-1.])).tolist()}
        atomic_json(self.output/'calibration.json',self.calibration);atomic_json(self.output/'profile.json',self.profile)
        atomic_json(self.output/'geometry_check.json',self.geometry_check)
    def _capture(self):
        if self.finalized:
            return
        sim=self.sim;rgb,_=sim.rgbd();buffer=io.BytesIO();Image.fromarray(rgb).save(buffer,format='JPEG',quality=95)
        now=time.time();q=np.degrees(sim.data.qpos[sim.qadr]).tolist()
        grip=float(np.clip(sim.data.qpos[sim.model.joint('left_finger').qposadr][0]/.0375,0,1))
        with self.cache_lock:
            self.cache={'top':buffer.getvalue(),'captured_at':now,'joints':q,'gripper':grip,
                        'simulation_time_s':float(sim.data.time),'stage':sim.stage}
    def snapshot(self):
        with self.cache_lock:return dict(self.cache)
    def execute(self,waypoints,progress):
        import mujoco
        sim=self.sim;sim.stage='protocol_trajectory'
        for point in waypoints:
            if self.cancel.is_set():raise RuntimeError('local simulator cancelled')
            target=np.radians(point['right_joints_deg']);grip=float(point.get('right_gripper',1.))
            old=sim.data.ctrl[:6].copy()
            # 10-Hz wire samples interpolate actuator controls at the simulator
            # timestep; no physical qpos is assigned and no object is attached.
            for i in range(50):
                if self.cancel.is_set():raise RuntimeError('local simulator cancelled')
                sim.data.ctrl[:6]=old+(target-old)*(i+1)/50
                sim.data.ctrl[6]=grip*.0375;sim.step(1)
            progress(point['step_id']);self._capture()
        sim.step(250);self._capture()
        joint_error=float(np.max(np.abs(np.degrees(sim.data.qpos[sim.qadr])-waypoints[-1]['right_joints_deg'])))
        if joint_error>1.0:raise RuntimeError(f'joint tracking error {joint_error:.3f} deg')
        return {'max_final_joint_error_deg':joint_error,'simulated_time_s':float(sim.data.time)}
    def finalize(self):
        # The worker serializes concurrent result/video requests. Reuse the
        # saved outcome after the renderer closes, without advancing physics.
        if self.finalized:
            return json.loads((self.output/'physical_result.json').read_text())
        sim=self.sim;sim.step(500);self._capture()
        xyz=sim.data.body('target_cube').xpos.copy();destination=np.array(sim.setup['destination'])
        error=float(np.linalg.norm(xyz-destination))
        contacts=[[sim.model.geom(c.geom1).name,sim.model.geom(c.geom2).name,float(c.dist)] for c in sim.data.contact]
        supported=any(set(c[:2])=={'table','target_geom'} for c in contacts)
        released=not any('target_geom' in c[:2] and any(n and n.endswith('collision') for n in c[:2]) for c in contacts)
        lift_seconds=sim.lift_seconds_by_object['target_cube']
        checks={'destination':error<.025,'supported':supported,'released':released,
            'bilateral_lift':lift_seconds>.5,'protected_stable':sim.max_protected_displacement<.015,
            'no_protected_robot_contact':not sim.protected_contacts,'no_arm_table_contact':not sim.forbidden_contacts,
            'no_self_contact':not sim.self_contacts,'target_penetration':sim.max_target_penetration<.003}
        result={'evidence_kind':'mujoco_protocol_gateway','physical_robot_contacted':False,
            'simulation_success':all(checks.values()),'checks':checks,'destination_error_m':error,
            'bilateral_lift_seconds':lift_seconds,'max_protected_displacement_m':sim.max_protected_displacement,
            'object_attachment_used':False,'object_pose_reset_after_initialization':False}
        sim.save(result);atomic_json(self.output/'physical_result.json',result);self.finalized=True;return result


def make_gateway(checkout,worker,origin):
    if urlsplit(origin).hostname!='127.0.0.1':raise ValueError('gateway is loopback only')
    _,session=import_official_runner(checkout)
    class PhysicsAPI(session.MockSessionAPI):
        def __init__(self):
            super().__init__(auto_activate=False);self.capabilities={};self.trace_events=[]
        def observation(self,robot='yam-local-1'):
            if not worker.finalized and not any(r['active_trajectory'] for r in self._sessions.values()):worker.rpc(worker._capture)
            c=worker.snapshot();active=any(r['active_trajectory'] for r in self._sessions.values())
            return {'schema_version':1,'jetson_id':robot,'source':'simulation','mode':'API_ACTIVE',
                'homed':True,'settled':not active,'queue_ready':True,'observed_at':c['captured_at'],
                'left_joints_deg':[0.]*6,'right_joints_deg':c['joints'],'left_gripper':1.,'right_gripper':c['gripper'],
                'safety':{'ok':True,'estop_engaged':False},'images':{'top':{'url':'/v1/robots/yam-local-1/cameras/top.jpg'}}}
        def create_session(self,prompt,**kwargs):
            if self._sessions:raise session.SessionAPIError('One local episode per gateway; start a fresh gateway for a new trial')
            observation=self.observation();self._observations=[observation]
            result=super().create_session(prompt,**kwargs);sid=result['session_id']
            self.capabilities[sid]=result['session_capability'];self.activate(sid)
            return result
        def get_session(self,sid):
            value=super().get_session(sid);r=self._record(sid)
            value.update(lease_id=r['lease_id'],latest_observation_step=r['next_command_step'])
            return value
        def get_queue_snapshot(self):
            value=super().get_queue_snapshot()
            for station in value.get('stations',[]):station['jetson_id']='yam-local-1'
            return value
        def submit_trajectory(self,sid,eid,lid,tid,cadence,waypoints):
            normalized=session._trajectory_waypoints(waypoints)
            if cadence!=10.:raise session.SessionAPIError('10-Hz simulation only')
            if any(p['left_joints_deg']!=[0.]*6 or p.get('left_gripper',1.)!=1. for p in normalized):
                raise session.SessionAPIError('The virtual other arm must remain held')
            with self._lock:
                r=self._record(sid)
                if r['status']!='running' or (eid,lid)!=(r['episode_id'],r['lease_id']):raise session.SessionAPIError('invalid lease')
                request={'episode_id':eid,'lease_id':lid,'cadence_hz':cadence,'waypoints':normalized}
                if tid in r['trajectory_receipts']:
                    original,receipt=r['trajectory_receipts'][tid]
                    if request!=original:raise session.SessionAPIError('trajectory idempotency body changed')
                    return {**receipt,'duplicate':True}
                if r['active_trajectory'] or normalized[0]['step_id']!=r['next_command_step']:
                    raise session.SessionAPIError('trajectory active or steps not contiguous')
                r['next_command_step']=normalized[-1]['step_id']+1
                r['active_trajectory']={'trajectory_id':tid,'state':'accepted'}
                packet={'trajectory_id':tid,'cadence_hz':cadence,'waypoints':normalized}
                self.trajectory_log.append(packet)
                ack={'accepted':True,'trajectory_id':tid,'waypoint_count':len(normalized)}
                r['trajectory_receipts'][tid]=(request,ack)
                self.record_result(r,{'trajectory_id':tid,'status':'accepted'})
            def progress(step):
                with self._lock:self._event(r,'trajectory_progress',{'trajectory_id':tid,'step_id':step,'executed_at':time.time()})
            future=worker.submit(lambda:worker.execute(normalized,progress))
            def done(future):
                try:details=future.result();status='completed'
                except Exception as error:details={'error':str(error)};status='aborted'
                with self._lock:
                    r['active_trajectory']=None
                    c=worker.snapshot()
                    self.record_result(r,{'trajectory_id':tid,'status':status,'details':details})
                    self._event(r,'observation',{'episode_id':eid,'lease_id':lid,'step_id':r['next_command_step'],
                        'source':'simulation','observed_at':c['captured_at'],'left_joints_deg':[0.]*6,'right_joints_deg':c['joints'],
                        'left_gripper':1.,'right_gripper':c['gripper'],'homed':True,'settled':True})
            future.add_done_callback(done)
            return ack
        def record_result(self,record,payload):
            payload={**payload,'reported_at':time.time()};self.trace_events.append({'type':'trajectory_result','payload':payload})
            self._event(record,'trajectory_result',payload)
        def get_episode_trace(self,sid):
            self._record(sid);return {'schema_version':1,'events':list(self.trace_events)}
        def stop_session(self,sid,*,reason='user_requested'):
            if reason!='policy_complete':worker.cancel.set()
            return super().stop_session(sid,reason=reason)
    api=PhysicsAPI();request_log=[]
    async def app(scope,receive,send):
        path=scope.get('path','');method=scope.get('method');headers=dict(scope.get('headers',[]))
        if scope['type']=='lifespan':
            while True:
                event=await receive()
                await send({'type':event['type']+'.complete'})
                if event['type']=='lifespan.shutdown':return
        match=re.match(r'/v1/sessions/([^/]+)',path);sid=match[1] if match else None
        if sid and (sid not in api.capabilities or headers.get(b'authorization',b'').decode()!='Bearer '+api.capabilities[sid]):
            if scope['type']=='websocket':await send({'type':'websocket.close','code':4401});return
            status=401;value={'error':'unauthorized'}
        elif scope['type']=='websocket':
            await receive();await send({'type':'websocket.accept'});events=api.open_events(sid)
            disconnected=asyncio.create_task(receive())
            try:
                while not disconnected.done():
                    try:event=await asyncio.to_thread(events.receive,.2)
                    except TimeoutError:continue
                    await send({'type':'websocket.send','text':json.dumps(event)})
            except (ConnectionError,RuntimeError):pass
            finally:disconnected.cancel();events.close()
            return
        else:
            body=b''
            while True:
                part=await receive();body+=part.get('body',b'')
                if not part.get('more_body'):break
            payload=json.loads(body) if body else {}
            request_log.append({'method':method,'path':path,'authorized':bool(sid),'time':time.time()})
            try:
                status=200
                if path=='/v1/robots':value={'robots':[{'robot_id':'yam-local-1','source':'simulation',
                    'calibration_url':'/v1/robots/yam-local-1/calibration'}]}
                elif path.endswith('/calibration'):value=worker.calibration
                elif path.endswith('/observation'):value=await asyncio.to_thread(api.observation)
                elif path.endswith('/top.jpg'):
                    cache=worker.snapshot();raw=cache['top']
                    await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'image/jpeg'),
                        (b'content-length',str(len(raw)).encode()),(b'x-captured-at',str(cache['captured_at']).encode())]})
                    await send({'type':'http.response.body','body':raw});return
                elif path.endswith('/queue'):value=api.get_queue_snapshot()
                elif path=='/v1/sessions' and method=='POST':
                    value=await asyncio.to_thread(api.create_session,payload['prompt'],run_duration_s=payload.get('run_duration_s',300))
                elif path.endswith('/trajectories') and method=='POST':
                    value=api.submit_trajectory(sid,payload['episode_id'],payload['lease_id'],payload['trajectory_id'],payload['cadence_hz'],payload['waypoints']);status=202
                elif path.endswith('/episode/trace'):value=api.get_episode_trace(sid)
                elif path.endswith('/stop') and method=='POST':value=api.stop_session(sid,reason=payload.get('reason','user_requested'))
                elif sid:value=api.get_session(sid)
                else:status=404;value={'error':'not_found'}
            except Exception as error:status=409;value={'error':{'code':'local_gateway_error','message':str(error),'status':409}}
        raw=json.dumps(value).encode()
        await send({'type':'http.response.start','status':status,'headers':[(b'content-type',b'application/json'),(b'content-length',str(len(raw)).encode())]})
        await send({'type':'http.response.body','body':raw})
    return app,api,request_log


def recommended_provider(config,worker,origin):
    from ..platform.runner_provider import load_provider
    from ..platform.config import PlatformConfig
    from ..runtime import TaskSpec,Subgoal
    provider=load_provider(config);provider.output=worker.output/'agent_runs'
    provider.profile=worker.profile;provider.platform_config=PlatformConfig('blupe','LOCAL_UNUSED',origin,'yam-local-1')
    provider.registry={k:dict(provider.registry[k]) for k in ('cube','vase')}
    provider.registry['cube']['visual_description']='red rigid cube or block'
    provider.registry['vase']['visual_description']='green upright cylindrical vessel'
    provider.candidate_provider=color_region_candidates
    provider.task=TaskSpec('recommended_protected_neighbor',
        'Move the red rigid cube to the blue destination area without touching or moving the green vessel.',
        (Subgoal('cube','tray'),),('vase',))
    provider.destinations={'tray':[.26,.12,.036]}
    return provider

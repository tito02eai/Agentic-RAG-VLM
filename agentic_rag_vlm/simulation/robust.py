"""Evidence-gated controller improvements; separate from the frozen r4 profile."""
import xml.etree.ElementTree as ET
import math
import numpy as np

from .extended import ExtendedPhysics, RetentionTransport, HALVES, extended_scene
from .yam import _rotation_log


def feedback_force_recovery(strategy, failure, attempt, *, rng=None):
    """An explicit extension: bounded supported parameter adaptation, not fake L2 skills."""
    from ..recovery import FailureType, RecoveryDecision, recover
    if not 1 <= attempt <= 3:
        raise ValueError('contact recovery is bounded to three retries')
    if failure not in {FailureType.SLIP, FailureType.RETENTION_MARGIN}:
        return recover(strategy, failure, attempt, rng=rng)
    signature = strategy.get('_semantic_signature', [])
    fragility = float(signature[3]) if len(signature) > 3 else 1.
    current = float(strategy['force_n']); corrected = dict(strategy)
    if not math.isfinite(fragility) or not math.isfinite(current) or current <= 0 or fragility > .25 or current*1.2 > 1.6:
        return RecoveryDecision(attempt, failure, corrected, 'unsupported_force_increase', True)
    corrected['force_n'] = current*1.2
    return RecoveryDecision(attempt, failure, corrected, 'bounded_force_escalation')


def varied_scene(path, *, variation='independent', mixed_shapes=False, upper_half_width=None, **kwargs):
    """Private sampled physics never enters public object metadata or recovery."""
    setup = extended_scene(path, **kwargs)
    seed = kwargs.get('seed', 0)
    rng = np.random.default_rng(seed+17000)
    tree = ET.parse(path); root = tree.getroot()
    many = setup['multiple']; stacked = setup['mode']=='stacked_sort'
    if upper_half_width is not None:
        if not many or not stacked or not math.isfinite(upper_half_width) or not .012<=upper_half_width<=.017:
            raise ValueError('occlusion width requires a stacked scene and half-width in [12,17] mm')
        # Change only the visible upper footprint; height, mass, friction and
        # support placement remain independently controlled and paired.
        root.find('.//geom[@name="target_geom2"]').set('size',f'{upper_half_width} {upper_half_width} .014')
    first = np.array([.25, -.055] if stacked else [.27, -.15]) + rng.uniform(-.015, .015, 2)
    second = first + rng.uniform(-.002, .002, 2) if stacked else np.array([.20, -.04]) + rng.uniform(-.015, .015, 2)
    third = np.array([.306, -.16 if stacked else -.035]) + rng.uniform(-.012, .012, 2)
    positions = [('target_cube', first, .019)]
    if many:
        positions += [('target_cube2', second, .051 if stacked else .015), ('target_cube3', third, .019)]
    for body, xy, height in positions:
        root.find(f'.//body[@name="{body}"]').set('pos', f'{xy[0]} {xy[1]} {height}')
    if many:
        protected = np.array([.43, -.1]) + rng.uniform(-.008, .008, 2)
        root.find('.//body[@name="protected_vase"]').set('pos', f'{protected[0]} {protected[1]} .04')
    for index, (name, xyz) in enumerate(setup['destinations'].items()):
        xyz[:2] = (np.asarray(xyz[:2]) + rng.uniform(-.012, .012, 2)).tolist()
        root.find(f'.//geom[@name="destination_{index}"]').set('pos', f'{xyz[0]} {xyz[1]} .001')
    contact = {}
    for body, geom_name in setup['target_geoms'].items():
        mass = float(rng.uniform(.055, .075)) if variation=='mixed_contact' else .06
        friction = float(rng.uniform(.6, 1.0)) if variation=='mixed_contact' else kwargs.get('friction', .7)
        geom = root.find(f'.//geom[@name="{geom_name}"]')
        geom.set('mass', str(mass)); geom.set('friction', f'{friction} .01 .001')
        contact[body] = {'mass_kg': mass, 'friction': friction}
    if many and mixed_shapes:
        geom=root.find('.//geom[@name="target_geom3"]')
        geom.set('type','cylinder');geom.set('size','.018 .018')
    tree.write(path, encoding='unicode')
    setup.update(scene_variation=variation, private_sampled_contact=contact,mixed_shapes=mixed_shapes)
    if upper_half_width is not None:setup['upper_half_width_m']=upper_half_width
    return setup


class RobustPhysics(ExtendedPhysics):
    """Keep existing IK tolerances; use a bound-aware solver on DLS stagnation."""
    def step(self,count):
        from .yam import PhysicsSafetyError
        for _ in range(count):
            super().step(1)
            if self.private_max_table_penetration>.003:
                raise PhysicsSafetyError('target/table penetration exceeded 3 mm; stopping at first observed violation')
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise PhysicsSafetyError('non-finite simulator state')

    def solve(self, position, *, rotation=None, seed=None):
        try:
            return super().solve(position, rotation=rotation, seed=seed)
        except RuntimeError as error:
            if not str(error).startswith('IK failed'):
                raise
            initial_error = str(error)
        import mujoco
        from scipy.optimize import least_squares
        rotation = np.diag([-1., 1., -1.]) if rotation is None else np.asarray(rotation)
        initial = self.data.qpos[self.qadr].copy() if seed is None else np.asarray(seed).copy()
        limits = self.model.jnt_range[self.joints]
        lower, upper = limits[:, 0] + .001, limits[:, 1] - .001

        def residual(q):
            self.ik_data.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, self.ik_data)
            site = self.ik_data.site(self.site)
            return np.r_[np.asarray(position) - site.xpos,
                         .2 * _rotation_log(rotation @ site.xmat.reshape(3, 3).T)]

        best = None
        for start in (initial, np.array([0, 1.2, 1.6, -.6, 0, 0]), limits.mean(axis=1)):
            solution = least_squares(residual, np.clip(start, lower, upper), bounds=(lower, upper),
                                     max_nfev=200, xtol=1e-11, ftol=1e-11, gtol=1e-11)
            value = residual(solution.x)
            ep, er = float(np.linalg.norm(value[:3])), float(np.linalg.norm(value[3:]) / .2)
            if best is None or ep + .1*er < best[0]:
                best = ep + .1*er, ep, er
            if ep < .0005 and er < .015:
                self.public_trace.append({'event': 'ik_bound_refinement', 'time': float(self.data.time),
                    'target_xyz': list(position), 'position_error_m': ep, 'rotation_error_rad': er,
                    'evaluations': solution.nfev, 'joint_limits_preserved': True})
                return solution.x
        raise RuntimeError(f'{initial_error}; bounded refinement pos={best[1]:.5f} ori={best[2]:.5f}')


class RobustRetentionTransport(RetentionTransport):
    """A partially unloaded grasp needs visual support and a guarded unload probe."""
    def _visible_support(self, target, destination):
        """Use rendered RGB-D only; no body poses, geom contacts or semantic IDs."""
        height = 2 * HALVES[target]
        from PIL import Image
        self._support_probe_id = getattr(self, '_support_probe_id', 0) + 1
        for camera_name in ('top', 'overview'):
            rgb, depth = self.sim.rgbd(camera_name)
            r, g, b = rgb.astype(float).transpose(2, 0, 1)
            masks = [(r > 40) & (r > g*1.8) & (r > b*1.8),
                     (g > 40) & (g > r*1.7) & (g > b*1.4),
                     (r > 50) & (g > 45) & (b < np.minimum(r, g)*.55),
                     (r > 40) & (b > 40) & (g < np.minimum(r, b)*.5)]
            camera = self.sim.data.cam(camera_name)
            fovy = float(self.sim.model.cam_fovy[camera.id])
            f = rgb.shape[0] / (2*np.tan(np.deg2rad(fovy)/2))
            candidates = []
            for mask in masks:
                rows, cols = np.where(mask & np.isfinite(depth) & (depth > .05) & (depth < 2))
                if len(rows) < 8:
                    continue
                z = depth[rows, cols]
                points = np.column_stack(((cols+.5-rgb.shape[1]/2)*z/f,
                                          -(rows+.5-rgb.shape[0]/2)*z/f, -z))
                points = points @ camera.xmat.reshape(3, 3).T + camera.xpos
                near = points[np.linalg.norm(points[:, :2] - destination[:2], axis=1) < .035]
                if len(near) >= 8:
                    candidates.append({'point_xyz_m': np.median(near, axis=0).tolist(),
                        'top_z_m': float(np.quantile(near[:, 2], .95)),
                        'bottom_z_m': float(np.quantile(near[:, 2], .05))})
            # An overhead occlusion may hide the upper face. The oblique
            # view can instead see the lower silhouette at the measured
            # support plane; never substitute an assumed top coordinate.
            nearby = [c for c in candidates
                if np.all(np.abs(np.asarray(c['point_xyz_m'][:2])-destination[:2]) < HALVES[target]+.005)
                and (abs(c['top_z_m']-height) < .004 or (
                    abs(c['bottom_z_m']) < .0035 and c['top_z_m']-c['bottom_z_m'] > HALVES[target]))]
            observation = np.asarray(nearby[0]['point_xyz_m']) if len(nearby) == 1 else None
            folder = self.sim.output / f'support_probe_{self._support_probe_id:03d}_{camera_name}'
            folder.mkdir()
            Image.fromarray(rgb).save(folder/'rgb.png'); np.save(folder/'depth_m.npy', depth)
            self.sim.public_trace.append({'event': 'placement_visual_support', 'time': float(self.sim.data.time),
                'target': target, 'destination_xyz_m': destination.tolist(), 'camera': camera_name,
                'rgb_path': str(folder/'rgb.png'), 'depth_path': str(folder/'depth_m.npy'),
                'camera_position': camera.xpos.tolist(), 'camera_rotation': camera.xmat.reshape(3, 3).tolist(),
                'camera_fovy_deg': fovy, 'visible_geometry_candidates': candidates,
                'matched_surface_xyz_m': observation.tolist() if observation is not None else None,
                'source': 'public_rgbd_destination_roi_and_known_height'})
            if observation is not None:
                return observation
        return None

    def _guarded_unload(self, target, destination, baseline):
        first = self._visible_support(target, destination)
        self.sim.step(25)
        second = self._visible_support(target, destination)
        if first is None or second is None or np.linalg.norm(first-second) > .004:
            return False
        # The object already visibly rests at the support plane and part of
        # its load has transferred. A small opening removes finger preload;
        # full release remains forbidden until load and visual evidence agree.
        for step in range(3):
            before = self.sim.tactile()
            opening = min(.0375, before['left_finger_q_m'] + .0005)
            self.sim.data.ctrl[6] = opening
            self.sim.stage = 'placement_unload_probe'
            self.sim.step(250)
            after = self.sim.tactile()
            visible = self._visible_support(target, destination)
            stable = bool(visible is not None and np.linalg.norm(visible-second) < .004)
            self.sim.public_trace.append({'event': 'placement_unload_probe', 'time': float(self.sim.data.time),
                'target': target, 'step': step+1, 'opening_increment_m': .0005,
                'normal_load_before_n': before['downward_load_n'],
                'normal_load_after_n': after['downward_load_n'], 'visible_support_stable': stable})
            if not stable:
                return False
            if abs(after['downward_load_n']) < .15 * baseline:
                return True
        return False

    def place(self, command):
        destination = np.asarray(command.parameters['destination_xyz_m'])
        goal = destination + self.grasp_offset
        over = goal.copy(); over[2] = .15
        self.sim.move(over, grip=0, duration=4, stage='transport')
        self.sim.step(150)
        baseline = self.sim.tactile()['downward_load_n']
        if baseline < .03:
            raise RuntimeError('object load absent before placement')
        supported = False
        heights = [max(float(h), goal[2]) for h in np.arange(.144, goal[2]-.006, -.006)]
        # Known collision-pad clearance is 2.2 mm at the nominal goal. A
        # bounded 1 mm fine approach resolves sub-mm suspension from grasp
        # offsets/curved contact; it does not allow arbitrary deeper descent.
        heights += [goal[2]-delta for delta in (.00025, .0005, .00075, .001)]
        for height in heights:
            point = goal.copy(); point[2] = height
            self.sim.move(point, grip=0, duration=.2, stage='place_compliant')
            loads = []
            for _ in range(3):
                self.sim.step(20); loads.append(self.sim.tactile()['downward_load_n'])
            if min(loads) < baseline*.75:
                for _ in range(10):
                    if all(load < baseline*.5 for load in loads[-3:]):
                        break
                    self.sim.step(20); loads.append(self.sim.tactile()['downward_load_n'])
            self.sim.public_trace.append({'event': 'placement_load', 'target': command.target,
                'time': float(self.sim.data.time), 'tcp_z_m': point[2],
                'reference_load_n': baseline, 'observed_loads_n': loads})
            if all(load < baseline*.5 for load in loads[-3:]):
                supported = True
                break
            if (all(0 < load < baseline*.75 for load in loads[-3:])
                    and np.std(loads[-3:]) < baseline*.05
                    and self._guarded_unload(command.target, destination, baseline)):
                supported = True
                break
        if not supported:
            raise RuntimeError('placement support not sensed; refusing blind release')
        self.sim.stage = 'release'; self.sim.data.ctrl[6] = .0375; self.sim.step(600)
        self.sim.move(over, grip=.0375, duration=2, stage='retract')
        self.sim.move([.2, -.2, .15], grip=.0375, duration=2, stage='park')
        self.sim.stage = 'settle_final'; self.sim.step(750)
        self.sim.evaluator_end_attempt()
        return {'accepted': True, 'execution_started': True, 'command_id': command.command_id}

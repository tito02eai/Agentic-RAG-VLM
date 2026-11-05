"""Opt-in low-height proof lift; r5 controller remains available unchanged."""
import numpy as np
import xml.etree.ElementTree as ET

from .robust import RobustRetentionTransport


def multicontact_scene(path, *, builder=None, **kwargs):
    """Use MuJoCo multi-contact convex collision without changing materials."""
    from .extended import extended_scene
    setup = (builder or extended_scene)(path, **kwargs)
    tree = ET.parse(path)
    option = tree.getroot().find('option')
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('multiccd', 'enable')
    world = tree.getroot().find('worldbody')
    ET.SubElement(world, 'camera', name='inspection', pos='.65 .65 .45',
                  xyaxes='-.879 .477 0 -.214 -.395 .894', fovy='48')
    tree.write(path, encoding='unicode')
    setup['multiccd'] = True
    return setup


class GripLost(RuntimeError):
    pass


class GuardedRetentionTransport(RobustRetentionTransport):
    def send_command(self, command):
        if command.command_type == 'pick':
            self._active_pick = command
            self._set_visual_reference(command)
        return super().send_command(command)

    def _set_visual_reference(self, command):
        """Bind appearance at the already-grounded metric target, before motion."""
        from PIL import Image
        from .extended import HALVES
        rgb, depth = self.sim.rgbd('top')
        expected = np.asarray(command.parameters['position_xyz_m']).copy()
        expected[2] += HALVES[command.target]
        proposals = self.sim.proposals(rgb, depth)
        compatible = [p for p in proposals
            if np.linalg.norm(np.asarray(p['top_xyz_m'])[:2]-expected[:2]) < .02
            and abs(p['top_xyz_m'][2]-expected[2]) < .008]
        self._reference_chromaticity = None
        if len(compatible) != 1:
            return
        proposal = compatible[0]
        x0, y0, x1, y1 = proposal['bbox_xyxy']
        # Top camera is calibrated vertical in this testbed. Select only the
        # observed surface's depth band, excluding an overlapping supporter.
        top_z = float(self.sim.data.cam('top').xpos[2])-depth[y0:y1, x0:x1]
        patch = rgb[y0:y1, x0:x1].astype(float)
        mask = (np.abs(top_z-proposal['top_xyz_m'][2]) < .004) & (patch.max(axis=2)-patch.min(axis=2) > 35)
        colours = patch[mask]
        if len(colours) < 8:
            return
        colours = colours-colours.min(axis=1, keepdims=True)
        reference = np.median(colours/np.maximum(colours.sum(axis=1, keepdims=True), 1), axis=0)
        reference /= reference.sum()
        self._reference_chromaticity = reference
        folder = self.sim.output/f'proof_reference_{len(self.attempts)+1:03d}'
        folder.mkdir()
        Image.fromarray(rgb).save(folder/'rgb.png'); np.save(folder/'depth_m.npy', depth)
        self.sim.public_trace.append({'event': 'proof_visual_reference', 'time': float(self.sim.data.time),
            'target': command.target, 'observed_top_xyz_m': proposal['top_xyz_m'],
            'bbox_xyxy': proposal['bbox_xyxy'], 'chromaticity': reference.tolist(),
            'rgb_path': str(folder/'rgb.png'), 'depth_path': str(folder/'depth_m.npy'),
            'source': 'appearance_at_already_grounded_target_no_new_semantic_identity'})

    def _proof_surface(self, grasp):
        """Track visible coloured surface over source ROI, without class IDs."""
        from PIL import Image
        target_xy = grasp[:2] - [.014, 0]
        self._proof_capture_id = getattr(self, '_proof_capture_id', 0)+1
        captures = []
        for name in ('top', 'overview', 'inspection'):
            if name == 'inspection':
                try:
                    self.sim.model.camera(name)
                except KeyError:
                    continue
            rgb, depth = self.sim.rgbd(name)
            values = rgb.astype(float)
            opponent = values-values.min(axis=2, keepdims=True)
            chromaticity = opponent/np.maximum(opponent.sum(axis=2, keepdims=True), 1)
            reference = self._reference_chromaticity
            foreground = (np.linalg.norm(chromaticity-reference, axis=2) < .10
                if reference is not None else np.zeros(depth.shape, dtype=bool))
            foreground &= values.max(axis=2)-values.min(axis=2) > 35
            rows, cols = np.where(foreground & np.isfinite(depth) & (depth > .05) & (depth < 2))
            camera = self.sim.data.cam(name)
            fovy = float(self.sim.model.cam_fovy[camera.id])
            f = depth.shape[0]/(2*np.tan(np.deg2rad(fovy)/2))
            z = depth[rows, cols]
            points = np.column_stack(((cols+.5-320)*z/f, -(rows+.5-240)*z/f, -z))
            points = points @ camera.xmat.reshape(3, 3).T + camera.xpos
            local = points[np.linalg.norm(points[:, :2]-target_xy, axis=1) < .025]
            height = float(np.quantile(local[:, 2], .95)) if len(local) >= 8 else None
            folder = self.sim.output/f'proof_surface_{self._proof_capture_id:03d}_{name}'
            folder.mkdir()
            Image.fromarray(rgb).save(folder/'rgb.png'); np.save(folder/'depth_m.npy', depth)
            captures.append({'camera': name, 'height_m': height, 'visible_pixels': len(local),
                'camera_position': camera.xpos.tolist(), 'camera_rotation': camera.xmat.reshape(3, 3).tolist(),
                'camera_fovy_deg': fovy, 'rgb_path': str(folder/'rgb.png'), 'depth_path': str(folder/'depth_m.npy')})
        self.sim.public_trace.append({'event': 'proof_surface', 'time': float(self.sim.data.time),
            'source_xy_m': target_xy.tolist(), 'captures': captures, 'source': 'public_rgbd_source_roi_no_class_override'})
        return {c['camera']: c['height_m'] for c in captures if c['height_m'] is not None}

    def get_observation(self):
        observation = super().get_observation()
        if self.use_vlm:
            camera = self.sim.data.cam('top')
            observation.receipt['camera_calibration'] = {
                'position': camera.xpos.tolist(), 'rotation': camera.xmat.reshape(3, 3).tolist(),
                'fovy_deg': float(self.sim.model.cam_fovy[camera.id])}
        return observation

    def _lift_and_test(self, grasp, lifted, before):
        samples = []
        monitor_samples = []
        lost_since = None
        initial_aperture = before['left_finger_q_m']
        last_sample = -1.

        def monitor():
            nonlocal lost_since, last_sample
            time = float(self.sim.data.time)
            tactile = self.sim.tactile()
            lost = ((not tactile['bilateral'] and tactile['downward_load_n'] < .02)
                    or tactile['left_finger_q_m'] < .004
                    or tactile['left_finger_q_m'] < initial_aperture-.0012)
            lost_since = (time if lost_since is None else lost_since) if lost else None
            if time-last_sample >= .0099:
                monitor_samples.append({'time': time, 'stage': self.sim.stage, **tactile})
                last_sample = time
            if lost_since is not None and time-lost_since >= .012-1e-8:
                # Hold measured arm position immediately. No object pose reset.
                self.sim.data.ctrl[:6] = self.sim.data.qpos[self.sim.qadr]
                raise GripLost('bilateral contact/aperture lost for 12 ms')

        proof = grasp.copy(); proof[2] += .008
        proof_passed = False
        proof_samples = []
        visual_proof = {}
        force_margin = {}
        failure_type = 'slip' if before['bilateral'] else 'position_error'
        returned_to_support = False
        force_adjustments = []
        initial_force = float(self.sim.model.actuator_forcerange[6, 1])
        abort = None
        try:
            if not before['bilateral']:
                raise GripLost('no bilateral contact before proof lift')
            surface_before = self._proof_surface(grasp)
            # Short Cartesian increments bound the excursion before proving
            # load retention. The final 15 cm lift is forbidden before proof.
            for dz in (.002, .004, .006, .008):
                point = grasp.copy(); point[2] += dz
                self.sim.move(point, grip=0, duration=.25, stage='proof_lift', step_monitor=monitor)
            for _ in range(10):
                for _ in range(10):
                    self.sim.step(1); monitor()
                proof_samples.append({'time': float(self.sim.data.time), **self.sim.tactile()})
            surface_first = self._proof_surface(grasp)
            for _ in range(250):
                self.sim.step(1); monitor()
            surface_second = self._proof_surface(grasp)
            matching = sorted(set(surface_before) & set(surface_first) & set(surface_second))
            visual_proof = {'before': surface_before, 'first': surface_first, 'second': surface_second,
                'required_rise_m': .006, 'maximum_drift_m': .0008,
                'valid_cameras': [c for c in matching
                    if min(surface_first[c], surface_second[c])-surface_before[c] >= .006
                    and abs(surface_first[c]-surface_second[c]) < .0008]}
            aperture = [s['left_finger_q_m'] for s in proof_samples]
            # Public pad calibration supplies a conservative friction floor
            # (0.7 in this testbed), not the sampled object's coefficient.
            # 1.7 * payload asks for reserve beyond static weight/0.7.
            minimum_normal = min(sum(s['normal_force_n'].values()) for s in proof_samples)
            payload = max(s['downward_load_n'] for s in proof_samples)
            force_margin = {'minimum_normal_force_n': minimum_normal, 'maximum_payload_load_n': payload,
                'required_normal_to_load_ratio': 1.7, 'calibrated_friction_floor': .7,
                'passed': bool(payload > .03 and minimum_normal >= 1.7*payload)}
            policy = getattr(self, '_active_pick', None)
            policy = policy.parameters.get('retention_force_policy', {}) if policy else {}
            if (all(s['bilateral'] and s['downward_load_n'] > .03 for s in proof_samples)
                    and policy.get('maximum_adjustments') == 3 and policy.get('scale') == 1.2
                    and policy.get('ceiling_n') == 1.6
                    and np.isfinite(policy.get('observed_fragility', float('nan')))
                    and policy.get('observed_fragility', 1.) <= .25):
                # Bounded reclosure while bilateral tactile load is present;
                # visual rise must still be proved after the adjustment. Never infer
                # private mass/friction; all feedback is measured pad/load.
                for _ in range(3):
                    if force_margin['passed']:
                        break
                    current = float(self.sim.model.actuator_forcerange[6, 1])
                    corrected = current*1.2
                    if corrected > 1.6:
                        break
                    self.sim.model.actuator_forcerange[6] = [-corrected, corrected]
                    correction = {'time': float(self.sim.data.time), 'before_n': current,
                        'after_n': corrected, 'margin_before': dict(force_margin),
                        'source': 'public_tactile_preventive_adjustment', 'adjustment': len(force_adjustments)+1}
                    force_adjustments.append(correction)
                    self.sim.public_trace.append({'event': 'retention_force_adjustment', **correction})
                    for _ in range(100):
                        self.sim.step(1); monitor()
                    proof_samples = []
                    for _ in range(10):
                        for _ in range(10):
                            self.sim.step(1); monitor()
                        proof_samples.append({'time': float(self.sim.data.time), **self.sim.tactile()})
                    minimum_normal = min(sum(s['normal_force_n'].values()) for s in proof_samples)
                    payload = max(s['downward_load_n'] for s in proof_samples)
                    force_margin = {'minimum_normal_force_n': minimum_normal, 'maximum_payload_load_n': payload,
                        'required_normal_to_load_ratio': 1.7, 'calibrated_friction_floor': .7,
                        'passed': bool(payload > .03 and minimum_normal >= 1.7*payload)}
                    surface_new = self._proof_surface(grasp)
                    visual_proof['second'] = surface_new
                    visual_proof['valid_cameras'] = [c for c in visual_proof['valid_cameras']
                        if c in surface_new and abs(surface_new[c]-surface_first[c]) < .0008]
            if force_margin['passed'] and force_adjustments and not visual_proof['valid_cameras']:
                # The first low probe may slide along an object still on its
                # support. After bounded reclosure, demand a NEW 8-mm visual
                # proof; do not reinterpret that failed probe as successful.
                previous_probe = dict(visual_proof)
                surface_before = self._proof_surface(grasp)
                for dz in (.010, .012, .014, .016):
                    point = grasp.copy(); point[2] += dz
                    self.sim.move(point, grip=0, duration=.25, stage='proof_lift_recheck', step_monitor=monitor)
                proof[2] = grasp[2]+.016
                surface_first = self._proof_surface(grasp)
                for _ in range(250):
                    self.sim.step(1); monitor()
                surface_second = self._proof_surface(grasp)
                matching = sorted(set(surface_before) & set(surface_first) & set(surface_second))
                visual_proof = {'before': surface_before, 'first': surface_first, 'second': surface_second,
                    'required_rise_m': .006, 'maximum_drift_m': .0008, 'previous_probe': previous_probe,
                    'valid_cameras': [c for c in matching
                        if min(surface_first[c], surface_second[c])-surface_before[c] >= .006
                        and abs(surface_first[c]-surface_second[c]) < .0008]}
            if not force_margin['passed']:
                failure_type = 'retention_margin'
            aperture = [s['left_finger_q_m'] for s in proof_samples]
            proof_passed = bool(all(s['bilateral'] and s['downward_load_n'] > .03 for s in proof_samples)
                                and max(aperture)-min(aperture) < .0006
                                and visual_proof['valid_cameras'] and force_margin['passed'])
            if not proof_passed:
                raise GripLost('proof lift did not establish stable positive payload load')
            # Joint-space endpoint interpolation can sweep the TCP sideways
            # even for a vertical request. Small Cartesian waypoints keep the
            # held object over its observed source and bound acceleration.
            heights = np.linspace(proof[2], lifted[2], max(2, int(np.ceil((lifted[2]-proof[2])/.01))+1))[1:]
            for height in heights:
                point = lifted.copy(); point[2] = height
                self.sim.move(point, grip=0, duration=.4, stage='lift', step_monitor=monitor)
        except GripLost as error:
            abort = str(error)
            self.sim.public_trace.append({'event': 'lift_aborted', 'time': float(self.sim.data.time),
                'stage': self.sim.stage, 'reason': abort, 'proof_passed': proof_passed,
                'source': 'public_tactile_2ms_monitor'})
            # A preventive margin failure is distinct from actual slip. If
            # vision proved the object followed the 8-mm lift and it remains
            # bilaterally held, reverse that short, monitored path before
            # releasing (especially important above a stacked supporter).
            if (failure_type == 'retention_margin' and visual_proof.get('valid_cameras')
                    and self.sim.tactile()['bilateral']):
                try:
                    for dz in (.006, .004, .002, 0):
                        point = grasp.copy(); point[2] += dz
                        self.sim.move(point, grip=0, duration=.25, stage='proof_return_to_support', step_monitor=monitor)
                    returned_to_support = True
                except GripLost:
                    pass
            # Actual loss never triggers a blind descent. Hold the arm;
            # the caller releases, retreats, and takes new observations.
            self.sim.data.ctrl[:6] = self.sim.data.qpos[self.sim.qadr]
        for _ in range(10):
            self.sim.step(20)
            samples.append({'time': float(self.sim.data.time), **self.sim.tactile()})
        retained = bool(abort is None and proof_passed
                        and all(s['bilateral'] and s['left_finger_q_m'] > .004 for s in samples))
        evidence = {'retention_contract': 'low_height_proof_v1', 'proof_lift_m': float(proof[2]-grasp[2]),
                    'proof_passed': proof_passed, 'lift_abort_reason': abort,
                    'proof_samples': proof_samples,
                    'visual_proof': visual_proof,
                    'force_margin': force_margin, 'failure_type': failure_type,
                    'returned_to_support': returned_to_support,
                    'force_adjustments': force_adjustments,
                    'initial_motor_force_cap_n': initial_force,
                    'motor_force_cap_n': float(self.sim.model.actuator_forcerange[6, 1]),
                    'lift_monitor_samples': monitor_samples,
                    'monitor_period_s': float(self.sim.model.opt.timestep)}
        return retained, samples, evidence

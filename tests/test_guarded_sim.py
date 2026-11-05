"""Independent geometry/safety counterexamples for the opt-in r6 profile."""
import os
from functools import partial

import numpy as np
import pytest

from agentic_rag_vlm.simulation.geometry import cylinder_cap_center


@pytest.mark.parametrize('tilt', [0., .2, .45])
def test_cylinder_metric_center_compensates_tilt_with_side_outliers(tilt):
    rng = np.random.default_rng(91)
    center = np.array([.4, -.1, .043])
    rotation = np.array([[1., 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                         [0, np.sin(tilt), np.cos(tilt)]])
    radius, height = .018, .04
    theta = rng.uniform(0, 2*np.pi, 2000)
    radii = radius*np.sqrt(rng.uniform(0, 1, 2000))
    cap = np.column_stack((radii*np.cos(theta), radii*np.sin(theta), np.full(2000, height)))
    side_theta = rng.uniform(0, 2*np.pi, 500)
    side = np.column_stack((radius*np.cos(side_theta), radius*np.sin(side_theta), rng.uniform(-height, height, 500)))
    points = np.vstack((cap, side)) @ rotation.T + center
    points += rng.normal(0, .00002, points.shape)
    fit = cylinder_cap_center(points, half_height_m=height, radius_m=radius)
    assert fit is not None
    assert np.linalg.norm(np.asarray(fit['center_xyz_m'])-center) < .001
    # The old surface-minus-vertical-height estimator errs by >7 mm at tilt.
    if tilt > .1:
        naive = cap.mean(axis=0) @ rotation.T + center - [0, 0, height]
        assert np.linalg.norm(naive-center) > .007


def test_cylinder_fit_rejects_insufficient_foreground_and_wrong_dimensions():
    rng = np.random.default_rng(5)
    points = rng.uniform(-.002, .002, (100, 3)); points[:, 2] = .08
    assert cylinder_cap_center(points, half_height_m=.04, radius_m=.018) is None
    assert cylinder_cap_center(np.full((100, 3), np.nan), half_height_m=.04, radius_m=.018) is None


def test_multicontact_does_not_change_materials_or_initial_layout(tmp_path):
    import xml.etree.ElementTree as ET
    from agentic_rag_vlm.simulation.robust import varied_scene
    from agentic_rag_vlm.simulation.guarded import multicontact_scene
    base = partial(varied_scene, variation='mixed_contact', mixed_shapes=True)
    first, second = tmp_path/'original.xml', tmp_path/'multiccd.xml'
    original = base(first, seed=713, mode='stacked_sort')
    adjusted = multicontact_scene(second, builder=base, seed=713, mode='stacked_sort')
    a, b = ET.parse(first), ET.parse(second)
    b.getroot().find('worldbody').remove(b.getroot().find('.//camera[@name="inspection"]'))
    assert ET.tostring(a.getroot().find('worldbody')) == ET.tostring(b.getroot().find('worldbody'))
    assert original['private_sampled_contact'] == adjusted['private_sampled_contact']
    assert b.getroot().find('option/flag').get('multiccd') == 'enable'


def test_guarded_lift_requires_proof_and_keeps_physical_weak_grasp_safe(tmp_path):
    pytest.importorskip('mujoco'); os.environ.setdefault('MUJOCO_GL', 'egl')
    from agentic_rag_vlm.simulation.robust import RobustPhysics, varied_scene
    from agentic_rag_vlm.simulation.guarded import GuardedRetentionTransport, multicontact_scene
    from agentic_rag_vlm.platform.commands import PlatformCommand
    builder = partial(multicontact_scene, builder=partial(varied_scene,
                      variation='mixed_contact', mixed_shapes=True))
    sim = RobustPhysics(tmp_path, seed=502, mode='stacked_sort', scene_builder=builder)
    try:
        sim.move([.2, -.2, .15], grip=.0375, stage='initial_view_clearance')
        target = sim.detect()['cube3']
        transport = GuardedRetentionTransport(sim, destinations_by_target=sim.setup['destinations'])
        receipt = transport.send_command(PlatformCommand('weak', 'pick', 'cube3', 'test', {
            'approach': 'top', 'position_xyz_m': target, 'force_n': .1, 'aperture_m': .075}))
        assert receipt['accepted'] is False
        attempt = transport.attempts[0]
        assert not attempt['retained'] and not attempt['proof_passed']
        assert not any(e.get('stage') == 'lift' and e['event'] == 'motion' for e in sim.public_trace)
        assert sim.private_max_table_penetration < .003
        assert sim.max_target_penetration < .003
    finally:
        sim.renderer.close()

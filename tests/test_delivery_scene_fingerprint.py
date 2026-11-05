"""Different checkout paths must not hide physical/asset changes."""
from scripts.finalize_visible_progress import physics_scene_fingerprint


def scene(root,*,mesh=b'same mesh',position='0 0 0'):
    root.mkdir();assets=root/'assets';assets.mkdir();(assets/'body.stl').write_bytes(mesh)
    xml=root/'scene.xml';xml.write_text(f'<mujoco><compiler meshdir="{assets}"/><asset><mesh file="body.stl"/></asset><worldbody><body pos="{position}"/></worldbody></mujoco>')
    return physics_scene_fingerprint(xml)


def test_checkout_paths_normalize_only_with_matching_assets(tmp_path):
    a=scene(tmp_path/'first');b=scene(tmp_path/'second')
    assert a['raw_xml_sha256']!=b['raw_xml_sha256']
    assert a['physics_and_assets_sha256']==b['physics_and_assets_sha256']


def test_asset_changes_remain_detectable(tmp_path):
    a=scene(tmp_path/'first');b=scene(tmp_path/'second',mesh=b'changed mesh')
    assert a['physics_and_assets_sha256']!=b['physics_and_assets_sha256']


def test_geometry_changes_remain_detectable(tmp_path):
    a=scene(tmp_path/'first');b=scene(tmp_path/'second',position='0 0 .1')
    assert a['physics_and_assets_sha256']!=b['physics_and_assets_sha256']

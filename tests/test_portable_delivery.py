import json
from pathlib import Path
import pytest
from scripts.run_platform_session import load_task


@pytest.mark.parametrize('xyz',[None,[0,1],['0',1,2],[True,1,2],[float('nan'),1,2]])
def test_measured_task_rejects_invalid_destinations_before_session(tmp_path,xyz):
    p=tmp_path/'task.json';p.write_text(json.dumps({'task_id':'test','instruction':'move',
        'subgoals':[{'target':'cube','destination':'tray'}],'destinations':{'tray':xyz}}))
    with pytest.raises(ValueError,match='finite'):load_task(p)


def test_shipped_hardware_template_has_no_invented_coordinates():
    root=Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError,match='measured finite'):load_task(root/'configs/blupe_measured_task.template.json')
    example=json.loads((root/'configs/blupe_stage1_task.example.json').read_text())
    assert example['remote_execution_allowed'] is False

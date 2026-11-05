import base64
from dataclasses import asdict
import json
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pytest
from agentic_rag_vlm.platform.http_policy import make_server,preview_request
from tests.test_platform_provider import fixture
from tests.test_delivery import calibration


def payload(provider,transport,geometry,task):
    observation=transport.get_observation()
    return {'task':{**asdict(task),'coordinate_frame':'arm_base','destinations':provider.destinations},
        'calibration':calibration(),'geometry':geometry,
        'image':{'data_base64':base64.b64encode(open(observation.image_paths[0],'rb').read()).decode()}}


def test_http_full_plan_never_dispatches_and_checks_auth(tmp_path):
    provider,transport,geometry,task,_=fixture(tmp_path);provider.prepare()
    server=make_server(provider,tmp_path/'http',port=0,token='contract-token')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    endpoint=f'http://127.0.0.1:{server.server_port}'
    try:
        with pytest.raises(HTTPError) as error:urlopen(endpoint+'/health')
        assert error.value.code==401
        headers={'Authorization':'Bearer contract-token','Content-Type':'application/json'}
        with urlopen(Request(endpoint+'/health',headers=headers)) as r:health=json.load(r)
        assert health['status']=='ready' and health['hardware_session_created'] is False
        request=Request(endpoint+'/v1/preview',data=json.dumps(payload(provider,transport,geometry,task)).encode(),headers=headers)
        with urlopen(request) as r:result=json.load(r)
        assert result['action']['source']=='haa_rag' and result['execution_performed'] is False
        assert not transport.commands
        with pytest.raises(HTTPError) as error:urlopen(Request(endpoint+'/v1/execute',data=b'{}',headers=headers))
        assert error.value.code==404
    finally:server.shutdown();server.server_close();thread.join()


def test_empty_scene_preview_returns_stop_without_coordinates(tmp_path):
    provider,transport,geometry,task,_=fixture(tmp_path);transport.positions={}
    result=preview_request(provider,payload(provider,transport,geometry,task),tmp_path/'http')
    assert result['decision']=='stop' and result['action'] is None and not transport.commands


def test_public_binding_requires_explicit_token(tmp_path):
    provider,*_=fixture(tmp_path)
    with pytest.raises(ValueError,match='TOKEN'):make_server(provider,tmp_path,host='0.0.0.0',port=0)


def test_vlm_veto_is_a_stop_preview_and_preserves_raw_response(tmp_path):
    from agentic_rag_vlm.qwen_adapter import VLMResult
    provider,transport,geometry,task,client=fixture(tmp_path);original=client.infer
    def infer(images,prompt):
        return original(images,prompt) if prompt.startswith('Detect visible') else VLMResult({'decision':'stop'},'{"decision":"stop"}',1.,'test','test',{})
    client.infer=infer
    result=preview_request(provider,payload(provider,transport,geometry,task),tmp_path/'http')
    assert result['decision']=='stop' and result['reasoner_calls'][-1]['parsed']['decision']=='stop'
    assert transport.commands==[]


def test_public_proposals_keep_model_identity_separate_from_pixels(tmp_path):
    from agentic_rag_vlm.qwen_adapter import VLMResult
    provider,transport,geometry,task,client=fixture(tmp_path);original=client.infer
    request=payload(provider,transport,geometry,task)
    request['object_candidates']=[{'candidate_id':'candidate_0','pixel_xy':[375,180],'bbox_xyxy':[350,160,400,200]},
        {'candidate_id':'candidate_1','pixel_xy':[570,125],'bbox_xyxy':[550,105,590,145]}]
    task_data=request['task'];task_data['subgoals']=task_data['subgoals'][:1]
    def infer(images,prompt):
        if prompt.startswith('These are object candidate'):
            value={'candidate_0':'cube','candidate_1':'vase'}
            return VLMResult(value,json.dumps(value),1.,'test','test',{})
        return original(images,prompt)
    client.infer=infer
    result=preview_request(provider,request,tmp_path/'http')
    assert result['decision']=='propose' and result['perception']['vlm_result']['identity_source']=='vlm_candidate_classification'
    assert result['perception']['vlm_result']['effective_objects'][0]['pixel_xy']==[375,180]
    assert result['action']['source']=='haa_rag' and not transport.commands


def test_invalid_model_response_is_stopped_and_saved(tmp_path):
    from agentic_rag_vlm.qwen_adapter import VLMResult
    provider,transport,geometry,task,client=fixture(tmp_path)
    client.infer=lambda *a:VLMResult(None,'unparseable model text',1.,'test','test',{},'no object')
    result=preview_request(provider,payload(provider,transport,geometry,task),tmp_path/'http')
    assert result['decision']=='stop' and result['perception']['raw_text']=='unparseable model text'
    assert not transport.commands


def test_unseen_protected_object_blocks_preview_before_review(tmp_path):
    provider,transport,geometry,task,client=fixture(tmp_path)
    transport.positions.pop('vase')
    result=preview_request(provider,payload(provider,transport,geometry,task),tmp_path/'http')
    assert result['decision']=='stop' and result['action'] is None
    assert result['trace'][-1]['payload']['reason']=='protected_object_not_visible'
    assert result['reasoner_calls']==[] and transport.commands==[]

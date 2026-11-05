"""Counterexamples against trusting reported destinations without raw evidence."""
import pytest
from scripts.audit_extended_benchmark import verify_endpoint


def sample(x=.3):
    return {'time':1.,'objects':{'target_cube':[x,.1,.018]},'contacts':[['table','target_geom',-.0001]]}


def test_sampled_endpoint_requires_task_evidence_not_exact_time_claim():
    valid,warning=verify_endpoint(sample(),1.01,[.3003,.1,.018],[.3,.1,.018],'target_geom')
    assert valid and warning['exact_final_pose_verified'] is False


@pytest.mark.parametrize('raw,time,reported',[
    (sample(.35),1.01,[.3,.1,.018]),  # reported goal cannot hide a raw miss
    (sample(),1.05,[.3,.1,.018]),    # stale trace cannot certify a later endpoint
    (sample(),1.,[.301,.1,.018]),    # exact-time snapshots must agree tightly
])
def test_endpoint_counterexamples_fail(raw,time,reported):
    valid,_=verify_endpoint(raw,time,reported,[.3,.1,.018],'target_geom')
    assert not valid

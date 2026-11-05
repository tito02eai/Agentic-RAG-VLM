"""Auditable VLM decision summaries around quantitative tool outputs.

Requests a concise decision justification, not private chain-of-thought. Model
output cannot change metric coordinates, physical limits or tool corrections.
"""
import json


class VLMReasoner:
    def __init__(self, client):
        self.client = client
        self.images = []
        self.target_images = {}
        self.calls = []
        self.negative_experiences = []

    def _call(self, stage, payload, expected, *, images=None):
        prompt = (
            "You supervise a robot grasp agent. Use only the supplied public observations and tool results. "
            "Give a brief decision justification (one sentence), not a reasoning trace. "
            "Do not propose coordinates or change quantitative tool corrections. "
            + expected + "\n" + json.dumps(payload,ensure_ascii=False,default=str)
        )
        response = self.client.infer(self.images if images is None else images,prompt)
        import hashlib
        supplied = self.images if images is None else images
        receipt = {"stage":stage,"prompt":prompt,**response.to_dict(),
            "images":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in supplied]}
        self.calls.append(receipt)
        if not isinstance(response.parsed,dict):
            raise ValueError("unparseable VLM agent decision")
        return receipt

    def think(self, action, evidence, history, attempt):
        retrieval = evidence.get("retrieval",{})
        shortlist = [{k:v for k,v in entry.items() if k in {
            "card_id","category_score","affordance_score","visual_score","final_score","outcome","strategy"}}
            for entry in retrieval.get("ranked_cards",[])]
        payload = {"attempt":attempt,"proposed_action":action.to_dict(),"retrieval_candidates":shortlist,
            "memory":evidence.get("memory",{}),"constraints":evidence.get("constraint",{}),
            "previous_failures":[event.get("reflection",{}) for event in history if event.get("reflection")]}
        payload["review_stage"] = "before execution; the arm will move to the target, not already grasping it"
        visibility=evidence.get('visibility_replanning')
        review_note=''
        if visibility:
            payload['visibility_replanning']=visibility
            review_note=(' The proposed upper target is explicitly identified. A different deferred target remains '
                'unidentified; do not assign it an identity. The tool proposes removing only the identified upper '
                'target to its configured destination, then observing again. Review this bounded order change. ')
            if visibility.get('policy')=='visible_target_progress_then_reobserve':
                review_note=(' The tool selects an identified, stable elevated task target, with scene constraints '
                    'checked, and proposes moving it to its existing destination before observing again. '
                    'No identity or support/occlusion relationship is inferred for the deferred unseen target. '
                    'Approve only this bounded visible-target action if compatible. ')
        result = self._call("thought",payload,
            'Return JSON {"decision":"execute" or "stop","summary":"brief justification"}. '
            "The image is the proposed target crop, not the current gripper pose. "
            "Geometry and identities are grounded by observation tools. Review parameter/constraint compatibility; "
            "do not infer a wrong grasp just because the arm is currently parked. "
            "Constraint active=true means protection is required and its listed corrections are already compiled "
            "into the proposal; it does not mean blocked. Inspect blocked, prerequisites and the specified "
            "approach offset/height/velocity corrections. An active constraint alone is not grounds for a veto. "
            "Final numerical planning and execution checks still apply; stop for incompatible parameters or "
            "insufficient evidence. Executor fields explicitly listed as unsupported are unapplied knowledge "
            "priors, not active hardware force controls. Execute if compatible."+review_note,
            images=self.target_images.get(action.target))
        if result["parsed"].get("decision") != "execute":
            raise ValueError("VLM agent did not approve the proposed action")
        return result

    def reflect(self, action, correction, history):
        return self._call("reflection",{"failed_action":action.to_dict(),"tool_correction":correction,
            "attempts_so_far":len(history),"retrieved_negative_experiences":self.negative_experiences},
            'Return JSON {"summary":"brief diagnosis and quantitative correction"}.')

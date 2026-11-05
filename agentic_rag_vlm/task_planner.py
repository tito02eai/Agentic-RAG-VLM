"""Natural-language task → validated object/destination subgoals.

Uses grounded object identities and operator-provided destination names. No
model-generated metric coordinates are executable through this interface.
"""
from dataclasses import asdict
import json
from .runtime import TaskSpec,Subgoal


class VLMTaskPlanner:
    def __init__(self, client):
        self.client=client

    def plan(self, task_id, instruction, objects, destinations, *, protected_objects=(),images=()):
        if not instruction.strip() or set(protected_objects)-set(objects):
            raise ValueError("instruction and observed protected object identities required")
        catalogue=[{"name":name,"category":obj.category,"material_prior":obj.material,
                    "fragility_prior":obj.fragility} for name,obj in objects.items()]
        prompt=("Create a finite ordered robot task plan from this instruction and grounded object catalogue. "
            "Use only exact provided object IDs and destination names. Never act on protected objects. "
            "Do not output coordinates, forces, hidden objects or unsupported skills. "
            'Return one JSON OBJECT: {"subgoals":[{"target":"object_id","destination":"zone_name"}]}. '
            "The only skill is grasp-and-place.\n"+json.dumps({"instruction":instruction,"objects":catalogue,
                "destinations":list(destinations),"protected_objects":list(protected_objects)}))
        result=self.client.infer(list(images),prompt)
        receipt={"prompt":prompt,**result.to_dict()}
        parsed=result.parsed
        if not isinstance(parsed,dict) or not isinstance(parsed.get("subgoals"),list) or not 1<=len(parsed["subgoals"])<=len(objects):
            raise ValueError("invalid or unbounded VLM task plan")
        goals=[]
        for item in parsed["subgoals"]:
            if not isinstance(item,dict) or set(item)!={"target","destination"}:
                raise ValueError("task goal schema violation")
            if item["target"] not in objects or item["target"] in protected_objects or item["destination"] not in destinations:
                raise ValueError("unknown/protected object or destination in VLM task plan")
            if any(g.target==item["target"] for g in goals):
                raise ValueError("repeated target in VLM task plan")
            goals.append(Subgoal(item["target"],item["destination"]))
        task=TaskSpec(task_id,instruction,tuple(goals),tuple(protected_objects))
        receipt["validated_task"]=asdict(task)
        return task,receipt

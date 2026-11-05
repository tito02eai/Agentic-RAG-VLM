"""Swept gripper-envelope check for the paper's geometric executor interface.

This validates an axis-aligned gripper envelope against observed object boxes,
not a full arm or an inter-arm collision certificate. Missing bounds reject.
"""
import numpy as np


def intersects_box(start,end,lower,upper):
    direction=end-start
    low,high=0.,1.
    for i in range(3):
        if abs(direction[i])<1e-10:
            if start[i]<lower[i] or start[i]>upper[i]:return False
            continue
        a,b=sorted(((lower[i]-start[i])/direction[i],(upper[i]-start[i])/direction[i]))
        low,high=max(low,a),min(high,b)
        if low>high:return False
    return True


class EnvelopePathValidator:
    def __init__(self, object_half_extents, *, gripper_half_extents, workspace_bounds, table_z):
        self.extents={k:np.asarray(v,dtype=float) for k,v in object_half_extents.items()}
        self.gripper=np.asarray(gripper_half_extents,dtype=float)
        self.bounds=np.asarray(workspace_bounds,dtype=float)
        self.table_z=float(table_z)
        if self.gripper.shape!=(3,) or np.any(self.gripper<=0) or self.bounds.shape!=(3,2):
            raise ValueError("measured gripper and workspace bounds required")
        if not np.isfinite(self.gripper).all() or not np.isfinite(self.bounds).all() or not np.isfinite(self.table_z):
            raise ValueError("finite bounds required")
        if np.any(self.bounds[:,0]>=self.bounds[:,1]):raise ValueError("invalid workspace")
        if any(v.shape!=(3,) or not np.isfinite(v).all() or np.any(v<=0) for v in self.extents.values()):
            raise ValueError("invalid object geometry")

    def __call__(self, proposal, objects):
        target=proposal["target"]
        if target not in objects or set(objects)-set(self.extents):return False
        goal=np.asarray(objects[target].center_xyz_m)
        offset=np.r_[proposal["approach_offset_xy_m"],0.]
        points=[goal+offset+[0,0,.10],goal+offset,goal]
        for point in points:
            if not np.isfinite(point).all() or np.any(point-self.gripper<self.bounds[:,0]) or np.any(point+self.gripper>self.bounds[:,1]):
                return False
            if point[2]-self.gripper[2] < self.table_z:return False
        for start,end in zip(points,points[1:]):
            for name,obj in objects.items():
                if name==target:continue
                center=np.asarray(obj.center_xyz_m)
                inflated=self.extents[name]+self.gripper
                if intersects_box(start,end,center-inflated,center+inflated):return False
        proposal.update(waypoints_xyz_m=[p.tolist() for p in points],validation_scope="swept_gripper_envelope_only")
        return True

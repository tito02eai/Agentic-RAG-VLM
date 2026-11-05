"""Development-only force/friction characterization, never counted as a benchmark."""
import argparse,json,sys
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agentic_rag_vlm.simulation.yam import YamPhysics,build_scene
from scripts.run_yam_physics import PhysicsTransport
from agentic_rag_vlm.platform.commands import PlatformCommand


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--force',type=float,required=True);p.add_argument('--friction',type=float,default=.7)
    p.add_argument('--mass',type=float,default=.06);p.add_argument('--record',action='store_true');args=p.parse_args()
    if args.output.exists():p.error('new output required')
    def builder(path,**kwargs):
        setup=build_scene(path,**kwargs);tree=ET.parse(path)
        for geom in tree.getroot().findall('.//geom'):
            name=geom.get('name','')
            if name.startswith(('lf_','rf_')):geom.set('friction','.7 .005 .0005')
            if name=='target_geom':geom.set('mass',str(args.mass));geom.set('friction',f'{args.friction} .01 .001')
        tree.write(path,encoding='unicode');return setup
    sim=YamPhysics(args.output,scene_builder=builder,record=args.record)
    transport=PhysicsTransport(sim)
    xyz=sim.detect()['cube']
    command=PlatformCommand('probe','pick','cube','development',{'position_xyz_m':xyz,'force_n':args.force,'aperture_m':.075,'approach':'top'})
    try:
        receipt=transport.send_command(command);samples=[]
        for _ in range(5):
            sim.step(100);samples.append(sim.tactile())
        result={'development_only':True,'force':args.force,'friction':args.friction,'mass':args.mass,
            'tactile_samples':samples,'private_final_xyz':sim.data.body('target_cube').xpos.copy().tolist(),
            'max_lift':sim.max_lift,'bilateral_lift_seconds':sim.bilateral_lift_seconds,'receipt':receipt}
    except Exception as error:result={'development_only':True,'error':str(error)}
    sim.save(result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

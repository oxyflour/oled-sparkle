"""Replay full-pupil saved runs and optionally time the retained legacy path.

Run from the repository with its CUDA-enabled Python environment:
  python test-results/benchmark-mitsuba-resident.py --with-legacy-baseline
All physical/sampling CLI arguments come from the original run.json files.
New solver outputs are written separately; historical results stay untouched.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'oled-ag-sparkle/scripts'))
import sparkle_multibackend as s

CASES={'pits':'oled-1mm-pits3to5um-periodic-50um',
       'original':'oled-1mm-original-periodic-50um',
       'etched':'oled-etched-ag/1um', 'sms-green':'oled-sms-reference/1um'}
LIMITS={'optical_relative_l2':1e-4,'measurement_relative_l2':1e-5,
        'cv_difference_percentage_points':1e-3}


def compare(m,image,reference):
    image,reference=image.astype(np.float64),reference.astype(np.float64)
    collected,_=s.collect_analysis_pixels(image,m.dx_m*1e6,m.display_pitch_m*1e6,m.roi_margin_m*1e6)
    original,_=s.collect_analysis_pixels(reference,m.dx_m*1e6,m.display_pitch_m*1e6,m.roi_margin_m*1e6)
    cv=float(100*collected.std()/collected.mean())
    ref_cv=float(100*original.std()/original.mean())
    return {'optical_relative_l2':float(np.linalg.norm(image-reference)/np.linalg.norm(reference)),
            'measurement_relative_l2':float(np.linalg.norm(collected-original)/np.linalg.norm(original)),
            'max_optical_absolute_error':float(np.max(np.abs(image-reference))),
            'cv_percent':cv,'reference_cv_percent':ref_cv,
            'cv_difference_percentage_points':abs(cv-ref_cv)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'test-results/mitsuba-resident-regression')
    parser.add_argument('--with-legacy-baseline',action='store_true',
                        help='also time a complete legacy pit-surface solve (several minutes)')
    args=parser.parse_args()
    root=args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    saved={name:json.loads((ROOT/'test-results'/path/'run.json').read_text(encoding='utf-8'))
           for name,path in CASES.items()}
    import mitsuba as mi
    import drjit as dr
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'git_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'solver_sha256':hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),
              'python':sys.version,'platform':platform.platform(),
              'mitsuba_version':mi.__version__,'drjit_version':dr.__version__,
              'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version',
                                             '--format=csv,noheader'],text=True).strip(),
              'accuracy_limits':LIMITS,'minimum_controlled_speedup':10,
              'timing':'Sequential solves; includes mesh/scene creation, JIT and final host image; excludes postprocessing.',
              'cases':{name:{'reference':path,'parameters':saved[name]['parameters'],
                             'input_files':saved[name]['input_files']} for name,path in CASES.items()}}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    result={'status':'running','protocol':'protocol.json','cases':[]}
    try:
        if args.with_legacy_baseline:
            m=s.model_from_args(SimpleNamespace(**saved['pits']['parameters']))
            m=replace(m,progress_every=400)
            mi.set_variant(m.mitsuba_variant)
            dr.sync_thread()
            start=time.perf_counter()
            mesh,_,_=s.create_mitsuba_mesh(mi,m)
            scene=mi.load_dict({'type':'scene','ag_surface':mesh})
            image,count=s._simulate_mitsuba_numpy(mi,scene,m)
            dr.sync_thread()
            elapsed=time.perf_counter()-start
            np.save(root/'legacy-pits-brightness.npy',image)
            reference=np.load(ROOT/'test-results'/CASES['pits']/'mitsuba/brightness.npy')
            np.testing.assert_array_equal(image,reference)
            result['legacy_baseline']={'case':'pits','solver_seconds':elapsed,
                                       'rays':count,'bitwise_equal_to_saved':True}
            print('Controlled legacy seconds:',elapsed,flush=True)
            del scene,mesh,image
        for name,path in CASES.items():
            original=saved[name]
            out=root/name
            argv=original['argv']+['--backend','mitsuba','--backend-workers','1',
                                   '--progress-every','0','--out',str(out)]
            stats=s.run(argv)['mitsuba']
            run=json.loads((out/'run.json').read_text(encoding='utf-8'))
            changed={key for key,value in original['parameters'].items()
                     if run['parameters'][key]!=value}
            assert changed <= {'backend','backend_workers','progress_every','out'},changed
            for field in ('height','emission'):
                assert run['input_files'][field]['sha256']==original['input_files'][field]['sha256']
            assert stats['execution_context']['integrator']=='drjit_resident'
            m=s.model_from_args(SimpleNamespace(**run['parameters']))
            image=np.load(out/'mitsuba/brightness.npy')
            reference=np.load(ROOT/'test-results'/path/'mitsuba/brightness.npy')
            errors=compare(m,image,reference)
            for key,limit in LIMITS.items():
                assert errors[key]<=limit,(name,key,errors[key],limit)
            dr.sync_thread()
            start=time.perf_counter()
            repeated,count=s.simulate_mitsuba(m)
            dr.sync_thread()
            repeated_seconds=time.perf_counter()-start
            np.testing.assert_array_equal(repeated,image)
            assert count==int(np.prod(m.shape)*m.ray_radial*m.ray_angular)
            collected,_=s.collect_analysis_pixels(image,m.dx_m*1e6,m.display_pitch_m*1e6,m.roi_margin_m*1e6)
            np.testing.assert_array_equal(collected,np.load(out/'mitsuba/analysis_brightness.npy'))
            assert abs(stats['sparkle_cv_percent']-errors['cv_percent'])<1e-10
            row={'case':name,'solver_seconds':[stats['elapsed_seconds'],repeated_seconds],
                 'rays':count,'grid':list(m.shape),'pupil_nodes':m.ray_radial*m.ray_angular,
                 'ray_chunk':m.ray_chunk,'mesh_stride':m.mesh_stride,**errors,
                 'input_hashes_unchanged':True,'physical_and_sampling_parameters_unchanged':True,
                 'repeat_bitwise_equal':True,'saved_analysis_and_cv_verified':True}
            if name=='pits' and 'legacy_baseline' in result:
                row['controlled_speedup']=result['legacy_baseline']['solver_seconds']/float(np.median(row['solver_seconds']))
                assert row['controlled_speedup']>=10,row['controlled_speedup']
            result['cases'].append(row)
            print(json.dumps(row),flush=True)
        result['status']='passed'
    except Exception as exc:
        result['status']='failed'
        result['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        (root/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

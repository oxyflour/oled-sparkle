#!/usr/bin/env python3
"""Run bundled example from ANY working directory, with consistent physical units."""
import argparse
import subprocess
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--example', choices=['quick','1mm'], default='quick')
    ap.add_argument('--emission-variant', choices=['basic','subpixel'], default='basic')
    ap.add_argument('--backend', choices=['torch','torchoptics','chromatix','mitsuba','all'], default='torch')
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    ap.add_argument('--out', default=None)
    ap.add_argument('--skip-unavailable', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--extra', nargs=argparse.REMAINDER, help='Any additional solver flags; must appear after --extra')
    a=ap.parse_args()
    if a.example=='quick':
        demo=SKILL_ROOT/'assets/demo_quick'
        pitch, height_range, em_pitch, display_pitch, roi, pad = ('2','0.5','64','64','0','64')
        em_file='emission.png'
    else:
        demo=SKILL_ROOT/'assets/demo_1mm'
        pitch, height_range, display_pitch, roi, pad = ('0.5','2','50','100','192')
        em_file='emission_subpixel.png' if a.emission_variant=='subpixel' else 'emission.png'
        em_pitch='10' if a.emission_variant=='subpixel' else '50'
    out=Path(a.out).expanduser().resolve() if a.out else SKILL_ROOT/'outputs'/('demo_'+a.example)
    cmd=[sys.executable,str(SKILL_ROOT/'scripts/sparkle_multibackend.py'),
         '--height',str(demo/'heightfield.png'), '--height-pitch-um',pitch,
         '--height-range-um',height_range,'--emission',str(demo/em_file),
         '--emission-pitch-um',em_pitch,'--analysis-pitch-um',display_pitch,
         '--gap-um','500','--wavelength-nm','530', '--n-stack','1.5',
         '--n-out','1.0','--na','0.08', '--pad-px',pad,
         '--roi-margin-um',roi,'--source-samples','1','--batch','2',
         '--device',a.device,'--backend',a.backend,'--out',str(out)]
    if a.dry_run: cmd.append('--dry-run')
    if a.skip_unavailable: cmd.append('--skip-unavailable')
    if a.extra: cmd.extend(a.extra)
    print('Command:', ' '.join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

if __name__=='__main__':
    main()

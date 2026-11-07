"""Audit the 1mm field, 1um height grid and 50um brightness pixels, then package."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime,timezone
import base64
import difflib
import hashlib
import io
import json
import math
import re
import shutil
import zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'test-results'
DATA=RESULTS/'oled-1mm-height1um-50um'
SKILL=ROOT/'oled-ag-sparkle'
NAMES=('torch','torchoptics','chromatix','mitsuba')


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.in_data=False; self.payload=[]; self.downloads={}; self.images=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='script' and attrs.get('id')=='report-data': self.in_data=True
        if tag=='img': self.images.append(attrs)
        if tag=='a' and 'download' in attrs:
            self.downloads[attrs['download']]=base64.b64decode(attrs['href'].split(',',1)[1])
    def handle_endtag(self,tag):
        if tag=='script': self.in_data=False
    def handle_data(self,value):
        if self.in_data: self.payload.append(value)


def test_summary():
    path=RESULTS/'logs/validation-wsl.log'
    if not path.exists(): path=RESULTS/'logs/validation-height1um.log'
    log=path.read_text(encoding='utf-8')
    match=re.search(r'Ran (\d+) tests',log)
    assert match and log.rstrip().endswith('OK'), 'All tests must pass without skips'
    count=int(match.group(1))
    return {'discovered':count,'passed':count,'skipped':0,'failed':0,
            'log':path.relative_to(ROOT).as_posix()}


def audit():
    test_summary()
    run=json.loads((DATA/'run.json').read_text(encoding='utf-8'))
    assert run['status']=='complete'
    assert set(run['backend_runs'])==set(NAMES)
    for role,info in run['input_files'].items():
        size,pitch=(1000,1) if role=='height' else (500,2)
        assert info['width_px']==info['height_px']==size
        assert math.isclose(info['physical_width_um'],1000)
        assert math.isclose(info['physical_height_um'],1000)
        assert math.isclose(info['pitch_um_per_texel'],pitch)
    previous=json.loads((RESULTS/'oled-1mm-50um/run.json').read_text(encoding='utf-8'))
    assert run['input_files']['emission']['sha256']==previous['input_files']['emission']['sha256']
    assert run['parameters']['analysis_pitch_um']==50
    assert run['parameters']['roi_margin_um']==0
    assert run['input_files']['height']['png_bits_per_channel']==16
    report=RESULTS/'report.html'
    parser=ReportParser(); parser.feed(report.read_text(encoding='utf-8'))
    payload=json.loads(''.join(parser.payload))
    records=payload['groups'][0]['records']
    assert len(records)==4 and all(r['status']=='complete' for r in records)
    assert len(parser.images)==6 and len(parser.downloads)==10
    assert all(image['src'].startswith('data:image/png;base64,') for image in parser.images)
    for role,info in run['input_files'].items():
        path=Path(info['path'])
        assert parser.downloads[path.name]==path.read_bytes()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
    summaries=[]
    for backend in NAMES:
        folder=DATA/backend
        fine=np.load(folder/'brightness.npy')
        collected=np.load(folder/'analysis_brightness.npy')
        mask=np.load(folder/'roi_mask.npy')
        stats=json.loads((folder/'stats.json').read_text(encoding='utf-8'))
        assert fine.shape==(1000,1000) and collected.shape==mask.shape==(20,20)
        assert mask.all() and stats['num_roi_pixels']==400
        assert stats['analysis_grid']==[20,20]
        assert math.isclose(stats['analysis_pixel_pitch_um'],50)
        np.testing.assert_allclose(stats['analysis_extent_um_xyxy'],[-500,-500,500,500],rtol=1e-14)
        expected=fine.astype(np.float64).reshape(20,50,20,50).mean(axis=(1,3))
        np.testing.assert_allclose(collected,expected,rtol=1e-13,atol=1e-16)
        assert math.isclose(collected.mean(),fine.astype(np.float64).mean(),rel_tol=1e-13)
        cv=100*collected.std(ddof=0)/collected.mean()
        assert math.isclose(stats['sparkle_cv_percent'],cv,rel_tol=1e-14)
        assert next(r['stats'] for r in records if r['backend']==backend)==stats
        for filename in ('brightness.npy','analysis_brightness.npy'):
            assert parser.downloads[backend+'-'+filename]==(folder/filename).read_bytes()
        summaries.append({'backend':backend,'sparkle_cv_percent':float(cv),
                          'solver_seconds':stats['elapsed_seconds'],
                          'postprocess_seconds':stats['postprocess_elapsed_seconds'],
                          'mean':stats['brightness_mean'],'std':stats['brightness_std'],
                          'analysis_grid':[20,20],'analysis_pixel_um':50,'roi_pixels':400})
    reference=np.load(DATA/'torch/analysis_brightness.npy')
    wave_errors={}
    for backend in ('torchoptics','chromatix'):
        other=np.load(DATA/backend/'analysis_brightness.npy')
        relative_l2=float(np.linalg.norm(other-reference)/np.linalg.norm(reference))
        assert relative_l2<1e-3
        wave_errors[backend]=relative_l2
    validation={'physical_field_mm':[1,1],'optical_grid':[1000,1000],'optical_pitch_um':1,
                'emission_grid':[500,500],'emission_pitch_um':2,'emission_bytes_unchanged':True,
                'measurement_grid':[20,20],'measurement_pixel_um':[50,50],
                'optical_samples_per_measurement_pixel':[50,50],'roi_pixels_per_backend':400,
                'inputs':2,'brightness_outputs':4,'display_images':len(parser.images),
                'float_downloads':8,'input_downloads':2,
                'area_average_matches_independent_reshape':True,
                'full_field_mean_preserved':True,'metrics_recomputed_from_400_pixels':True,
                'raw_downloads_match_saved_arrays':True,'input_downloads_match_source_pngs':True,
                'wave_backend_relative_l2_brightness_50um':wave_errors,
                'formula':'100 * std(I_50um, ddof=0) / mean(I_50um)',
                'records':summaries,'report_bytes':report.stat().st_size}
    (RESULTS/'validation-height1um.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    return run,validation


def package(run,validation):
    archive=RESULTS/'history/height2um'; archive.mkdir(parents=True,exist_ok=True)
    if (RESULTS/'report.json').exists() and not (archive/'report.json').exists():
        shutil.copy2(RESULTS/'report.json',archive/'report.json')
    baseline=ROOT/'oled-ag-sparkle-skill.zip'
    manifest={'schema_version':6,'date':datetime.now(timezone.utc).isoformat(),
              'metric':'direct CV of 50um collected-brightness pixels',
              'metric_definition':validation['formula'],'input_files':run['input_files'],
              'parameters':run['parameters'],'effective_model':run['effective_model'],
              'environment':run['environment'],'run_wall_seconds':run['wall_elapsed_seconds'],
              'brightness_results':validation['records'],'validation':validation,
              'tests':test_summary(),
              'comparison':json.loads((DATA/'comparison.json').read_text(encoding='utf-8')),
              'original_zip_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
              'artifacts':{'html_report':'test-results/report.html',
                           'results_directory':'test-results/oled-1mm-height1um-50um',
                           'comparison_figure':'test-results/brightness-comparison-height1um-50um.png'},
              'scope':'1mm square field; height 1000x1000 at 1um from existing height data; emission 500x500 at 2um; all 400 collected 50um pixels directly enter CV'}
    runtime_validation=RESULTS/'chromatix-wsl-bridge/validation.json'
    if runtime_validation.exists():
        manifest['windows_chromatix_wsl']=json.loads(runtime_validation.read_text(encoding='utf-8'))
        manifest['reference_run_started_at_utc']=run['started_at_utc']
        manifest['reference_run_note']='Top-level multi-backend images and timings retain the recorded reference run. Current Windows Chromatix routing and bridge measurements are in windows_chromatix_wsl.'
        manifest['artifacts']['chromatix_wsl_bridge']='test-results/chromatix-wsl-bridge/validation.json'
    for path in (RESULTS/'report.json',SKILL/'VALIDATION.json'):
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(DATA/'comparison.csv',RESULTS/'comparison-summary.csv')
    patch=[]
    with zipfile.ZipFile(baseline) as original:
        old={name.split('/',1)[1]:original.read(name) for name in original.namelist()
             if name.startswith('oled-ag-sparkle/') and not name.endswith('/')}
        for path in sorted(SKILL.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or 'outputs' in path.parts: continue
            relative=path.relative_to(SKILL).as_posix()
            if path.suffix not in ('.py','.md','.txt','.json','.yaml','.yml'): continue
            before=old.get(relative,b'').decode('utf-8-sig',errors='replace').splitlines(keepends=True)
            after=path.read_text(encoding='utf-8').splitlines(keepends=True)
            patch.extend(difflib.unified_diff(before,after,fromfile='a/'+relative,tofile='b/'+relative))
    (RESULTS/'changes.patch').write_text(''.join(patch),encoding='utf-8')
    target=ROOT/'oled-ag-sparkle-skill-fixed.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(SKILL.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and 'outputs' not in path.parts:
                z.write(path,'oled-ag-sparkle/'+path.relative_to(SKILL).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        saved=json.loads(z.read('oled-ag-sparkle/VALIDATION.json'))
        assert saved['schema_version']==6 and saved['tests']==manifest['tests']
        assert 'oled-ag-sparkle/assets/oled-1mm-height1um/heightfield.png' in z.namelist()
        assert 'oled-ag-sparkle/scripts/chromatix_wsl.py' in z.namelist()
        assert 'oled-ag-sparkle/references/windows-wsl.md' in z.namelist()
    print(json.dumps({'report':str(RESULTS/'report.html'),'skill':str(target),
                      'tests_passed':manifest['tests']['passed'],'results':validation['records']},ensure_ascii=False,indent=2))


if __name__=='__main__': package(*audit())

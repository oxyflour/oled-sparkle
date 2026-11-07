"""Verify the offline report, record this experiment, and rebuild the skill zip."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime, timezone
import base64
import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'test-results'
SKILL = ROOT/'oled-ag-sparkle'
DATA = RESULTS/'oled-1mm-pits3to5um-periodic-50um'


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = None
        self.payloads = {}
        self.downloads = {}
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script' and attrs.get('type') == 'application/json':
            self.active = attrs['id']
            self.payloads[self.active] = []
        if tag == 'a' and 'download' in attrs:
            self.downloads[attrs['download']] = base64.b64decode(attrs['href'].split(',', 1)[1])
        if tag == 'img':
            self.images.append(attrs)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = None

    def handle_data(self, text):
        if self.active:
            self.payloads[self.active].append(text)


def main():
    global DATA
    parser_args = argparse.ArgumentParser(description=__doc__)
    parser_args.add_argument('--legacy-stretched', action='store_true')
    parser_args.add_argument('--legacy-whole-tile', action='store_true')
    args = parser_args.parse_args()
    periodic_case = not (args.legacy_stretched or args.legacy_whole_tile)
    prefix = 'random-pit-periodic'
    log = RESULTS/'logs/validation-periodic.log'
    if args.legacy_stretched:
        DATA = RESULTS/'oled-1mm-pits3to5um-50um'
        prefix = 'random-pit'
        log = RESULTS/'logs/validation-pits.log'
    elif args.legacy_whole_tile:
        DATA = RESULTS/'oled-1mm-pits3to5um-tiled-50um'
        prefix = 'random-pit-tiled'
        log = RESULTS/'logs/validation-tiled.log'
    summary = json.loads((RESULTS/f'{prefix}-comparison.json').read_text(encoding='utf-8'))
    report = DATA/'report.html'
    parser = ReportParser()
    parser.feed(report.read_text(encoding='utf-8'))
    embedded = {key: json.loads(''.join(chunks)) for key, chunks in parser.payloads.items()}
    assert embedded['surface-comparison-data'] == summary
    expected_images = 6+len(embedded['report-data'].get('figures', []))
    assert expected_images == 10 if periodic_case else expected_images in (9, 10)
    assert len(parser.images) == expected_images
    assert all(image['src'].startswith('data:image/png;base64,') for image in parser.images)
    groups = embedded['report-data']['groups']
    assert len(groups) == 1 and len(groups[0]['records']) == 4
    run = groups[0]['run']
    for info in run['input_files'].values():
        path = Path(info['path'])
        raw = path.read_bytes()
        assert parser.downloads[path.name] == raw
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
    for record in groups[0]['records']:
        backend = record['backend']
        assert record['status'] == 'complete'
        for filename in ('brightness.npy', 'analysis_brightness.npy'):
            assert parser.downloads[backend+'-'+filename] == (DATA/backend/filename).read_bytes()
    assert len(parser.downloads) == 10
    test_text = log.read_text(encoding='utf-8-sig')
    count = int(re.search(r'Ran (\d+) tests', test_text).group(1))
    assert count >= 25 and test_text.rstrip().endswith('OK') and 'skipped=' not in test_text
    summary['report_integrity'] = {'display_images': expected_images, 'original_input_downloads': 2,
        'raw_float_downloads': 8, 'all_downloads_identical_to_saved_files': True,
        'report': report.relative_to(ROOT).as_posix(),
        'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest()}
    summary['tests'] = {'passed': count, 'skipped': 0, 'failed': 0,
                        'log': log.relative_to(ROOT).as_posix()}
    focused_log = RESULTS/'logs/validation-periodic-input.log'
    if periodic_case and focused_log.exists():
        focused_text = focused_log.read_text(encoding='utf-8-sig')
        assert focused_text.rstrip().endswith('OK')
        summary['tests']['focused_input_checks'] = {
            'passed': int(re.search(r'Ran (\d+) tests', focused_text).group(1)),
            'log': focused_log.relative_to(ROOT).as_posix()}
    if periodic_case:
        annotations = []
        for previous in (RESULTS/'oled-1mm-height1um-50um/report.html',
                         RESULTS/'oled-1mm-pits3to5um-50um/report.html',
                         RESULTS/'oled-1mm-original-tiled-50um/report.html',
                         RESULTS/'oled-1mm-pits3to5um-tiled-50um/report.html'):
            if not previous.exists():
                continue
            document = previous.read_text(encoding='utf-8')
            if 'id="superseded-emission-period"' in document:
                continue
            before = hashlib.sha256(previous.read_bytes()).hexdigest()
            document = re.sub(r'<aside id="superseded-emission-scale"[\s\S]*?</aside>', '', document)
            relative = Path(os.path.relpath(report, previous.parent)).as_posix()
            reason = ('原图块未裁成完整重复周期，整块平铺存在接缝。' if '-tiled-' in previous.parent.name else
                      '原 256 µm 纹理被拉伸到 1 mm，改变了发光单元尺寸与间距。')
            notice = ('<aside id="superseded-emission-period" style="padding:22px;background:#fff0bd;'
                      'border-bottom:2px solid #d09116;font:16px sans-serif;line-height:1.7">'
                      '<strong>历史结果：发光图构造已修正。</strong>'+reason+
                      '当前案例已先裁出 70 × 70 完整重复单元，再以原 2 µm 采样平铺。此处原始数值仅作历史记录。'
                      f'<a href="{relative}">查看周期单元裁剪后的完整对比报告</a></aside>')
            previous.write_text(document.replace('<body>', '<body>'+notice, 1), encoding='utf-8')
            annotations.append({'report': previous.relative_to(ROOT).as_posix(),
                                'sha256_before_notice': before,
                                'sha256_after_notice': hashlib.sha256(previous.read_bytes()).hexdigest()})
        previous_validation = RESULTS/f'{prefix}-validation.json'
        if not annotations and previous_validation.exists():
            annotations = json.loads(previous_validation.read_text(encoding='utf-8')).get('superseded_report_annotations', [])
        summary['superseded_report_annotations'] = annotations
    # Keep the report's embedded summary stable; integrity is recorded separately.
    (RESULTS/f'{prefix}-validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    validation_path = SKILL/'VALIDATION.json'
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    if not args.legacy_stretched and 'random_circular_pits' in validation:
        validation.setdefault('superseded_stretched_emission_random_pits', validation['random_circular_pits'])
    if periodic_case and 'random_circular_pits' in validation:
        validation.setdefault('superseded_uncropped_tile_random_pits', validation['random_circular_pits'])
    validation['current_example'] = ('random_circular_pits_3to5um_diameter_50pct_coverage_periodic_cell' if periodic_case else
                                     'legacy_uncropped_tile' if args.legacy_whole_tile else 'legacy_stretched_emission')
    validation['latest_experiment_at_utc'] = datetime.now(timezone.utc).isoformat()
    validation['random_circular_pits'] = summary
    validation['latest_tests'] = summary['tests']
    validation['latest_results_report'] = summary['report_integrity']['report']
    if not args.legacy_stretched:
        validation['emission_scale_correction'] = {
            'previous_layout': 'same illustrated content stretched from 256um to 1000um',
            'previous_linear_enlargement': 500/128,
            'current_layout': 'crop repeat cell [26,58,96,128], keep 2um/texel, repeat 70x70 cell 8x8, crop to 500x500' if periodic_case else 'legacy whole 128x128 tile',
            'previous_cv_comparison_superseded': True,
            'historical_top_level_fields': 'Original reference measurements retained; current results are in random_circular_pits.'}
    if periodic_case:
        validation['emission_period_crop_correction'] = summary['emission_field_construction']
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest_path = RESULTS/'report.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    for key in ('current_example', 'latest_experiment_at_utc', 'random_circular_pits',
                'latest_tests', 'latest_results_report', 'emission_scale_correction',
                'superseded_stretched_emission_random_pits', 'superseded_uncropped_tile_random_pits',
                'emission_period_crop_correction'):
        if key in validation:
            manifest[key] = validation[key]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(report, RESULTS/'report.html')
    shutil.copy2(DATA/'comparison.csv', RESULTS/'comparison-summary.csv')
    target = ROOT/'oled-ag-sparkle-skill-fixed.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SKILL.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and 'outputs' not in path.parts:
                archive.write(path, 'oled-ag-sparkle/'+path.relative_to(SKILL).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert archive.read('oled-ag-sparkle/assets/oled-1mm-pits3to5um/heightfield.png') == (
            SKILL/'assets/oled-1mm-pits3to5um/heightfield.png').read_bytes()
        assert archive.read('oled-ag-sparkle/assets/oled-1mm-pits3to5um/pits.csv') == (
            SKILL/'assets/oled-1mm-pits3to5um/pits.csv').read_bytes()
        assert 'oled-ag-sparkle/scripts/prepare_random_pits.py' in archive.namelist()
        if periodic_case:
            assert 'oled-ag-sparkle/scripts/prepare_tiled_emission.py' in archive.namelist()
            for name in ('emission.png', 'periodic-cell.png'):
                assert archive.read('oled-ag-sparkle/assets/oled-1mm-periodic/'+name) == (
                    SKILL/'assets/oled-1mm-periodic'/name).read_bytes()
    print(json.dumps({'report': str(report), 'skill_zip': str(target),
                      'report_integrity': summary['report_integrity'], 'tests': summary['tests']}, indent=2))


if __name__ == '__main__':
    main()

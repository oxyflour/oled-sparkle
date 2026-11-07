"""Audit saved floats and create the original/random-pit surface comparison."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
import argparse
import hashlib
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT/'oled-ag-sparkle'
RESULTS = ROOT/'test-results'
OLD = RESULTS/'oled-1mm-original-periodic-50um'
NEW = RESULTS/'oled-1mm-pits3to5um-periodic-50um'
EMISSION = SKILL/'assets/oled-1mm-periodic/emission.png'
SAMPLING = RESULTS/'pit-sampling-periodic-0p5um'
PREFIX = 'random-pit-periodic'
NAMES = ('torch', 'torchoptics', 'chromatix', 'mitsuba')
sys.path.insert(0, str(SKILL/'scripts'))
from reporting import write_html_report


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def emission_figure():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), layout='constrained')
    source = np.asarray(Image.open(SKILL/'assets/oled-diamond/emission.png').convert('RGB'))
    cell = np.asarray(Image.open(SKILL/'assets/oled-1mm-periodic/periodic-cell.png').convert('RGB'))
    axes[0].imshow(source, extent=(0, 128, 128, 0), interpolation='nearest')
    axes[0].add_patch(Rectangle((26, 58), 70, 70, linewidth=2, edgecolor='#ffdc40', facecolor='none'))
    axes[0].set(title='Source patch and selected repeat cell', xlabel='x (source texels)', ylabel='y (source texels)')
    axes[1].imshow(cell, extent=(0, 140, 140, 0), interpolation='nearest')
    axes[1].set(title='70 x 70 texels | 140 x 140 um\n2 red + 4 green + 2 blue bodies', xlabel='x (um)', ylabel='y (um)')
    axes[2].imshow(np.tile(cell, (3, 3, 1)), extent=(0, 420, 420, 0), interpolation='nearest')
    axes[2].add_patch(Rectangle((140, 140), 140, 140, linewidth=1, edgecolor='#ffdc40', facecolor='none'))
    axes[2].set(title='3 x 3 repetition | joins preserved\nCenter cell outlined', xlabel='x (um)', ylabel='y (um)')
    fig.suptitle('Crop the complete repeat cell before expanding the OLED field\n'
                 'Source crop [26, 58, 96, 128]; original 2 um/texel; no rescaling, blending or pixel repainting.', fontsize=13)
    path = RESULTS/'emission-periodic-crop.png'
    fig.savefig(path, dpi=170, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return path


def height_figure():
    old = np.asarray(Image.open(SKILL/'assets/oled-1mm-height1um/heightfield.png'))/65535*.5
    new = np.asarray(Image.open(SKILL/'assets/oled-1mm-pits3to5um/heightfield.png'))/65535*.5
    fine = np.asarray(Image.open(SKILL/'assets/oled-1mm-pits3to5um-pitch0p5/heightfield.png'))/65535*.5
    geometry = np.loadtxt(SKILL/'assets/oled-1mm-pits3to5um/pits.csv', delimiter=',', skiprows=1)
    emission = np.asarray(Image.open(EMISSION).convert('RGB'))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
    for ax, image, title in ((axes[0, 0], old, 'Original surface | 1 um grid'),
                              (axes[0, 1], new, 'Random pits | 1 um grid')):
        im = ax.imshow(image, origin='lower', extent=(0, 1000, 0, 1000),
                       cmap='viridis', vmin=0, vmax=.5)
        ax.set(title=title, xlabel='x (um)', ylabel='y (um)')
        fig.colorbar(im, ax=ax, label='Height (um)', shrink=.8)
    axes[0, 2].imshow(emission, origin='lower', extent=(0, 1000, 0, 1000))
    axes[0, 2].set(title='OLED emission | fixed 2 um grid', xlabel='x (um)', ylabel='y (um)')
    for ax, image, title in ((axes[1, 0], old[450:500, 450:500], 'Original | 50 x 50 um detail'),
                              (axes[1, 1], fine[900:1000, 900:1000], 'Pits | same detail, 0.5 um sampling')):
        im = ax.imshow(image, origin='lower', extent=(450, 500, 450, 500),
                       cmap='viridis', vmin=0, vmax=.5, interpolation='nearest')
        ax.set(title=title, xlabel='x (um)', ylabel='y (um)')
        fig.colorbar(im, ax=ax, label='Height (um)', shrink=.8)
    axes[1, 2].hist(geometry[:, 2], bins=np.linspace(3, 5, 21), color='#168584', edgecolor='white')
    axes[1, 2].set(title='39,048 pits | diameter distribution', xlabel='Diameter (um)', ylabel='Pit count')
    fig.suptitle('Random circular pits: diameter 3-5 um, depth 0.5 um, footprint coverage 50.0001%\n'
                 'Nonoverlapping random positions; smooth bowls; seed 20261107', fontsize=14)
    path = RESULTS/f'{PREFIX}-height-comparison.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def audit_geometry():
    folder = SKILL/'assets/oled-1mm-pits3to5um'
    metadata = read(folder/'heightfield.source.json')
    pits = np.loadtxt(folder/'pits.csv', delimiter=',', skiprows=1)
    radii = pits[:, 2]/2
    pairs = np.array(list(cKDTree(pits[:, :2]).query_pairs(5.)), dtype=int)
    gaps = np.linalg.norm(pits[pairs[:, 0], :2]-pits[pairs[:, 1], :2], axis=1)
    gaps -= radii[pairs[:, 0]]+radii[pairs[:, 1]]
    assert float(gaps.min()) >= -1e-12
    assert np.all(pits[:, :2] >= radii[:, None])
    assert np.all(pits[:, :2]+radii[:, None] <= 1000)
    assert np.all((pits[:, 2] >= 3) & (pits[:, 2] <= 5))
    fraction = float(np.pi*np.sum(radii**2)/1e6)
    assert .5 <= fraction < .5+np.pi*2.5**2/1e6
    fine_metadata = read(SKILL/'assets/oled-1mm-pits3to5um-pitch0p5/heightfield.source.json')
    assert metadata['geometry_sha256'] == fine_metadata['geometry_sha256']
    return {'metadata': metadata, 'independent_coverage': fraction,
            'minimum_edge_separation_um': float(gaps.min()),
            'all_circles_inside_field': True, 'no_overlapping_circles': True,
            'fine_grid_geometry_identical': True}


def audit_run(folder):
    run = read(folder/'run.json')
    assert run['status'] == 'complete'
    results = {}
    for backend, outcome in run['backend_runs'].items():
        assert outcome['status'] == 'complete'
        stats = read(folder/backend/'stats.json')
        assert stats['run_id'] == run['run_id']
        fine = np.load(folder/backend/'brightness.npy')
        image = np.load(folder/backend/'analysis_brightness.npy')
        mask = np.load(folder/backend/'roi_mask.npy')
        block = round(50/run['parameters']['height_pitch_um'])
        expected = fine.astype(np.float64).reshape(20, block, 20, block).mean(axis=(1, 3))
        np.testing.assert_allclose(image, expected, rtol=1e-12, atol=1e-16)
        assert image.shape == mask.shape == (20, 20) and mask.all()
        cv = float(100*image.std(ddof=0)/image.mean())
        assert math.isclose(cv, stats['sparkle_cv_percent'], rel_tol=1e-13)
        assert stats['num_roi_pixels'] == 400
        results[backend] = {'cv_percent': cv, 'brightness_mean': float(image.mean()),
                            'solver_seconds': stats['elapsed_seconds'],
                            'execution_context': stats['execution_context']}
    for role, info in run['input_files'].items():
        assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest() == info['sha256']
    return run, results


def brightness_figure(rows):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4), layout='constrained')
    for col, row in enumerate(rows):
        name = row['backend']
        images = [np.load(folder/name/'analysis_brightness.npy') for folder in (OLD, NEW)]
        vmax = max(float(image.max()) for image in images)
        for n, (label, image) in enumerate(zip(('Original', 'Random pits'), images)):
            ax = axes[n, col]
            im = ax.imshow(image, origin='lower', extent=(-500, 500, -500, 500),
                           cmap='gray', vmin=0, vmax=vmax, interpolation='nearest')
            cv = row['old_cv_percent'] if n == 0 else row['new_cv_percent']
            title = f'{name} | {label}\nCV = {cv:.6f}%'
            if n == 1:
                title += (f'\nChange: {row["delta_percentage_points"]:+.3f} pp '
                          f'({row["relative_change_percent"]:+.2f}%)')
            ax.set(xlabel='x (um)', ylabel='y (um)')
            ax.set_title(title, fontsize=10.5)
            fig.colorbar(im, ax=ax, label='Relative linear brightness', shrink=.7)
    fig.suptitle('Original surface vs random 3-5 um circular pits\n'
                 '50 um measurement pixels; 20 x 20 full-field ROI; shared display scale within each backend', fontsize=14)
    path = RESULTS/f'{PREFIX}-brightness-comparison.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def cv_figure(rows, sampling):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout='constrained')
    x = np.arange(len(rows))
    axes[0].bar(x-.18, [r['old_cv_percent'] for r in rows], width=.36, label='Original', color='#90a4b8')
    axes[0].bar(x+.18, [r['new_cv_percent'] for r in rows], width=.36, label='Random pits', color='#168584')
    axes[0].set(xticks=x, xticklabels=[r['backend'] for r in rows], ylabel='Direct brightness CV (%)',
                title='Four-backend comparison | 1 um optical grid')
    peak = max(max(r['old_cv_percent'], r['new_cv_percent']) for r in rows)
    axes[0].set_ylim(0, peak*1.16)
    for i, row in enumerate(rows):
        axes[0].text(i, max(row['old_cv_percent'], row['new_cv_percent'])+peak*.025,
                     f'{row["delta_percentage_points"]:+.3f} pp', ha='center', fontsize=9)
    fine = sampling['fine_comparison']
    coarse = rows[0]
    width = .3
    for offset, label, key, color in ((-width/2, 'Original', 'old_cv_percent', '#90a4b8'),
                                      (width/2, 'Random pits', 'new_cv_percent', '#168584')):
        axes[1].bar(np.arange(2)+offset, [coarse[key], fine[key]], width=width, label=label, color=color)
    axes[1].set(xticks=[0, 1], xticklabels=['1 um', '0.5 um'], ylabel='Direct brightness CV (%)',
                title='Optical-grid sensitivity | Torch CUDA')
    fine_peak = max(coarse['old_cv_percent'], coarse['new_cv_percent'], fine['old_cv_percent'], fine['new_cv_percent'])
    axes[1].set_ylim(0, fine_peak*1.16)
    for i, row in enumerate((coarse, fine)):
        axes[1].text(i, max(row['old_cv_percent'], row['new_cv_percent'])+fine_peak*.025,
                     f'{row["delta_percentage_points"]:+.3f} pp', ha='center')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Identical OLED input, 0.5 um height range, 500 um gap, NA 0.08 and 50 um measurement pixels\n'
                 'pp = percentage-point change in CV; all 400 ROI pixels included', fontsize=12)
    path = RESULTS/f'{PREFIX}-cv-comparison.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def delta(old, new):
    return {'old_cv_percent': old, 'new_cv_percent': new,
            'delta_percentage_points': new-old, 'relative_change_percent': 100*(new/old-1)}


def main():
    global OLD, NEW, EMISSION, SAMPLING, PREFIX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true')
    parser.add_argument('--legacy-stretched', action='store_true')
    parser.add_argument('--legacy-whole-tile', action='store_true')
    args = parser.parse_args()
    periodic_case = not (args.legacy_stretched or args.legacy_whole_tile)
    if args.legacy_stretched:
        OLD = RESULTS/'oled-1mm-height1um-50um'
        NEW = RESULTS/'oled-1mm-pits3to5um-50um'
        EMISSION = SKILL/'assets/oled-1mm/emission.png'
        SAMPLING = RESULTS/'pit-sampling-0p5um'
        PREFIX = 'random-pit'
    elif args.legacy_whole_tile:
        OLD = RESULTS/'oled-1mm-original-tiled-50um'
        NEW = RESULTS/'oled-1mm-pits3to5um-tiled-50um'
        EMISSION = SKILL/'assets/oled-1mm-tiled/emission.png'
        SAMPLING = RESULTS/'pit-sampling-tiled-0p5um'
        PREFIX = 'random-pit-tiled'
    height_path = height_figure()
    emission_path = emission_figure() if periodic_case else None
    if args.inputs_only:
        print(height_path)
        return
    old_run, old = audit_run(OLD)
    new_run, new = audit_run(NEW)
    assert set(new) == set(NAMES)
    assert old_run['input_files']['emission']['sha256'] == new_run['input_files']['emission']['sha256']
    tiling_verified = None
    if not args.legacy_stretched:
        tile = np.asarray(Image.open(SKILL/'assets/oled-diamond/emission.png'))
        if periodic_case:
            cropped = tile[58:128, 26:96]
            tile = np.asarray(Image.open(SKILL/'assets/oled-1mm-periodic/periodic-cell.png'))
            np.testing.assert_array_equal(tile, cropped)
        expanded = np.asarray(Image.open(EMISSION))
        expected = tile[np.arange(500)[:, None] % tile.shape[0],
                        np.arange(500)[None, :] % tile.shape[1]]
        np.testing.assert_array_equal(expanded, expected)
        info = new_run['input_files']['emission']
        assert info['sha256'] == hashlib.sha256(EMISSION.read_bytes()).hexdigest()
        assert info['pitch_um_per_texel'] == 2
        assert info['provenance']['source_tile_field_um'] == ([140, 140] if periodic_case else [256, 256])
        assert info['provenance']['pixel_scale_changed'] is False
        tiling_verified = True
    excluded = {'height', 'out', 'device'}
    assert {k: v for k, v in old_run['parameters'].items() if k not in excluded} == {
        k: v for k, v in new_run['parameters'].items() if k not in excluded}
    rows = [{'backend': name, **delta(old[name]['cv_percent'], new[name]['cv_percent']),
             'old_brightness_mean': old[name]['brightness_mean'],
             'new_brightness_mean': new[name]['brightness_mean'],
             'mean_brightness_relative_change_percent': 100*(new[name]['brightness_mean']/old[name]['brightness_mean']-1),
             'new_solver_seconds': new[name]['solver_seconds'],
             'new_execution_context': new[name]['execution_context']} for name in NAMES]
    wave_errors = {}
    reference = np.load(NEW/'torch/analysis_brightness.npy')
    for name in ('torchoptics', 'chromatix'):
        other = np.load(NEW/name/'analysis_brightness.npy')
        error = float(np.linalg.norm(other-reference)/np.linalg.norm(reference))
        assert error < .001
        wave_errors[name] = error
    fine_runs, fine_values = {}, {}
    for label in ('original', 'pits'):
        fine_runs[label], results = audit_run(SAMPLING/label)
        assert fine_runs[label]['input_files']['emission']['sha256'] == new_run['input_files']['emission']['sha256']
        fine_values[label] = results['torch']['cv_percent']
    sampling = {'optical_pitch_um': .5, 'measurement_pitch_um': 50,
                'original_cv_percent': fine_values['original'], 'pits_cv_percent': fine_values['pits'],
                'fine_comparison': delta(fine_values['original'], fine_values['pits']),
                'coarse_to_fine_original_cv_change_pp': fine_values['original']-old['torch']['cv_percent'],
                'coarse_to_fine_pits_cv_change_pp': fine_values['pits']-new['torch']['cv_percent'],
                'note': 'Two grid levels measure sensitivity; they do not establish full convergence. '
                        'Original height is bilinearly refined from its 1um samples; pits are analytically resampled. '
                        'Physical FFT padding remains 140um per edge; emission-source positions follow solver grid quantization.'}
    figures = [height_path, brightness_figure(rows), cv_figure(rows, sampling)]
    if emission_path:
        figures.insert(0, emission_path)
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'baseline_results': str(OLD), 'new_results': str(NEW),
               'emission_field_construction': new_run['input_files']['emission']['provenance'],
               'supersedes_stretched_emission_comparison': not args.legacy_stretched,
               'supersedes_uncropped_tile_comparison': periodic_case,
               'baseline_input_files': old_run['input_files'],
               'baseline_effective_model': old_run['effective_model'],
               'height_rms_original_um': old_run['effective_model']['height_rms_about_mean_um'],
               'height_rms_pits_um': new_run['effective_model']['height_rms_about_mean_um'],
               'metric': '100 * std(all 400 area-averaged 50um brightness pixels, ddof=0) / mean',
               'geometry': audit_geometry(), 'comparison': rows, 'sampling_check': sampling,
               'wave_backend_relative_l2_brightness_50um': wave_errors,
               'audit': {'all_metrics_recomputed_from_saved_floats': True,
                         'measurement_pixels_match_independent_area_average': True,
                         'emission_bytes_unchanged': True, 'optical_parameters_unchanged': True,
                         'emission_expansion_preserves_repeat_cell_texels_and_scale': tiling_verified,
                         'all_four_backends_complete': True},
               'new_run_wall_seconds': new_run['wall_elapsed_seconds'],
               'figures': [str(p) for p in figures]}
    (RESULTS/f'{PREFIX}-comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    title = '随机圆坑 3–5 µm · 50% 覆盖率 · Sparkle 对比'
    if not args.legacy_stretched:
        title += ' · 固定尺度平铺发光图'
    if periodic_case:
        title += ' · 完整周期单元裁剪'
    report = write_html_report(NEW, title=title, figures=figures)
    table_rows = ''.join(f'<tr><td>{r["backend"]}</td><td>{r["old_cv_percent"]:.6f}%</td>'
                         f'<td>{r["new_cv_percent"]:.6f}%</td><td>{r["delta_percentage_points"]:+.6f}</td>'
                         f'<td>{r["relative_change_percent"]:+.4f}%</td>'
                         f'<td>{r["mean_brightness_relative_change_percent"]:+.4f}%</td></tr>' for r in rows)
    note = ('当前统计包含 OLED 发光图本身的不均匀性，CV 不单独分离表面引入的闪点。'
            '三种波动后端与 Mitsuba 几何模型分开解读。本次只比较一个固定随机种子。')
    section = ('<section id="surface-comparison"><h2>原高度图与随机圆坑的比较</h2>'
               '<p>直径 3–5 µm，坑深 0.5 µm，覆盖率 50.0001%，39,048 个互不重叠的平滑碗形圆坑。'
               '采用随机位置、均匀直径分布，种子 20261107。以下主比较使用 1 µm 光学网格。</p>'
               '<div class="scroll"><table><thead><tr><th>后端</th><th>原高度 CV</th><th>随机圆坑 CV</th>'
               '<th>变化（百分点）</th><th>CV 相对变化</th><th>平均亮度相对变化</th></tr></thead><tbody>'+table_rows+'</tbody></table></div>'
               f'<p>0.5 µm 网格复核（Torch）：原高度 {fine_values["original"]:.6f}%，'
               f'随机圆坑 {fine_values["pits"]:.6f}%，变化 {sampling["fine_comparison"]["delta_percentage_points"]:+.6f} 个百分点。'
               '两级网格用于检查采样敏感性，尚未证明完全收敛。</p><p class="note">'+escape(note)+'</p></section>')
    if periodic_case:
        section = ('<section><h2>完整周期单元裁剪</h2><p>先从原 128 × 128 图块的 '
                   '[26,58,96,128] 矩形裁出 70 × 70 单元，保留 2 µm/texel，单元尺寸为 140 × 140 µm。'
                   '单元内包含 2 红、4 绿、2 蓝完整高亮区域；高亮主体不跨裁剪边界，'
                   '跨边界的相邻像素变化小于单元内部的相邻变化。无边缘混合、补色或重画。'
                   '8 × 8 整数重复后裁至 500 × 500，得到 1 mm × 1 mm 发光图。'
                   '此前直接重复整块 128 × 128 裁图的结果已被替代。</p></section>')+section
    elif args.legacy_whole_tile:
        section = ('<section><h2>发光图尺度修正</h2><p>原 128 × 128 发光纹理保持 2 µm/texel，'
                   '每块覆盖 256 × 256 µm；4 × 4 整数平铺到 512 × 512，再从左上裁取 500 × 500，'
                   '得到 1 mm × 1 mm 区域。扩展时没有缩放或插值，逐 texel 保持原值。'
                   '此前把同一内容拉伸到 1 mm 的比较已被本次计算替代。</p>'
                   '<p>原纹理是示意图裁剪块，平铺保留其图块边界，未重建经过标定的 OLED 原胞。</p></section>')+section
    document = report.read_text(encoding='utf-8')
    document = document.replace('<section id="input-images">', section+'<section id="input-images">', 1)
    payload = json.dumps(summary, ensure_ascii=False).replace('<', '\\u003c')
    document = document.replace('</body>', f'<script id="surface-comparison-data" type="application/json">{payload}</script></body>')
    report.write_text(document, encoding='utf-8')
    print(json.dumps({'report': str(report), 'comparison': rows, 'sampling_check': sampling}, indent=2))


if __name__ == '__main__':
    main()

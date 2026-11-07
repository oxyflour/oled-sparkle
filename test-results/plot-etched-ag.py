"""Plot the etched-AG experiment, audit saved floats and build an offline report."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / 'oled-ag-sparkle'
RESULTS = ROOT / 'test-results/oled-etched-ag'
sys.path.insert(0, str(SKILL / 'scripts'))
from reporting import write_html_report
from sparkle_multibackend import collect_analysis_pixels

BACKENDS = ('torch', 'torchoptics', 'chromatix', 'mitsuba')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def surface_plot():
    folder = SKILL / 'assets/oled-1mm-etched-ag'
    meta = read_json(folder / 'heightfield.source.json')
    height = np.load(folder / 'height_um.npy') * 1000
    relative = height - height.mean()
    low, high = relative.min(), relative.max()
    fig = plt.figure(figsize=(15, 8.8), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=(1.12, .8))
    ax = fig.add_subplot(grid[0, 0])
    im = ax.imshow(relative, extent=(0, 1000, 1000, 0), cmap='gray', vmin=low, vmax=high)
    ax.set(title='1 mm field | 1,600 irregular bowls', xlabel='x [um]', ylabel='y [um]')
    ax.add_patch(plt.Rectangle((400, 400), 200, 200, fill=False, edgecolor='#df7639', lw=1.5))
    fig.colorbar(im, ax=ax, label='Height relative to mean [nm]', shrink=.8)
    ax = fig.add_subplot(grid[0, 1])
    im = ax.imshow(relative[400:600, 400:600], extent=(400, 600, 600, 400),
                   cmap='gray', vmin=low, vmax=high)
    ax.set(title='200 x 200 um detail', xlabel='x [um]', ylabel='y [um]')
    fig.colorbar(im, ax=ax, label='Height [nm]', shrink=.8)
    ax = fig.add_subplot(grid[0, 2], projection='3d')
    axis = np.arange(440, 560) + .5
    xx, yy = np.meshgrid(axis, axis)
    ax.plot_surface(xx, yy, relative[440:560, 440:560], cmap='viridis',
                    rcount=120, ccount=120, linewidth=0, antialiased=True)
    ax.set(title='120 um detail | height axis exaggerated', xlabel='x [um]', ylabel='y [um]', zlabel='Height [nm]')
    ax.set_box_aspect((1, 1, .6))
    ax.view_init(37, -53)
    ax = fig.add_subplot(grid[1, :2])
    for row, color in zip((425, 500, 575), ('#167e98', '#db7738', '#557643')):
        ax.plot(np.arange(400, 600) + .5, relative[row, 400:600], color=color, lw=1.2, label=f'y = {row + .5} um')
    ax.set(title='Surface profiles through the detail region', xlabel='x [um]', ylabel='Height relative to mean [nm]')
    ax.legend(ncol=3, frameon=False, fontsize=9)
    ax.grid(alpha=.18)
    ax = fig.add_subplot(grid[1, 2])
    ax.hist(relative.ravel(), bins=65, color='#368496', density=True)
    surf = meta['measured_surface']
    ax.set(title=f"Sa = {surf['Sa_um'] * 1000:.1f} nm | Sq = {surf['Sq_um'] * 1000:.1f} nm",
           xlabel='Height relative to mean [nm]', ylabel='Probability density [1/nm]')
    fig.suptitle('Synthetic etched-AG proxy: random polygonal bowls with rounded rims', fontsize=17)
    fig.supxlabel('1 um/texel, 16-bit PNG; nominal site spacing 25 um; 1.2 um rounding; illustrative, not measured glass', fontsize=10)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / 'surface-overview.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def brightness_plot():
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout='constrained')
    for column, backend in enumerate(BACKENDS):
        folder = RESULTS / '1um' / backend
        stats = read_json(folder / 'stats.json')
        for row, filename in enumerate(('brightness.npy', 'analysis_brightness.npy')):
            values = np.load(folder / filename)
            im = axes[row, column].imshow(values, extent=(0, 1000, 1000, 0), cmap='inferno',
                                           vmin=0, vmax=values.max(), interpolation='nearest')
            fig.colorbar(im, ax=axes[row, column], shrink=.76, format='%.2g')
            axes[row, column].set(xlabel='x [um]', ylabel='y [um]')
        axes[0, column].set_title(f'{backend}\n1 um optical grid')
        axes[1, column].set_title(f"50 um pixels | CV {stats['sparkle_cv_percent']:.4f}%")
    fig.suptitle('Collected OLED brightness through the synthetic etched surface', fontsize=17)
    fig.supxlabel('Linear intensity, own scale per panel. CV uses all 400 float measurement pixels; no OLED-pattern removal.', fontsize=10)
    path = RESULTS / 'brightness-overview.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def audit():
    records = []
    arrays = {}
    for grid, backends, pitch in (('1um', BACKENDS, 1.), ('0p5um', ('torch',), .5)):
        run = read_json(RESULTS / grid / 'run.json')
        assert run['status'] == 'complete'
        for role in ('height', 'emission'):
            info = run['input_files'][role]
            path = Path(info['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
            if role == 'height':
                assert path.read_bytes()[24:26] == bytes((16, 0))
        for backend in backends:
            folder = RESULTS / grid / backend
            stats = read_json(folder / 'stats.json')
            raw = np.load(folder / 'brightness.npy')
            values = np.load(folder / 'analysis_brightness.npy')
            assert np.isfinite(raw).all() and np.all(raw >= 0)
            assert values.shape == (20, 20)
            assert np.load(folder / 'roi_mask.npy').all()
            integrated, _ = collect_analysis_pixels(raw, pitch, 50., 0.)
            np.testing.assert_allclose(values, integrated, rtol=1e-12, atol=1e-14)
            cv = 100 * values.std(ddof=0) / values.mean()
            assert abs(cv - stats['sparkle_cv_percent']) < 1e-10
            assert stats['num_roi_pixels'] == 400
            assert run['backend_runs'][backend]['status'] == 'complete'
            context = stats['execution_context']
            if backend in ('torch', 'torchoptics'):
                assert str(context['device']).startswith('cuda')
            elif backend == 'chromatix':
                assert context['runtime'] == 'wsl' and context['default_backend'] == 'gpu'
            else:
                assert context['variant'] == 'cuda_ad_rgb'
            arrays[(grid, backend)] = raw
            records.append({'grid': grid, 'backend': backend, 'CV_percent': float(cv),
                            'solver_seconds': stats['elapsed_seconds'],
                            'actual_execution_context': stats['execution_context']})
    coarse_meta = read_json(SKILL / 'assets/oled-1mm-etched-ag/heightfield.source.json')
    fine_meta = read_json(SKILL / 'assets/oled-1mm-etched-ag-pitch0p5/heightfield.source.json')
    assert coarse_meta['calibration'] == fine_meta['calibration']
    assert (SKILL / 'assets/oled-1mm-etched-ag/sites.csv').read_bytes() == (
        SKILL / 'assets/oled-1mm-etched-ag-pitch0p5/sites.csv').read_bytes()
    for meta in (coarse_meta, fine_meta):
        assert abs(meta['measured_surface']['Sa_um'] - .1) < .002
    coarse = next(row['CV_percent'] for row in records if row['grid'] == '1um' and row['backend'] == 'torch')
    fine = next(row['CV_percent'] for row in records if row['grid'] == '0p5um')
    wave_errors = {}
    ref = arrays[('1um', 'torch')]
    for backend in ('torchoptics', 'chromatix'):
        error = float(np.linalg.norm(arrays[('1um', backend)] - ref) / np.linalg.norm(ref))
        assert error < 1e-3
        wave_errors[backend] = error
    emission_hashes = [read_json(RESULTS / grid / 'run.json')['input_files']['emission']['sha256']
                       for grid in ('1um', '0p5um')]
    assert len(set(emission_hashes)) == 1
    summary = {'status': 'passed', 'surface': coarse_meta, 'runs': records,
               'wave_relative_L2_vs_torch': wave_errors,
               'sampling_check': {'coarse_CV_percent': coarse, 'fine_CV_percent': fine,
                                  'difference_percentage_points': fine - coarse,
                                  'relative_difference_percent': 100 * (fine / coarse - 1),
                                  'scope': 'Two-level optical-grid sensitivity only; source sampling, padding, ray mesh and pupil convergence are not established.'},
               'metric': '100*std(I, ddof=0)/mean(I), directly from all 400 area-averaged 50 um pixels',
               'verified': ['16-bit PNG input hashes', 'unchanged periodic OLED emission',
                            'same surface geometry and calibration across grids',
                            'independent area integration', 'CV from saved floats',
                            'three-wave-backend agreement', 'all five computations complete']}
    test_log = ROOT / 'test-results/logs/validation-etched-ag.log'
    if test_log.is_file():
        log_bytes = test_log.read_bytes()
        log = log_bytes.decode('utf-16' if log_bytes.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        assert 'Ran 31 tests' in log and 'OK (skipped=4)' in log
        summary['regression_tests'] = {'run': 31, 'passed': 27, 'skipped': 4,
                                       'log': str(test_log),
                                       'note': 'Optional integration tests skipped in the CPU test environment; four production backends were run and audited separately on GPU.'}
    (RESULTS / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def finalize(surface, brightness, summary):
    report = write_html_report(RESULTS, recursive=True,
                              title='化学蚀刻 AG 示意表面 · OLED 亮度仿真', figures=[surface, brightness])
    document = report.read_text(encoding='utf-8')
    sampling = summary['sampling_check']
    model = f'''<section id="surface-model"><h2>本次生成的表面与结果解释</h2>
    <p>采用 1 mm × 1 mm 周期区域内的 1,600 个随机 Voronoi 浅坑；名义中心间距 25 µm，
    单元等面积直径均值 27.56 µm，坑深系数独立变化，坑沿用 1.2 µm 高斯尺度平滑。
    面粗糙度 Sa = 0.100 µm、Sq ≈ 0.119 µm、实际采样峰谷差 Sz ≈ 0.515 µm。
    PNG 的黑至白编码范围约 0.536185 µm，与实际 Sz 分别记录。</p>
    <p>这是参考蚀刻 AG 不规则形貌及公开粗糙度量级建立的合成模型。没有拟合商业样品的实测形貌；
    生成的 Sa 和单元等面积直径，分别不等同于厂家数据表中的轮廓 Ra 和 Rsm。
    形貌参考：<a href="https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf">SID 2025，图 2</a>；
    粗糙度量级参考：<a href="https://www.agc-yourglass.com/sites/default/files/agc_docs/Factsheet_Anti-glare_1020_EN.pdf">AGC 显示玻璃数据表</a>。</p>
    <p>保留 OLED 重复单元及 2 µm 输入采样，采用 500 µm 间距、530 nm 有效波长、NA 0.08；
    直接统计 20 × 20 个 50 µm 测量像素，包含 OLED 图案本身的不均匀性。</p>
    <p>同一几何的 Torch 光学网格检查：1 µm 为 {sampling['coarse_CV_percent']:.6f}%，
    0.5 µm 为 {sampling['fine_CV_percent']:.6f}%，变化 {sampling['difference_percentage_points']:+.6f} 个百分点。
    此检查只衡量两级光学网格的敏感性；光源采样、FFT 边界与光线积分的收敛性仍有限。
    三个波动后端实现同一标量模型；Mitsuba 采用几何光线模型，结果应分别解读。</p></section>'''
    document = document.replace('<main>', '<main>' + model, 1)
    report.write_text(document, encoding='utf-8')
    # Audit portable input downloads against their original bytes, including uint16.
    for group in ('1um', '0p5um'):
        meta = read_json(RESULTS / group / 'run.json')
        for role in ('height', 'emission'):
            data = base64.b64encode(Path(meta['input_files'][role]['path']).read_bytes()).decode('ascii')
            assert data in document
    for download in re.findall(r'href="data:application/octet-stream;base64,([^\"]+)"', document):
        assert base64.b64decode(download).startswith(b'\x93NUMPY')
    assert len(re.findall(r'href="data:application/octet-stream;base64,', document)) == 10
    summary['verified'].append('offline HTML preserves original PNG bytes and all 10 NumPy downloads')
    (RESULTS / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    validation_path = SKILL / 'VALIDATION.json'
    validation = read_json(validation_path)
    validation['latest_experiment_at_utc'] = read_json(RESULTS / '0p5um/run.json')['finished_at_utc']
    validation['latest_results_report'] = 'test-results/oled-etched-ag/report.html'
    validation['etched_ag_proxy'] = {
        'report': 'test-results/oled-etched-ag/report.html',
        'audit': 'test-results/oled-etched-ag/validation.json',
        'height_asset': 'oled-ag-sparkle/assets/oled-1mm-etched-ag/heightfield.png',
        'synthetic_model': summary['surface']['model_status'],
        'height_sha256': summary['surface']['sha256'],
        'surface_statistics': summary['surface']['measured_surface'],
        'brightness_results': [{k: row[k] for k in ('grid', 'backend', 'CV_percent', 'solver_seconds')}
                               for row in summary['runs']],
        'sampling_check': summary['sampling_check'],
        'wave_relative_L2_vs_torch': summary['wave_relative_L2_vs_torch'],
        'verified': summary['verified'],
    }
    if 'regression_tests' in summary:
        validation['latest_tests'] = summary['regression_tests']
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Report:', report)
    print(json.dumps({'runs': [{k: row[k] for k in ('grid', 'backend', 'CV_percent', 'solver_seconds')}
                               for row in summary['runs']], 'sampling': summary['sampling_check']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    surface = surface_plot()
    if args.inputs_only:
        print(surface)
        return
    brightness = brightness_plot()
    summary = audit()
    finalize(surface, brightness, summary)


if __name__ == '__main__':
    main()

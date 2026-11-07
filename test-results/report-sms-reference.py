"""Audit SMS-reference calculations and generate a self-contained comparison."""
from pathlib import Path
import base64
import hashlib
from html import escape
import json
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / 'oled-ag-sparkle'
OUT = ROOT / 'test-results/oled-sms-reference'
sys.path.insert(0, str(SKILL / 'scripts'))
from reporting import CSS, write_html_report
from sms_reference import analyze, flat_wave_control
from sparkle_multibackend import model_from_args, parse_args

BACKENDS = ('torch', 'torchoptics', 'chromatix', 'mitsuba')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze_saved():
    records = []
    arrays = {}
    source_hashes = set()
    for grid, names, pitch in (('1um', BACKENDS, 1.), ('0p5um', ('torch',), .5),
                                ('old-na-0p08', ('torch',), 1.)):
        run = read(OUT / grid / 'run.json')
        assert run['status'] == 'complete'
        assert run['parameters']['emission_channel'] == 'green'
        for role in ('height', 'emission'):
            info = run['input_files'][role]
            assert hashlib.sha256(Path(info['path']).read_bytes()).hexdigest() == info['sha256']
            if role == 'emission':
                source_hashes.add(info['sha256'])
        for backend in names:
            folder = OUT / grid / backend
            raw = np.load(folder / 'brightness.npy')
            assert np.isfinite(raw).all() and np.all(raw >= 0)
            result = analyze(raw, pitch)
            data = result['summary']
            direct = read(folder / 'stats.json')
            context = direct['execution_context']
            if backend in ('torch', 'torchoptics'):
                assert str(context['device']).startswith('cuda')
            elif backend == 'chromatix':
                assert context['default_backend'] == 'gpu' and context['runtime'] == 'wsl'
            else:
                assert context['variant'] == 'cuda_ad_rgb'
            assert abs(data['direct_unseparated_camera_CV_percent'] - direct['sparkle_cv_percent']) < 1e-9
            pixels = result['measurement_pixels']
            cv = 100 * pixels.std(ddof=0) / pixels.mean()
            assert abs(cv - data['primary']['sparkle_contrast_percent']) < 1e-10
            mwa_pixels = result['mwa_measurement_pixels']
            mwa_cv = 100 * mwa_pixels.std(ddof=0) / mwa_pixels.mean()
            assert abs(mwa_cv-data['sensitivities'][2]['sparkle_contrast_percent']) < 1e-10
            assert pixels.shape == (21, 21)
            assert abs(data['primary_filter']['mean_preservation_error']) < 1e-12
            for filename, values in (('sms_filtered_optical_roi.npy', result['separated']),
                                     ('sms_measurement_pixels.npy', pixels),
                                     ('sms_frequency_mask.npy', result['transfer']),
                                     ('sms_mwa_optical_roi.npy', result['mwa_filtered_optical_roi']),
                                     ('sms_mwa_measurement_pixels.npy', mwa_pixels)):
                np.save(folder / filename, values)
            data.update({'grid': grid, 'backend': backend, 'object_NA': run['parameters']['na'],
                         'physics': direct['physics_group'], 'solver_seconds': direct['elapsed_seconds'],
                         'measured_ROI_global_um_xyxy': [-275.625, -275.625, 275.625, 275.625],
                         'filtered_ROI_global_um_xyxy': [-280., -280., 280., 280.]})
            (folder / 'sms_stats.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
            arrays[(grid, backend)] = result
            records.append(data)
    assert len(source_hashes) == 1
    for grid in ('1um', 'old-na-0p08'):
        run = read(OUT / grid / 'run.json')
        model = model_from_args(parse_args(run['argv']))
        control = flat_wave_control(model)
        result = analyze(control, model.dx_m * 1e6)
        folder = OUT / 'controls' / grid
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'brightness.npy', control)
        np.save(folder / 'sms_measurement_pixels.npy', result['measurement_pixels'])
        (folder / 'sms_stats.json').write_text(json.dumps(result['summary'], indent=2), encoding='utf-8')
        arrays[(grid, 'flat')] = result
    return records, arrays


def figures(records, arrays):
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), layout='constrained')
    for column, backend in enumerate(BACKENDS):
        result = arrays[('1um', backend)]
        data = result['summary']
        raw = result['optical_roi']
        im = axes[0, column].imshow(raw, extent=(-280, 280, 280, -280), cmap='gray', interpolation='nearest')
        fig.colorbar(im, ax=axes[0, column], shrink=.7, format='%.2g')
        axes[0, column].set(title=f'{backend}\ngreen-only optical ROI', xlabel='x [um]', ylabel='y [um]')
        values = result['measurement_pixels']
        deviation = 100 * (values / values.mean()-1)
        limit = float(np.max(np.abs(deviation)))
        im = axes[1, column].imshow(deviation, extent=(-275.625, 275.625, 275.625, -275.625),
                                     cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        fig.colorbar(im, ax=axes[1, column], shrink=.7, label='Deviation [%]')
        axes[1, column].set(xlabel='x [um]', ylabel='y [um]')
        axes[1, column].set_title(f"Fourier CV = {data['primary']['sparkle_contrast_percent']:.3f}%", fontsize=11)
        values = result['mwa_measurement_pixels']
        deviation = 100*(values/values.mean()-1)
        limit = float(np.max(np.abs(deviation)))
        im = axes[2, column].imshow(deviation, extent=(-275.625, 275.625, 275.625, -275.625),
                                     cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        fig.colorbar(im, ax=axes[2, column], shrink=.7, label='Deviation [%]')
        axes[2, column].set(xlabel='x [um]', ylabel='y [um]')
        axes[2, column].set_title(f"3x MWA CV = {data['sensitivities'][2]['sparkle_contrast_percent']:.3f}%", fontsize=11)
    fig.suptitle('Green OLED + existing etched AG | no LDF | predeclared Fourier method', fontsize=16)
    fig.supxlabel('F/8, 400 mm object distance, object NA 0.007812; virtual detector pitch 26.25 um. Independent scales per panel.', fontsize=10)
    main = OUT / 'sms-brightness-comparison.png'
    fig.savefig(main, dpi=160)
    plt.close(fig)

    wave = next(x for x in records if x['grid'] == '1um' and x['backend'] == 'torch')
    labels = ['Fourier\nprimary', 'Fourier\nnotch guard', 'Fourier\n0.8x cutoff', '3x MWA\n140 um']
    values = [wave['primary']['sparkle_contrast_percent'],
              *[x['sparkle_contrast_percent'] for x in wave['sensitivities']]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout='constrained')
    axes[0].axhspan(4.8, 6., color='#e9bc69', alpha=.35, label='Cheng 2025 no-LDF readings: 4.8-6.0')
    bars = axes[0].bar(labels, values, color=['#187e99', '#639eaa', '#84b2b9', '#5d8561'])
    axes[0].bar_label(bars, fmt='%.3f%%', padding=4)
    axes[0].set(title='Same brightness, different evaluation bandwidths', ylabel='Sparkle contrast [%]', ylim=(0, max(values)*1.22))
    axes[0].legend(fontsize=8, loc='upper right')
    result = arrays[('1um', 'torch')]
    transfer = np.fft.fftshift(result['transfer'])
    n = transfer.shape[0]
    c = n//2
    axes[1].imshow(transfer[c-12:c+13, c-12:c+13], cmap='gray', origin='lower',
                   extent=(-12.5/560, 12.5/560, -12.5/560, 12.5/560), interpolation='nearest')
    axes[1].set(title='Predeclared mask: white = retained\nDC retained; 140 um lattice bins removed', xlabel='fx [cycles/um]', ylabel='fy [cycles/um]')
    diagnostic = OUT / 'filter-sensitivity.png'
    fig.savefig(diagnostic, dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout='constrained')
    deviations = [100*(arrays[('1um', name)]['mwa_measurement_pixels'] /
                        arrays[('1um', name)]['mwa_measurement_pixels'].mean()-1) for name in BACKENDS]
    limit = max(float(np.max(np.abs(values))) for values in deviations)
    for ax, name, values in zip(axes, BACKENDS, deviations):
        im = ax.imshow(values, extent=(-275.625, 275.625, 275.625, -275.625),
                       cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        stat = arrays[('1um', name)]['summary']['sensitivities'][2]
        ax.set(title=f"{name}\nSparkle {stat['sparkle_contrast_percent']:.3f}%", xlabel='x [um]', ylabel='y [um]')
    fig.colorbar(im, ax=axes, shrink=.85, label='Random brightness deviation [%]')
    fig.suptitle('Green OLED + AG, no LDF | 3 x 140 um moving averages', fontsize=14)
    sparkle_plot = OUT / 'mwa-sparkle.png'
    fig.savefig(sparkle_plot, dpi=150)
    plt.close(fig)
    return main, diagnostic, sparkle_plot


def row_for(record):
    mwa = record['sensitivities'][2]['sparkle_contrast_percent']
    return f"<tr><td>{record['grid']}</td><td>{record['backend']}</td><td>{record['object_NA']:.7f}</td><td>{record['direct_unseparated_camera_CV_percent']:.6f}</td><td>{record['primary']['sparkle_contrast_percent']:.6f}</td><td>{mwa:.6f}</td></tr>"


def report(records, arrays, plots):
    coarse = next(x for x in records if x['grid'] == '1um' and x['backend'] == 'torch')
    fine = next(x for x in records if x['grid'] == '0p5um')
    flat = arrays[('1um', 'flat')]['summary']
    baseline = flat['primary']['sparkle_contrast_percent']
    baseline_mwa = flat['sensitivities'][2]['sparkle_contrast_percent']
    summary = {'status': 'passed', 'protocol': read(OUT / 'protocol.json'),
               'protocol_sha256': hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest(),
               'records': records,
               'flat_control': {'primary_Fourier_percent': baseline, 'three_MWA_percent': baseline_mwa,
                                'not_used_for_division_or_subtraction': True},
               'sampling_check': {'primary_difference_percentage_points': fine['primary']['sparkle_contrast_percent']-coarse['primary']['sparkle_contrast_percent'],
                                  'MWA_difference_percentage_points': fine['sensitivities'][2]['sparkle_contrast_percent']-coarse['sensitivities'][2]['sparkle_contrast_percent']},
               'scope': 'SMS-inspired reference implementation. Different methods have different retained bandwidths; equality is not expected. Agreement with target in one method is not measurement calibration.',
               'verified': ['all six optical runs complete on GPU', 'same green G bytes and height hashes',
                            'separated-image mean retained', '441 measurement pixels all included',
                            'saved measurement floats independently reproduce CV',
                            'fast flat limit independently tested against wave propagation']}
    summary['processing_software'] = {'python': sys.version, 'executable': sys.executable,
                                      'numpy': np.__version__, 'scipy': scipy.__version__,
                                      'matplotlib': matplotlib.__version__}
    summary['processing_code'] = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                  for name, path in [('analysis', SKILL / 'scripts/sms_reference.py'),
                                                     ('runner', SKILL / 'scripts/run_sms_reference.py'),
                                                     ('report', Path(__file__).resolve())]}
    wave_reference = arrays[('1um', 'torch')]['measurement_pixels']
    wave_errors = {}
    for name in ('torchoptics', 'chromatix'):
        measured = arrays[('1um', name)]['measurement_pixels']
        error = float(np.linalg.norm(measured-wave_reference)/np.linalg.norm(wave_reference))
        assert error < 1e-3
        wave_errors[name] = error
    summary['wave_relative_L2_vs_torch'] = wave_errors
    log = ROOT / 'test-results/logs/validation-sms-reference.log'
    if log.is_file():
        data = log.read_bytes()
        content = data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        match = re.search(r'Ran (\d+) tests', content)
        if match and 'OK (skipped=4)' in content:
            summary['regression_tests'] = {'status': 'passed', 'run': int(match.group(1)), 'skipped': 4,
                                           'passed': int(match.group(1))-4, 'log': str(log)}
        elif 'FAILED' in content:
            raise AssertionError('Regression test log reports a failure')
        else:
            summary['regression_tests'] = {'status': 'uncomputed_or_still_running', 'log': str(log)}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    target = write_html_report(OUT, output=OUT / 'sparkle-report.html', recursive=True,
                              title='Sparkle · 全绿 / 无 LDF', figures=plots[:2])
    document = target.read_text(encoding='utf-8')
    document = document.replace('闪点值 CV (%)', '分离前总亮度 CV (%)').replace('闪点值：', '分离前 CV：')
    document = document.replace('主指标直接由亮度图 ROI 像素计算：100 × std(I) / mean(I)，单位 %；',
                                '本次分离后的 sparkle 见下方 SMS 参考测量；原始计算表另列分离前 CV。')
    document = document.replace('四种算法收集到的亮度图 · 闪点统计像素', '原始光学计算 · 分离前测量像素')
    document = document.replace('闪点值与计算时间', '原始计算 CV 与耗时')
    header = f'''<section id="sms-reference"><h2>主结果：全绿、无 LDF、分离 OLED 周期调制</h2>
    <p>计算前固定的主方案为频域分离：保留 DC，去除 140 µm 发光强度重复单元的离散周期频率，
    再限制在绿色中心基频 1/70 µm⁻¹ 以下。使用中心 560 µm 方形区域，
    随后按 26.25 µm 虚拟探测像素作面积平均，全部 21 × 21 = 441 个像素直接计算 100 σ/μ。</p>
    <p><strong>主方案 Torch：{coarse['primary']['sparkle_contrast_percent']:.4f}%；
    三次完整周期移动平均：{coarse['sensitivities'][2]['sparkle_contrast_percent']:.4f}%。</strong>
    两种方法的随机信号频带不同；这里只报告比较，不以接近文献区间作为选取滤波器的依据。
    原始计算块中的 CV 是分离前的总亮度 CV，与本节指标分列。</p>
    <p>公开方法文献 3.4 节讨论三次 MWA，结论列出多次 MWA 且窗口 w=2/f₀ 的例子。
    取绿色中心几何基频 f₀=1/70 µm⁻¹ 时，w=140 µm，亦等于本输入的完整强度重复周期。
    三次 MWA 有 sinc 型平滑频响，其随机信号通带比本次频域主方案更窄；两者分离周期后仍可能给出不同 CV。</p>
    <div class="scroll"><table><thead><tr><th>网格/工况</th><th>后端</th><th>物方 NA</th><th>分离前 CV (%)</th>
    <th>频域主方案 (%)</th><th>三次 140 µm MWA (%)</th></tr></thead><tbody>
    {''.join(row_for(x) for x in records)}</tbody></table></div>
    <p>波动模型平面控制：频域 {baseline:.6f}%，MWA {baseline_mwa:.6f}%。
    控制结果用于检查残余周期和边界影响，不用于除法、相减或改变 AG 测量像素。
    同几何细网格变化：频域 {summary['sampling_check']['primary_difference_percentage_points']:+.6f} 个百分点，
    MWA {summary['sampling_check']['MWA_difference_percentage_points']:+.6f} 个百分点。</p>
    <h3>固定参考条件与公开方法的边界</h3>
    <p>只保留原始发光图 G 通道，R/B 置零；G 像素逐字节保留。530 nm 单波长，OLED 到 AG 距离 500 µm，
    高度图保持 Sa=0.10 µm、名义中心间距 25 µm，无 LDF。全绿强度的完整重复周期为 140 µm，
    四个绿色主体的几何中心间距约 70 µm，但其形状不完全相同。</p>
    <p>参考相机为 50 mm、F/8、400 mm 物距、3.75 µm 传感器像素；薄透镜近似给出倍率 1/7、
    物方像素 26.25 µm、绿色像素 ISR≈2.667，物方 NA≈0.007812。
    这些参数在计算前固定，属于公开范围内的参考选择，并非 Cheng 论文未公布的实际设置。
    当前求解器使用理想物方孔径；频域处理发生在光学网格，再积分至探测像素，省略了滤波前传感器混叠。
    Mitsuba 为无衍射的几何光线模型，小孔径下尤其需要与标量波动结果分别解读。</p>
    <p>频域掩模是线性操作，不能保证分离后的信号始终非负。所有分离后样本均原样纳入面积积分和 CV，
    不作截断或亮度选择；每组 sms_stats.json 记录负值像素数。实际原始光强仍要求全部非负。</p>
    <p>Cheng 2025 无 LDF 读数 4.8–6.0 按 SMS 通常百分比标度解读；原表未明示百分号或算法设置。
    其表面 Ra=0.25–0.35 µm、雾度 20–40%，与这里的合成 Sa=0.10 µm 不同。
    1/70 截止、离散 notch 和三次 140 µm MWA 均为本次明确选择，未声称复现专有软件或满足 IEC 标准。</p>
    <p>方法依据：<a href="https://www.display-messtechnik.de/fileadmin/template/main/docs/10-Years-SparkleMeasurement--edC-2021-V2.pdf">Becker 2021，3.3–3.5 节</a>；
    相机范围：<a href="https://www.display-messtechnik.de/fileadmin/template/main/docs/SMS1000-05.2023-7.pdf">SMS-1000 公开规格</a>；
    对比数据：<a href="https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf">Cheng 2025，表 3</a>。</p>
    <h3>滤波与带宽敏感性：Torch 1 µm</h3><div class="scroll"><table><thead><tr><th>方案</th><th>结果 (%)</th></tr></thead><tbody>
    {''.join('<tr><td>'+escape(x['case'])+'</td><td>'+format(x['sparkle_contrast_percent'],'.6f')+'</td></tr>' for x in coarse['sensitivities'])}
    </tbody></table></div><h3>分离后原始浮点下载</h3>'''
    for record in records:
        folder = OUT / record['grid'] / record['backend']
        header += f"<p>{record['grid']} / {record['backend']}: "
        for filename in ('sms_filtered_optical_roi.npy', 'sms_measurement_pixels.npy', 'sms_frequency_mask.npy',
                         'sms_mwa_optical_roi.npy', 'sms_mwa_measurement_pixels.npy'):
            payload = base64.b64encode((folder / filename).read_bytes()).decode('ascii')
            header += f'<a href="data:application/octet-stream;base64,{payload}" download="{record["grid"]}-{record["backend"]}-{filename}">{filename}</a> · '
        header += '</p>'
    header += '</section>'
    header += '<details class="group"><summary>分离后处理软件与代码哈希</summary><div class="content"><pre>'+escape(
        json.dumps({'software': summary['processing_software'], 'code': summary['processing_code'],
                    'protocol_sha256': summary['protocol_sha256']}, ensure_ascii=False, indent=2))+'</pre></div></details>'
    mwa = coarse['sensitivities'][2]['sparkle_contrast_percent']
    mwa_fine = fine['sensitivities'][2]['sparkle_contrast_percent']
    plot_data = base64.b64encode(plots[2].read_bytes()).decode('ascii')
    front = f'''<section id="sparkle-result"><h2>Sparkle · 全绿、无 LDF</h2>
    <div class="card"><small>显示口径：三次 140 µm 移动平均（MWA），分离 OLED 周期纹理后计算</small>
    <strong>{mwa:.3f}%</strong><p>同一 AG 表面；全部 441 个测量像素计算随机亮度的标准差 / 均值。</p></div>
    <table><thead><tr><th>后端</th><th>Sparkle (%) · 1 µm 网格</th></tr></thead><tbody>
    {''.join('<tr><td>'+x['backend']+'</td><td>'+format(x['sensitivities'][2]['sparkle_contrast_percent'],'.3f')+'</td></tr>' for x in records if x['grid']=='1um')}
    </tbody></table><p>0.5 µm 网格复核：<strong>{mwa_fine:.3f}%</strong>；
    平面控制残余：{baseline_mwa:.3f}%。结果处于文献无 LDF 读数 4.8–6.0 的量级。</p>
    <img style="width:100%;height:auto" src="data:image/png;base64,{plot_data}" alt="去周期后闪点分布">
    <p>参考测量条件：50 mm、F/8、400 mm 物距，26.25 µm 物方测量像素，530 nm。
    这是固定 MWA 口径下的参考实现值，尚未按原论文仪器设置完成实测校准。</p></section>'''
    appendix_open = '<details class="group"><summary>方法对照、原始统计与复现数据（附录）</summary><div class="content">'
    document = re.sub(r'<header>.*?</header>', '<header><h1>全绿、无 LDF：Sparkle</h1><p>去除 OLED 周期纹理后的随机亮度起伏。</p></header>', document, count=1, flags=re.S)
    document = document.replace('<main>', '<main>'+front+appendix_open+header, 1)
    document = document.replace('</main>', '</div></details></main>', 1)
    target.write_text(document, encoding='utf-8')
    # Check self-contained original 16-bit PNGs and all filtered-data downloads.
    for grid in ('1um', '0p5um', 'old-na-0p08'):
        meta = read(OUT / grid / 'run.json')
        for role in ('height', 'emission'):
            encoded = base64.b64encode(Path(meta['input_files'][role]['path']).read_bytes()).decode('ascii')
            assert encoded in document
    downloads = re.findall(r'href="data:application/octet-stream;base64,([^\"]+)"', document)
    assert len(downloads) == 42
    assert all(base64.b64decode(value).startswith(b'\x93NUMPY') for value in downloads)
    summary['verified'].append('offline report preserves input PNGs and all 42 NumPy downloads, including MWA floats')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # A small entry page keeps the inline NumPy downloads in the full report.
    overview = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    overview += '<title>全绿无 LDF · SMS 公开方法参考结果</title><style>'+CSS+'</style></head><body>'
    overview += '<header><h1>全绿、无 LDF：Sparkle</h1><p>去除 OLED 周期纹理后的随机亮度起伏。</p></header><main>'+front
    overview += appendix_open+header.split('<h3>分离后原始浮点下载</h3>', 1)[0]+'</section>'
    for plot in plots[:2]:
        encoded = base64.b64encode(plot.read_bytes()).decode('ascii')
        overview += '<section><img style="width:100%;height:auto" src="data:image/png;base64,'+encoded+'" alt="'+escape(plot.name)+'"></section>'
    overview += '</div></details><section><p><a href="sparkle-report.html">完整离线报告与浮点下载</a></p>'
    overview += '<p><a href="summary.json">机器可读结果及验证</a> · <a href="protocol.json">计算前固定的测量方案</a></p></section></main></body></html>'
    (OUT / 'sparkle.html').write_text(overview, encoding='utf-8')
    validation = read(SKILL / 'VALIDATION.json')
    validation['latest_results_report'] = 'test-results/oled-sms-reference/sparkle-report.html'
    validation['latest_experiment_at_utc'] = read(OUT / 'old-na-0p08/run.json')['finished_at_utc']
    validation['sms_reference'] = {'report': 'test-results/oled-sms-reference/sparkle-report.html',
                                 'overview': 'test-results/oled-sms-reference/sparkle.html',
                                 'summary': 'test-results/oled-sms-reference/summary.json',
                                 'primary_Torch_percent': coarse['primary']['sparkle_contrast_percent'],
                                 'MWA_Torch_percent': coarse['sensitivities'][2]['sparkle_contrast_percent'],
                                 'flat_control': summary['flat_control'], 'sampling': summary['sampling_check'],
                                 'scope': summary['scope']}
    if 'regression_tests' in summary:
        validation['latest_tests'] = summary['regression_tests']
    (SKILL / 'VALIDATION.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('REPORT:', target)
    print(json.dumps({'records': [{'grid': x['grid'], 'backend': x['backend'],
                                  'primary': x['primary']['sparkle_contrast_percent'],
                                  'MWA': x['sensitivities'][2]['sparkle_contrast_percent']} for x in records],
                      'flat': summary['flat_control'], 'sampling': summary['sampling_check']}, indent=2))


def main():
    records, arrays = analyze_saved()
    plots = figures(records, arrays)
    report(records, arrays, plots)


if __name__ == '__main__':
    main()

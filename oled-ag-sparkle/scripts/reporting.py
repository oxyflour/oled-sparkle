"""Capture reproducible run inputs and build a portable offline HTML report."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
from html import escape
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import uuid

import numpy as np
from PIL import Image


PACKAGES = ('numpy', 'Pillow', 'torch', 'torchoptics', 'jax', 'jaxlib',
            'jax-cuda13-plugin', 'jax-cuda13-pjrt', 'chromatix', 'mitsuba', 'drjit')
LABELS = {
    'height': '高度图路径', 'height_pitch_um': '高度图横向采样 [µm/texel]',
    'height_range_um': '灰度黑至白的高度差 [µm]', 'emission': '发光强度图路径',
    'emission_pitch_um': '发光图采样 [µm/texel]', 'analysis_pitch_um': '闪点统计像素边长 [µm]',
    'emission_channel': '发光通道', 'gamma_decode': 'sRGB 解码开关',
    'emission_offset_x_um': '发光图 X 偏移 [µm]', 'emission_offset_y_um': '发光图 Y 偏移 [µm]',
    'gap_um': 'OLED 至平均 AG 距离 [µm]', 'wavelength_nm': '真空波长 [nm]',
    'n_stack': '层叠有效折射率', 'n_out': '外部介质折射率', 'na': '相机物侧 NA',
    'source_samples': '每个发光 texel 每轴点源数 K', 'emission_threshold': '发光模式筛选阈值',
    'source_margin_um': '额外发光源边界 [µm]', 'max_sources': '波动模式数量上限',
    'pad_px': 'FFT 每侧 padding [px]', 'batch': '波动计算批量', 'backend': '请求后端',
    'backend_workers':'同时计算的后端数',
    'device': 'Torch 设备设置', 'ray_radial': '光瞳径向节点', 'ray_angular': '光瞳角向节点',
    'ray_chunk': '每批光线数', 'mesh_stride': '几何网格采样步长',
    'mitsuba_variant': 'Mitsuba variant 设置', 'roi_margin_um': '统计 ROI 裁边 [µm]',
    'progress_every': '日志输出间隔', 'out': '输出目录',
    'skip_unavailable': '跳过缺失可选依赖', 'dry_run': '仅规模估算',
}
STATUS = {'complete': '完成', 'failed': '失败', 'skipped': '跳过',
          'not_run': '未计算', 'dry_run': '仅估算', 'interrupted': '中断'}
DETAIL_LABELS = {
    'started_at_utc':'开始时间 [UTC]', 'finished_at_utc':'结束时间 [UTC]',
    'wall_elapsed_seconds':'整次运行耗时 [s]，截至 HTML 生成前',
    'elapsed_seconds':'后端求解耗时 [s]', 'postprocess_elapsed_seconds':'后处理与结果保存耗时 [s]',
    'sparkle_cv_percent':'输出闪点值：亮度图直接 CV [%]',
    'brightness_mean':'ROI 亮度均值', 'brightness_std':'ROI 亮度标准差', 'num_roi_pixels':'ROI 像素数',
    'analysis_pixel_pitch_um':'统计像素边长 [µm]', 'analysis_pixel_area_um2':'统计像素面积 [µm²]',
    'analysis_grid':'统计亮度图尺寸 [行, 列]', 'analysis_resampling':'统计像素收集方式',
    'field_width_um':'计算区域宽度 [µm]', 'field_height_um':'计算区域高度 [µm]',
    'analysis_extent_um_xyxy':'统计区域物理边界 [x0, y0, x1, y1] [µm]',
    'analysis_source_bounds_px_xyxy':'统计区域在光学计算网格中的边界 [px]',
    'effective_roi_margin_um_xy':'实际统计区域裁边 [X, Y] [µm]',
    'metric_source_file':'闪点值直接使用的浮点亮度文件',
    'num_sources_or_rays':'实际波动模式 / 光线数量', 'source_representation':'光源表示方式',
    'source_node_pitch_um':'波动点源间距 [µm]', 'execution_context':'实际执行设备 / variant',
    'sha256':'文件 SHA-256', 'physical_width_um':'物理宽度 [µm]',
    'physical_height_um':'物理高度 [µm]', 'png_bits_per_channel':'PNG 每通道位深',
    'total_wall_seconds':'全部工况驱动程序总耗时 [s]',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def environment_snapshot():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'versions': versions,
            'environment_settings': {key: os.environ.get(key) for key in
                                     ('PYTHONPATH','OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'JAX_PLATFORMS',
                                      'JAX_ENABLE_X64', 'DRJIT_LIBLLVM_PATH',
                                      'SPARKLE_WSL_DISTRO', 'SPARKLE_WSL_PYTHON',
                                      'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION')}}


def image_metadata(path, pitch_um):
    path = Path(path).expanduser().resolve()
    result = {'path': str(path), 'pitch_um_per_texel': pitch_um}
    try:
        result.update({'sha256': sha256(path), 'file_bytes': path.stat().st_size})
        with Image.open(path) as image:
            array = np.asarray(image)
            result.update({'format': image.format, 'mode': image.mode, 'width_px': image.width,
                           'height_px': image.height, 'array_shape': list(array.shape),
                           'dtype': str(array.dtype), 'raw_min': float(array.min()),
                           'raw_max': float(array.max()), 'raw_std': float(array.std()),
                           'physical_width_um': image.width * pitch_um,
                           'physical_height_um': image.height * pitch_um})
        sidecar=path.with_suffix('.source.json')
        if sidecar.is_file():
            result['provenance']=json.loads(sidecar.read_text(encoding='utf-8'))
        if result['format'] == 'PNG':
            with path.open('rb') as stream:
                header = stream.read(26)
            result.update({'png_bits_per_channel': header[24], 'png_color_type': header[25]})
    except (OSError, ValueError) as error:
        result['read_error'] = str(error)
    return result


def capture_run(args, argv, solver_path):
    parameters = {key: value for key, value in vars(args).items() if not key.startswith('_')}
    command=[sys.executable,str(solver_path),*argv]
    return {'schema_version': 1, 'run_id': uuid.uuid4().hex, 'status': 'running',
            'started_at_utc': utc_now(), 'parameter_provenance': 'captured_from_actual_cli_with_defaults',
            'parameters': parameters, 'argv': list(argv),
            'command_display': subprocess.list2cmdline(command) if os.name=='nt' else shlex.join(command),
            'solver': {'path': str(Path(solver_path).resolve()), 'sha256': sha256(solver_path)},
            'input_files': {'height': image_metadata(args.height, args.height_pitch_um),
                            'emission': image_metadata(args.emission, args.emission_pitch_um)},
            'environment': environment_snapshot(), 'backend_runs': {},
            'timing_definition': 'Solver elapsed includes any library/JIT/device setup occurring inside the solver; excludes postprocessing and HTML writing. Run wall time is measured up to report generation.'}


def model_metadata(model, source_count):
    return {'height_grid': list(model.shape), 'fft_grid': list(model.padded_shape),
            'field_width_um':model.shape[1]*model.dx_m*1e6,
            'field_height_um':model.shape[0]*model.dx_m*1e6,
            'analysis_pixel_pitch_um':model.display_pitch_m*1e6,
            'wave_source_modes': source_count,
            'wave_source_node_pitch_um': model.em_dx_m * 1e6 / model.source_samples,
            'one_complex64_field_mib': np.prod(model.padded_shape).item() * 8 / 2**20,
            'ray_queries': int(np.prod(model.shape) * model.ray_radial * model.ray_angular),
            'pupil_nodes': model.ray_radial * model.ray_angular,
            'height_min_um': float(model.h_m.min() * 1e6),
            'height_max_um': float(model.h_m.max() * 1e6),
            'height_rms_about_mean_um': float(model.h_m.std() * 1e6),
            'linear_emission_min': float(model.emission.min()),
            'linear_emission_max': float(model.emission.max()),
            'active_emission_texels': int(np.sum(model.emission > model.threshold)),
            'resolved_torch_device_setting': model.device}


def execution_context(backend, model):
    context = {'backend': backend}
    if backend in ('torch', 'torchoptics'):
        torch = sys.modules['torch']
        context.update({'device': model.device, 'threads': torch.get_num_threads()})
        if model.device == 'cuda':
            context['gpu_name'] = torch.cuda.get_device_name()
    elif backend == 'chromatix':
        remote_context = getattr(model, '_chromatix_execution_context', None)
        if remote_context is not None:
            return dict(remote_context)
        jax = sys.modules['jax']
        context.update({'devices': [str(device) for device in jax.devices()],
                        'x64_enabled': bool(jax.config.jax_enable_x64)})
    elif backend == 'mitsuba':
        context['variant'] = sys.modules['mitsuba'].variant()
    return context


def collect_groups(root, recursive=False):
    root = Path(root).resolve()
    candidates = {root} if (root / 'run.json').exists() else set()
    stats_paths = root.rglob('stats.json') if recursive else root.glob('*/stats.json')
    candidates.update(path.parent.parent for path in stats_paths)
    if recursive:
        candidates.update(path.parent for path in root.rglob('run.json'))
    groups = []
    for directory in sorted(candidates):
        metadata_path = directory / 'run.json'
        meta = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {
            'status': 'complete', 'parameter_provenance': 'legacy_partial',
            'parameters': {}, 'input_files': {}, 'backend_runs': {}}
        records = []
        outcomes = meta.get('backend_runs', {})
        names = list(outcomes) if outcomes else [path.parent.name for path in directory.glob('*/stats.json')]
        for name in names:
            outcome = outcomes.get(name, {'status': 'complete'})
            stat_path = directory / name / 'stats.json'
            stat = None
            if outcome['status'] == 'complete' and stat_path.exists():
                stat = json.loads(stat_path.read_text(encoding='utf-8'))
                if meta.get('run_id') and stat.get('run_id') != meta['run_id']:
                    outcome = {'status': 'failed', 'error': {'message': 'Stats run_id does not match this run; stale output excluded.'}}
                    stat = None
            elif outcome['status'] == 'complete':
                outcome = {'status': 'failed', 'error': {'message': 'stats.json is missing.'}}
            records.append({'backend': name, 'status': outcome['status'], 'outcome': outcome, 'stats': stat})
        comparison_path = directory / 'comparison.json'
        comparison = (json.loads(comparison_path.read_text(encoding='utf-8'))
                      if comparison_path.exists() and meta.get('comparison_written', True) else {})
        groups.append({'name': directory.relative_to(root).as_posix(), 'directory': str(directory),
                       'run': meta, 'records': records, 'comparison': comparison})
    return groups


def _value(value):
    if value is None:
        return '未记录'
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _table(data, labels=None):
    labels = DETAIL_LABELS if labels is None else labels
    rows = ''.join(f'<tr><th><code>{escape(str(key))}</code><span>{escape(labels.get(key,""))}</span></th>'
                   f'<td>{escape(_value(value))}</td></tr>' for key, value in data.items())
    return f'<div class="scroll"><table class="kv"><tbody>{rows}</tbody></table></div>'


def _number(value, digits=8):
    return '—' if value is None else f'{float(value):.{digits}g}'


def _preview(path, expected_hash=None, max_size=(512,512)):
    path = Path(path)
    if not path.exists():
        return '<p class="muted">预览文件不可用</p>'
    if expected_hash and sha256(path) != expected_hash:
        return '<p class="muted">当前文件与运行时 SHA-256 不一致，未显示预览</p>'
    with Image.open(path) as original:
        array = np.asarray(original)
        if array.ndim == 2 and (array.dtype != np.uint8):
            top = 65535 if array.dtype == np.uint16 or original.mode.startswith('I') else max(float(array.max()), 1)
            image = Image.fromarray(np.clip(array.astype(float) / top * 255, 0, 255).astype(np.uint8))
        else:
            image = original.convert('RGB')
        image.thumbnail(max_size)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'<img loading="lazy" src="data:image/png;base64,{encoded}" alt="{escape(path.name, quote=True)}">'


def _input_gallery(groups):
    """Expose source images before results and preserve original PNG bytes."""
    unique={}
    for group in groups:
        parameters=group['run'].get('parameters',{})
        for role,info in group['run'].get('input_files',{}).items():
            if not info.get('path'):
                continue
            interpretation=(parameters.get('height_range_um') if role=='height' else
                            (parameters.get('emission_channel'),parameters.get('gamma_decode')))
            key=(role,info['path'],info.get('sha256'),info.get('pitch_um_per_texel'),str(interpretation))
            item=unique.setdefault(key,{'role':role,'file':info,'uses':[],
                                       'height_range_um':parameters.get('height_range_um')})
            name=group['name'] if group['name']!='.' else '当前运行'
            if name not in item['uses']:
                item['uses'].append(name)
    html=['<section id="input-images"><h2>输入高度图与发光图</h2>'
          '<p class="muted">以下为本次计算实际使用的输入图。预览采用屏幕显示位深；'
          '“下载原始文件”保留原始 PNG 像素和位深，已内嵌到本 HTML，离线可用。</p>']
    if not unique:
        html.append('<p class="muted">此旧结果未记录输入图路径，无法追溯输入图像。</p>')
    entries=[]
    for role in dict.fromkeys(['height','emission',*[item['role'] for item in unique.values()]]):
        items=[item for item in unique.values() if item['role']==role]
        if not items:
            continue
        label={'height':'高度图','emission':'发光图'}.get(role,role)
        html.append(f'<h3>{escape(label)}</h3><div class="input-grid">')
        for item in items:
            info=item['file']; path=Path(info['path'])
            original=None
            if path.is_file():
                raw=path.read_bytes()
                if info.get('sha256') and hashlib.sha256(raw).hexdigest()==info['sha256']:
                    original=base64.b64encode(raw).decode('ascii')
            width=_number(info.get('width_px')); height=_number(info.get('height_px'))
            depth=(f'{info["png_bits_per_channel"]} 位/通道' if info.get('png_bits_per_channel') is not None
                   else info.get('dtype','位深未记录'))
            html.append('<figure class="input-card">'
                        f'<h4>{escape(label)} · {escape(path.name)}</h4><div class="input-preview">'+
                        _preview(path,info.get('sha256'),max_size=(768,768))+'</div><figcaption>'
                        f'<strong>{width} × {height} px · {escape(depth)}</strong><br>'
                        f'横向采样：{_number(info.get("pitch_um_per_texel"))} µm/texel<br>'
                        f'物理范围：{_number(info.get("physical_width_um"))} × '
                        f'{_number(info.get("physical_height_um"))} µm<br>')
            if role=='height':
                html.append(f'灰度黑 → 白高度差：{_number(item["height_range_um"])} µm<br>')
            html.append(f'原始数值范围：{_number(info.get("raw_min"))} → {_number(info.get("raw_max"))}<br>'
                        '<span class="muted">用于：'+escape('、'.join(item['uses']))+'</span><br>')
            provenance=info.get('provenance',{})
            source_url=provenance.get('source_page','')
            if source_url.startswith('https://'):
                html.append('<a href="'+escape(source_url,quote=True)+'">来源：'+
                            escape(provenance.get('title','图像来源'))+'</a><br>')
            if original is not None:
                mime=Image.MIME.get(info.get('format'),'application/octet-stream')
                html.append(f'<a class="original-download" href="data:{escape(mime,quote=True)};base64,{original}" '
                            f'download="{escape(path.name,quote=True)}">下载原始文件（保留原始位深）</a>')
            else:
                html.append('<span class="muted">原始文件不可用，或与运行时 SHA-256 不符，未嵌入原文件。</span>')
            html.append('</figcaption></figure>')
            entries.append({**item,'original_embedded':original is not None})
        html.append('</div>')
    html.append('</section>')
    return ''.join(html),entries


CSS = """
:root{font-family:Inter,'Segoe UI','Microsoft YaHei',sans-serif;color:#172c43;background:#f2f6fa;font-size:14px}
*{box-sizing:border-box}body{margin:0}header{background:#102c46;color:white;padding:38px max(5vw,24px)}
header p{color:#c0d2e3;max-width:1000px;line-height:1.8}.eyebrow{letter-spacing:.13em;color:#63d4c8;font-size:12px}
h1{font-size:30px;margin:12px 0}h2{font-size:21px;margin:0 0 16px}h3{font-size:16px;margin:22px 0 10px}
main{max-width:1460px;margin:auto;padding:26px 32px 60px}nav{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:22px}
a{color:#117b82;text-decoration:none}section,.group{background:white;border:1px solid #dce5ee;border-radius:12px;padding:24px;margin-bottom:22px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}.card{background:white;border:1px solid #dce5ee;border-radius:10px;padding:19px}
.card strong{font-size:26px;display:block;margin:8px 0}.muted,.card small{color:#5b7084;line-height:1.7}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #e5ebf2;vertical-align:top}
thead{background:#edf3f8}thead th{white-space:nowrap;color:#405a70;font-size:12px}.numeric{text-align:right;white-space:nowrap}
.metric{font-weight:700;color:#087c76}.badge{font-size:11px;padding:4px 9px;border-radius:12px;background:#e3f5ec;color:#167448;white-space:nowrap}
.failed{background:#ffe8e8;color:#9d2828}.skipped,.not_run,.dry_run{background:#edf0f5;color:#596b7d}
.kv th{width:40%;font-weight:500}.kv th span{display:block;color:#5b7084;font-size:12px;margin-top:5px}
td,code,pre{overflow-wrap:anywhere}code{font:12px ui-monospace,Consolas,monospace}pre{white-space:pre-wrap;background:#f3f6fa;padding:16px;border-radius:6px;font-size:12px;line-height:1.6}
details.group{padding:0}summary{cursor:pointer;padding:20px 24px;font-weight:650}.content{padding:0 24px 24px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px}.gallery figure{margin:0;background:#f5f8fb;padding:12px;border-radius:8px}
.gallery img{display:block;width:100%;max-height:260px;object-fit:contain}.gallery figcaption{color:#506b80;font-size:12px;margin-top:9px}
.input-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:20px;margin-bottom:22px}
.input-card{margin:0;border:1px solid #d9e3ed;border-radius:9px;padding:16px}.input-card h4{margin:0 0 13px;font-size:15px}
.input-preview{min-height:220px;display:flex;align-items:center;justify-content:center;border:1px solid #cbd6e2;background:repeating-conic-gradient(#edf1f5 0% 25%,#fff 0% 50%) 50%/18px 18px}
.input-preview img{display:block;width:100%;height:300px;object-fit:contain;image-rendering:pixelated}.input-card figcaption{line-height:1.9;padding-top:13px;overflow-wrap:anywhere}
.original-download{display:inline-block;margin-top:9px;padding:7px 11px;border:1px solid #9fcfc9;background:#edf8f7;border-radius:6px}
.comparison-figure{margin:18px 0}.comparison-figure img{display:block;width:100%;height:auto}.comparison-figure figcaption{color:#5b7084;font-size:12px;margin-top:10px}
.gallery img[alt="analysis_brightness.png"]{image-rendering:pixelated}
.note{border-left:3px solid #22a69b;background:#edf8f7;padding:14px 16px;line-height:1.8}.notes li{line-height:1.8;margin-bottom:4px}
footer{color:#657c91;margin-top:20px;font-size:12px}@media(max-width:720px){main{padding:18px 12px}.cards{grid-template-columns:1fr}header{padding:28px 20px}h1{font-size:24px}section,.content{padding:16px}}
@media print{header{background:white;color:#172c43}header p{color:#405a70}body{background:white}nav{display:none}.scroll{overflow:visible}details{break-inside:avoid}}
"""


def write_html_report(root, output=None, recursive=False, title='OLED / AG 闪点仿真报告', figures=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / 'report.html'
    groups = collect_groups(root, recursive)
    if not groups:
        raise ValueError('No run.json or backend stats.json found in the results directory')
    records = [(group, record) for group in groups for record in group['records']]
    completed = [record for _, record in records if record['stats'] is not None]
    known_times = [record['stats']['elapsed_seconds'] for record in completed
                   if record['stats'].get('elapsed_seconds') is not None]
    solver_sum = sum(float(value) for value in known_times)
    solver_total = f'{solver_sum:.4f} s' if known_times else ('未记录' if completed else '未计算')
    input_gallery,input_images=_input_gallery(groups)
    body = [f'<header><div class="eyebrow">OLED AG · SIMULATION REPORT</div><h1>{escape(title)}</h1>'
            '<p>详细输入参数、计算耗时与闪点输出。主指标直接由亮度图 ROI 像素计算：100 × std(I) / mean(I)，单位 %；'
            '三套波动后端用于数值一致性检查，Mitsuba 几何光线模型单独解读。</p></header><main>'
            '<nav><a href="#input-images">输入图像</a><a href="#results">结果与耗时</a><a href="#inputs">完整输入参数</a><a href="#limits">指标与模型说明</a></nav>',
            '<div class="cards">'
            f'<div class="card">成功计算<strong>{len(completed)} / {len(records)}</strong><small>{len(groups)} 组运行；跳过与失败单独标记</small></div>'
            f'<div class="card">已记录求解耗时合计<strong>{solver_total}</strong><small>各后端实测求解耗时之和，含其中的首次加载 / JIT</small></div>'
            '<div class="card">闪点值定义<strong>100 × σ(I) / μ(I)</strong><small>按指定物理像素面积平均收集亮度后，直接统计全部 ROI 像素，ddof = 0</small></div></div>',
            input_gallery,
            '<section id="results"><h2>闪点值与计算时间</h2><div class="scroll"><table><thead><tr>'
            '<th>运行 / 工况</th><th>后端</th><th>物理模型</th><th>状态</th><th>闪点值 CV (%)</th>'
            '<th>统计像素 (µm)</th><th>亮度均值</th><th>亮度标准差</th><th>ROI 像素数</th><th>模式 / 光线数</th><th>求解 (s)</th><th>后处理 (s)</th></tr></thead><tbody>']
    for group, record in records:
        stat = record['stats'] or {}
        status = record['status']
        row = [escape(group['name'] if group['name'] != '.' else '当前运行'), escape(record['backend']),
               escape(stat.get('physics_group', '—')),
               f'<span class="badge {escape(status)}">{escape(STATUS.get(status,status))}</span>',
               _number(stat.get('sparkle_cv_percent')), _number(stat.get('analysis_pixel_pitch_um')),
               _number(stat.get('brightness_mean')),
               _number(stat.get('brightness_std')), _number(stat.get('num_roi_pixels')), _number(stat.get('num_sources_or_rays')),
               _number(stat.get('elapsed_seconds'), 7), _number(stat.get('postprocess_elapsed_seconds'), 7)]
        body.append('<tr>' + ''.join(f'<td class="{("numeric metric" if index==4 else "numeric") if index>=4 else ""}">{value}</td>'
                                    for index,value in enumerate(row)) + '</tr>')
    body.append('</tbody></table></div><p class="muted">求解耗时不含结果后处理与 HTML 写入；'
                '具体首次加载、JIT 和设备初始化计时以每组运行记录为准。“—”表示未记录或未计算，绝不以 0 代替。</p></section>'
                )
    if any(group['run'].get('parameters',{}).get('backend_workers',1)>1 for group in groups):
        body.append('<p class="note">本次包含后端并行计算。后端耗时为共享计算资源下的实测墙钟时长，'
                    '包含资源等待；各后端耗时之和与整次运行耗时分别记录。</p>')
    body.append('<section id="brightness-images"><h2>四种算法收集到的亮度图 · 闪点统计像素</h2><div class="gallery">')
    for group,record in records:
        if record['stats'] is None: continue
        stat=record['stats']; folder=Path(group['directory'])/record['backend']
        preview=folder/'analysis_brightness.png'
        if not preview.exists(): preview=folder/'brightness.png'
        body.append('<figure>'+_preview(preview)+
                    '<figcaption><strong>'+escape(record['backend'])+'</strong><br>闪点值：'+
                    _number(stat.get('sparkle_cv_percent'))+'% · 求解：'+_number(stat.get('elapsed_seconds'))+' s<br>')
        if stat.get('analysis_grid'):
            rows,cols=stat['analysis_grid']; pitch=stat['analysis_pixel_pitch_um']
            body.append(f'{cols} × {rows} 像素；每像素 {_number(pitch)} × {_number(pitch)} µm<br>'
                        f'统计区域 {_number(cols*pitch/1000)} × {_number(rows*pitch/1000)} mm<br>')
        for filename,label in (('analysis_brightness.npy','下载统计像素浮点亮度图'),
                               ('brightness.npy','下载光学计算网格原始亮度图')):
            raw=folder/filename
            if raw.exists():
                data=base64.b64encode(raw.read_bytes()).decode('ascii')
                body.append('<a href="data:application/octet-stream;base64,'+data+'" download="'+escape(record['backend'])+'-'+filename+'">'+label+'</a><br>')
        body.append('</figcaption></figure>')
    body.append('</div><p class="muted">每个统计像素收集其物理面积内的平均线性亮度。显示图按各自亮度范围作线性缩放；闪点值直接使用统计像素的浮点亮度，不使用显示 PNG 的量化值。</p></section>')
    figure_metadata=[]
    if figures:
        body.append('<section><h2>对比图</h2>')
        for figure in figures:
            path=Path(figure).resolve()
            figure_metadata.append({'path':str(path),'sha256':sha256(path)})
            body.append('<figure class="comparison-figure">'+_preview(path,max_size=(1920,1600))+
                        '<figcaption>'+escape(path.name)+'</figcaption></figure>')
        body.append('</section>')
    benchmark=None
    if (root/'manifest.json').exists():
        benchmark=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        body.append('<details class="group"><summary>多工况统一设置与总耗时</summary><div class="content">'+
                    _table({'total_wall_seconds':benchmark.get('total_wall_seconds'),
                            'started_at':benchmark.get('date')})+
                    _table(benchmark.get('settings',{}))+'</div></details>')
    body.append('<h2 id="inputs">完整输入、执行参数与输出图</h2>')
    for index, group in enumerate(groups):
        meta = group['run']
        group_name = group['name'] if group['name'] != '.' else '当前运行'
        body.append(f'<details class="group"{ " open" if len(groups)==1 else ""}><summary>{escape(group_name)} · '
                    f'{escape(STATUS.get(meta.get("status"),meta.get("status","未记录")))}</summary><div class="content">')
        if meta.get('parameter_provenance') == 'legacy_partial':
            body.append('<p class="note">旧结果没有完整运行快照。缺失输入参数和总耗时不作推定；使用新版求解器重跑可补齐。</p>')
        timing = {key: meta.get(key) for key in ('run_id','started_at_utc','finished_at_utc','wall_elapsed_seconds',
                                                'parameter_provenance','timing_definition')}
        body.append('<h3>运行与计时口径</h3>' + _table(timing))
        if meta.get('error'):
            body.append('<h3>运行错误</h3>' + _table(meta['error']))
        if meta.get('command_display'):
            body.append('<h3>执行命令</h3><pre>' + escape(meta['command_display']) + '</pre>')
        body.append('<h3>全部 CLI 参数（包括默认值）</h3>' + _table(meta.get('parameters', {}), LABELS))
        body.append('<h3>实际网格、采样与输入强度</h3>' + _table(meta.get('effective_model', {})))
        for name, info in meta.get('input_files', {}).items():
            body.append(f'<h3>输入文件 · {escape(name)}</h3>' + _table(info))
            body.append('<p><a href="#input-images">查看上方输入原图</a></p>')
        body.append('<h3>执行环境与代码版本</h3>' + _table(meta.get('environment', {})) + _table(meta.get('solver', {})))
        for record in group['records']:
            body.append(f'<h3>后端输出 · {escape(record["backend"])}</h3>')
            if record['stats'] is None:
                body.append(_table(record['outcome']))
                continue
            stat = record['stats']
            body.append(_table({key: value for key,value in stat.items() if key != 'warnings'}))
            folder = Path(group['directory']) / record['backend']
            body.append('<p><a href="#brightness-images">查看上方亮度图与原始浮点数据</a></p>')
        pairs = group.get('comparison', {}).get('pairwise_brightness', [])
        if pairs:
            body.append('<h3>后端配对误差</h3>' + ''.join(_table(pair) for pair in pairs))
        body.append('</div></details>')
    warnings = sorted({warning for record in completed for warning in record['stats'].get('warnings', [])})
    body.append('<section id="limits"><h2>指标与模型说明</h2><p class="note">'
                '本报告先按指定物理像素尺寸，对光学网格亮度作面积平均收集，再直接对保存的统计亮度图计算 100 × 标准差 / 均值。'
                '所有完整统计像素（包括暗区）都参与计算；ROI 只由几何裁边及完整像素范围确定。PNG 的显示缩放不参与计算。</p>'
                '<ul class="notes">' + ''.join(f'<li>{escape(item)}</li>' for item in warnings) + '</ul></section>')
    payload = {'schema_version': 1, 'generated_at_utc': utc_now(), 'results_root': str(root),
               'groups': groups,'input_images':input_images,'figures':figure_metadata,'benchmark':benchmark}
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    body.append(f'<script id="report-data" type="application/json">{data}</script>'
                '<footer>独立离线 HTML · 所有预览图已内嵌 · 完整机器可读数据位于 report-data JSON</footer></main>')
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">' \
               '<meta name="viewport" content="width=device-width,initial-scale=1">' \
               f'<title>{escape(title)}</title><style>{CSS}</style></head><body>' + ''.join(body) + '</body></html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding='utf-8')
    return output

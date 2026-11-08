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
    'emission_scale': '线性发光强度倍率（通道选择及解码之后）',
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
            'linear_emission_mean': float(model.emission.mean()),
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
        context['integrator'] = ('drjit_resident' if context['variant'].startswith('cuda')
                                 else 'numpy')
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


# SMS reference presentation uses saved optical runs; importing this module does
# not load Matplotlib or the SMS analysis/solver modules.
_SMS_BACKENDS = ('torch', 'torchoptics', 'chromatix', 'mitsuba')
_SMS_DISPLAY_NAMES = {'torch': 'PyTorch FFT', 'torchoptics': 'TorchOptics',
                      'chromatix': 'Chromatix', 'mitsuba': 'Mitsuba'}



def write_sms_report(results, output=None):
    """Audit saved SMS-reference runs and rebuild the input-first comparison.

    Writes sparkle.html, solver-comparison.html, sparkle-report.html, figures
    and summary.json inside results. output optionally copies the overview to a
    separate entry HTML, updating its relative links. Does not run optics,
    prepare inputs, overwrite protocol.json, package files or edit skill metadata.
    """
    from sms_reference import analyze_saved_runs

    results = Path(results).expanduser().resolve()
    if json.loads((results/'protocol.json').read_text(encoding='utf-8')).get('kind') == 'oled-layout-comparison':
        return _write_layout_report(results, output)
    if output and Path(output).expanduser().resolve() == results / 'sparkle-report.html':
        raise ValueError('--out is the overview destination; the full report is sparkle-report.html')
    records, arrays = analyze_saved_runs(results)
    plots = _sms_figures(results, records, arrays)
    entry = _write_sms_report(results, records, arrays, plots)
    if output:
        destination = Path(output).expanduser().resolve()
        if destination != entry:
            document = entry.read_text(encoding='utf-8')
            for name in ('sparkle-report.html', 'summary.json', 'protocol.json'):
                relative = os.path.relpath(results / name, destination.parent).replace(os.sep, '/')
                document = document.replace(f'href="{name}"', f'href="{escape(relative, quote=True)}"')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(document, encoding='utf-8')
        return destination
    return entry



def _write_layout_report(results, output=None):
    """Same report CLI, dedicated equal-pitch/common-bandwidth comparison view."""
    from sms_reference import analyze_layout_runs, measure_mwa
    from sparkle_multibackend import read_emission, read_height
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    protocol, records, arrays = analyze_layout_runs(results)
    camera_mode = protocol['analysis'].get('primary_metric') == 'sampled_camera_common_notch'
    optical_root = Path(protocol.get('optical_root', results))
    comparison = protocol.get('height_scale_comparison') or protocol.get('surface_comparison')
    # The retained layout metadata also defines the shared periodic mask. A
    # surface-control batch may compute only a subset while preserving that mask.
    names = list(dict.fromkeys(plan['layout'] for plan in protocol['runs']))
    labels = {'diamond': 'Diamond', 'rgb-stripe': 'RGB stripe', 'rgb-delta': 'RGB-Delta', 'windmill': 'Windmill'}
    display_backend = 'torch' if any(r['backend'] == 'torch' and r['grid'] == '1um' for r in records) else records[0]['backend']
    primary = [next(r for r in records if r['layout'] == name and r['grid'] == '1um' and r['backend'] == display_backend) for name in names]
    fine = [next(r for r in records if r['layout'] == name and r['grid'] == '0p5um') for name in names]
    measurement = records[0]['primary_filter']
    if camera_mode:
        method_title = 'Sampled camera | common aliased lattice notches'
    else:
        x_windows = '/'.join(f'{pair[0]:g}' for pair in measurement['windows_um_xy'])
        y_windows = '/'.join(f'{pair[1]:g}' for pair in measurement['windows_um_xy'])
        reference_window = measurement['reference_window_um']
        method_title = f'Common three-pass MWA | X: {x_windows} um; Y: {y_windows} um'
    peak = max(float(read_emission(protocol['layouts'][name]['emission_path'], 'green', True).max())*
               protocol['layouts'][name]['linear_emission_scale'] for name in names)
    fig, axes = plt.subplots(2, len(names), figsize=(max(7.5, 3.9*len(names)), 7.3), squeeze=False, layout='constrained')
    for col, name in enumerate(names):
        info = protocol['layouts'][name]
        pitch = info['assigned_emission_texel_um']
        rgb = np.asarray(Image.open(info['rgb_path']).convert('RGB'))
        signal = read_emission(info['emission_path'], 'green', True)*info['linear_emission_scale']
        size = round(4*protocol['nominal_pixel_pitch_um']/pitch)
        start = (rgb.shape[0]-size)//2
        end = start+size
        half = size*pitch/2
        extent = (-half, half, half, -half)
        axes[0, col].imshow(rgb[start:end, start:end], extent=extent, interpolation='nearest')
        axes[0, col].set_title(labels[name]+' | RGB geometry')
        image = axes[1, col].imshow(signal[start:end, start:end], extent=extent, cmap='gray',
                                    vmin=0, vmax=peak, interpolation='nearest')
        axes[1, col].set_title('Full green | field mean = 1')
        for ax in axes[:, col]:
            ax.set(xlabel='x [um]', ylabel='y [um]')
    fig.colorbar(image, ax=axes[1, :], shrink=.75, label='Linear green intensity (shared scale)')
    fig.suptitle(f'Equal nominal address pitch P = {protocol["nominal_pixel_pitch_um"]:g} um'+
                 ('\n' if len(names) == 1 else ' | ')+'Identical physical field of view')
    inputs_plot = results/'layout-inputs.png'
    fig.savefig(inputs_plot, dpi=150)
    plt.close(fig)

    values = [arrays[('1um', name, display_backend)]['primary_measurement_pixels'] for name in names]
    deviations = [100*(a/a.mean()-1) for a in values]
    limit = max(float(np.max(np.abs(a))) for a in deviations)
    fig, axes = plt.subplots(1, len(names), figsize=(max(7.5, 3.9*len(names)), 4), squeeze=False, layout='constrained')
    for ax, name, deviation, record in zip(axes[0], names, deviations, primary):
        x0, y0, x1, y1 = record['primary']['analysis_extent_um_xyxy']
        image = ax.imshow(deviation, extent=(x0, x1, y1, y0), cmap='RdBu_r', vmin=-limit, vmax=limit,
                          interpolation='nearest')
        ax.set(title=f'{labels[name]}\nSparkle = {record["primary"]["sparkle_contrast_percent"]:.3f}%',
               xlabel='x [um]', ylabel='y [um]')
    fig.colorbar(image, ax=axes[0], shrink=.8, label='Brightness deviation [%], shared scale')
    fig.suptitle('Same AG + camera + ROI + common filter'+
                 ('\n' if len(names) == 1 else ' | ')+f'{display_backend} | no LDF')
    sparkle_plot = results/'layout-sparkle.png'
    fig.savefig(sparkle_plot, dpi=150)
    plt.close(fig)

    backends = list(dict.fromkeys(r['backend'] for r in records if r['grid'] == '1um'))
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    width = .8/(len(backends)+1)
    centers = np.arange(len(names))
    for index, backend in enumerate(backends):
        heights = [next(r['primary']['sparkle_contrast_percent'] for r in records
                        if r['layout'] == name and r['grid'] == '1um' and r['backend'] == backend) for name in names]
        ax.bar(centers-.4+width*(index+.5), heights, width, label=backend+' (1 um)')
    ax.bar(centers+.4-width/2, [r['primary']['sparkle_contrast_percent'] for r in fine], width, label='torch (0.5 um)')
    ax.set(xticks=centers, xticklabels=[labels[name] for name in names], ylabel='Separated sparkle [%]',
           title=method_title)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, -.12))
    metrics_plot = results/'layout-metrics.png'
    fig.savefig(metrics_plot, dpi=150)
    plt.close(fig)

    if camera_mode:
        from sms_reference import circular_pupil_mtf
        fig, axes = plt.subplots(2, len(names), figsize=(max(7.5, 3.9*len(names)), 7.3), squeeze=False, layout='constrained')
        raw_pixels = [arrays[('1um', name, display_backend)]['camera_pixels'] for name in names]
        raw_deviations = [100*(a/a.mean()-1) for a in raw_pixels]
        bound = max(float(np.abs(a).max()) for a in raw_deviations+deviations)
        for col, (name, raw, separated, record) in enumerate(zip(names, raw_deviations, deviations, primary)):
            for row, data in enumerate((raw, separated)):
                im = axes[row, col].imshow(data, cmap='RdBu_r', vmin=-bound, vmax=bound,
                                           extent=(-420, 420, 420, -420), interpolation='nearest')
                value = record['direct_unseparated_camera_CV_percent'] if row == 0 else record['primary']['sparkle_contrast_percent']
                axes[row, col].set(title=labels[name]+'\n'+
                                   f'{"Captured CV" if row == 0 else "Separated contrast"} = {value:.3f}%',
                                   xlabel='Object x [um]', ylabel='Object y [um]')
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Relative signal deviation [%], one shared scale', shrink=.8)
        fig.suptitle('Top: lens -> square pixel integration -> sampling'+
                     ('\n' if len(names) == 1 else ' | ')+'Bottom: same samples after common notch mask')
        camera_plot = results/'camera-capture.png'
        fig.savefig(camera_plot, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
        cutoff = measurement['optical_intensity_cutoff_cycles_per_um']
        f = np.linspace(0, cutoff*1.1, 600)
        optical = circular_pupil_mtf(f, cutoff)
        detector = np.sinc(f*measurement['camera_pitch_um'])
        axes[0].plot(f*1000, optical, label='Ideal lens, incoherent circular pupil')
        axes[0].plot(f*1000, detector, label='Square pixel, axial sinc')
        axes[0].plot(f*1000, optical*detector, label='Pre-sampling camera MTF (axial)')
        mtf = primary[0]['camera_mtf_audit']
        axes[0].plot(np.array(mtf['frequency_cycles_per_um'])*1000, mtf['numerical_pupil_MTF'], '.',
                     label='Actual FFT pupil audit', markersize=3)
        axes[0].axvline(measurement['sensor_nyquist_cycles_per_um']*1000, color='gray', ls='--', label='Sensor Nyquist')
        axes[0].set(xlabel='Object spatial frequency [cycles/mm]', ylabel='Modulation transfer', ylim=(0, 1.04))
        axes[0].legend(fontsize=8)
        mask = arrays[('1um', names[0], display_backend)]['transfer']
        nyquist = measurement['sensor_nyquist_cycles_per_um']*1000
        frequencies = np.fft.fftshift(np.fft.fftfreq(mask.shape[0], measurement['camera_pitch_um']))*1000
        df = frequencies[1]-frequencies[0]
        axes[1].imshow(np.fft.fftshift(mask), cmap='gray', vmin=0, vmax=1,
                       extent=(frequencies[0]-df/2, frequencies[-1]+df/2,
                               frequencies[-1]+df/2, frequencies[0]-df/2), interpolation='nearest')
        axes[1].set(title=f'Common mask | {measurement["removed_non_DC_bins"]} removed bins\nWhite = retain; black = periodic lines and aliases',
                    xlabel='fx [cycles/mm]', ylabel='fy [cycles/mm]')
        transfer_plot = results/'camera-transfer.png'
        fig.savefig(transfer_plot, dpi=150)
        plt.close(fig)

    errors = {}
    for name in names:
        reference = np.load(optical_root/'1um'/name/display_backend/'brightness.npy')
        for backend in backends:
            other = np.load(optical_root/'1um'/name/backend/'brightness.npy')
            errors[name+'/'+backend] = float(np.linalg.norm(other-reference)/np.linalg.norm(reference))
    reference_check = None
    diamond = next((r for r in records if r['grid'] == '1um' and r['layout'] == 'diamond' and r['backend'] == 'torch'), None)
    historical = Path(__file__).resolve().parents[2]/'test-results/oled-sms-current/1um/torch/brightness.npy'
    if not camera_mode and diamond and historical.is_file() and measurement['reference_window_um'] == 140 and measurement['roi_width_um'] == 560:
        _, _, historical_stats = measure_mwa(np.load(historical), 1.)
        old_run = json.loads((historical.parent.parent/'run.json').read_text(encoding='utf-8'))
        reference_check = {'historical_three_140_MWA_percent': historical_stats['sparkle_contrast_percent'],
                           'current_diamond_same_three_140_MWA_percent': diamond['reference_mwa']['sparkle_contrast_percent'],
                           'current_diamond_fourier_percent': diamond['fourier']['sparkle_contrast_percent'],
                           'historical_raw_sha256': sha256(historical),
                           'same_AG_hash': old_run['input_files']['height']['sha256'] == diamond['height_sha256'],
                           'note': 'Old MWA definition and ROI reproduced directly from saved floats. Current optical run additionally normalizes input and includes all positive emission samples.'}
    summary = {'status': 'passed', 'protocol': protocol, 'records': records,
               'protocol_sha256': sha256(results/'protocol.json'),
               'primary_metric': 'sampled_camera_common_notch' if camera_mode else 'common_three_pass_MWA', 'measurement': measurement,
               'legacy_diamond_check': reference_check,
               'postprocessing_revision': ('Camera pixels captured before spectral separation; no MWA and no extra low-pass. '+
                                           ('Historical optical fields reused after identity checks.' if 'optical_root' in protocol else
                                            'New optical runs verified against the predeclared protocol.') if camera_mode else
                                           'Restore three-pass MWA as headline; original Fourier protocol retained without modification as a diagnostic. Window schedule is derived only from periods and legacy support.'),
               'shared_mask_verified': True,
               'fine_minus_coarse_percentage_points': {name: b['primary']['sparkle_contrast_percent']-a['primary']['sparkle_contrast_percent']
                                                       for name, a, b in zip(names, primary, fine)},
               'relative_L2_vs_display_backend': errors,
               'analysis_code_sha256': sha256(Path(__file__).with_name('sms_reference.py')),
               'report_code_sha256': sha256(Path(__file__)), 'processing_software': environment_snapshot()}
    if camera_mode:
        summary['validation'] = {
            'optical_reuse_manifest_verified': 'reused_files_sha256' in protocol,
            'captured_wave_results': len(records),
            'max_input_mean_absolute_error': max(abs(r['normalized_input_mean']-1.) for r in records),
            'max_absolute_MTF_error': max(r['camera_mtf_audit']['max_absolute_MTF_error'] for r in records),
            'max_mean_preservation_absolute_error': max(abs(r['primary']['mean_preservation_error']) for r in records),
            'max_periodic_flat_null_percent': max(r['periodic_flat_null_percent'] for r in records),
            'max_finite_flat_control_percent': max(r['flat_control_percent'] for r in records),
            'max_wave_backend_relative_L2': max(errors.values()),
            'identical_main_fine_masks': all(np.array_equal(arrays[('1um', name, display_backend)]['transfer'],
                                                            arrays[('0p5um', name, 'torch')]['transfer']) for name in names),
            'note': 'Numerical consistency and stated controls; this does not establish camera calibration or a commercial-panel ranking.'}
    (results/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    input_rows = []
    for name in names:
        info = protocol['layouts'][name]
        input_rows.append(f'<tr><td>{escape(labels[name])}</td><td>{info["nominal_pixel_pitch_um"]:g}</td>'
                          f'<td>{escape(str(info["full_rgb_period_um_xy"]))}</td><td>{escape(str(info["green_period_um_xy"]))}</td>'
                          f'<td>{info["linear_emission_scale"]:.8f}</td><td>{info["unscaled_field_linear_green_mean"]*info["linear_emission_scale"]:.9f}</td></tr>')
    result_rows = []
    for r in records:
        diagnostic = r['direct_unseparated_camera_CV_percent'] if camera_mode else r['reference_mwa']['sparkle_contrast_percent']
        result_rows.append(f'<tr><td>{escape(labels[r["layout"]])}</td><td>{escape(r["grid"])}</td><td>{escape(r["backend"])}</td>'
                           f'<td class="metric">{r["primary"]["sparkle_contrast_percent"]:.6f}</td>'
                           f'<td>{r["flat_control_percent"]:.6f}</td><td>{diagnostic:.6f}</td>'
                           f'<td>{r["solver_seconds"]:.3f}</td><td>{r["normalized_input_mean"]:.9f}</td></tr>')
    plot_html = lambda path: '<figure class="comparison-figure">'+_preview(path, max_size=(2400, 1600))+'</figure>'
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    title = 'OLED · 相机闪点与表面对照' if comparison else 'OLED 排列 · 同 AG 闪点对比'
    document += '<title>'+title+'</title><style>'+CSS+'</style></head><body>'
    document += '<header><h1>'+title+'</h1><p>全绿、530 nm、无 LDF；同一 AG 高度图、相机、名义物理像素间距和输入平均线性亮度。'
    document += ('先成像，再做方形像元积分与离散采样，最后用同一张含混叠频率的周期掩膜去周期。理想单色相机，无额外 MWA、低通或 LDF。</p></header><main>' if camera_mode else
                 '主结果恢复为三次移动平均（MWA）。所有排列使用相同窗口序列，并保留旧三次 140 µm MWA 的对照核验。</p></header><main>')
    metrology_path = results/'metrology-audit.json'
    if camera_mode and metrology_path.exists():
        audit = json.loads(metrology_path.read_text(encoding='utf-8'))
        spectrum = json.loads((results/'camera-spectrum.json').read_text(encoding='utf-8'))
        if audit['kind'] != 'SMS1000-public-definition-audit':
            raise ValueError('Unexpected SMS1000 metrology audit kind')
        document += '<section><h2>SMS1000 口径核对：32% 尚不是论文的等价读数</h2><p>厂商公开说明明确包含采集图像上的数值低通，以限制参与评价的空间频率。当前共同陷波方案只去除周期频点，保留其间的高频随机分量，因此不能当作 SMS1000 等价测量。本次保留原始 32% 指标，未新增或调节主结果滤波器，未重跑光学求解。</p><ul>'
        for item in audit['findings']:
            document += '<li>'+escape(item['finding'])+'</li>'
        document += '</ul><h3>参数来源逐项核对</h3><div class="scroll"><table><thead><tr><th>量</th><th>厂商/公开方法</th><th>Cheng 2025</th><th>当前模型</th><th>判定</th></tr></thead><tbody>'
        for row in audit['parameter_rows']:
            document += '<tr><td>'+escape(row['quantity'])+'</td><td>'+escape(row['vendor'])+'</td><td>'+escape(row['paper'])+'</td><td>'+escape(row['current'])+'</td><td>'+escape(row['status'])+' <a href="'+escape(row['source_url'], quote=True)+'">来源</a></td></tr>'
        document += '</tbody></table></div><p>145 µm 是 LDF—像素距离；当前 500 µm 是 AG 平均界面—发光层距离，二者不是同一参数。50 mm、3.75 µm 来自厂商典型设备信息；F/8、400 mm 和由此导出的 26.25 µm 物方像元不是已确认的论文设置。表1的 Width/Depth Variance 不能直接当平均纹理周期和坑深。</p>'
        document += '<h3>固定几何频率下的方差审计</h3><p>只按已知 70 µm 名义像素、140 µm 精确全绿周期及相机采样节距划分频段，没有反求能得到 5% 的截止频率。下表各分量用原信号均值归一，属于方差诊断，不是新的 SMS1000 校正值。</p><div class="scroll"><table><thead><tr><th>表面 / 网格</th><th>原主值 [%]</th><th>高于1/70的方差 [%]</th><th>1/70以下分量 [%]</th><th>1/140以下分量 [%]</th><th>Nyquist矩形内分量 [%]</th><th>仅DUT周期陷波 [%]</th></tr></thead><tbody>'
        for row in spectrum['records']:
            bands = row['frequency_boundaries']
            document += (f'<tr><td>{row["surface"]} / {row["grid"]}</td><td>{row["headline_sparkle_percent"]:.5f}</td>'
                f'<td>{bands["nominal_address_pitch_70um"]["above_retained_variance_percent"]:.3f}</td>'
                f'<td>{bands["nominal_address_pitch_70um"]["within_component_contrast_percent_total_mean"]:.5f}</td>'
                f'<td>{bands["exact_green_repeat_140um"]["within_component_contrast_percent_total_mean"]:.5f}</td>'
                f'<td>{row["Nyquist_decomposition"]["low_component_contrast_percent_total_mean"]:.5f}</td>'
                f'<td>{row["diamond_only_notch_diagnostic"]["contrast_percent"]:.5f}</td></tr>')
        document += '</tbody></table></div><p>缺少带宽限制是明确的定义差异，但简单剔除1/70以上频率后，三倍深度仍约30%。不能把全部差距归给这一项。Nyquist 诊断把1 mm光学浮点图按相机二维矩形 Nyquist 区域分解后再采样；存在有限DFT周期延拓假设，并保留两分量协方差，因此不是实测抗混叠镜头替代。它显示当前模型的混叠分量不足以解释32%到5%的全部差距。</p>'
        fig, ax = plt.subplots(figsize=(8.5, 4.6), layout='constrained')
        for row in spectrum['records']:
            if row['grid'] != '0p5um':
                continue
            curve = row['cumulative_variance']
            ax.plot([r['frequency_cycles_per_mm'] for r in curve],
                    [r['within_retained_variance_percent'] for r in curve], label=row['surface'])
        ax.axvline(1000/140, ls='--', color='gray', label='1 / 140 um G repeat')
        ax.axvline(1000/70, ls=':', color='black', label='1 / 70 um nominal pixel')
        ax.set(xlabel='Object radial spatial frequency [cycles/mm]', ylabel='Cumulative retained non-DC variance [%]',
               xlim=(0, 27), ylim=(0, 101), title='Unchanged camera signal | frequency audit, no fitted cutoff')
        ax.legend(fontsize=8)
        path = results/'metrology-frequency-variance.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        document += plot_html(path)+'<p>当前公共并集掩膜剔除89个非DC频点，而仅Diamond的周期掩膜剔除56个；后者使三倍深度上升到约33.07%，说明并集掩膜并不是32%偏高的原因。后端与网格检查仍只能审查数值一致性，不验证实际AG与测量链。</p><h3>仍缺少的独立证据</h3><ul>'
        for item in audit['required_for_identified_comparison']:
            document += '<li>'+escape(item)+'</li>'
        document += '</ul><p>公开资料不足以唯一分摊剩余差距。没有得到论文实际设置前，不输出声称已复现仪器的约5%结果。</p><p><a href="metrology-audit.json">参数与结论</a> · <a href="camera-spectrum.json">方差与采样诊断</a> · <a href="sources.json">来源清单、位置与校验值</a></p>'
        for source in audit['sources']:
            document += '<p><a href="'+escape(source['url'], quote=True)+'">'+escape(Path(source['local_file']).stem)+'</a> · <a href="'+source['local_file']+'">本地核对副本</a>：'+escape(source['relevant_location'])+'</p>'
        document += '</section>'
    if reference_check:
        document += '<section><h2>先核对原来约 5% 的钻石基准</h2><p>'
        document += f'旧报告原始浮点图按三次 140×140 µm MWA、560 µm ROI 重评：<strong>{reference_check["historical_three_140_MWA_percent"]:.6f}%</strong>；'
        document += f'本批次钻石在完全相同的后处理下：<strong>{reference_check["current_diamond_same_three_140_MWA_percent"]:.6f}%</strong>。'
        document += f'上一版展示的 {diamond["fourier"]["sparkle_contrast_percent"]:.6f}% 来自另一种频域滤波与 420 µm ROI，不能替代这个 MWA 基准。'
        document += ('两批次 AG 哈希相同；' if reference_check['same_AG_hash'] else '两批次 AG 哈希不同，仅作方法对照；')
        document += '本批次仍保留线性亮度归一化与全部正值发光样本。光学结果未重跑。</p></section>'
    document += '<section><h2>输入与归一化</h2><p>采用已保存的发光布局；钻石保留仓库原有完整重复单元，其他布局为公开图示启发的几何近似。'
    document += 'P 表示名义逻辑像素间距，不表示各色子像素最近邻距离相同。按全 1 mm 输入场的线性 G 均值归一至 1；倍率在 sRGB 解码后应用，峰值不裁剪，全部正值发光样本参与求解。</p>'
    document += '<div class="scroll"><table><thead><tr><th>排列</th><th>P [µm]</th><th>RGB 精确周期 X/Y [µm]</th><th>G 精确周期 X/Y [µm]</th><th>线性倍率</th><th>输入均值</th></tr></thead><tbody>'+''.join(input_rows)+'</tbody></table></div>'
    document += plot_html(inputs_plot)+'</section>'
    if camera_mode and comparison:
        height_only = 'height_scale' in comparison
        baseline_label = '原深度' if height_only else comparison['baseline_label']
        surface_label = '三倍深度' if height_only else comparison['surface_label']
        value_key = 'scaled_height_sparkle_percent' if height_only else 'surface_sparkle_percent'
        comparison_file = 'depth-comparison.json' if height_only else 'surface-comparison.json'
        baseline = Path(comparison['baseline_results'])
        assert sha256(baseline/'protocol.json') == comparison['baseline_protocol_sha256']
        assert sha256(baseline/'summary.json') == comparison['baseline_summary_sha256']
        previous = json.loads((baseline/'summary.json').read_text(encoding='utf-8'))
        assert previous['protocol']['analysis'] == protocol['analysis']
        assert previous['protocol']['camera'] == protocol['camera']
        comparisons = []
        for record in records:
            original = next(r for r in previous['records'] if all(r[k] == record[k] for k in ('layout', 'grid', 'backend')))
            assert original['primary_metric'] == record['primary_metric']
            # In-memory bin coordinates include tuples; saved JSON uses lists.
            assert original['primary_filter'] == json.loads(json.dumps(record['primary_filter']))
            base_value = original['primary']['sparkle_contrast_percent']
            value = record['primary']['sparkle_contrast_percent']
            comparisons.append({'layout': record['layout'], 'grid': record['grid'], 'backend': record['backend'],
                'baseline_sparkle_percent': base_value, value_key: value,
                'difference_percentage_points': value-base_value,
                'same_frequency_mask': np.array_equal(np.load(baseline/record['grid']/record['layout']/record['backend']/'camera_frequency_mask.npy'),
                                                     arrays[(record['grid'], record['layout'], record['backend'])]['transfer'])})
        assert all(row['same_frequency_mask'] for row in comparisons)
        (results/comparison_file).write_text(json.dumps({'definition': comparison, 'records': comparisons}, ensure_ascii=False, indent=2), encoding='utf-8')
        comparison_title = f'高度起伏 ×{comparison["height_scale"]:g}：同口径闪点对照' if height_only else comparison['title']
        description = ('只改变 AG 高度标定；PNG 灰度、横向采样、发光布局、相机、ROI 与频率掩膜保持一致。原深度的已保存相机浮点结果经身份和掩膜核对后作为对照，没有重写原文件。保留四种参考排列的周期集合来固定原共同掩膜，本次仅重算表中布局。' if height_only else comparison['description'])
        document += '<section><h2>'+escape(comparison_title)+'</h2><p>'+escape(description)+'</p><table><thead><tr><th>布局 / 网格 / 后端</th><th>'+escape(baseline_label)+' [%]</th><th>'+escape(surface_label)+' [%]</th><th>差 [百分点]</th></tr></thead><tbody>'
        for row in comparisons:
            document += f'<tr><td>{row["layout"]} / {row["grid"]} / {row["backend"]}</td><td>{row["baseline_sparkle_percent"]:.6f}</td><td>{row[value_key]:.6f}</td><td>{row["difference_percentage_points"]:+.6f}</td></tr>'
        document += '</tbody></table><p>主值为相机采样后的周期分离对比度，不是原始亮度 CV、雾度或历史三次 140 µm MWA。<a href="'+comparison_file+'">完整表面对照</a>；校准用例、执行命令与所有原始结果见 protocol.json 和各 run.json。</p>'
        if not height_only:
            document += '<p>本用例是探索性参数赋值，不是 Table 1 微结构统计的确定性复现；当前仍不是 SMS1000 等价测量。<a href="'+escape(comparison['haze_report'], quote=True)+'">同一输入的雾度报告</a>。</p>'
        first = primary[0]
        original = next(r for r in previous['records'] if all(r[k] == first[k] for k in ('layout', 'grid', 'backend')))
        original_run = json.loads((Path(original['raw_optical_file']).parents[1]/'run.json').read_text(encoding='utf-8'))
        scaled_plan = next(p for p in protocol['runs'] if p['layout'] == first['layout'] and p['grid'] == first['grid'])
        scaled_run = json.loads((optical_root/scaled_plan['output']/'run.json').read_text(encoding='utf-8'))
        height_maps = []
        for run in (original_run, scaled_run):
            params = run['parameters']
            values = read_height(params['height'], params['height_range_um']).astype(np.float64)*1e6
            height_maps.append(values-values.mean())
        if height_only:
            np.testing.assert_allclose(height_maps[1], comparison['height_scale']*height_maps[0], atol=3e-7, rtol=3e-6)
        limit = max(float(np.max(np.abs(h))) for h in height_maps)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
        plot_labels = ('Original depth', f'Height x{comparison["height_scale"]:g}') if height_only else (comparison['baseline_plot_label'], comparison['surface_plot_label'])
        field_um = 1000 if height_only else 100
        for ax, values, label, run in zip(axes, height_maps, plot_labels, (original_run, scaled_run)):
            if not height_only:
                size = round(field_um/run['parameters']['height_pitch_um'])
                start = (values.shape[0]-size)//2
                values = values[start:start+size, start:start+size]
            image = ax.imshow(values, cmap='viridis', vmin=-limit, vmax=limit,
                              extent=(-field_um/2, field_um/2, field_um/2, -field_um/2), interpolation='nearest')
            ax.set(title=label, xlabel='x [um]', ylabel='y [um]')
        fig.colorbar(image, ax=axes, shrink=.8, label='Height relative to mean [um], shared scale')
        height_plot = results/('height-depth-comparison.png' if height_only else 'height-surface-comparison.png')
        fig.savefig(height_plot, dpi=150)
        plt.close(fig)
        document += plot_html(height_plot)
        for label, run in ((baseline_label, original_run), (surface_label, scaled_run)):
            params = run['parameters']
            link = os.path.relpath(params['height'], results).replace(os.sep, '/')
            metadata = json.loads(Path(params['height']).with_suffix('.source.json').read_text(encoding='utf-8'))
            stats = metadata['measured_surface']
            document += f'<p>{escape(label)}：名义坑间距 {metadata["nominal_site_spacing_um"]:g} µm，网格 {params["height_pitch_um"]:g} µm，Sa={stats["Sa_um"]:.6f}、Sq={stats["Sq_um"]:.6f}、Sz={stats["Sz_um"]:.6f} µm；PNG 黑白高度范围 {params["height_range_um"]:.12f} µm。<a href="{link}">原始 16-bit 高度 PNG</a>；<a href="{str(Path(link).with_suffix(".source.json")).replace(os.sep, "/")}">物理标定 sidecar</a>。</p>'
        document += '</section>'
    if camera_mode:
        document += '<section><h2>相机采样后的随机闪点对比度</h2>'+plot_html(camera_plot)+plot_html(sparkle_plot)+plot_html(metrics_plot)
        document += '<p>主列为同一频率掩膜后的 100×σ/μ，采用总体标准差 ddof=0；直接采样 CV 仍包含显示周期，不能作为闪点。分离后的有符号信号不截断、不再次积分。波动平面控制的残余单列，不从结果扣除。</p>'
        document += '<div class="scroll"><table><thead><tr><th>排列</th><th>光学网格</th><th>后端</th><th>采样后去周期 [%]</th><th>有限视场平面控制 [%]</th><th>未去周期采样 CV [%]</th><th>原光学求解 [s]</th><th>输入均值</th></tr></thead><tbody>'+''.join(result_rows)+'</tbody></table></div></section>'
        document += '<section><h2>相机模型与实际处理顺序</h2><p>沿用 50 mm、F/8、400 mm 物距与 3.75 µm 像元。薄透镜像距 57.142857 mm，放大率 1/7，物方像元边长 26.25 µm。理想无像差圆形光瞳对焦于发光平面，物方 NA=0.0078122616；该光瞳已在原波动求解中实施，不再叠加一次 Airy 模糊。</p>'
        document += '<p>按同一模型的每个不相干点源，依次计算 OLED→AG 传播、薄相位屏、对焦与光瞳成像，再累加强度；在这个连续像面近似的 1/0.5 µm 物方网格上，对每个方形感光区做面积重叠积分，然后以相机节距取样。采用单色线性、100% 填充率、无噪声、无量化的传感器模型，未使用实测镜头 MTF，也未模拟 Bayer 或 ISP。</p>'
        document += f'<p>530 nm 下光学强度截止约 {cutoff*1000:.6f} cycles/mm；传感器 Nyquist 为 {nyquist:.6f} cycles/mm，故高于 Nyquist 的光学分量可以折叠。方形像元的 sinc 响应会衰减这些分量，但不会使其全部消失。下图是平面输入时的系统 MTF；AG 的传播作用仍由逐源波动求解给出。</p>'+plot_html(transfer_plot)
        document += '<h3>明确的去周期依据与参数</h3><p>采用 IEC 62977-3-9:2023 信息性附录 B 的频域掩膜方法；该标准没有规定本报告这张掩膜的具体参数。本实现先采样，再按每种排列精确 G 周期枚举光学带宽内的倒格矢，将其模传感器采样频率折回基带。取四种排列频点的并集，每个精确 DFT 频点剔除一个 bin，保留 DC。所有排列使用同一掩膜；不做峰值阈值搜索、保护带扩展、窗函数或额外低通。</p>'
        document += f'<p>中心 840×840 µm ROI 对应 32×32 个传感器像元，且同时包含 70、140、210 µm 的整数周期。因此理论周期线恰落 DFT bin，无需拼接滤波窗口。共同掩膜剔除 {measurement["removed_non_DC_bins"]} 个非 DC bin，保留 {measurement["retained_non_DC_bins"]} 个非 DC bin。周期能量与随机能量在同一 bin 重叠时无法分辨，该 bin 的随机分量也会一起被去除；这就是本指标明示的频率选择。</p>'
        document += '<p>Becker (2021) §3.1–3.4 讨论了相机 MTF、采样混叠和电子图像上的频域分离。其用于视觉评价的额外高频抑制未应用于本次相机域指标。因此这里的数值不能与历史三次 140 µm MWA 的约 5.29% 等同；没有将旧读数作为调参目标。本报告也不宣称是实测相机标定或完整 IEC 符合性试验。</p><ul>'
        for ref in protocol['analysis']['references']:
            document += '<li><a href="'+escape(ref['url'], quote=True)+'">'+escape(ref['title'])+'</a> — '+escape(ref['use'])+'</li>'
        document += '</ul></section><section><h2>网格、采样相位与视场检查</h2>'
        document += f'<p>实际 FFT 光瞳的轴向 MTF 与解析圆形光瞳 MTF 的最大绝对差为 {mtf["max_absolute_MTF_error"]:.6f}，反映有限光瞳频率采样误差。相位检查仅移动传感器采样网格，不移动 AG 或发光排列；物方偏移依次为 (0,0)、(7,0)、(0,7)、(7,7) µm，7 µm 约为 0.267 像元。</p>'
        document += '<p>420 µm 中心区（16×16 像元）给出两种诊断：在完整 32×32 去周期结果上取中心区，检验空间变化；独立采集 16×16 并重新做频域分离，同时改变了 ROI 和 DFT bin 宽度。后者沿用相同物理频点规则，但单个 bin 变宽，不能把其变化全部归因于空间不均匀。</p>'
        document += '<div class="scroll"><table><thead><tr><th>排列</th><th>1 µm 主值 [%]</th><th>0.5 µm [%]</th><th>网格差 [百分点]</th><th>四相位范围 [%]</th><th>同掩膜中心区 [%]</th><th>独立 420 µm ROI [%]</th><th>420 µm 平面残余 [%]</th><th>无限周期平面残余 [%]</th></tr></thead><tbody>'
        for name, r, fine_r in zip(names, primary, fine):
            phase_values = [c['sparkle_contrast_percent'] for c in r['phase_checks']]
            document += f'<tr><td>{labels[name]}</td><td>{r["primary"]["sparkle_contrast_percent"]:.6f}</td><td>{fine_r["primary"]["sparkle_contrast_percent"]:.6f}</td><td>{summary["fine_minus_coarse_percentage_points"][name]:+.6f}</td><td>{min(phase_values):.6f}–{max(phase_values):.6f}</td><td>{r["nested_ROI_same_full_mask"]["sparkle_contrast_percent"]:.6f}</td><td>{r["nested_ROI"]["sparkle_contrast_percent"]:.6f}</td><td>{r["nested_ROI_flat_percent"]:.6f}</td><td>{r["periodic_flat_null_percent"]:.3g}</td></tr>'
        document += '</tbody></table></div><p>有限 1 mm 发光场的衍射边缘及源离散化可能产生非周期残余；有限平面控制包含这些误差，无限周期平面控制用解析 Fourier 像元积分检查掩膜本身。0.5 µm 检查只细化光学/AG 网格，发光纹理与点源间距仍为 2 µm；单个 AG 实现、单个发光相位及小视场的统计差异不构成商用面板质量排名。</p>'
        document += '<p>本报告比较 Torch、TorchOptics 和 Chromatix 的同一标量波动模型。Mitsuba 几何光线结果缺少衍射，未并入这次理想衍射相机比较。用于分析的原光学文件保持不变，来源、哈希和运行 ID 记录于 JSON。</p>'
        document += '<p><a href="summary.json">完整统计与浮点结果索引</a> · <a href="protocol.json">采样与去周期前固定的方案</a></p></section>'
        document += '<details class="group"><summary>完整掩膜、折叠频率与复现参数</summary><pre>'+escape(json.dumps(measurement, ensure_ascii=False, indent=2))+'</pre></details>'
    else:
        document += '<section><h2>三次共同 MWA 闪点主结果</h2>'+plot_html(sparkle_plot)+plot_html(metrics_plot)
        document += f'<div class="scroll"><table><thead><tr><th>排列</th><th>光学网格</th><th>后端</th><th>共同三次 MWA 闪点 [%]</th><th>波动平面控制 [%]</th><th>三次 {reference_window:g} µm MWA [%] · 对照</th><th>求解 [s]</th><th>实际输入均值</th></tr></thead><tbody>'+''.join(result_rows)+'</tbody></table></div></section>'
        document += '<section><h2>周期、测量与验证</h2><p>三次 MWA 的共同矩形窗口由精确 G 周期确定，所有排列逐次使用完全相同的窗口序列。'
        document += f'当前 X 窗口为 {x_windows} µm，Y 为 {y_windows} µm。对每个已检测轴周期，至少一个窗口是其整数倍，确保所有周期谐波都有精确零点。'
        document += f'两轴总支撑宽度均为 {measurement["total_support_width_um_xy"][0]:g} µm，与三次 {reference_window:g} µm MWA 相同。中心 ROI 为 {measurement["roi_width_um"]:g} µm，使用 {primary[0]["primary"]["analysis_grid"]} 个 26.25 µm 测量像素，无反射边界进入测量区。'
        document += '每种输入的纯周期图均经过残余检查。窗口由几何和支撑范围决定，没有按目标闪点值调参；共同 MWA 与旧 140³ MWA 的传递函数仍有差别，故两者分列。'
        document += 'Delta 的 210 µm 周期不能由旧 140 µm 窗口精确消除，旧口径列仅用于对照，不据此给四种排列排名。</p>'
        document += '<p>每种布局均有同输入的波动平面控制与 0.5 µm Torch 网格复核。一个 AG 样本、一个发光相位，仅反映这些假设输入；不能外推为真实手机屏幕优劣。</p>'
        document += '<p>Torch、TorchOptics、Chromatix 求解同一标量波动模型；Mitsuba 是不含衍射的几何光线近似，两类结果分列，不把它们之间的差异作为同一模型的数值误差。</p>'
        document += '<pre>'+escape(json.dumps({'primary_measurement': measurement, 'camera': protocol['camera'],
                                                'batch_id': protocol['execution']['batch_id'], 'height_sha256': protocol['height_sha256'],
                                                'fine_minus_coarse_percentage_points': summary['fine_minus_coarse_percentage_points']}, ensure_ascii=False, indent=2))+'</pre>'
        document += '<p><a href="summary.json">全部浮点统计与验证 JSON</a> · <a href="protocol.json">计算前固定方案</a></p></section>'
        document += '<details class="group"><summary>原频域结果与后处理修订记录</summary><div class="content"><p>原光学 protocol.json 保持不变。'
        document += '其中的频域主结果已改列为诊断，修订后的主指标与窗口记录在 summary.json 的 measurement 和 postprocessing_revision 中。</p>'
        document += '<div class="scroll"><table><thead><tr><th>排列</th><th>网格</th><th>后端</th><th>原共同频域 [%]</th><th>频域平面控制 [%]</th></tr></thead><tbody>'
        document += ''.join(f'<tr><td>{escape(labels[r["layout"]])}</td><td>{r["grid"]}</td><td>{r["backend"]}</td>'
                            f'<td>{r["fourier"]["sparkle_contrast_percent"]:.6f}</td><td>{r["fourier_flat_percent"]:.6f}</td></tr>' for r in records)
        document += '</tbody></table></div><pre>'+escape(json.dumps(protocol['analysis'], ensure_ascii=False, indent=2))+'</pre></div></details>'
    document += '<details class="group"><summary>来源、几何定义与原始计算</summary><div class="content">'
    for name in names:
        info = protocol['layouts'][name]
        document += f'<h3>{escape(labels[name])}</h3><a href="{escape(info["source_page"], quote=True)}">公开参考来源</a><pre>'+escape(json.dumps(info['geometry'], ensure_ascii=False, indent=2))+'</pre>'
    for plan in protocol['runs']:
        for backend in plan['backends']:
            if camera_mode and backend == 'mitsuba':
                continue
            base = plan['output']+'/'+backend+'/'
            document += '<p>'+escape(plan['grid']+' / '+plan['layout']+' / '+backend)+' · '
            filenames = (('camera_pixels.npy', 'camera_separated.npy', 'camera_frequency_mask.npy', 'camera_stats.json') if camera_mode else
                         ('brightness.npy', 'sms_common_mwa_measurement_pixels.npy',
                          'sms_reference_mwa_measurement_pixels.npy', 'sms_measurement_pixels.npy', 'sms_stats.json'))
            if camera_mode:
                raw_link = os.path.relpath(optical_root/base/'brightness.npy', results).replace(os.sep, '/')
                document += '<a href="'+raw_link+'">原镜头成像 brightness.npy</a> · '
            for filename in filenames:
                document += f'<a href="{base+filename}">{filename}</a> · '
            document += '</p>'
    document += '</div></details></main></body></html>'
    entry = results/'sparkle.html'
    entry.write_text(document, encoding='utf-8')
    if output:
        destination = Path(output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Keep both JSON and floating-result links usable from an optional copy.
        prefix = os.path.relpath(results, destination.parent).replace(os.sep, '/')+'/'
        import re
        document = re.sub(r'href="(?!https:|data:)([^"]+)"', lambda m: 'href="'+prefix+m[1]+'"', document)
        destination.write_text(document, encoding='utf-8')
        return destination
    return entry


def _sms_figures(results, records, arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))

    inputs = read(results / '1um/run.json')['input_files']
    measurement = arrays[('1um', 'torch')]['summary']['sensitivities'][2]
    x0, y0, x1, y1 = measurement['analysis_extent_um_xyxy']
    input_plots = []
    for role in ('height', 'emission'):
        info = inputs[role]
        values = np.asarray(Image.open(info['path']))
        width = info['physical_width_um']
        height = info['physical_height_um']
        fig, ax = plt.subplots(figsize=(6.2, 4.9), layout='constrained')
        options = {'extent': (-width/2, width/2, height/2, -height/2),
                   'interpolation': 'nearest'}
        if role == 'height':
            height_range = info['provenance']['height_range_um']
            values = values.astype(np.float64) / 65535 * height_range
            im = ax.imshow(values, cmap='viridis', vmin=0, vmax=height_range, **options)
            fig.colorbar(im, ax=ax, label='Height [µm]', shrink=.88)
        else:
            ax.imshow(values, **options)
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='white',
                               linestyle='--', linewidth=1.6))
        ax.set(xlabel='x [µm]', ylabel='y [µm]')
        target = results / f'input-{role}.png'
        fig.savefig(target, dpi=150)
        plt.close(fig)
        input_plots.append(target)

    fig, axes = plt.subplots(3, 4, figsize=(15, 11), layout='constrained')
    for column, backend in enumerate(_SMS_BACKENDS):
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
    main = results / 'sms-brightness-comparison.png'
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
    diagnostic = results / 'filter-sensitivity.png'
    fig.savefig(diagnostic, dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout='constrained')
    deviations = [100*(arrays[('1um', name)]['mwa_measurement_pixels'] /
                        arrays[('1um', name)]['mwa_measurement_pixels'].mean()-1) for name in _SMS_BACKENDS]
    limit = max(float(np.max(np.abs(values))) for values in deviations)
    for ax, name, values in zip(axes, _SMS_BACKENDS, deviations):
        stat = arrays[('1um', name)]['summary']['sensitivities'][2]
        x0, y0, x1, y1 = stat['analysis_extent_um_xyxy']
        im = ax.imshow(values, extent=(x0, x1, y1, y0),
                       cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        ax.set(title=f"{_SMS_DISPLAY_NAMES[name]}\nSparkle {stat['sparkle_contrast_percent']:.3f}%", xlabel='x [µm]', ylabel='y [µm]')
        ax.set_xticks([x0, 0, x1], [f'{x0:g}', '0', f'{x1:g}'])
        ax.set_yticks([y0, 0, y1], [f'{y0:g}', '0', f'{y1:g}'])
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, shrink=.85, label='Random brightness deviation [%]')
    fig.suptitle('Green OLED + AG, no LDF | 3 x 140 um moving averages', fontsize=14)
    fig.supxlabel(f'Measurement ROI: {x1-x0:g} × {y1-y0:g} µm; 21 × 21 pixels at 26.25 µm. Origin: center of 1 mm field.', fontsize=10)
    sparkle_plot = results / 'mwa-sparkle.png'
    fig.savefig(sparkle_plot, dpi=150)
    plt.close(fig)
    return main, diagnostic, sparkle_plot, *input_plots


def _sms_row_for(record):
    mwa = record['sensitivities'][2]['sparkle_contrast_percent']
    return f"<tr><td>{record['grid']}</td><td>{record['backend']}</td><td>{record['object_NA']:.7f}</td><td>{record['direct_unseparated_camera_CV_percent']:.6f}</td><td>{record['primary']['sparkle_contrast_percent']:.6f}</td><td>{mwa:.6f}</td></tr>"


def _write_sms_report(results, records, arrays, plots):
    import re
    import sys
    import scipy
    import matplotlib
    skill_root = Path(__file__).resolve().parents[1]
    workspace = skill_root.parent
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    height_meta = read(results/'1um/run.json')['input_files']['height']['provenance']

    coarse = next(x for x in records if x['grid'] == '1um' and x['backend'] == 'torch')
    fine = next(x for x in records if x['grid'] == '0p5um')
    flat = arrays[('1um', 'flat')]['summary']
    baseline = flat['primary']['sparkle_contrast_percent']
    baseline_mwa = flat['sensitivities'][2]['sparkle_contrast_percent']
    summary = {'status': 'passed', 'protocol': read(results / 'protocol.json'),
               'protocol_sha256': hashlib.sha256((results / 'protocol.json').read_bytes()).hexdigest(),
               'records': records,
               'flat_control': {'primary_Fourier_percent': baseline, 'three_MWA_percent': baseline_mwa,
                                'not_used_for_division_or_subtraction': True},
               'sampling_check': {'primary_difference_percentage_points': fine['primary']['sparkle_contrast_percent']-coarse['primary']['sparkle_contrast_percent'],
                                  'MWA_difference_percentage_points': fine['sensitivities'][2]['sparkle_contrast_percent']-coarse['sensitivities'][2]['sparkle_contrast_percent']},
               'scope': 'SMS-inspired reference implementation. Different methods have different retained bandwidths; equality is not expected. Agreement with target in one method is not measurement calibration.',
               'verified': ['all six optical runs complete on GPU in this batch',
                            'current solver SHA matches every run',
                            'run and backend timestamps follow batch declaration',
                            'same green G bytes and height hashes',
                            'separated-image mean retained', '441 measurement pixels all included',
                            'saved measurement floats independently reproduce CV',
                            'fast flat limit independently tested against wave propagation']}
    summary['processing_software'] = {'python': sys.version, 'executable': sys.executable,
                                      'numpy': np.__version__, 'scipy': scipy.__version__,
                                      'matplotlib': matplotlib.__version__}
    summary['execution'] = summary['protocol']['execution']
    assert {record['batch_id'] for record in records} == {summary['execution']['batch_id']}
    assert {record['solver_sha256'] for record in records} == {summary['execution']['solver_sha256']}
    started_at = min(record['started_at_utc'] for record in records)
    finished_at = max(record['finished_at_utc'] for record in records)
    summary['processing_code'] = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                  for name, path in [('analysis', skill_root / 'scripts/sms_reference.py'),
                                                     ('runner', skill_root / 'scripts/run_sms_reference.py'),
                                                     ('report', Path(__file__).resolve())]}
    wave_reference = arrays[('1um', 'torch')]['measurement_pixels']
    wave_errors = {}
    for name in ('torchoptics', 'chromatix'):
        measured = arrays[('1um', name)]['measurement_pixels']
        error = float(np.linalg.norm(measured-wave_reference)/np.linalg.norm(wave_reference))
        assert error < 1e-3
        wave_errors[name] = error
    summary['wave_relative_L2_vs_torch'] = wave_errors
    log = results.parent / 'logs/validation-sms-reference.log'
    if log.is_file():
        data = log.read_bytes()
        content = data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        match = re.search(r'Ran (\d+) tests', content)
        completed = re.search(r'^OK(?: \(skipped=(\d+)\))?\s*$', content, re.M)
        if match and completed:
            skipped = int(completed.group(1) or 0)
            summary['regression_tests'] = {'status': 'passed', 'run': int(match.group(1)), 'skipped': skipped,
                                           'passed': int(match.group(1))-skipped, 'log': str(log)}
        elif 'FAILED' in content:
            raise AssertionError('Regression test log reports a failure')
        else:
            summary['regression_tests'] = {'status': 'uncomputed_or_still_running', 'log': str(log)}
    (results / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    target = write_html_report(results, output=results / 'sparkle-report.html', recursive=True,
                              title='Sparkle · 全绿 / 无 LDF', figures=plots[:2])
    document = target.read_text(encoding='utf-8')
    document = document.replace('id="inputs"', 'id="raw-input-parameters"').replace(
        'href="#inputs"', 'href="#raw-input-parameters"')
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
    {''.join(_sms_row_for(x) for x in records)}</tbody></table></div>
    <p>波动模型平面控制：频域 {baseline:.6f}%，MWA {baseline_mwa:.6f}%。
    控制结果用于检查残余周期和边界影响，不用于除法、相减或改变 AG 测量像素。
    同几何细网格变化：频域 {summary['sampling_check']['primary_difference_percentage_points']:+.6f} 个百分点，
    MWA {summary['sampling_check']['MWA_difference_percentage_points']:+.6f} 个百分点。</p>
    <h3>固定参考条件与公开方法的边界</h3>
    <p>只保留原始发光图 G 通道，R/B 置零；G 像素逐字节保留。530 nm 单波长，OLED 到 AG 距离 500 µm，
    高度图 Sa={height_meta['measured_surface']['Sa_um']:.6f} µm、Sz={height_meta['measured_surface']['Sz_um']:.6f} µm，
    名义中心间距 {height_meta['nominal_site_spacing_um']:g} µm，无 LDF。全绿强度的完整重复周期为 140 µm，
    四个绿色主体的几何中心间距约 70 µm，但其形状不完全相同。</p>
    <p>参考相机为 50 mm、F/8、400 mm 物距、3.75 µm 传感器像素；薄透镜近似给出倍率 1/7、
    物方像素 26.25 µm、绿色像素 ISR≈2.667，物方 NA≈0.007812。
    这些参数在计算前固定，属于公开范围内的参考选择，并非 Cheng 论文未公布的实际设置。
    当前求解器使用理想物方孔径；频域处理发生在光学网格，再积分至探测像素，省略了滤波前传感器混叠。
    Mitsuba 为无衍射的几何光线模型，小孔径下尤其需要与标量波动结果分别解读。</p>
    <p>频域掩模是线性操作，不能保证分离后的信号始终非负。所有分离后样本均原样纳入面积积分和 CV，
    不作截断或亮度选择；每组 sms_stats.json 记录负值像素数。实际原始光强仍要求全部非负。</p>
    <p>Cheng 2025 无 LDF 读数 4.8–6.0 按 SMS 通常百分比标度解读；原表未明示百分号或算法设置。
    其表面 Ra=0.25–0.35 µm、雾度 20–40%，不能直接与这里的合成表面 Sa 等同。
    1/70 截止、离散 notch 和三次 140 µm MWA 均为本次明确选择，未声称复现专有软件或满足 IEC 标准。</p>
    <p>方法依据：<a href="https://www.display-messtechnik.de/fileadmin/template/main/docs/10-Years-SparkleMeasurement--edC-2021-V2.pdf">Becker 2021，3.3–3.5 节</a>；
    相机范围：<a href="https://www.display-messtechnik.de/fileadmin/template/main/docs/SMS1000-05.2023-7.pdf">SMS-1000 公开规格</a>；
    对比数据：<a href="https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf">Cheng 2025，表 3</a>。</p>
    <h3>滤波与带宽敏感性：Torch 1 µm</h3><div class="scroll"><table><thead><tr><th>方案</th><th>结果 (%)</th></tr></thead><tbody>
    {''.join('<tr><td>'+escape(x['case'])+'</td><td>'+format(x['sparkle_contrast_percent'],'.6f')+'</td></tr>' for x in coarse['sensitivities'])}
    </tbody></table></div><h3>分离后原始浮点下载</h3>'''
    for record in records:
        folder = results / record['grid'] / record['backend']
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
    measurement = coarse['sensitivities'][2]
    x0, y0, x1, y1 = measurement['analysis_extent_um_xyxy']
    roi_width = coarse['roi_width_um']
    roi_margin = (roi_width-(x1-x0))/2
    plot_data = base64.b64encode(plots[2].read_bytes()).decode('ascii')
    solver_records = [x for x in records if x['grid']=='1um']
    assert [x['backend'] for x in solver_records] == list(_SMS_BACKENDS)
    cards, table_rows, optical_panels = [], [], []
    for record in solver_records:
        backend = record['backend']
        label = _SMS_DISPLAY_NAMES[backend]
        value = record['sensitivities'][2]['sparkle_contrast_percent']
        physics = '几何光线' if backend=='mitsuba' else '标量波动'
        cards.append(f'<div class="card" data-solver="{backend}"><h3>{label}</h3>'
                     f'<strong>{value:.3f}<span class="unit"> %</span></strong>'
                     f'<small>Sparkle · {physics}</small></div>')
        table_rows.append(f'<tr data-solver-row="{backend}"><td>{label}</td><td>{physics}</td>'
                          f'<td class="metric numeric">{value:.6f}</td><td class="numeric">{value-mwa:+.6f}</td>'
                          f'<td class="numeric">{record["solver_seconds"]:.3f}</td></tr>')
        brightness = results / '1um' / backend / 'brightness.png'
        encoded = base64.b64encode(brightness.read_bytes()).decode('ascii')
        optical_panels.append(f'<figure data-solver-image="{backend}"><img src="data:image/png;base64,{encoded}" '
                              f'alt="{label} 全绿光学亮度图"><figcaption><strong>{label}</strong><br>'
                              '1 mm × 1 mm 光学计算区域</figcaption></figure>')
    run = read(results / '1um/run.json')
    run_inputs = run['input_files']
    parameters = run['parameters']
    camera = summary['protocol']['camera']
    input_panels = []
    for role, label, preview in (('height', 'AG 表面高度图', plots[3]),
                                  ('emission', 'OLED 全绿发光图', plots[4])):
        info = run_inputs[role]
        raw = Path(info['path']).read_bytes()
        encoded = base64.b64encode(raw).decode('ascii')
        preview_data = base64.b64encode(preview.read_bytes()).decode('ascii')
        bits = info['png_bits_per_channel']
        rows = [('物理范围', f'{info["physical_width_um"]:g} × {info["physical_height_um"]:g} µm'),
                ('图像规格', f'{info["width_px"]} × {info["height_px"]} px · {bits} 位/通道'),
                ('横向采样', f'{info["pitch_um_per_texel"]:g} µm/texel')]
        provenance = info['provenance']
        if role == 'height':
            surface = provenance['measured_surface']
            rows.extend([
                ('表面形貌', f'{provenance["cell_count"]:,} 个不规则 Voronoi 浅坑；名义间距 {provenance["nominal_site_spacing_um"]:g} µm'),
                ('面粗糙度', f'Sa = {surface["Sa_um"]:.4f} µm；Sq = {surface["Sq_um"]:.4f} µm'),
                ('采样峰谷差 Sz', f'{surface["Sz_um"]:.6f} µm'),
                ('高度编码范围', f'{provenance["height_range_um"]} µm（黑 → 白）')])
            note = '合成蚀刻 AG 表面；色标为 PNG 编码对应的物理高度。'
        else:
            rows.extend([
                ('发光通道', '保留原图 G 通道；R = B = 0'),
                ('强度周期', f'{provenance["full_emission_supercell_um"]:g} µm；绿色几何中心间距约 {provenance["green_centers_geometric_pitch_um"]:g} µm'),
                ('强度处理', 'sRGB 解码为线性强度；530 nm 单波长'),
                ('来源', f'<a href="{escape(provenance["source_page"], quote=True)}">Samsung Display · Diamond Pixel</a>')])
            note = '按原 texel 尺寸重复完整单元；物理尺寸采用示例假设。'
        metadata = ''.join(f'<tr><th scope="row">{key}</th><td>{value}</td></tr>' for key, value in rows)
        try:
            local_path = Path(info['path']).relative_to(workspace).as_posix()
        except ValueError:
            local_path = str(info['path'])
        input_panels.append(f'<figure class="report-input" data-input="{role}"><h3>{label}</h3>'
                            f'<img src="data:image/png;base64,{preview_data}" alt="{label}，含物理坐标">'
                            f'<figcaption><p class="muted">{note}</p><table class="input-spec">{metadata}</table>'
                            f'<p class="input-file"><span>输入文件</span><code>{escape(local_path)}</code>'
                            f'<span>SHA-256</span><code>{info["sha256"]}</code></p>'
                            f'<a class="original-download" href="data:image/png;base64,{encoded}" '
                            f'download="{role}-original.png">下载原始 PNG · {bits} 位/通道</a>'
                            '</figcaption></figure>')
    front = f'''<nav class="report-nav" aria-label="报告目录"><a href="#inputs">01 输入</a>
    <a href="#measurement">02 计算与测量条件</a><a href="#sparkle-result">03 四求解器结果</a>
    <a href="#validation">04 验证与参考范围</a><a href="#methods">方法与复现附录</a></nav>
    <section id="inputs"><h2><span class="section-number">01</span> 输入与表面参数</h2>
    <p class="section-intro">四个求解器使用相同的高度图、发光图和物理尺度。计算区域为 1 mm × 1 mm，不含 LDF。</p>
    <div class="report-inputs">{''.join(input_panels)}</div>
    <p class="muted">图 1 · 共同输入。高度预览以物理高度着色；下载链接保留原始 PNG 字节和位深。
    白色虚线框为图 2 对应的 {x1-x0:g} × {y1-y0:g} µm 测量区域，坐标原点是整个 1 mm 计算区域的中心。
    AG 为合成形貌，OLED 为示意光源，两者均未按特定商业样品标定。</p></section>
    <section id="measurement"><h2><span class="section-number">02</span> 光学计算与闪点测量条件</h2>
    <div class="condition-grid"><div><h3>传播与采样</h3><table class="input-spec">
    <tr><th scope="row">OLED → AG 距离</th><td>{parameters['gap_um']:g} µm</td></tr>
    <tr><th scope="row">真空波长</th><td>{parameters['wavelength_nm']:g} nm</td></tr>
    <tr><th scope="row">折射率</th><td>层叠 {parameters['n_stack']:g} / 外部 {parameters['n_out']:g}</td></tr>
    <tr><th scope="row">光学网格</th><td>1 µm；1000 × 1000</td></tr>
    <tr><th scope="row">波动 / 光线模型</th><td>标量 ASM + 相位屏 / 几何折射</td></tr></table></div>
    <div><h3>参考相机</h3><table class="input-spec">
    <tr><th scope="row">镜头与物距</th><td>{camera['focal_length_mm']:g} mm · F/{camera['f_number']:g} · {camera['object_distance_mm']:g} mm</td></tr>
    <tr><th scope="row">传感器像素</th><td>{camera['sensor_pitch_um']:g} µm</td></tr>
    <tr><th scope="row">倍率</th><td>1/7（薄透镜近似）</td></tr>
    <tr><th scope="row">物方像素</th><td>{camera['object_pitch_um']:g} µm</td></tr>
    <tr><th scope="row">物方 NA</th><td>{camera['object_NA']:.8f}</td></tr></table></div>
    <div><h3>闪点评价</h3><table class="input-spec">
    <tr><th scope="row">周期纹理分离</th><td>3 次 140 µm 移动平均（MWA）</td></tr>
    <tr><th scope="row">滤波区域</th><td>中心 560 × 560 µm</td></tr>
    <tr><th scope="row">有效测量区域</th><td>551.25 × 551.25 µm</td></tr>
    <tr><th scope="row">测量网格</th><td>21 × 21；全部 441 像素</td></tr>
    <tr><th scope="row">物理像素积分</th><td>滤波后作面积平均</td></tr></table></div></div>
    <p class="metric-definition"><strong>Sparkle (%) = 100 × σ(I) / μ(I)</strong>
    <span>I 为分离周期调制并按物方像素积分后的亮度；σ 使用总体标准差（ddof = 0）。平面控制仅用于验证。</span></p>
    <p id="roi-explanation"><strong>图 2 的坐标来源：</strong>1 mm 计算区域为 −500 至 +500 µm；
    本算例预先固定中心 {roi_width:g} µm 区域（−{roi_width/2:g} 至 +{roi_width/2:g} µm），覆盖 4 个完整的 140 µm 强度周期。
    {roi_width:g} µm 内可容纳 floor({roi_width:g} / {camera['object_pitch_um']:g}) = 21 个完整测量像素，
    因此实际宽度为 21 × {camera['object_pitch_um']:g} = {x1-x0:g} µm，左右各再居中收进 {roi_margin:g} µm，
    最终 X、Y 边界均为 <strong>{x0:g} 至 +{x1:g} µm</strong>。图 2 展示全部 441 个测量像素；该裁取范围是本参考算例的选择。</p>
    <p class="muted">本页按公开 MWA 方法报告闪点。预先固定的 Fourier 方案及带宽敏感性完整保留在附录；
    两种处理保留的随机信号频带不同。本参考实现未复现 SMS-1000 的专有算法。</p></section>
    <section id="sparkle-result"><h2><span class="section-number">03</span> 四求解器结果与图像对比</h2>
    <p class="section-intro">统一输入、物方孔径、测量区域和 MWA 处理。三个波动后端实现同一标量模型，Mitsuba 使用几何光线模型。</p>
    <p class="muted" id="execution-batch">本次批次：<code>{summary['execution']['batch_id']}</code>；
    计算时间（UTC）：{escape(started_at)} 至 {escape(finished_at)}。主表和全部对照均使用此批次的新计算。
    求解器 SHA-256：<code>{summary['execution']['solver_sha256']}</code>。</p>
    <div class="cards solver-cards">{''.join(cards)}</div>
    <h3>表 1 · 闪点结果与求解耗时</h3>
    <div class="scroll"><table id="solver-comparison"><thead><tr><th>求解器</th><th>物理模型</th>
    <th class="numeric">Sparkle (%)</th><th class="numeric">相对 PyTorch FFT（百分点）</th><th class="numeric">本次求解耗时 (s)</th></tr></thead><tbody>
    {''.join(table_rows)}
    </tbody></table></div><p class="muted">差值用于数值比较；显示位数不代表测量精度。
    四后端按顺序使用同一块 GPU。本次求解耗时统一取后端墙钟记录，包含求解器初始化、JIT 和最终图像回读，
    不含闪点后处理与报告写入；Chromatix 通过 WSL CUDA 执行，其记录包含 worker 启动与传输时间。</p>
    <figure class="comparison-figure"><img src="data:image/png;base64,{plot_data}" alt="PyTorch FFT、TorchOptics、Chromatix、Mitsuba 闪点分布对比">
    <figcaption>图 2 · 分离后的闪点分布。X、Y 物理边界均为 {x0:g} 至 +{x1:g} µm；
    即图 1 虚线框中的 {x1-x0:g} × {y1-y0:g} µm 区域，21 × 21 个测量像素，像素中心为 −262.5 至 +262.5 µm。
    四图采用相同色标，颜色表示相对各自均值的亮度偏差 (%)。</figcaption></figure>
    <h3>图 3 · 四求解器的光学亮度输出</h3><div class="gallery solver-gallery">{''.join(optical_panels)}</div>
    <p class="muted">光学亮度预览各用自身刻度。所有闪点统计均由原始浮点结果计算。</p></section>
    <section id="validation"><h2><span class="section-number">04</span> 验证与参考范围</h2>
    <div class="scroll"><table><thead><tr><th>检查项</th><th>结果</th><th>解释</th></tr></thead><tbody>
    <tr><td>0.5 µm 网格复核 · PyTorch FFT</td><td class="numeric">{mwa_fine:.6f}%</td><td>相对 1 µm 网格变化 {mwa_fine-mwa:+.6f} 个百分点</td></tr>
    <tr><td>平面控制残余</td><td class="numeric">{baseline_mwa:.6f}%</td><td>检查周期残余和边界影响；不用于扣除或归一化</td></tr>
    <tr><td>文献无 LDF 参考读数</td><td class="numeric">4.8–6.0</td><td><a href="https://sid.org/Portals/sid/Files/SID-Digest/docs/68-2.pdf">Cheng 2025 · 表 3</a>，按 SMS 通常百分比标度解读</td></tr>
    </tbody></table></div>
    <p>本次四个求解器的 MWA 闪点均处于文献读数的量级。原文未公开仪器与滤波设置，
    其表面粗糙度和显示面板也与本算例不同，因此该比较用于量级检查，尚不能视为实测校准。</p></section>'''
    comparison_css = '''
    :root{background:#f4f6f8;color:#233247}p{line-height:1.8}
    header{background:white;color:#172c43;border-bottom:1px solid #dce5ee;padding:28px max(5vw,24px)}
    header p{color:#617186;margin:5px 0}header h1{font-size:29px;margin:4px 0 8px;font-weight:650}
    main{max-width:1380px}section{scroll-margin-top:20px;border-radius:8px;padding:28px}
    h2{font-size:21px;display:flex;align-items:baseline;gap:12px}.section-number{font-size:14px;color:#63829a;font-weight:500}
    .section-intro{color:#617186;margin-top:-3px}.report-nav{margin:0 0 22px;padding:0 4px;font-size:13px;gap:24px}
    .report-inputs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}
    .report-input{margin:0;min-width:0;border:1px solid #dce5ee;border-radius:6px;padding:18px}
    .report-input h3{margin:0 0 10px}.report-input>img{display:block;width:100%;height:auto}
    .report-input figcaption p{font-size:12px;line-height:1.65}.input-spec th{font-size:12px;font-weight:500;color:#617186;width:33%;white-space:normal}
    .input-spec td{font-size:13px}.input-spec th,.input-spec td{padding:9px 8px}
    .input-file{display:grid;gap:4px;margin:14px 0 4px}.input-file span{color:#617186;margin-top:4px}
    .input-file code{font-size:11px;word-break:break-all;color:#526477}.original-download{font-size:12px}
    .condition-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}
    .condition-grid h3{margin:4px 0 10px}.condition-grid .input-spec th{width:40%}
    .metric-definition{background:#f4f7fa;border-left:3px solid #66889f;padding:14px 18px;margin-top:22px}
    .metric-definition strong{font-size:15px;font-variant-numeric:tabular-nums}.metric-definition span{display:block;font-size:12px;color:#617186;margin-top:5px}
    .solver-cards{grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}
    .solver-cards .card{background:#fafcfd;border-radius:6px;padding:18px}.solver-cards h3{margin:0;font-size:14px}
    .solver-cards strong{font-variant-numeric:tabular-nums;font-size:30px;color:#244d68;font-weight:600}
    .solver-cards .unit{font-size:18px;color:#617186;font-weight:400}.solver-gallery{grid-template-columns:repeat(4,minmax(0,1fr))}
    .comparison-figure figcaption{line-height:1.7}.muted{font-size:12px}th,td{line-height:1.6}
    @media(max-width:900px){.condition-grid{grid-template-columns:1fr}.solver-cards,.solver-gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:640px){.report-inputs{grid-template-columns:1fr}.report-nav{gap:12px}section{padding:18px}header h1{font-size:23px}}
    '''
    document = document.replace('</style>', comparison_css+'</style>', 1)
    appendix_open = '<details class="group" id="methods"><summary>方法对照、原始统计与复现数据（附录）</summary><div class="content">'
    report_header = '<header><h1>OLED / AG 闪点仿真报告</h1><p>全绿画面 · 无 LDF · 四求解器对比 · SMS 公开方法参考口径</p></header>'
    document = re.sub(r'<header>.*?</header>', report_header, document, count=1, flags=re.S)
    document = document.replace('<main>', '<main>'+front+appendix_open+header, 1)
    document = document.replace('</main>', '</div></details></main>', 1)
    target.write_text(document, encoding='utf-8')
    # Check self-contained original 16-bit PNGs and all filtered-data downloads.
    for grid in ('1um', '0p5um', 'old-na-0p08'):
        meta = read(results / grid / 'run.json')
        for role in ('height', 'emission'):
            encoded = base64.b64encode(Path(meta['input_files'][role]['path']).read_bytes()).decode('ascii')
            assert encoded in document
    downloads = re.findall(r'href="data:application/octet-stream;base64,([^\"]+)"', document)
    assert len(downloads) == 42
    assert all(base64.b64decode(value).startswith(b'\x93NUMPY') for value in downloads)
    summary['verified'].append('offline report preserves input PNGs and all 42 NumPy downloads, including MWA floats')
    (results / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # A small entry page keeps the inline NumPy downloads in the full report.
    overview = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    overview += '<title>OLED / AG 闪点仿真报告 · 四求解器对比</title><style>'+CSS+comparison_css+'</style></head><body>'
    overview += report_header+'<main>'+front
    overview += appendix_open+header.split('<h3>分离后原始浮点下载</h3>', 1)[0]+'</section>'
    for plot in plots[:2]:
        encoded = base64.b64encode(plot.read_bytes()).decode('ascii')
        overview += '<section><img style="width:100%;height:auto" src="data:image/png;base64,'+encoded+'" alt="'+escape(plot.name)+'"></section>'
    overview += '</div></details><section><p><a href="sparkle-report.html">完整离线报告与浮点下载</a></p>'
    overview += '<p><a href="summary.json">机器可读结果及验证</a> · <a href="protocol.json">计算前固定的测量方案</a></p></section></main></body></html>'
    (results / 'sparkle.html').write_text(overview, encoding='utf-8')
    (results / 'solver-comparison.html').write_text(overview, encoding='utf-8')
    return results / 'sparkle.html'


def write_haze_report(results, output=None):
    """Audit and plot saved four-backend haze data; never launch a solver."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from calculate_haze import transmission_haze, load_surface, surface_diagnostics

    results = Path(results).resolve()
    entry = Path(output).resolve() if output else results/'haze.html'
    artifacts = entry.parent
    artifacts.mkdir(parents=True, exist_ok=True)
    data = json.loads((results/'haze.json').read_text(encoding='utf-8'))
    if data['kind'] != 'ag-haze-four-backend':
        raise ValueError('--kind haze requires a saved native-backend comparison')
    protocol = json.loads((results/'protocol.json').read_text(encoding='utf-8'))
    assert data['protocol_sha256'] == sha256(results/'protocol.json')
    records = data['records']
    declared = datetime.fromisoformat(protocol['declared_at_utc'])
    curves, validations, diagnostics = {}, [], {}
    for record in records:
        assert datetime.fromisoformat(record['started_at_utc']) >= declared
        assert sha256(record['input']['path']) == record['input']['sha256']
        folder = (results/record['angular_curve_csv']).parent
        assert json.loads((folder/'stats.json').read_text(encoding='utf-8')) == record
        with (results/record['angular_curve_csv']).open(encoding='utf-8', newline='') as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        at_cutoff = next(row for row in rows if row['half_angle_deg'] == record['acceptance_half_angle_deg'])
        error = abs(at_cutoff['outside_cone_percent']-record['haze_percent'])
        assert error < 1e-10
        if record['physics_group'] == 'scalar_phase_screen_wave':
            power = np.load(folder/'angular_power.npy', allow_pickle=False)
            h, pitch, _ = load_surface(record['input']['path'], record['input']['height_pitch_um'],
                record['input']['height_range_um'], record['input']['periodicity'] == 'explicit user assumption')
            measured, _ = transmission_haze(h, pitch*1e-6, record['wavelength_nm']*1e-9,
                record['n_in'], record['n_out'], record['acceptance_half_angle_deg'], angular_power=power)
            assert abs(measured['haze_percent']-record['haze_percent']) < 1e-10
            audit_key = f'{record["surface_id"]}/{record["wavelength_nm"]:g}nm'
            if audit_key not in diagnostics:
                surface, surface_curves = surface_diagnostics(h, pitch*1e-6, record['wavelength_nm']*1e-9,
                    record['n_in'], record['n_out'], record['acceptance_half_angle_deg'])
                diagnostics[audit_key] = {'input': record['input'], 'surface': surface, 'curves': surface_curves,
                    'angular_flux': {k: measured[k] for k in ('zero_order_forward_flux_percent',
                        'within_cone_nonzero_order_flux_percent', 'haze_percent',
                        'nonzero_order_forward_flux_percent', 'haze_without_cosine_weight_percent')}}
        else:
            assert record['incident_rays'] == record['hit_rays'] == record['mesh_faces']
            assert record['energy_balance_error'] < 1e-6
        assert record['flat_control_percent'] < 1e-12
        key = (record['surface_id'], record['backend'], record['wavelength_nm'])
        curves[key] = rows
        validations.append({'surface_id': record['surface_id'], 'backend': record['backend'],
                            'curve_metric_absolute_error': error, 'passed': True})
    surfaces = list(dict.fromkeys(r['surface_id'] for r in records))
    backends = list(dict.fromkeys(r['backend'] for r in records))
    wavelengths = list(dict.fromkeys(r['wavelength_nm'] for r in records))
    figures = []
    for wavelength in wavelengths:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        x = np.arange(len(backends))
        width = .75/len(surfaces)
        for index, surface in enumerate(surfaces):
            selection = [next(r for r in records if r['surface_id'] == surface and
                         r['backend'] == b and r['wavelength_nm'] == wavelength) for b in backends]
            axes[0].bar(x-.375+width*(index+.5), [r['haze_percent'] for r in selection], width,
                        label=f'{selection[0]["height_pitch_um"]:g} um height grid')
        reference = next(r['reference_numpy_percent'] for r in records if r['wavelength_nm'] == wavelength)
        axes[0].axhline(reference, color='gray', ls='--', label=f'NumPy phase screen: {reference:.6f}%')
        axes[0].set(xticks=x, xticklabels=backends, ylabel='Transmission haze [%]', title='Actual native backend runs')
        axes[0].legend(fontsize=8)
        for backend in backends:
            rows = curves[(surfaces[0], backend, wavelength)]
            angles = np.array([r['half_angle_deg'] for r in rows])
            outside = np.array([r['outside_cone_percent'] for r in rows])
            valid = (angles <= 6) & (outside > 1e-8)
            axes[1].plot(angles[valid], outside[valid], label=backend)
        axes[1].axvline(records[0]['acceptance_half_angle_deg'], color='gray', ls='--')
        axes[1].set(xlabel='Acceptance half-angle [degrees]', ylabel='Power outside cone [%]',
                    xlim=(0, 6), yscale='log', title='Angular power integral | main height grid')
        axes[1].legend(fontsize=8)
        fig.suptitle(f'Uniform normal incidence | {wavelength:g} nm | wave phase-screen vs geometric refraction')
        path = artifacts/f'haze-comparison-{wavelength:g}nm.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    table = []
    for r in records:
        flat = r['flat_control_percent']
        physics = '薄相位屏波动' if r['physics_group'] == 'scalar_phase_screen_wave' else '网格折射 / Fresnel'
        table.append(f'<tr><td>{r["height_pitch_um"]:g}</td><td>{r["wavelength_nm"]:g}</td><td>{escape(r["backend"])}</td>'
                     f'<td>{physics}</td><td class="metric">{r["haze_percent"]:.6f}</td>'
                     f'<td>{r["reference_numpy_percent"]:.6f}</td><td>{r["difference_vs_numpy_percentage_points"]:+.6f}</td>'
                     f'<td>{flat:.3g}</td><td>{r["solver_seconds"]:.3f}</td></tr>')
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    title = 'AG 雾度 · 纹理尺度、深度与形貌' if (artifacts/'texture-sensitivity.json').exists() else 'AG 雾度 · 四后端实际计算'
    document += '<title>'+title+'</title><style>'+CSS+'</style></head><body><header><h1>'+title+'</h1>'
    document += f'<p>同一校准高度图、均匀正入射、n={records[0]["n_in"]:g}→{records[0]["n_out"]:g}、波长 {escape(str(wavelengths))} nm、{records[0]["acceptance_half_angle_deg"]:g}° 接收半角。三个波动后端调用各自原生传播；Mitsuba 调用网格求交及原生折射/Fresnel。无 OLED 发光图、相机光瞳、像元积分或额外滤波。</p></header><main>'
    document += '<section><h2>结果与 NumPy 近似对照</h2><div class="scroll"><table><thead><tr><th>网格 [µm]</th><th>波长 [nm]</th><th>后端</th><th>模型</th><th>雾度 [%]</th><th>NumPy [%]</th><th>差 [百分点]</th><th>平面控制 [%]</th><th>求解 [s]</th></tr></thead><tbody>'+''.join(table)+'</tbody></table></div>'
    for path in figures:
        document += '<figure class="comparison-figure">'+_preview(path, max_size=(2200, 1000))+'</figure>'
    document += f'</section><section><h2>如何理解后端一致与差异</h2><p>Torch、TorchOptics 和 Chromatix 仍然使用同一个薄相位屏近似；实际执行出射介质中的 {protocol["parameters"]["propagation_um"]:g} µm ASM 传播后，再读取角谱功率。均匀介质传播只改变传播模的相位，不改变其角功率，因此与未传播的 NumPy 估算一致是预期的数值检查，不是对薄相位面物理准确性的独立验证。</p>'
    document += '<p>Mitsuba 将同一周期高度场补齐端点，建立完整周期投影范围的三角网格。每个三角形按投影面积在重心发射一条正入射光线，使用几何面法线和原生 Fresnel/refract 计算单次透射。透射光线携带的功率直接进入角度积分，不再乘一次 cosθ。这个模型包含局部折射和 Fresnel 权重，但缺少衍射；数值差异不等同于任一模型已验证为真实玻璃。</p>'
    document += '<p>波动模型的 Cartesian 角谱功率按 cosθ 换算法向通量；2.5° 锥内的 DC 与漫射通量均包含在分母。波动角度覆盖受高度采样限制，几何模型则受三角网格法线离散化影响。0.5 µm 是同一生成表面的独立采样，不是对粗图插值。两种模型均未模拟体散射、多次界面作用或标准白光积分球的仪器校正。</p></section>'
    geometric = sorted((r for r in records if r['physics_group'] == 'geometric_rays' and
                        r['wavelength_nm'] == wavelengths[0]), key=lambda r: r['height_pitch_um'], reverse=True)
    if len(geometric) == 2 and geometric[0]['haze_percent'] > 0:
        delta = geometric[1]['haze_percent']-geometric[0]['haze_percent']
        document += f'<section><h2>几何模型的网格敏感性</h2><p>Mitsuba 从 {geometric[0]["height_pitch_um"]:g} µm 到 {geometric[1]["height_pitch_um"]:g} µm 的变化为 {delta:+.6f} 个百分点，相对变化 {100*delta/geometric[0]["haze_percent"]:+.2f}%。这两级结果仍显示可见的网格敏感性，不能据此声称几何结果已充分收敛。波动与几何模型的差值也不构成真实雾度的误差上下界。</p></section>'
    if diagnostics:
        document += '<section><h2>差异来源：通量分区、横向尺度与坡度尾部</h2><p>下表从保存的角谱与原校准高度图离线计算。非 DC 阶包含窄角散射；不能将其整体作为 2.5° 雾度。周期单元的 DC 也不等同于随机表面系综的严格相干分量。</p><div class="scroll"><table><thead><tr><th>表面 / 波长</th><th>DC [%]</th><th>锥内非 DC [%]</th><th>锥外 [%]</th><th>去掉 cosθ 后 [%]</th><th>ACF 首次 1/e [µm]</th><th>频谱 RMS 坡度</th><th>解析网格折射 [%]</th></tr></thead><tbody>'
        for key, audit in diagnostics.items():
            flux, surface = audit['angular_flux'], audit['surface']
            document += (f'<tr><td>{escape(key)}</td><td>{flux["zero_order_forward_flux_percent"]:.6f}</td>'
                f'<td>{flux["within_cone_nonzero_order_flux_percent"]:.6f}</td><td>{flux["haze_percent"]:.6f}</td>'
                f'<td>{flux["haze_without_cosine_weight_percent"]:.6f}</td>'
                f'<td>{escape(str(surface["circular_radial_acf_first_1e_length_um"]))}</td>'
                f'<td>{surface["spectral_rms_slope"]:.6f}</td><td>{surface["facet_model"]["single_hit_haze_percent"]:.6f}</td></tr>')
        document += '</tbody></table></div><p>ACF 使用周期二维高度自相关的径向平均，报告首次 1/e 交点，没有拟合高斯相关长度。解析折射逐一使用原三角面的坡度和投影面积，只审查同一单次界面几何近似；它没有运行新原生后端，也没有验证 Maxwell 边界。</p>'
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        for key, audit in diagnostics.items():
            # Omit the mean-removed DC only from this log plot; retain every
            # shell in diagnostics.json and in all numerical integrations.
            rows = [r for r in audit['curves']['radial_height_spectrum']
                    if 0 < r['frequency_cycles_per_um'] <= .3]
            axes[0].semilogy([r['frequency_cycles_per_um'] for r in rows],
                             np.maximum([r['height_variance_um2'] for r in rows], 1e-30), label=key)
            axes[0].axvline(audit['surface']['acceptance_frequency_cycles_per_um'], color='gray', ls='--', alpha=.4)
            rows = audit['curves']['radial_height_acf']
            axes[1].plot([r['radius_um'] for r in rows], [r['correlation'] for r in rows], label=key)
            rows = audit['curves']['facet_slope_survival']
            axes[2].semilogy([r['slope'] for r in rows], np.maximum([r['area_above_percent'] for r in rows], 1e-8), label=key)
            threshold = audit['surface']['facet_model']['slope_threshold_for_acceptance']
            if threshold is not None:
                axes[2].axvline(threshold, color='gray', ls='--', alpha=.4)
        axes[0].set(xlim=(0, .3), xlabel='Radial spatial frequency [cycles/um]', ylabel='Height variance per shell [um^2]')
        axes[1].set(xlim=(0, 40), xlabel='Periodic radial displacement [um]', ylabel='Normalized height ACF')
        axes[1].axhline(1/np.e, color='gray', ls='--')
        axes[2].set(xlim=(0, .15), ylim=(1e-4, 100), xlabel='Facet slope magnitude', ylabel='Projected area above slope [%]')
        axes[0].legend(fontsize=6)
        path = artifacts/'haze-surface-diagnostics.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        document += '<figure class="comparison-figure">'+_preview(path, max_size=(2400, 1000))+'</figure><p>高度谱按径向壳层积分，其纵轴是高度方差而非光学角功率。相位 RMS 另存于诊断中；相位起伏并非必然远小于 1 rad，不能直接把高度 PSD 尾部比例当作雾度。详细矩、坡度阈值及曲线保存在 <a href="diagnostics.json">diagnostics.json</a>。</p></section>'
    # Optional retained research experiments are data, never executable report
    # helpers. --out keeps these and all newly derived artifacts together.
    local_artifacts = {'diagnostics.json', 'validation.json'}
    grid_path = artifacts/'geometric-grid.json'
    if grid_path.exists():
        grid = json.loads(grid_path.read_text(encoding='utf-8'))
        if grid['kind'] != 'analytic-facet-grid-sensitivity':
            raise ValueError('Unexpected geometric-grid.json experiment kind')
        local_artifacts.add(grid_path.name)
        document += '<section><h2>独立细采样：解析几何网格实验</h2><p>复用原生成器，固定坑位置、深度、平滑尺度及参考高度标定，分别直接计算每个网格；未插值粗 PNG，未写入或覆盖高度输入。这里运行的是 float64 解析 Snell/Fresnel 面积分，没有新增 Mitsuba GPU 作业。</p><table><thead><tr><th>网格 [µm]</th><th>解析雾度 [%]</th><th>RMS 面坡度</th><th>超过阈值的投影面积 [%]</th></tr></thead><tbody>'
        for row in grid['records']:
            document += f'<tr><td>{row["pitch_um"]:g}</td><td>{row["single_hit_haze_percent"]:.6f}</td><td>{row["rms_slope"]:.8f}</td><td>{row["area_above_slope_threshold_percent"]:.6f}</td></tr>'
        document += '</tbody></table><p>更细网格约束的是同一个单次界面几何模型的离散误差，仍未包含衍射或真实玻璃中缺失的细结构。<a href="geometric-grid.json">固定标定、采样方案与结果</a></p></section>'
    spectral_path = artifacts/'spectral-sensitivity.json'
    if spectral_path.exists():
        spectral = json.loads(spectral_path.read_text(encoding='utf-8'))
        if spectral['kind'] != 'illustrative-phase-screen-spectral-sensitivity':
            raise ValueError('Unexpected spectral-sensitivity.json experiment kind')
        local_artifacts.update(('spectral-sensitivity.json', 'spectral-sensitivity.csv'))
        document += '<section><h2>白光与色散：明确权重的示例积分</h2><p>使用独立 0.5 µm 高度图，380–780 nm、5 nm 步长、CIE D65 与 V(λ)，以 Q=T₀(n)×已表示的标量法向通量作近似透射权重，计算 ∫D65·V·Q·H dλ / ∫D65·V·Q dλ。T₀ 是平面界面 Fresnel 功率，Q 不是实测绝对透射率。此积分截断波段，未实现仪器校正，因此仅作模型敏感性研究。</p><table><thead><tr><th>折射率模型</th><th>D65/V 加权 [%]</th><th>10 nm 步长 [%]</th><th>单色范围 [%]</th></tr></thead><tbody>'
        for row in spectral['summaries']:
            document += f'<tr><td>{escape(row["model"])}</td><td>{row["D65_V_weighted_380_780nm_percent"]:.6f}</td><td>{row["10nm_quadrature_percent"]:.6f}</td><td>{row["minimum_monochromatic_percent"]:.4f}–{row["maximum_monochromatic_percent"]:.4f}</td></tr>'
        document += '</tbody></table><p>N-BK7 仅为有制造商色散系数的示例材料，不能据此认定论文玻璃为 N-BK7。冻结 n(530 nm) 与使用同种材料色散的差异单独体现色散影响；与 n=1.5 的比较同时改变基准折射率。给定非负光谱及透射权重，积分比值是 H(λ) 的加权平均。这个模型的几百分点单色结果不会仅因白光加权变成 20–40%。</p>'
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        for model in dict.fromkeys(r['model'] for r in spectral['records']):
            rows = [r for r in spectral['records'] if r['model'] == model]
            ax.plot([r['wavelength_nm'] for r in rows], [r['haze_percent'] for r in rows], label=model)
        ax.set(xlabel='Vacuum wavelength [nm]', ylabel='Scalar surface haze [%]', title='Same heightfield | spectral and index sensitivity')
        ax.legend(fontsize=8)
        path = artifacts/'haze-spectral-sensitivity.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        document += '<figure class="comparison-figure">'+_preview(path, max_size=(1400, 800))+'</figure><p>数据来源：<a href="https://cie.co.at/datatable/cie-standard-illuminant-d65">CIE D65</a>、<a href="https://cie.co.at/datatable/cie-spectral-luminous-efficiency-photopic-vision">CIE V(λ)</a>、<a href="https://media.schott.com/api/public/content/41e799d0bf874807a0bb8e702fbb75b5?v=54856406">SCHOTT N-BK7</a>。输入校验值、公式及全部单色结果见 <a href="spectral-sensitivity.json">JSON</a> / <a href="spectral-sensitivity.csv">CSV</a>。</p></section>'
    interface_path = artifacts/'interface-benchmark.json'
    if interface_path.exists():
        benchmark = json.loads(interface_path.read_text(encoding='utf-8'))
        if benchmark['kind'] != 'shallow-1d-Maxwell-interface-Rayleigh-collocation':
            raise ValueError('Unexpected interface-benchmark.json experiment kind')
        local_artifacts.add(interface_path.name)
        document += '<section><h2>浅正弦界面的 TE/TM 电磁边界基准</h2><p>530 nm、n=1.5→1、2.5° 半角；正弦振幅 0.12 µm，沿 y 无限延伸。用含传播及倏逝阶的反射/透射 Rayleigh 展开，在实际起伏界面匹配 TE 或 TM Maxwell 边界。TE/TM 按等入射功率的透射通量合并。结果仅验证这两个浅一维基准，不能外推为二维 AG 的电磁解。</p><table><thead><tr><th>周期 [µm]</th><th>薄相位屏 [%]</th><th>界面电磁解 [%]</th><th>相位屏差 [百分点]</th><th>解析几何 [%]</th></tr></thead><tbody>'
        for row in benchmark['records']:
            document += f'<tr><td>{row["period_um"]:g}</td><td>{row["phase_screen_haze_percent"]:.6f}</td><td>{row["unpolarized_interface_haze_percent"]:.6f}</td><td>{row["phase_minus_interface_percentage_points"]:+.6f}</td><td>{row["analytic_facet_haze_percent"]:.6f}</td></tr>'
        document += '</tbody></table><p>检查包含平面 Fresnel T=0.96/R=0.04、阶数 ±8/16/32/48、配点加密与独立偏移配点残差、能量平衡。该测试独立施加界面边界条件，超过均匀介质传播一致性检查；Rayleigh 展开在一般陡峭界面上的适用性没有在此建立。16 µm 基准的几何结果为零而电磁解非零，直接展示锥外衍射可在所有局部折射偏角都很小时存在。</p><p>参考 <a href="https://arxiv.org/html/physics/0408050">Depine &amp; Lakhtakia 的边界与 Rayleigh 展开</a>；可复现实现在现有 test_haze.py 的 InterfaceBenchmarkTests 中。<a href="interface-benchmark.json">逐阶功率、收敛及残差</a></p></section>'
    texture_path = artifacts/'texture-sensitivity.json'
    if texture_path.exists():
        texture = json.loads(texture_path.read_text(encoding='utf-8'))
        if texture['kind'] != 'controlled-AG-texture-sensitivity':
            raise ValueError('Unexpected texture-sensitivity.json experiment kind')
        local_artifacts.update(('texture-sensitivity.json', 'texture-sensitivity.csv', 'texture-protocol.json'))
        rows = texture['matrix']
        lateral = sorted(set(r['lateral_scale'] for r in rows))
        heights = sorted(set(r['height_scale'] for r in rows))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
        for ax, model, metric, title in zip(axes, ('wave', 'geometric'),
                ('haze_percent', 'single_hit_haze_percent'), ('Scalar phase screen', 'Analytic single-hit geometry')):
            values = np.array([[next(r[model][metric] for r in rows if r['lateral_scale'] == s and r['height_scale'] == d)
                                for s in lateral] for d in heights])
            plot = ax.imshow(values, origin='lower', aspect='auto', vmin=0, vmax=100, cmap='viridis')
            for iy, ix in np.ndindex(values.shape):
                ax.text(ix, iy, f'{values[iy, ix]:.2f}', ha='center', va='center',
                        color='white' if values[iy, ix] < 50 else 'black', fontsize=9)
            ax.set(xticks=range(len(lateral)), xticklabels=[f'{25*s:g}' for s in lateral],
                   yticks=range(len(heights)), yticklabels=[f'{d:g}x' for d in heights],
                   xlabel='Scaled nominal site spacing [um]', ylabel='Height factor', title=title)
            fig.colorbar(plot, ax=ax, label='Model haze [%]', shrink=.8)
        path = artifacts/'haze-texture-sensitivity.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        document += '<section><h2>受控因素：横向尺度与深度</h2><p>对原 0.5 µm 高度图施加 h_new(x,y)=d·h_original(x/s,y/s)。s 改变整个纹理的横向尺度，包括名义坑间距、1.2 µm 圆滑尺度和计算域；d 只改变高度幅值。原始 25 µm 是名义坑间距，不是严格光栅周期。所有数值仍为 530 nm、n=1.5→1、2.5° 半角下的模型估算。</p>'
        document += '<figure class="comparison-figure">'+_preview(path, max_size=(2100, 900))+'</figure><p>两因素效应相互耦合；相同深宽比可给出相同几何折射，却因相对于波长的绝对尺度不同而具有不同衍射雾度。图中最陡的反事实含少量全反射，单次界面近似的适用范围也会改变，不能把这些数值视为真实玻璃预测。</p>'
        document += '<p>将原图精确重复 2×2，计算域从 1 mm 变为 2 mm，雾度不变。这检查扩大盒子但重复同一纹理的数学一致性，并不检验独立更大随机样本的代表性。将全部凹坑反号为凸起，在当前正入射相位屏与单次几何模型中也保持雾度不变；不能将这一对称性外推到一般电磁多界面问题。</p><p>全部定义与结果：<a href="texture-sensitivity.json">JSON</a> / <a href="texture-sensitivity.csv">CSV</a>；<a href="texture-protocol.json">本次因素实验方案</a>。这些是分别记录的参数扰动，没有按论文雾度优化原表面。</p></section>'
    spacing_path = artifacts/'spacing-sensitivity.json'
    if spacing_path.exists():
        spacing = json.loads(spacing_path.read_text(encoding='utf-8'))
        if spacing['kind'] != 'spacing-at-fixed-rounding' or not spacing['complete']:
            raise ValueError('Incomplete or unexpected fixed-rounding spacing experiment')
        local_artifacts.add(spacing_path.name)
        document += '<section><h2>分离坑间距与圆滑尺度</h2><p>整体横向缩放会同时改变坑间距和圆滑尺度。这里另固定物理高斯圆滑 σ=1.2 µm、Sq≈0.1185 µm、原坑的相对布局与深度，只改变布局及计算域的横向尺寸，并由 raw_surface 独立采样。对照明确区分两个横向因素；名义间距仍不是严格光栅周期。</p><table><thead><tr><th>名义坑间距 [µm]</th><th>固定 σ=1.2 µm 波动 [%]</th><th>解析几何 [%]</th><th>Sa [µm]</th><th>ACF 1/e [µm]</th></tr></thead><tbody>'
        for row in spacing['records']:
            document += f'<tr><td>{row["nominal_site_spacing_um"]:g}</td><td>{row["wave"]["haze_percent"]:.6f}</td><td>{row["surface"]["facet_model"]["single_hit_haze_percent"]:.6f}</td><td>{row["input"]["roughness"]["Sa_um"]:.6f}</td><td>{row["surface"]["circular_radial_acf_first_1e_length_um"]:.4f}</td></tr>'
        document += '</tbody></table><p>固定的是 Sq 和相位 RMS，Sa、峰谷深度及完整高度分布可随间距/圆滑的相对尺度变化。<a href="spacing-sensitivity.json">完整定义、校准用例与结果</a></p></section>'
    morphology_path = artifacts/'morphology-sensitivity.json'
    if morphology_path.exists():
        morphology = json.loads(morphology_path.read_text(encoding='utf-8'))
        if morphology['kind'] != 'AG-morphology-sensitivity' or not morphology['complete']:
            raise ValueError('Incomplete or unexpected morphology experiment')
        local_artifacts.update(('morphology-sensitivity.json', 'morphology-protocol.json'))
        rows = morphology['records']
        document += '<section><h2>受控因素：等高度统计下的坑形与圆滑程度</h2><p>固定原 1600 个坑位置及相对深度，复用 prepare_etched_ag 的同一生成函数。原始坑形为 −depth·(1−(d1/d2)²)^p；p=1 在高斯圆滑前保留坑缘斜率，p=2 为原形状，p=4 更集中于坑中心。每个新用例均明确匹配 Sq 或 Sa，并另存校准 PNG 与 sidecar；匹配 Sq 不表示匹配峰谷深度或完整高度分布。</p><div class="scroll"><table><thead><tr><th>用例</th><th>匹配量</th><th>Sa [µm]</th><th>Sq [µm]</th><th>Sz [µm]</th><th>波动 [%]</th><th>解析几何 [%]</th><th>ACF 1/e [µm]</th><th>频谱 RMS 坡度</th></tr></thead><tbody>'
        for row in rows:
            rough, surface = row['input']['roughness'], row['surface']
            document += (f'<tr><td>{escape(row["case_id"])}</td><td>{row["matched_metric"]}</td>'
                f'<td>{rough["Sa_um"]:.5f}</td><td>{rough["Sq_um"]:.5f}</td><td>{rough["Sz_um"]:.5f}</td>'
                f'<td>{row["wave"]["haze_percent"]:.5f}</td><td>{surface["facet_model"]["single_hit_haze_percent"]:.5f}</td>'
                f'<td>{surface["circular_radial_acf_first_1e_length_um"]:.4f}</td><td>{surface["spectral_rms_slope"]:.6f}</td></tr>')
        document += '</tbody></table></div>'
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
        x = np.arange(len(rows))
        axes[0].bar(x-.18, [r['wave']['haze_percent'] for r in rows], .36, label='Scalar phase screen')
        axes[0].bar(x+.18, [r['surface']['facet_model']['single_hit_haze_percent'] for r in rows], .36, label='Analytic geometry')
        axes[0].set(xticks=x, xticklabels=[r['case_id'] for r in rows], ylabel='Model haze [%]', title='Same sites | explicit Sa or Sq matching')
        axes[0].tick_params(axis='x', labelrotation=45)
        axes[0].legend(fontsize=8)
        for row in rows:
            if row['case_id'] in ('power1', 'power2', 'power4'):
                axes[1].plot(row['profile_x_um'], row['profile_height_um'], label=f'p={row["profile_power"]:g}, matched Sq')
        axes[1].set(xlim=(0, 100), xlabel='Lateral coordinate [um]', ylabel='Height relative to mean [um]', title='Same cross-section, different bowl profiles')
        axes[1].legend(fontsize=8)
        path = artifacts/'haze-morphology-sensitivity.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        document += '<figure class="comparison-figure">'+_preview(path, max_size=(2300, 900))+'</figure><p><a href="morphology-sensitivity.json">用例、输入校准、形貌与光学统计</a>；<a href="morphology-protocol.json">固定方案</a>。这些模型不能由一个 Sa、Sq 或名义间距唯一标定。</p></section>'
    sampling_path = artifacts/'sampling-check.json'
    if sampling_path.exists():
        sampling = json.loads(sampling_path.read_text(encoding='utf-8'))
        if sampling['kind'] != 'independent-morphology-sampling-check':
            raise ValueError('Unexpected morphology sampling check')
        local_artifacts.add(sampling_path.name)
        document += '<section><h2>最小圆滑尺度的采样检查</h2><p>σ=0.3 µm、p=2 用例保留相同原始深度标定与编码步长，在两级网格独立计算；未插值较粗高度图，也没有新增原生 GPU 作业。</p><table><thead><tr><th>网格 [µm]</th><th>波动 [%]</th><th>解析几何 [%]</th></tr></thead><tbody>'
        for row in sampling['records']:
            document += f'<tr><td>{row["pitch_um"]:g}</td><td>{row["wave_haze_percent"]:.8f}</td><td>{row["geometric_haze_percent"]:.8f}</td></tr>'
        document += '</tbody></table><p>细节与精度说明见 <a href="sampling-check.json">sampling-check.json</a>。</p></section>'
    surrogate_path = artifacts/'phase-surrogates.json'
    if surrogate_path.exists():
        surrogate = json.loads(surrogate_path.read_text(encoding='utf-8'))
        if surrogate['kind'] != 'same-height-PSD-phase-surrogates':
            raise ValueError('Unexpected phase-surrogates.json experiment kind')
        local_artifacts.add(surrogate_path.name)
        document += '<section><h2>保持高度 PSD 的统计对照</h2><p>保持原二维高度 Fourier 幅值及 Sq，仅以共轭对称随机相位重排。它保持二维高度 PSD 和二点相关，但改变高度分布及更高阶空间相关，不能视为只改变高度 PDF 的实验。当前 exp(i·phase) 的非线性意味着等高度 PSD 也不严格保证等光学角谱。</p><table><thead><tr><th>seed</th><th>PSD 相对 L2 误差</th><th>Sa [µm]</th><th>Sq [µm]</th><th>波动 [%]</th><th>偏度</th><th>峰度</th></tr></thead><tbody>'
        for row in surrogate['records']:
            document += f'<tr><td>{row["seed"]}</td><td>{row["same_height_PSD_relative_L2"]:.2g}</td><td>{row["Sa_um"]:.6f}</td><td>{row["Sq_um"]:.6f}</td><td>{row["wave"]["haze_percent"]:.6f}</td><td>{row["surface"]["height_skewness"]:.5f}</td><td>{row["surface"]["height_kurtosis"]:.5f}</td></tr>'
        document += '</tbody></table><p><a href="phase-surrogates.json">对照定义与全部结果</a>。这个有限对照不能排除其他形貌或未采样微结构造成更大的变化。</p></section>'
    document += '<section><h2>实际运行与输入追溯</h2>'
    for r in records:
        folder = (results/r['angular_curve_csv']).parent
        relative = folder.relative_to(results).as_posix()+'/'
        document += '<details><summary>'+escape(f'{r["surface_id"]} / {r["backend"]} / {r["wavelength_nm"]:g} nm')+'</summary>'
        document += '<pre>'+escape(json.dumps({'input': r['input'], 'execution_context': r['execution_context'],
            'started_at_utc': r['started_at_utc'], 'finished_at_utc': r['finished_at_utc']}, ensure_ascii=False, indent=2))+'</pre>'
        for filename in ('stats.json', 'angular.csv')+(('angular_power.npy',) if r['physics_group'] != 'geometric_rays' else ()):
            document += '<a href="'+relative+filename+'">'+filename+'</a> · '
        document += '</details>'
    document += '<p><a href="haze.json">全部原始统计</a> · <a href="protocol.json">计算前固定方案</a> · <a href="validation.json">浮点及曲线审计</a></p></section></main></body></html>'
    (artifacts/'validation.json').write_text(json.dumps({'status': 'passed', 'checks': validations}, indent=2), encoding='utf-8')
    (artifacts/'diagnostics.json').write_text(json.dumps({'source_results': str(results),
        'source_haze_sha256': sha256(results/'haze.json'), 'records': diagnostics}, ensure_ascii=False, indent=2), encoding='utf-8')
    if output:
        import re
        prefix = os.path.relpath(results, artifacts).replace(os.sep, '/')+'/'
        document = re.sub(r'href="(?!https:|data:)([^"]+)"',
            lambda m: m[0] if m[1] in local_artifacts else 'href="'+prefix+m[1]+'"', document)
    entry.write_text(document, encoding='utf-8')
    return entry

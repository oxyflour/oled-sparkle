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

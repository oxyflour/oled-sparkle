# OLED 闪点仿真

## 当前闪点报告与复现

当前主报告使用**全绿画面、不加 LDF**，按三次 140 µm 移动平均（MWA）分离 OLED 周期纹理，随后对全部 441 个测量像素计算 `sparkle (%) = 100 × std(I, ddof=0) / mean(I)`。三个波动后端约为 **5.292%**，Mitsuba 为 **5.409%**，0.5 µm 网格的 Torch 复核为 **5.136%**。这些是固定参考测量方法下的仿真值，尚未按原论文的实际仪器参数校准；其他方法、孔径对照和原始统计保留在报告附录。

直接查看已提交的[闪点简报](test-results/oled-sms-reference/sparkle.html)。完整离线报告含输入原图和浮点下载，可按下文命令生成，输出到 `test-results/oled-sms-reference/sparkle-report.html`。

Mitsuba CUDA 求解器现已采用 GPU 常驻积分，保留原有物理模型、全部瞳孔/网格采样和 `ray_chunk` 上限。RTX 3090 Ti 上，完整随机坑算例由 **223.33 秒降至 0.41–0.75 秒**，按两次新实现耗时中位数约快 **387 倍**。四组完整算例的统计亮度图最大相对 L2 误差为 **7.74×10⁻⁷**，CV 最大变化为 **0.000006 个百分点**；42 项回归测试全部通过。[精度与性能记录](test-results/mitsuba-resident-regression/validation.json)、[复现脚本](test-results/benchmark-mitsuba-resident.py)和[详细验证清单](oled-ag-sparkle/VALIDATION.json)已保存。

### 1. 准备环境

从仓库根目录执行以下 PowerShell 命令。完整光学复算需要 Python 3.12、`uv`、支持 CUDA 的 NVIDIA GPU，以及 Windows 下的 WSL2 Chromatix CUDA 环境。

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend cu130 "torch==2.13.0" -r oled-ag-sparkle\requirements-core.txt -r oled-ag-sparkle\requirements-optional.txt
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

最后一行应报告 CUDA 可用。本次 Windows 计算使用 Torch `2.13.0+cu130`、TorchOptics `1.0.2`、Mitsuba `3.9.1`；完整版本和设备记录见各次运行的 `run.json`。脚本优先使用 `.venv` 中的 CUDA Torch；本机旧缓存仅作为兼容回退，新环境无需复用缓存路径。

Chromatix 的 WSL2 安装步骤见 [Windows / WSL 环境说明](oled-ag-sparkle/references/windows-wsl.md)。使用其他发行版或虚拟环境时，运行前设置：

```powershell
$env:SPARKLE_WSL_DISTRO = 'Ubuntu-22.04'
$env:SPARKLE_WSL_PYTHON = '/home/your-user/.venvs/oled-sparkle-chromatix/bin/python'
```

### 2. 完整复算报告

```powershell
& .\test-results\run-sms-reference.ps1 *> .\test-results\logs\sms-reference.log
```

该命令依次完成：

1. 按种子 `20261107` 重新生成 1 µm / 0.5 µm 两级高度图，保持同一几何与物理高度标尺。
2. 保留原始发光图 G 通道、将 R/B 置零，写出 `protocol.json` 并执行 dry-run。
3. 在 1 µm 网格上运行 Torch、TorchOptics、Chromatix（WSL CUDA）和 Mitsuba。
4. 运行 0.5 µm 网格的 Torch 复核，以及原 NA=0.08 的 Torch 孔径对照，共六次光学计算。
5. 分离周期纹理、统计 sparkle、绘图并审计浮点结果，生成主视图只展示闪点的简报和完整报告。

仅检查 CUDA、重建输入并做 dry-run，可用 `& .\test-results\run-sms-reference.ps1 -PrepareOnly`。该操作不运行光学求解。

### 3. 不重跑光学，仅重建报告

仓库保留全部六次运行的原始浮点亮度、参数和输入图。若只需重建 HTML、图表及 sparkle 统计，执行：

```powershell
& .\test-results\run-sms-reference.ps1 -ReportOnly
```

此模式只需本地 Python 的 NumPy、SciPy、Pillow、Matplotlib 等依赖，不要求当前机器具有 CUDA 或 WSL。它读取保存的计算结果，不重新生成高度图或改写计算前的 `protocol.json`。也可直接运行 `.venv\Scripts\python.exe test-results\report-sms-reference.py`。

### 4. 输出、口径与验证

| 路径 | 内容 |
|---|---|
| `test-results/oled-sms-reference/sparkle.html` | 轻量闪点报告；方法对照与原始统计折叠在附录 |
| `test-results/oled-sms-reference/sparkle-report.html` | 完整离线 HTML，内嵌原始输入及 42 个 NumPy 下载；作为大型可重建产物，不纳入 Git |
| `test-results/oled-sms-reference/protocol.json` | 计算前固定的相机、滤波和比较条件 |
| `test-results/oled-sms-reference/summary.json` | 六次计算的结果、平面控制、网格检查、代码哈希与审计记录 |
| `test-results/oled-sms-reference/{1um,0p5um,old-na-0p08}/<backend>/brightness.npy` | 原始光学亮度，用于重新处理 |
| 同目录的 `sms_mwa_measurement_pixels.npy` / `sms_stats.json` | 去周期后的 MWA 测量像素及 sparkle 统计 |
| `test-results/logs/sms-reference.log` | 完整复算日志 |

固定参考条件为：相同合成 AG 表面（Sa=0.10 µm）、500 µm OLED–AG 间距、530 nm、50 mm 镜头、F/8、400 mm 物距、3.75 µm 传感器像素；物方测量像素为 26.25 µm。主视图采用三次 140 µm MWA 的 sparkle；原先计算前固定的频域方案、孔径敏感性与原始统计完整保留在附录和 `protocol.json`，没有改写原始实验方案。

运行回归测试：

```powershell
.venv\Scripts\python.exe -m unittest discover -s oled-ag-sparkle\tests -v
```

已验证 37 项测试，33 项通过、4 项可选集成测试跳过；六次大尺寸 GPU 计算另行完成并审计。小数值可能随 GPU、依赖版本与浮点运算顺序略有差异，应结合已记录的网格敏感性判断。

## 其他实验与历史基准

使用 PyTorch FFT、TorchOptics、Chromatix 和 Mitsuba 计算 OLED 经高度表面后收集到的亮度图，并生成包含输入图、完整参数、实际设备、耗时和闪点值的离线 HTML 报告。

Windows 下的 Chromatix 自动调用 **WSL2 CUDA** 环境；Torch/TorchOptics 和 Mitsuba 在 Windows 中运行。WSL 安装与环境覆盖配置见 [Windows / WSL 说明](oled-ag-sparkle/references/windows-wsl.md)。

当前计算区域为 **1 mm × 1 mm**；默认高度图为**直径 3–5 µm、深度 0.5 µm、覆盖率约 50% 的随机圆坑**。39,048 个圆坑互不重叠，采用平滑碗形剖面，随机种子为 20261107。高度图为 **1000 × 1000、1 µm 采样、16 位 PNG**。发光图先从原 128 × 128 图块裁出 **[26,58,96,128] 的 70 × 70 完整重复单元**，保留 **2 µm/texel、140 µm × 140 µm**，**8 × 8 整数重复后裁取 500 × 500**，扩展时不缩放或插值。每个 **50 µm × 50 µm** 统计像素对 50 × 50 个计算像素作面积平均，随后对全部 400 个统计像素直接计算：

`CV (%) = 100 × std(I, ddof=0) / mean(I)`

[周期单元裁剪后的四后端报告](test-results/oled-1mm-pits3to5um-periodic-50um/report.html) · [裁剪位置及重复检查](test-results/emission-periodic-crop.png) · [高度图对照](test-results/random-pit-periodic-height-comparison.png) · [亮度图对照](test-results/random-pit-periodic-brightness-comparison.png) · [结果与网格检查](test-results/random-pit-periodic-comparison.json) · [计算与验证记录](oled-ag-sparkle/VALIDATION.json)

原高度图的[周期发光图基准报告](test-results/oled-1mm-original-periodic-50um/report.html)可通过 `--height-surface original` 复算。随机圆坑还使用 **0.5 µm 光学网格**对原高度和新高度作两级采样检查，统计像素保持 50 µm。

新增[化学蚀刻 AG 示意表面仿真](test-results/oled-etched-ag/report.html)：1 mm 方形区域内生成 1600 个不规则 Voronoi 浅坑，名义中心间距 25 µm，面粗糙度 Sa = 0.10 µm，坑沿平滑尺度 1.2 µm。这是参考公开形貌建立的合成模型；其 Sa 与单元等面积直径分别不等于厂家轮廓 Ra 与 Rsm。报告包含 1 µm 网格的四后端结果与同一几何在 0.5 µm 网格的 Torch 检查。[高度形貌图](test-results/oled-etched-ag/surface-overview.png)、[亮度对照图](test-results/oled-etched-ag/brightness-overview.png)、[独立浮点审计](test-results/oled-etched-ag/validation.json)及[复现脚本](test-results/run-etched-ag.ps1)一并保留。生成器为 `oled-ag-sparkle/scripts/prepare_etched_ag.py`；执行器为 `scripts/run_etched_ag_demo.py`，从 PNG 元数据读取实际高度编码范围。

[全绿、无 LDF 的 SMS 公开方法参考报告](test-results/oled-sms-reference/sparkle-report.html)采用固定的 50 mm / F8 / 400 mm 相机近似、3.75 µm 传感器像素和相同 AG 表面。先写 `protocol.json` 再计算；中心 560 µm 区域去除 OLED 周期后，在 21×21 个 26.25 µm 物方测量像素上计算随机对比度。预先固定的频域方法为 **23.575%**；三次 140 µm 完整重复单元移动平均为 **5.292%**，Mitsuba 的移动平均结果为 **5.409%**。0.5 µm 网格 Torch 分别为 **23.422% / 5.136%**；平面控制分别为 **0.119% / 0.150%**。后者接近文献 4.8–6.0 的读数量级，但方法带宽和相机孔径影响明显，不能据此认定实测校准成功。原 NA=0.08 时，频域和移动平均分别为 **5.181% / 0.686%**。报告包含原始与两种分离后浮点图、完整参数及敏感性，主算法在 `oled-ag-sparkle/scripts/sms_reference.py`，复现入口为 `test-results/run-sms-reference.ps1`。本次回归测试运行 37 项，33 项通过、4 项可选集成测试跳过；六次大尺寸 GPU 光学计算另行完成并审计。

此前的拉伸输入把发光单元放大了 **3.90625 倍**；随后直接重复原 128 × 128 图块也会产生相位接缝。两种历史输入的 CV 结论均不能代替当前案例。现在先提取完整周期单元：其中有 2 红、4 绿、2 蓝完整高亮区域，边缘不截断这些主体，横纵跨边界变化都小于图内相邻像素平均变化。未做边缘混合或补色，物理尺度仍为示意图假设。

当前 1 µm 网格结果：三种波动后端为 **39.051% → 22.626%**，相对下降 **42.06%**；Mitsuba 为 **42.347% → 21.055%**，相对下降 **50.28%**。0.5 µm 网格的 Torch 复核为 **39.239% → 22.793%**，相对下降 **41.91%**。CV 使用全部 400 个统计像素，包含 OLED 图案本身的不均匀性；两级网格用于检查采样敏感性。

安装与模型说明见 [skill 文档](oled-ag-sparkle/README.md) 和 [SKILL.md](oled-ag-sparkle/SKILL.md)。运行示例：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\run_oled_demo.py --backend all --out test-results\oled-1mm-pits3to5um-periodic-50um
```

工作区的完整平铺发光图计算、两种高度比较、网格检查、绘图与打包流程为 `test-results/run-pit-comparison.ps1`；绘图审计需要 matplotlib 和 scipy。`test-results/run-comparison.ps1` 显式使用 `--emission-layout legacy-stretched`，仅用于复现历史拉伸输入。不重算光学结果、重新生成当前对照表及对照图：

```powershell
.venv\Scripts\python.exe test-results\compare-random-pits.py
```

仓库保存输入素材、源码、测试、复现脚本及最新完整计算的浮点结果、报告和日志。原始 skill 压缩包作为打包脚本的基准输入保留；重新生成的发布包、虚拟环境、依赖缓存和历史批量输出由 Git 忽略。

周期单元基准验证为 **28 项测试全部通过**，包含周期检测、完整亮区与拼接连续性、裁剪像素保持、圆坑几何、深度极性、Windows → WSL GPU 转发、中文／空格路径、失败处理和四后端对照，记录位于 `test-results/logs/validation-periodic.log`。对应结果的几何、浮点 CV、面积平均与离线下载审计见 `test-results/random-pit-periodic-validation.json`。新增蚀刻示意模型后，回归测试运行 31 项，27 项通过、4 项可选集成测试跳过；本次另完成并审计四后端 GPU 仿真及 Torch 细网格复核，记录见 `test-results/logs/validation-etched-ag.log` 和 `test-results/oled-etched-ag/validation.json`。

# OLED 闪点仿真

## 脚本入口

以下命令从仓库根目录执行。输入准备、求解、后处理和报告各自复用已有实现；`test-results/` 保存实验记录，历史复现脚本不作为新的默认入口。

| 入口（位于 `oled-ag-sparkle/scripts/`） | 职责与用法 |
|---|---|
| `run_oled_demo.py` | 当前默认 OLED 用例。默认蚀刻 AG、1 µm 网格、四后端；`--height-surface` 选择表面，`--pitch-um .5` 作网格复核。输出原始亮度 CV。 |
| `run_sms_reference.py` | 全绿、无 LDF 的分离闪点流程。默认 `--stage all` 运行六个参考工况；加 `--layouts` 进行统一间距、亮度与测量带宽的多排列比较。 |
| `generate_report.py` | 唯一报告 CLI。默认重建原始亮度报告；`--kind sms-reference` 按 `protocol.json` 重建原参考流程或多排列分离闪点报告，不重跑光学。 |
| `sparkle_multibackend.py` | 自定义输入的通用四后端求解 CLI，支持 `--help` 查看所有物理与数值参数。 |
| `run_legacy_demo.py` | 仅保留历史测试素材，必须指定 `--example quick` 或 `--example 1mm`；默认只选 Torch。当前 OLED 用例使用 `run_oled_demo.py`。 |
| `prepare_etched_ag.py`、`prepare_random_pits.py`、`prepare_oled_1mm.py`、`prepare_tiled_emission.py` | 输入生成器，由运行器调用；需要单独生成或修改输入时才直接使用。 |

`reporting.py` 是离线报告实现，`sms_reference.py` 是周期分离与统计实现，`chromatix_wsl.py` 是 WSL 适配模块。它们由上述入口导入，不是额外的用户运行入口。蚀刻 AG 专用执行器已并入 `run_oled_demo.py`，SMS 报告也统一由 `generate_report.py` 生成。

`run_legacy_demo.py` 中的 `quick` 是 128×128、2 µm 采样的合成小网格，用于 CLI、报告及后端回归测试；`1mm` 是 2000×2000、0.5 µm 采样的旧合成粗糙面，搭配普通像素或合成绿色子像素光源，用于旧参数检查和历史复现。两者保留原始测试参数。

## 四种 OLED 排列：先相机采样，再去周期

[相机过程比较报告](test-results/oled-camera-comparison/sparkle.html)采用现有相机参数：50 mm、F/8、400 mm 物距、3.75 µm 单色像元。薄透镜放大率为 1/7，物方像元为 26.25 µm；理想圆形光瞳的衍射已在波动求解器内实施。逐源计算 AG 传播及成像后，**先做方形感光区的面积积分和离散采样，再做周期分离**。未使用额外 MWA、低通或 LDF；采用无像差、100% 填充率、线性、无噪声和无量化的明确理想模型，没有实测镜头 MTF。

沿用同一蚀刻 AG、1 mm 输入场、OLED–AG 间距 500 µm 和 530 nm 全绿光。名义逻辑像素间距 **P=70 µm** 定义为每个逻辑地址的面积 P²，并非子像素最近邻距离。2 µm 发光纹理在 sRGB 解码后将全场 G 均值归一至 1，`--emission-scale` 在求解器中施加，不裁剪峰值，阈值为 0。三个新增排列是公开图示启发的几何近似；钻石保留原始单元。

| 排列 | 精确 G 周期 X×Y [µm] | 1 µm 网格对比度 [%] | 0.5 µm 复核 [%] |
|---|---:|---:|---:|
| diamond | 140×140 | 24.016988 | 23.977983 |
| rgb-stripe | 70×70 | 8.880589 | 8.715334 |
| rgb-delta | 210×140 | 26.721904 | 26.486939 |
| windmill | 140×140 | 24.147108 | 24.082596 |

去周期采用 [IEC 62977-3-9:2023](https://webstore.iec.ch/en/publication/64780) 信息性附录 B 所列的**频域掩膜方法**，相机 MTF、采样混叠及采样后去周期的依据见 [Becker 2021 §3.1–3.4](https://www.display-messtechnik.de/fileadmin/template/main/docs/10-Years-SparkleMeasurement--edC-2021-V2.pdf)。标准没有指定本实现的具体掩膜。本实现按精确周期枚举光学强度带宽内的倒格矢，模采样频率折回基带，再取四种排列频点的并集。所有排列剔除相同的精确 DFT bin，保留 DC；不根据输出幅度、目标读数或峰值阈值调整。周期和随机分量重合的 bin 无法分辨，会一同剔除。

中心 ROI 为 **840×840 µm = 32×32 像元**，同时包含所有精确周期的整数倍；共同掩膜剔除 89 个、保留 934 个非 DC bin。光学强度截止为 29.480 cycles/mm，传感器 Nyquist 为 19.048 cycles/mm，故必须考虑采样折叠。最终直接从去周期后的传感器浮点数组计算 `100×std(ddof=0)/mean`，不截断负值、不再次积分。这是声明频率选择下的**相机域随机对比度**，未加视觉 CSF 权重，不是校准的 SMS-1000 读数，也不宣称完整 IEC 符合性。

三个波动后端复用经哈希校验的原理想光瞳浮点结果，最大相对 L2 差约 1.37×10⁻⁵。0.5 µm 网格差异最大 0.235 个百分点；该检查未细化 2 µm 的点源间距。实际离散光瞳与解析圆形光瞳的轴向 MTF 最大绝对差为 0.00333。有限视场无 AG 控制残余最高 0.642%，无限周期无 AG 控制低于 3×10⁻¹³%。这些残余不从主结果扣除。报告另列四种传感器相位、完整去周期图的中心子区，以及独立 420 µm ROI 的检查；独立小 ROI 同时改变 DFT bin 宽度，不能把其差异全归为空间不均匀。单个 AG 实现及小视场的结果不代表商业面板排名；钻石与风车的差距也小于这些敏感性。

从仓库根目录复现本次采样分析，不重跑已完成的光学段：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\run_sms_reference.py --layouts diamond rgb-stripe rgb-delta windmill --measurement sampled-camera --reuse-optics test-results\oled-layout-comparison --stage all --out test-results\oled-camera-new
```

`--reuse-optics` 仅接受相同相机、布局、间距及求解器版本；冻结来源 protocol、run、stats 和原始亮度文件的 SHA-256，并写入**新的**结果目录。省略此参数可从现有入口重算光学；默认主网格 `--backend all` 会运行四后端，加四次 0.5 µm Torch 共 20 次，其中 Mitsuba 几何光线结果缺少衍射，不进入此次相机报告；`--backend torch` 可缩小运行范围。CUDA 环境配置见下文。`--stage prepare/main/fine` 可分段运行新光学批次；复用批次的 prepare 只固定采样方案，之后用报告 CLI 重评。固定的 840 µm ROI 要求输入周期与之相容。

只重建本次已有报告：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\generate_report.py --kind sms-reference --results test-results\oled-camera-comparison
```

`test-results/oled-camera-comparison/` 只保存本次直接需要的结果：`protocol.json` 固定测量方法及光学来源；`summary.json` 和逐运行 `camera_stats.json` 保存审计和统计；`camera_pixels.npy` 保存积分采样结果，`camera_separated.npy` 保存去周期结果，`camera_frequency_mask.npy` 保存掩膜；`controls/` 保存有限/无限周期无 AG 控制；PNG 和 `sparkle.html` 由已有报告 CLI 生成。没有新增脚本或入口，原光学数组和历史报告保持原样。

旧[共同 MWA 报告](test-results/oled-layout-comparison/sparkle.html)保留用于追溯，明确以 `--measurement optical-mwa` 选择历史方案。其 70/140/210 µm 拼接窗口不作为本次相机评价方法。历史钻石按三次 140 µm MWA 为 **5.291803%**，本批次同口径为 **5.287834%**；不能把这些数值等同于保留更多相机频带的本次结果。

`assets/oled-layouts/<layout>/` 继续由 `prepare_tiled_emission.py` 维护：RGB 周期单元、1 mm RGB/full-green 输入与 `.source.json` 保存几何、周期、亮度倍率和哈希。`assets/oled-layout-reference/` 保存公开参考图及来源，参考图不直接作为光学强度。单独生成输入可用 `prepare_tiled_emission.py --layout rgb-delta`。

## 单排列 MWA 参考报告与复现

以下保留的单排列参考报告使用**全绿画面、不加 LDF**，按三次 140 µm 移动平均（MWA）分离 OLED 周期纹理，随后对全部 441 个测量像素计算 `sparkle (%) = 100 × std(I, ddof=0) / mean(I)`。三个波动后端约为 **5.292%**，Mitsuba 为 **5.409%**，0.5 µm 网格的 Torch 复核为 **5.136%**。这些是固定参考测量方法下的仿真值，尚未按原论文的实际仪器参数校准；其他方法、孔径对照和原始统计保留在报告附录。

直接查看[本批次闪点报告](test-results/oled-sms-current/sparkle.html)。四后端在同一 GPU 上串行重跑，主工况运行 ID 为 `87e679353334459a9bedb168895129e3`，批次为 `30ef6b2cac9c4acda85c13c9b523de0c`。

| 求解器 | 本次闪点 (%) | 本次求解耗时 (s) |
|---|---:|---:|
| PyTorch FFT | 5.291803 | 28.453979 |
| TorchOptics | 5.291731 | 24.865079 |
| Chromatix | 5.291656 | 54.366454 |
| Mitsuba | 5.409319 | 1.537723 |

本次六次光学计算的图像、浮点结果和耗时均位于 `test-results/oled-sms-current/`。完整离线报告含输入原图和浮点下载，可按下文命令生成，输出到 `test-results/oled-sms-current/sparkle-report.html`。

### 默认 heightfield 用例

skill 的默认高度图为 `oled-ag-sparkle/assets/oled-1mm-etched-ag/heightfield.png`：1 mm × 1 mm、1000×1000、16 位、1600 个不规则 Voronoi 浅坑，名义间距 25 µm、Sa=0.10 µm。黑至白编码高度差为 **0.5361845861372564 µm**，由相邻元数据读取；实际峰谷差约 0.515 µm。唯一默认用例入口为 `run_oled_demo.py`。原圆坑通过 `--height-surface random-pits` 显式选择，快速测试通过 `run_legacy_demo.py --example quick` 选择。

默认高度图的四后端原始亮度报告可这样刷新；闪点主报告继续使用前文的全绿 MWA 测量流程：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\run_oled_demo.py --backend all --out test-results\oled-1mm-etched-ag-periodic-50um
```

完整复现环境需要先按下节安装 CUDA Torch。当前默认用例已实际跑完四个后端，[输入与原始计算报告](test-results/oled-1mm-etched-ag-periodic-50um/report.html)记录每个后端的实际设备和耗时。

同一蚀刻几何的细网格复核使用同一个入口，不另设表面专用脚本：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\run_oled_demo.py --pitch-um .5 --backend torch --out test-results\oled-etched-ag\0p5um
```

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

该 PowerShell 包装只负责本机 CUDA 环境和已有结果路径。也可以直接使用 skill 内入口：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\run_sms_reference.py --stage all --out test-results\oled-sms-current
```

该命令依次完成：

1. 按种子 `20261107` 重新生成 1 µm / 0.5 µm 两级高度图，保持同一几何与物理高度标尺。
2. 保留原始发光图 G 通道、将 R/B 置零，写出 `protocol.json` 并执行 dry-run。
3. 在 1 µm 网格上运行 Torch、TorchOptics、Chromatix（WSL CUDA）和 Mitsuba。
4. 运行 0.5 µm 网格的 Torch 复核，以及原 NA=0.08 的 Torch 孔径对照，共六次光学计算。
5. 分离周期纹理、统计 sparkle、绘图并审计浮点结果，生成主视图只展示闪点的简报和完整报告。

仅检查 CUDA、重建输入并做 dry-run，可用 `& .\test-results\run-sms-reference.ps1 -PrepareOnly`。该操作不运行光学求解。

### 3. 不重跑光学，仅重建报告

当前结果目录保留本次六次运行的原始浮点亮度、参数和输入图。生成器核对计算均晚于本批次声明时间、求解器代码哈希一致、后端运行 ID 匹配。若只需重建 HTML、图表及 sparkle 统计，执行：

```powershell
& .\test-results\run-sms-reference.ps1 -ReportOnly
```

此模式只需本地 Python 的 NumPy、SciPy、Pillow、Matplotlib 等依赖，不要求当前机器具有 CUDA 或 WSL。它读取保存的计算结果，不重新生成高度图或改写计算前的 `protocol.json`。对应的统一报告入口为：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\generate_report.py --kind sms-reference --results test-results\oled-sms-current --out test-results\report.html
```

报告依次展示输入、计算与测量条件、四求解器结果、验证和附录。四后端默认串行运行，主表仅有本次求解耗时一列；所有图像、闪点和对照来自同一新计算批次。`--out` 可选，用于复制一份轻量入口并修正相对链接；完整报告始终写入结果目录的 `sparkle-report.html`。该命令只更新结果目录及指定入口，不打包文件，也不改写 skill 的验证清单。


图 1 的白色虚线框标出实际测量区域。1 mm 输入范围为 ±500 µm；预先选取中心 560 µm（4 个完整的 140 µm 周期），再居中放入 21 个 26.25 µm 测量像素，得到宽度 551.25 µm，因此图 2 的边界为 **±275.625 µm**，全部 441 个测量像素均参与闪点统计。

### 4. 输出、口径与验证

| 路径 | 内容 |
|---|---|
| `test-results/oled-sms-current/sparkle.html` | 轻量闪点报告；方法对照与原始统计折叠在附录 |
| `test-results/oled-sms-current/solver-comparison.html` | 同一轻量报告的四求解器对比入口 |
| `test-results/oled-sms-current/sparkle-report.html` | 完整离线 HTML，内嵌原始输入及 42 个 NumPy 下载；作为大型可重建产物，不纳入 Git |
| `test-results/oled-sms-current/protocol.json` | 计算前固定的相机、滤波和比较条件 |
| `test-results/oled-sms-current/summary.json` | 六次计算的结果、平面控制、网格检查、代码哈希与审计记录 |
| `test-results/oled-sms-current/{1um,0p5um,old-na-0p08}/<backend>/brightness.npy` | 原始光学亮度，用于重新处理 |
| 同目录的 `sms_mwa_measurement_pixels.npy` / `sms_stats.json` | 去周期后的 MWA 测量像素及 sparkle 统计 |
| `test-results/logs/sms-reference.log` | 完整复算日志 |

固定参考条件为：相同合成 AG 表面（Sa=0.10 µm）、500 µm OLED–AG 间距、530 nm、50 mm 镜头、F/8、400 mm 物距、3.75 µm 传感器像素；物方测量像素为 26.25 µm。主视图采用三次 140 µm MWA 的 sparkle；原先计算前固定的频域方案、孔径敏感性与原始统计完整保留在附录和 `protocol.json`，没有改写原始实验方案。

运行回归测试：

```powershell
.venv\Scripts\python.exe -m unittest discover -s oled-ag-sparkle\tests -v
```

本次四后端主工况、细网格及孔径对照全部重新计算，并记录统一批次、代码哈希、设备和实际耗时。小数值可能随 GPU、依赖版本与浮点运算顺序略有差异，应结合已记录的网格敏感性判断。

## 其他实验与历史基准

历史 Mitsuba 优化基准：Mitsuba CUDA 求解器现已采用 GPU 常驻积分，保留原有物理模型、全部瞳孔/网格采样和 `ray_chunk` 上限。RTX 3090 Ti 上，完整随机坑算例由 **223.33 秒降至 0.41–0.75 秒**，按两次新实现耗时中位数约快 **387 倍**。四组完整算例的统计亮度图最大相对 L2 误差为 **7.74×10⁻⁷**，CV 最大变化为 **0.000006 个百分点**；42 项回归测试全部通过。[精度与性能记录](test-results/mitsuba-resident-regression/validation.json)、[复现脚本](test-results/benchmark-mitsuba-resident.py)和[详细验证清单](oled-ag-sparkle/VALIDATION.json)已保存。

使用 PyTorch FFT、TorchOptics、Chromatix 和 Mitsuba 计算 OLED 经高度表面后收集到的亮度图，并生成包含输入图、完整参数、实际设备、耗时和闪点值的离线 HTML 报告。

Windows 下的 Chromatix 自动调用 **WSL2 CUDA** 环境；Torch/TorchOptics 和 Mitsuba 在 Windows 中运行。WSL 安装与环境覆盖配置见 [Windows / WSL 说明](oled-ag-sparkle/references/windows-wsl.md)。

历史圆坑基准的计算区域为 **1 mm × 1 mm**；高度图为**直径 3–5 µm、深度 0.5 µm、覆盖率约 50% 的随机圆坑**。39,048 个圆坑互不重叠，采用平滑碗形剖面，随机种子为 20261107。高度图为 **1000 × 1000、1 µm 采样、16 位 PNG**。发光图先从原 128 × 128 图块裁出 **[26,58,96,128] 的 70 × 70 完整重复单元**，保留 **2 µm/texel、140 µm × 140 µm**，**8 × 8 整数重复后裁取 500 × 500**，扩展时不缩放或插值。每个 **50 µm × 50 µm** 统计像素对 50 × 50 个计算像素作面积平均，随后对全部 400 个统计像素直接计算：

`CV (%) = 100 × std(I, ddof=0) / mean(I)`

[周期单元裁剪后的四后端报告](test-results/oled-1mm-pits3to5um-periodic-50um/report.html) · [裁剪位置及重复检查](test-results/emission-periodic-crop.png) · [高度图对照](test-results/random-pit-periodic-height-comparison.png) · [亮度图对照](test-results/random-pit-periodic-brightness-comparison.png) · [结果与网格检查](test-results/random-pit-periodic-comparison.json) · [计算与验证记录](oled-ag-sparkle/VALIDATION.json)

原高度图的[周期发光图基准报告](test-results/oled-1mm-original-periodic-50um/report.html)可通过 `--height-surface original` 复算。随机圆坑还使用 **0.5 µm 光学网格**对原高度和新高度作两级采样检查，统计像素保持 50 µm。

默认高度图采用[化学蚀刻 AG 示意表面仿真](test-results/oled-etched-ag/report.html)：1 mm 方形区域内生成 1600 个不规则 Voronoi 浅坑，名义中心间距 25 µm，面粗糙度 Sa = 0.10 µm，坑沿平滑尺度 1.2 µm。这是参考公开形貌建立的合成模型；其 Sa 与单元等面积直径分别不等于厂家轮廓 Ra 与 Rsm。报告包含 1 µm 网格的四后端结果与同一几何在 0.5 µm 网格的 Torch 检查。[高度形貌图](test-results/oled-etched-ag/surface-overview.png)、[亮度对照图](test-results/oled-etched-ag/brightness-overview.png)、[独立浮点审计](test-results/oled-etched-ag/validation.json)及[复现脚本](test-results/run-etched-ag.ps1)一并保留。生成器为 `oled-ag-sparkle/scripts/prepare_etched_ag.py`；执行统一使用 `scripts/run_oled_demo.py`，从 PNG 元数据读取实际高度编码范围。

[全绿、无 LDF 的 SMS 公开方法参考报告](test-results/oled-sms-reference/sparkle-report.html)采用固定的 50 mm / F8 / 400 mm 相机近似、3.75 µm 传感器像素和相同 AG 表面。先写 `protocol.json` 再计算；中心 560 µm 区域去除 OLED 周期后，在 21×21 个 26.25 µm 物方测量像素上计算随机对比度。预先固定的频域方法为 **23.575%**；三次 140 µm 完整重复单元移动平均为 **5.292%**，Mitsuba 的移动平均结果为 **5.409%**。0.5 µm 网格 Torch 分别为 **23.422% / 5.136%**；平面控制分别为 **0.119% / 0.150%**。后者接近文献 4.8–6.0 的读数量级，但方法带宽和相机孔径影响明显，不能据此认定实测校准成功。原 NA=0.08 时，频域和移动平均分别为 **5.181% / 0.686%**。报告包含原始与两种分离后浮点图、完整参数及敏感性，主算法在 `oled-ag-sparkle/scripts/sms_reference.py`，复现入口为 `test-results/run-sms-reference.ps1`。本次回归测试运行 37 项，33 项通过、4 项可选集成测试跳过；六次大尺寸 GPU 光学计算另行完成并审计。

此前的拉伸输入把发光单元放大了 **3.90625 倍**；随后直接重复原 128 × 128 图块也会产生相位接缝。两种历史输入的 CV 结论均不能代替当前案例。现在先提取完整周期单元：其中有 2 红、4 绿、2 蓝完整高亮区域，边缘不截断这些主体，横纵跨边界变化都小于图内相邻像素平均变化。未做边缘混合或补色，物理尺度仍为示意图假设。

当前 1 µm 网格结果：三种波动后端为 **39.051% → 22.626%**，相对下降 **42.06%**；Mitsuba 为 **42.347% → 21.055%**，相对下降 **50.28%**。0.5 µm 网格的 Torch 复核为 **39.239% → 22.793%**，相对下降 **41.91%**。CV 使用全部 400 个统计像素，包含 OLED 图案本身的不均匀性；两级网格用于检查采样敏感性。

安装与模型说明见 [skill 文档](oled-ag-sparkle/README.md) 和 [SKILL.md](oled-ag-sparkle/SKILL.md)。运行示例：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\run_oled_demo.py --height-surface random-pits --backend all --out test-results\oled-1mm-pits3to5um-periodic-50um
```

工作区的完整平铺发光图计算、两种高度比较、网格检查、绘图与打包流程为 `test-results/run-pit-comparison.ps1`；绘图审计需要 matplotlib 和 scipy。`test-results/run-comparison.ps1` 显式使用 `--emission-layout legacy-stretched`，仅用于复现历史拉伸输入。不重算光学结果、重新生成当前对照表及对照图：

```powershell
.venv\Scripts\python.exe test-results\compare-random-pits.py
```

仓库保存输入素材、源码、测试、复现脚本及最新完整计算的浮点结果、报告和日志。原始 skill 压缩包作为打包脚本的基准输入保留；重新生成的发布包、虚拟环境、依赖缓存和历史批量输出由 Git 忽略。

周期单元基准验证为 **28 项测试全部通过**，包含周期检测、完整亮区与拼接连续性、裁剪像素保持、圆坑几何、深度极性、Windows → WSL GPU 转发、中文／空格路径、失败处理和四后端对照，记录位于 `test-results/logs/validation-periodic.log`。对应结果的几何、浮点 CV、面积平均与离线下载审计见 `test-results/random-pit-periodic-validation.json`。新增蚀刻示意模型后，回归测试运行 31 项，27 项通过、4 项可选集成测试跳过；本次另完成并审计四后端 GPU 仿真及 Torch 细网格复核，记录见 `test-results/logs/validation-etched-ag.log` 和 `test-results/oled-etched-ag/validation.json`。

## 一次性打包 skill

发布包从当前 skill 源码和素材生成；不保留独立打包脚本。以下 PowerShell 命令排除缓存、运行输出和可重建的 NumPy 高度数组：

```powershell
@'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
skill = Path('oled-ag-sparkle')
with ZipFile('oled-ag-sparkle-skill-fixed.zip', 'w', ZIP_DEFLATED) as archive:
    for path in sorted(skill.rglob('*')):
        if path.is_file() and not {'__pycache__', 'outputs'}.intersection(path.relative_to(skill).parts) and path.suffix not in ('.npy', '.pyc'):
            archive.write(path, path.as_posix())
'@ | .venv\Scripts\python.exe -
```

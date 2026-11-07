# OLED 闪点仿真

使用 PyTorch FFT、TorchOptics、Chromatix 和 Mitsuba 计算 OLED 经高度表面后收集到的亮度图，并生成包含输入图、完整参数、实际设备、耗时和闪点值的离线 HTML 报告。

Windows 下的 Chromatix 自动调用 **WSL2 CUDA** 环境；Torch/TorchOptics 和 Mitsuba 在 Windows 中运行。WSL 安装与环境覆盖配置见 [Windows / WSL 说明](oled-ag-sparkle/references/windows-wsl.md)。

当前计算区域为 **1 mm × 1 mm**；默认高度图为**直径 3–5 µm、深度 0.5 µm、覆盖率约 50% 的随机圆坑**。39,048 个圆坑互不重叠，采用平滑碗形剖面，随机种子为 20261107。高度图为 **1000 × 1000、1 µm 采样、16 位 PNG**。发光图先从原 128 × 128 图块裁出 **[26,58,96,128] 的 70 × 70 完整重复单元**，保留 **2 µm/texel、140 µm × 140 µm**，**8 × 8 整数重复后裁取 500 × 500**，扩展时不缩放或插值。每个 **50 µm × 50 µm** 统计像素对 50 × 50 个计算像素作面积平均，随后对全部 400 个统计像素直接计算：

`CV (%) = 100 × std(I, ddof=0) / mean(I)`

[周期单元裁剪后的四后端报告](test-results/oled-1mm-pits3to5um-periodic-50um/report.html) · [裁剪位置及重复检查](test-results/emission-periodic-crop.png) · [高度图对照](test-results/random-pit-periodic-height-comparison.png) · [亮度图对照](test-results/random-pit-periodic-brightness-comparison.png) · [结果与网格检查](test-results/random-pit-periodic-comparison.json) · [计算与验证记录](oled-ag-sparkle/VALIDATION.json)

原高度图的[周期发光图基准报告](test-results/oled-1mm-original-periodic-50um/report.html)可通过 `--height-surface original` 复算。随机圆坑还使用 **0.5 µm 光学网格**对原高度和新高度作两级采样检查，统计像素保持 50 µm。

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

最新验证为 **28 项测试全部通过**，包含周期检测、完整亮区与拼接连续性、裁剪像素保持、圆坑几何、深度极性、Windows → WSL GPU 转发、中文／空格路径、失败处理和四后端对照，记录位于 `test-results/logs/validation-periodic.log`。本次结果的几何、浮点 CV、面积平均与离线下载审计见 `test-results/random-pit-periodic-validation.json`。

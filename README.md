# OLED 闪点仿真

使用 PyTorch FFT、TorchOptics、Chromatix 和 Mitsuba 计算 OLED 经高度表面后收集到的亮度图，并生成包含输入图、完整参数、实际设备、耗时和闪点值的离线 HTML 报告。

当前计算区域为 **1 mm × 1 mm**；高度图由现有数据插值至 **1000 × 1000、1 µm 采样**，发光图为 **500 × 500、2 µm 采样**。每个 **50 µm × 50 µm** 统计像素对 50 × 50 个计算像素作面积平均，随后对全部 400 个统计像素直接计算：

`CV (%) = 100 × std(I, ddof=0) / mean(I)`

[最新完整 HTML 报告](test-results/oled-1mm-height1um-50um/report.html) · [四后端对比图](test-results/brightness-comparison-height1um-50um.png) · [计算与验证记录](oled-ag-sparkle/VALIDATION.json)

安装与模型说明见 [skill 文档](oled-ag-sparkle/README.md) 和 [SKILL.md](oled-ag-sparkle/SKILL.md)。运行示例：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\run_oled_demo.py --backend all --out test-results\oled-1mm-height1um-50um
```

工作区的完整计算与绘图流程为 `test-results/run-comparison.ps1`。不重算光学结果、只重新生成 HTML：

```powershell
.venv\Scripts\python.exe oled-ag-sparkle\scripts\generate_report.py --results test-results\oled-1mm-height1um-50um
```

仓库保存输入素材、源码、测试、复现脚本及最新完整计算的浮点结果、报告和日志。原始 skill 压缩包作为打包脚本的基准输入保留；重新生成的发布包、虚拟环境、依赖缓存和历史批量输出由 Git 忽略。

最新验证为 **20 项测试全部通过**，记录位于 `test-results/logs/validation-height1um.log`。

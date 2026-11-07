# 仓库维护规则

适用于本仓库内的所有 agent。目标是让代码的用途、入口和复现方式清楚，禁止为完成一次任务随意堆积脚本。

## 变更范围与新增代码

- 只实现当前用户要求。禁止顺手添加无关功能、依赖、抽象层、兼容分支或重构。
- 优先修改现有函数、命令和参数。相同流程应复用已有实现，禁止复制求解参数、输入准备或报告逻辑，再新增一套近似入口。
- **禁止新增用途不明、没有调用方或没有明确运行方式的持久化代码。** 新增文件必须直接服务当前需求，并说明：做什么、由谁调用或如何运行、为何现有文件不能承担。无法说明就不要添加。
- 禁止为了单次绘图、审计、改版、修补、刷新或打包，随手留下新的 `run_*`、`report_*`、`finalize_*`、`fix_*`、`package_*` 等脚本。这条同样适用于 `test-results/`，不得把结果目录当成脚本堆放区。一次性操作使用临时命令；确需长期复用时，先整合进现有入口。新增独立工具须有明确需求依据。
- 新用例优先使用现有入口的参数或配置。禁止为每种表面、网格或报告样式增加一个包装脚本。
- 不得仅因发现历史重复就擅自删除或搬迁已有代码、结果和素材；整理旧入口须在用户授权的整理范围内完成，并迁移已知调用方。

## 现有入口与职责

修改前先读实际实现和调用方；以下说明不构成新增入口的理由。

| 文件 | 职责 |
| --- | --- |
| `oled-ag-sparkle/scripts/sparkle_multibackend.py` | 四后端求解器及通用 CLI；求解行为在这里维护。 |
| `oled-ag-sparkle/scripts/run_oled_demo.py` | 当前默认 OLED 用例入口；选择表面、发光布局、网格和后端，默认运行四后端。通过 `--pitch-um` 统一原蚀刻 AG 的两级网格实验；共享求解命令由 SMS 运行器复用。 |
| `oled-ag-sparkle/scripts/run_legacy_demo.py` | 仅保留历史测试素材，必须指定 `--example quick` 或 `--example 1mm`；默认仅 Torch。不要扩展为第二套默认用例实现。 |
| `oled-ag-sparkle/scripts/run_sms_reference.py` | 全绿、无 LDF 的 SMS 参考流程，负责输入、固定方案及各光学工况；`--stage all` 完成整套流程。 |
| `oled-ag-sparkle/scripts/sms_reference.py` | SMS 参考流程的周期分离、sparkle 计算与已保存浮点结果审计。 |
| `oled-ag-sparkle/scripts/reporting.py`、`generate_report.py` | 离线报告实现及唯一报告 CLI；`--kind brightness` 和 `--kind sms-reference` 明确区分原始 CV 与分离后的闪点，不再在结果目录放置另一套报告生成器。 |
| `oled-ag-sparkle/scripts/prepare_etched_ag.py`、`prepare_random_pits.py`、`prepare_oled_1mm.py`、`prepare_tiled_emission.py` | 已有表面和发光输入生成器。 |
| `oled-ag-sparkle/scripts/chromatix_wsl.py` | Windows 调用 WSL Chromatix 的运行适配。 |

普通 OLED 示例的原始亮度 CV 与全绿 SMS 参考流程分离周期后的 sparkle 含义不同。不得因入口名称相近就混用两者，或悄悄替换统计口径。

## 交付与检查

- 修改入口、参数或职责后，同步更新已有 README 的对应说明；禁止用新增说明文件掩盖入口混乱。
- 运行与改动相称的检查。单纯报告排版不重跑光学求解，不为低风险文案或排版改动新增镜像实现的测试。
- 最终说明实际改了什么、如何使用、如何验证。新增文件逐一交代用途；不要只列文件名或声称“已优化”。
- 不覆盖其他正在进行的修改，不擅自提交或推送。

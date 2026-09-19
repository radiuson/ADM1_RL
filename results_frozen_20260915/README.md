# 论文定稿所用的冻结结果（2026-09-15）

本目录是论文全部表格与图的唯一数据来源。后续实验不写入此处。

| | |
|---|---|
| 算法族 | 26（24 进主表 + 2 消融） |
| 每族 | 4 设定点 × 10 种子 = 40 次运行 |
| 已评估运行 | 1080（evres 517 + evcmdp 563） |
| 常规控制配置 | 113（evbase）→ 44 点帕累托上包络 |

- `evbase/` 常规控制配置的评估记录
- `evres/`  奖励惩罚表述下各算法族的逐次运行记录
- `evcmdp/` 约束表述下各算法族的逐次运行记录（每族 40 次，无重复）
- `envelope.py` 包络构造（全部 113 配置）
- `metrics.py`  约束指标（有符号 J_C−d 与绝对达成偏差）
- `FINAL_NUMBERS.txt` 由上述数据生成的定稿数字

复现论文表格：
    python3 -c "import sys; sys.path.insert(0,'.')" 后按 FINAL_NUMBERS.txt 中的脚本重算，
    将 glob 路径指向本目录下的 evbase/ evres/ evcmdp/。


## 2026-09-18 的两项归档

两处改动均为移出不应参与统计的文件，未删除任何数据。

### 1. 重复评估 → `_archive_duplicate_evals/`

`evcmdp/` 原有 804 份文件，但唯一 (algo, seed, cost_limit) 组合只有 563 个：
223 个组合各有 2–3 份逐位相同、`run_dir` 指向同一次训练的副本。按文件计数
会把有效样本量虚增约 40 %，人为收窄 bootstrap 区间。保留每组最早的一份，
其余 241 份连同 `MANIFEST.json`（含 MD5 校验）移入归档目录。

对常规控制上包络的甲烷差值，95 % 区间排除零的族数由 8/10 变为 6/10；
CPPOPID、FOCOPS、OnCRPO 去重后区间跨零，P3O 为 −23 [−37, −0]。

`evres/` 与 `evbase/` 经同样检查后确认无重复：evres 按 `model` 路径唯一，
`models_mem_*`（五步观测堆叠）与 `models_seed_*` 是不同配置；evbase 中同一
`name` 覆盖不同 PI 整定。两者均未改动。

### 2. 离策略约束族 → `_archive_offpolicy_old_dualgains/`

SACLag / SACPID / DDPGLag / TD3Lag 原有的 160 份评估产自 OmniSafe 出厂对偶
设置（`lambda_lr` 1e-5、PID 增益 1e-6/1e-7、`warmup_epochs` 100 而预算 75），
乘子在训练中几乎不动。替换为对齐尺度后重跑的 160 份（文件名前缀 `corr_`）。

替换后 evcmdp 共 563 份，15 个族各 40 次运行，无重复。

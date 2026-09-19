"""统一的约束指标计算。

两个量必须分开报告，因为定义不同：

  J_C - d        有符号的经验约束违反量（safe-RL 文献标准，非正即安全）。
                 低于 cost limit 不计为违反，只反映保守程度。
  |J_C - d|      指定值达成偏差（本文定义）。同时惩罚超出与保守，用于支撑
                 "越限率可指定"这一主张。safe-RL 文献无对应指标，因为该
                 文献只关心不越界，不关心是否过度保守。
"""
import numpy as np
def constraint_metrics(runs, episode_steps=60):
    """runs: [{'viol': 实测越限率%, 'cost_limit': 每回合允许越限次数}]"""
    sv, av = [], []
    for x in runs:
        if not x.get('cost_limit'): continue
        tgt = 100.0 * x['cost_limit'] / episode_steps
        sv.append(x['viol'] - tgt)       # J_C - d, 有符号
        av.append(abs(x['viol'] - tgt))  # 达成偏差
    if not sv: return None
    return {'signed_mean': float(np.mean(sv)),      # 正=超出, 负=保守
            'signed_max': float(np.max(sv)),
            'frac_safe': float(np.mean([s <= 0 for s in sv])),  # 非正比例
            'abs_mean': float(np.mean(av))}

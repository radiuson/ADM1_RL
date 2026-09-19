"""统一的常规控制上包络：全部 113 个配置（恒定进料 / 规则式 / PI 三族）。

评估脚本一律从此处取包络，避免正文与表格用不同基准。
"""
import json, glob
import numpy as np

def load_envelope(path=None, families=None):
    if path is None:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from paths import glob as dglob
        files = dglob('evbase')
    else:
        files = glob.glob(path)
    B = [json.load(open(f)) for f in files]
    if families:
        B = [b for b in B if b['kind'] in families]
    pts = sorted((b['viol'], b['ch4'], b['name']) for b in B)
    env, best = [], -1.0
    for v, c, n in pts:
        if c > best:
            env.append((v, c, n)); best = c
    return B, env

def interpolator(env):
    xs = [a for a, _, _ in env]; ys = [b for _, b, _ in env]
    def I(p):
        return float(np.interp(p, xs, ys)) if xs[0] <= p <= xs[-1] else None
    return I

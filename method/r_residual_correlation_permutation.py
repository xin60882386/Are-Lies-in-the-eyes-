# -*- coding: utf-8 -*-
r"""
pair_and_analyze_recursive_fixed_roles.py

一键完成：
1) 递归扫描 --src
2) 仅在满足筛选条件的目录里配对（可选：只在叶子目录；csv 数量 eq2 或 ge2）
3) 固定角色配对：
   - 数字开头 = 观察者（detectors / observer）
   - 字母开头 = 被观察者（deceivers / observed）
   支持配对策略：
     - zip_order（默认）
     - prefix_match（用 --prefix_regex 提取 trial 前缀来配对）
     - nearest_time（按 mtime 最接近贪心）
4) 复制/移动到 --dst（镜像结构，每对一个子文件夹）
5) 对每对文件执行：预处理 → AR 预白化 → “最优时滞”残差 Pearson 相关 → 循环平移置换检验
6) 输出：
   - 每层目录：pairing_report.csv / unmatched.csv
   - 结果目录 dst/ar_results：
       per_pair_feature_results.csv（逐组逐特征；含 abs_r_res、lag、FDR 等）
       per_pair_mean_over_features.csv（组水平均值；含 Fisher-z 纠偏均值）
7) 进度体验：tqdm 进度条 + 每特征实时日志

稳健性与统计学改进：
- safe_corr 避免 ±inf/NaN（方差≈0 或样本不足时返回 NaN）
- 在 [-max_lag, +max_lag] 搜索“最优时滞”，报告 lag 与对应相关
- 置换检验（循环平移）在“最优时滞”下进行
- 逐单元 FDR（Benjamini–Hochberg），输出 significant_fdr
- Fisher-z 平均并反变换，避免直接平均 r 的偏差
"""

import os, re, sys, json, shutil, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm

# ===================== 文件与角色工具 =====================

def is_alpha_start(name: str) -> bool:
    stem = Path(name).stem
    return len(stem) > 0 and stem[0].isalpha()

def is_digit_start(name: str) -> bool:
    stem = Path(name).stem
    return len(stem) > 0 and stem[0].isdigit()

def list_csv_in_dir(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower()==".csv"])

def split_fixed_roles(files: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """固定角色：数字=observer（detector），字母=observed（deceiver）"""
    observers = [f for f in files if is_digit_start(f.name)]
    observed  = [f for f in files if is_alpha_start(f.name)]
    others    = [f for f in files if f not in observers and f not in observed]
    return observers, observed, others

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ===================== 配对策略 =====================

def pair_zip_order(obs: List[Path], vid: List[Path]) -> Tuple[List[Tuple[Path,Path]], List[Path]]:
    obs_sorted = sorted(obs, key=lambda p: p.name)
    vid_sorted = sorted(vid, key=lambda p: p.name)
    n = min(len(obs_sorted), len(vid_sorted))
    pairs = list(zip(obs_sorted[:n], vid_sorted[:n]))
    unmatched = obs_sorted[n:] + vid_sorted[n:]
    return pairs, unmatched

def extract_prefix(name: str, regex: str) -> str:
    stem = Path(name).stem
    m = re.match(regex, stem)
    return m.group(0) if m else stem

def pair_prefix_match(obs: List[Path], vid: List[Path], prefix_regex: str) -> Tuple[List[Tuple[Path,Path]], List[Path]]:
    om: Dict[str, List[Path]] = {}
    vm: Dict[str, List[Path]] = {}
    for f in obs:
        om.setdefault(extract_prefix(f.name, prefix_regex), []).append(f)
    for f in vid:
        vm.setdefault(extract_prefix(f.name, prefix_regex), []).append(f)
    pairs, unmatched = [], []
    keys = sorted(set(om.keys()) | set(vm.keys()))
    for k in keys:
        os_list = om.get(k, [])
        vs_list = vm.get(k, [])
        if os_list and vs_list:
            pairs.append((os_list[0], vs_list[0]))
            unmatched.extend(os_list[1:])
            unmatched.extend(vs_list[1:])
        else:
            unmatched.extend(os_list or vs_list)
    return pairs, unmatched

def pair_nearest_time(obs: List[Path], vid: List[Path]) -> Tuple[List[Tuple[Path,Path]], List[Path]]:
    obs_sorted = sorted(obs, key=lambda p: p.stat().st_mtime)
    vid_sorted = sorted(vid, key=lambda p: p.stat().st_mtime)
    used_v = set()
    pairs=[]
    for fo in obs_sorted:
        mt = fo.stat().st_mtime
        best_i, best_d = None, float("inf")
        for i, fv in enumerate(vid_sorted):
            if i in used_v: continue
            d = abs(fv.stat().st_mtime - mt)
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            pairs.append((fo, vid_sorted[best_i]))
            used_v.add(best_i)
    unmatched = [v for i, v in enumerate(vid_sorted) if i not in used_v]
    paired_obs = {a for a,_ in pairs}
    unmatched.extend([o for o in obs_sorted if o not in paired_obs])
    return pairs, unmatched

# ===================== 时间序列预处理与检验 =====================

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def detrend_lin(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    try:
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return x - (m*t + b)
    except Exception:
        return x - np.nanmean(x)

def clean_series(s: pd.Series) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce")
    s = s.interpolate(limit_direction="both").bfill().ffill()
    arr = s.values.astype(float)
    arr = detrend_lin(arr)
    arr = zscore(arr)
    return arr

def fit_ar_resid(x: np.ndarray, pmax: int = 10, ic: str = "aic"):
    best_ic = np.inf
    best_resid, best_p = None, 0
    for p in range(1, pmax+1):
        try:
            m = AutoReg(x, lags=p, old_names=False).fit()
            cur_ic = getattr(m, ic)
            if cur_ic < best_ic:
                best_ic, best_resid, best_p = cur_ic, np.asarray(m.resid, float), p
        except Exception:
            pass
    if best_resid is None:
        try:
            m = AutoReg(x, lags=1, old_names=False).fit()
            return np.asarray(m.resid, float), 1
        except Exception:
            return x - np.nanmean(x), 0
    return best_resid, best_p

def circ_shift(a: np.ndarray, k: int) -> np.ndarray:
    k %= len(a)
    if k == 0: return a
    return np.concatenate([a[-k:], a[:-k]])

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m]; b = b[m]
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def align_with_lag(x: np.ndarray, y: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    # 正 lag 表示 y 滞后（y 往右移）
    if lag > 0:
        return x[:-lag], y[lag:]
    elif lag < 0:
        k = -lag
        return x[k:], y[:-k]
    else:
        return x, y

def resid_corr_perm_with_lag(x: np.ndarray, y: np.ndarray, pmax=10, ic="aic",
                             max_lag: int = 60, n_perm: int = 1000, seed: int = 42):
    xs, ys = clean_series(pd.Series(x)), clean_series(pd.Series(y))
    rx, pX = fit_ar_resid(xs, pmax=pmax, ic=ic)
    ry, pY = fit_ar_resid(ys, pmax=pmax, ic=ic)

    n = min(len(rx), len(ry))
    rx, ry = rx[:n], ry[:n]
    if n < 8:
        return dict(r_res=np.nan, pval=np.nan, n=int(n), pX=int(pX), pY=int(pY), lag=0)

    # 1) 搜索最优时滞
    lags = range(-max_lag, max_lag + 1)
    best = {"lag": 0, "r": np.nan}
    for L in lags:
        ax, ay = align_with_lag(rx, ry, L)
        r = safe_corr(ax, ay)
        if np.isfinite(r):
            if (not np.isfinite(best["r"])) or (abs(r) > abs(best["r"])):
                best = {"lag": L, "r": r}

    if not np.isfinite(best["r"]):
        return dict(r_res=np.nan, pval=np.nan, n=int(n), pX=int(pX), pY=int(pY), lag=0)

    # 2) 固定该时滞做置换检验（循环平移）
    ax, ay = align_with_lag(rx, ry, best["lag"])
    r_obs = safe_corr(ax, ay)

    rng = np.random.default_rng(seed)
    T = []
    m = len(ax)
    if m < 8:
        return dict(r_res=np.nan, pval=np.nan, n=int(m), pX=int(pX), pY=int(pY), lag=int(best["lag"]))
    ks = rng.integers(1, m, size=int(n_perm))
    for k in ks:
        r_perm = safe_corr(ax, circ_shift(ay, int(k)))
        if np.isfinite(r_perm):
            T.append(abs(r_perm))
    T = np.asarray(T, float)
    if T.size == 0 or not np.isfinite(r_obs):
        pval = np.nan
    else:
        pval = (1.0 + np.sum(T >= abs(r_obs))) / (1.0 + T.size)

    return dict(r_res=float(r_obs), pval=float(pval), n=int(m), pX=int(pX), pY=int(pY), lag=int(best["lag"]))

# ===================== 目录筛选与配对执行 =====================

def collect_candidate_dirs(src_root: Path, only_leaf_dirs: bool, csv_count_filter: str) -> List[Path]:
    """收集需要处理的目录：
       - only_leaf_dirs=True 仅保留叶子目录（自身有 CSV，且其任何子目录不再有 CSV）
       - csv_count_filter: 'eq2' 仅恰好2个CSV；'ge2' 至少2个CSV
    """
    dirs_with_csv = set()
    for dirpath, _, filenames in os.walk(src_root):
        if any(f.lower().endswith(".csv") for f in filenames):
            dirs_with_csv.add(Path(dirpath))

    if only_leaf_dirs:
        leaf_dirs = []
        dirs_sorted = sorted(dirs_with_csv)
        for d in dirs_sorted:
            # 如果存在另一个包含 CSV 的目录是 d 的后代，则 d 不是叶子
            if not any(str(o).startswith(str(d) + os.sep) for o in dirs_sorted if o != d):
                leaf_dirs.append(d)
        candidates = leaf_dirs
    else:
        candidates = sorted(dirs_with_csv)

    def pass_count_filter(d: Path) -> bool:
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower()==".csv")
        return (n == 2) if (csv_count_filter == "eq2") else (n >= 2)

    return [d for d in candidates if pass_count_filter(d)]

def do_pairing_in_dir(src_dir: Path, dst_dir: Path, move: bool, dry: bool,
                      pair_strategy: str, prefix_regex: str):
    files = list_csv_in_dir(src_dir)
    if len(files) < 2:
        return [], []
    observers, observed, others = split_fixed_roles(files)
    if len(observers)==0 or len(observed)==0:
        print(f"⚠️ 角色分组异常：{src_dir} | observers={len(observers)}, observed={len(observed)}")

    if pair_strategy == "zip_order":
        pairs, unmatched = pair_zip_order(observers, observed)
    elif pair_strategy == "prefix_match":
        pairs, unmatched = pair_prefix_match(observers, observed, prefix_regex)
    elif pair_strategy == "nearest_time":
        pairs, unmatched = pair_nearest_time(observers, observed)
    else:
        pairs, unmatched = pair_zip_order(observers, observed)

    unmatched.extend(others)
    ensure_dir(dst_dir)

    # 目录级报告
    rows = []
    for i, (fo, fv) in enumerate(pairs, 1):
        rows.append(dict(src_dir=str(src_dir), pair_idx=i, observer=fo.name, observed=fv.name))
    if rows:
        pd.DataFrame(rows).to_csv(dst_dir/"pairing_report.csv", index=False, encoding="utf-8-sig")
    if unmatched:
        pd.DataFrame({"src_dir": [str(src_dir)]*len(unmatched),
                      "unmatched": [p.name for p in unmatched]}).to_csv(dst_dir/"unmatched.csv", index=False, encoding="utf-8-sig")

    # 复制/移动
    for i, (fo, fv) in enumerate(pairs, 1):
        sub = dst_dir / f"pair_{i:03d}__{fo.stem}__{fv.stem}"
        print(("DRY " if dry else "") + f"[{src_dir.name}] → {sub.name}")
        if not dry:
            ensure_dir(sub)
            (shutil.move if move else shutil.copy2)(fo, sub/fo.name)
            (shutil.move if move else shutil.copy2)(fv, sub/fv.name)
    return pairs, unmatched

# ===================== 统计工具 =====================

def fisher_z(r):
    if not np.isfinite(r):
        return np.nan
    r = np.clip(r, -0.999999, 0.999999)
    return float(np.arctanh(r))

def inv_fisher_z(z):
    if not np.isfinite(z):
        return np.nan
    return float(np.tanh(z))

def try_fdr(pvals: pd.Series):
    try:
        from statsmodels.stats.multitest import multipletests
        mask = np.isfinite(pvals.values)
        p_adj = np.full_like(pvals.values, np.nan, dtype=float)
        if mask.sum() >= 1:
            p_adj[mask] = multipletests(pvals.values[mask], method="fdr_bh")[1]
        return pd.Series(p_adj, index=pvals.index)
    except Exception:
        return pd.Series([np.nan]*len(pvals), index=pvals.index)

# ===================== 分析（带进度条） =====================

def analyze_pairs(root: Path, out: Path, features: List[str], pmax: int, ic: str,
                  n_perm: int, seed: int, max_lag: int, condition_regex: str):
    ensure_dir(out)
    results = []

    # 预收集所有“恰好2个CSV”的子目录（因为配对后的子文件夹就是2个CSV）
    candidate_dirs = []
    for dirpath, _, files in os.walk(root):
        csvs = [f for f in files if f.lower().endswith(".csv")]
        if len(csvs) == 2:
            candidate_dirs.append((dirpath, csvs))

    for dirpath, csvs in tqdm(candidate_dirs, desc="分析中", unit="pair"):
        dirpath = Path(dirpath)
        f1, f2 = [dirpath/f for f in csvs]
        # 固定角色：数字=observer，字母=observed
        if is_digit_start(f1.name) and is_alpha_start(f2.name):
            f_obs, f_vid = f1, f2
        elif is_digit_start(f2.name) and is_alpha_start(f1.name):
            f_obs, f_vid = f2, f1
        else:
            f_obs, f_vid = sorted([f1, f2], key=lambda p: p.name)

        # 读入（高容错）
        try:
            df_obs = pd.read_csv(f_obs, engine="python", on_bad_lines="skip")
            df_vid = pd.read_csv(f_vid, engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"读取失败：{f_obs} | {f_vid} | {e}")
            continue

        # 列名映射（大小写/空白不敏感；含常见别名兜底）
        def norm_cols(cols):
            return {re.sub(r"\s+", " ", c.strip()).lower(): c for c in cols}
        map_obs = norm_cols(df_obs.columns)
        map_vid = norm_cols(df_vid.columns)

        def find_col(m, target):
            key = re.sub(r"\s+", " ", target.strip()).lower()
            if key in m: return m[key]
            alias = {
                "pupilsize": ["pupil size","pupil_size"],
                "irissize": ["iris size","iris_size"],
                "innereyedistance": ["inner eye distance","inner_eye_distance","inner- eye distance","inner-eye-distance"],
                "eyeliddistance": ["eyelid distance","eye lid distance","eyelid_distance"],
                "theeyeopening": ["the eye opening","eye opening","eye_opening"],
                "earratio": ["ear","ear ratio","ear_ratio"]
            }
            base = re.sub(r"[^a-z0-9]", "", key)
            if base in alias:
                for cand in alias[base]:
                    k2 = re.sub(r"\s+", " ", cand.strip()).lower()
                    if k2 in m: return m[k2]
            return None

        # 情境标签（可选）：从父目录名里用正则抽取（如 tt/tf/ft/ff）
        condition = None
        try:
            m = re.search(condition_regex, dirpath.name, flags=re.I)
            if m:
                condition = m.group(0)
        except Exception:
            condition = None

        # 遍历特征
        for feat in features:
            c_obs = find_col(map_obs, feat)
            c_vid = find_col(map_vid, feat)
            if c_obs is None or c_vid is None:
                continue

            x = pd.to_numeric(df_obs[c_obs], errors="coerce").values
            y = pd.to_numeric(df_vid[c_vid], errors="coerce").values
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]

            print(f"正在分析 {f_obs.name} × {f_vid.name} | 特征 {feat}")

            res = resid_corr_perm_with_lag(
                x, y, pmax=pmax, ic=ic, max_lag=max_lag,
                n_perm=n_perm, seed=seed
            )
            results.append(dict(
                pair=f"{f_obs.name}|{f_vid.name}",
                observer_file=f_obs.name,
                observed_file=f_vid.name,
                feature=feat,
                condition=condition,
                **res
            ))

    # 写结果
    cols_detail = ["pair","observer_file","observed_file","feature","condition",
                   "r_res","pval","lag","n","pX","pY","abs_r_res","significant","pval_fdr","significant_fdr","r_fisher_z"]
    cols_group  = ["pair","observer_file","observed_file","condition",
                   "mean_abs_r_res","mean_r_fisher_z","mean_r_res_bias_corrected",
                   "k_features","mean_p","mean_p_fdr","k_sig","k_sig_fdr"]

    if not results:
        pd.DataFrame(columns=cols_detail).to_csv(out/"per_pair_feature_results.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=cols_group).to_csv(out/"per_pair_mean_over_features.csv", index=False, encoding="utf-8-sig")
        return

    df = pd.DataFrame(results)
    df["abs_r_res"] = df["r_res"].abs()
    df["significant"] = (df["pval"] < 0.05).astype(int)

    # FDR
    df["pval_fdr"] = try_fdr(df["pval"])
    df["significant_fdr"] = (df["pval_fdr"] < 0.05).astype(float)  # 用 float 便于 NaN

    # Fisher-z
    df["r_fisher_z"] = df["r_res"].apply(fisher_z)

    # 细表
    df[cols_detail].to_csv(out/"per_pair_feature_results.csv", index=False, encoding="utf-8-sig")

    # 组级汇总（Fisher-z 纠偏均值）
    g = df.groupby(["pair","observer_file","observed_file","condition"], as_index=False).agg(
        mean_abs_r_res=("abs_r_res", lambda x: float(np.nanmean(x))),
        mean_r_fisher_z=("r_fisher_z", lambda x: float(np.nanmean(x))),
        k_features=("feature", "count"),
        mean_p=("pval", lambda x: float(np.nanmean(x))),
        mean_p_fdr=("pval_fdr", lambda x: float(np.nanmean(x))),
        k_sig=("significant", "sum"),
        k_sig_fdr=("significant_fdr", "sum")
    )
    g["mean_r_res_bias_corrected"] = g["mean_r_fisher_z"].apply(inv_fisher_z)
    g[cols_group].to_csv(out/"per_pair_mean_over_features.csv", index=False, encoding="utf-8-sig")

# ===================== 主程序 =====================

def main():
    ap = argparse.ArgumentParser(description="递归配对（数字=observer, 字母=observed）+ AR预白化 + 最优时滞残差相关 + 置换检验（带筛选与进度条）")
    ap.add_argument("--src", default=r"", help="源根目录（递归扫描）")
    ap.add_argument("--dst", default=r"", help="输出根目录（镜像结构，原始不动）")

    # 目录筛选
    ap.add_argument("--only_leaf_dirs", type=str, default="true",
                    help="只在叶子目录配对（true/false）")
    ap.add_argument("--csv_count_filter", type=str, choices=["eq2","ge2"], default="ge2",
                    help="目录里的CSV数量过滤：eq2=恰好2个；ge2=至少2个")

    # 配对策略
    ap.add_argument("--pair_strategy", type=str, choices=["zip_order","prefix_match","nearest_time"], default="zip_order",
                    help="配对策略")
    ap.add_argument("--prefix_regex", type=str, default=r"^([A-Za-z]+|[0-9]+)",
                    help="prefix_match 用的前缀抽取正则")

    # 分析参数
    ap.add_argument("--features", nargs="+", default=[
        "PupilSize","IrisSize","InnerEyedistance","Eyeliddistance","TheEyeOpening","EarRatio"
    ], help="要分析的特征列名（大小写/空白不敏感）")
    ap.add_argument("--pmax", type=int, default=10, help="AR 最大阶（自动选阶上限）")
    ap.add_argument("--ic", type=str, default="aic", choices=["aic","bic"], help="选阶准则")
    ap.add_argument("--max_lag", type=int, default=60, help="最优时滞搜索范围（±max_lag）")
    ap.add_argument("--n_perm", type=int, default=1000, help="置换次数（建议≥1000）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--condition_regex", type=str, default=r"(tt|tf|ft|ff)$",
                    help="从 pair 子目录名中抽取情境标签（如 'tt','tf','ft','ff'），可留空关闭")

    # I/O 行为
    ap.add_argument("--move", type=str, default="false", help="是否移动（默认复制）：true/false")
    ap.add_argument("--dry_run", type=str, default="false", help="是否仅演练：true/false")
    args = ap.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    ensure_dir(dst_root)

    only_leaf_dirs = args.only_leaf_dirs.lower() in ("1","true","yes","y")
    do_move = args.move.lower() in ("1","true","yes","y")
    dry_run = args.dry_run.lower() in ("1","true","yes","y")

    # 目录候选集
    targets = collect_candidate_dirs(src_root, only_leaf_dirs, args.csv_count_filter)

    global_pairs, global_unmatched = [], []

    # 逐目录配对并复制
    for src_dir in targets:
        rel = src_dir.relative_to(src_root)
        dst_dir = dst_root / rel
        pairs, unmatched = do_pairing_in_dir(src_dir, dst_dir, move=do_move, dry=dry_run,
                                             pair_strategy=args.pair_strategy,
                                             prefix_regex=args.prefix_regex)
        global_pairs.extend([dict(src_dir=str(src_dir), dst_dir=str(dst_dir),
                                  observer=a.name, observed=b.name) for a,b in pairs])
        global_unmatched.extend([dict(src_dir=str(src_dir), file=u.name) for u in unmatched])

    # 全局报告
    if global_pairs:
        pd.DataFrame(global_pairs).to_csv(dst_root/"global_pairing_report.csv", index=False, encoding="utf-8-sig")
    if global_unmatched:
        pd.DataFrame(global_unmatched).to_csv(dst_root/"global_unmatched.csv", index=False, encoding="utf-8-sig")

    # 分析（配对子文件夹里都是“恰好2个CSV”，因此内部自动筛选）
    if not dry_run:
        analyze_pairs(
            dst_root, dst_root/"ar_results",
            features=args.features, pmax=args.pmax, ic=args.ic,
            n_perm=args.n_perm, seed=args.seed, max_lag=args.max_lag,
            condition_regex=args.condition_regex
        )

    # 保存运行配置
    with open(dst_root/"run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "src": str(src_root), "dst": str(dst_root),
            "features": args.features, "pmax": args.pmax, "ic": args.ic,
            "n_perm": args.n_perm, "seed": args.seed, "max_lag": args.max_lag,
            "move": do_move, "dry_run": dry_run,
            "only_leaf_dirs": only_leaf_dirs, "csv_count_filter": args.csv_count_filter,
            "pair_strategy": args.pair_strategy, "prefix_regex": args.prefix_regex,
            "condition_regex": args.condition_regex,
            "role_rule": "digit=observer(detector), alpha=observed(deceiver)"
        }, f, ensure_ascii=False, indent=2)

    print("\n 完成：递归配对 + 预白化残差相关 + 最优时滞 + 置换检验！")
    print(f" 结果目录：{dst_root}")
    print(f" 配对汇总：{dst_root/'global_pairing_report.csv'}")
    print(f" 分析结果：{(dst_root/'ar_results'/'per_pair_feature_results.csv')}")
    print(f" 组级汇总：{(dst_root/'ar_results'/'per_pair_mean_over_features.csv')}")

if __name__ == "__main__":
    main()

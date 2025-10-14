# -*- coding: utf-8 -*-
"""
final_hier_corr_eyefeatures_only.py
一次性计算（仅眼部特征）：
   六个眼部特征：PupilSize, IrisSize, InnerEyedistance, Eyeliddistance, TheEyeOpening, EarRatio

方法：
  - 逐试次：插补→去趋势→z-score→按较短长度对齐→Pearson r
  - 群体层：Fisher z → MixedLM(随机截距=Subject) + 被试聚类稳健SE（失败则回退普通 OLS）
  - 显著性：被试内循环平移置换（只滚动“被观察者”时间轴）

输出：
  - per_trial_eye_features_corr.csv         （逐试次×特征）
  - summary_overall.csv                     （六特征合并的总体 r̄ 与置换 p）
  - summary_by_feature.csv                  （六个特征各自的 r̄ 与置换 p）
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.signal import detrend
from scipy.stats import pearsonr

# ================== 配置 ==================
ROOT = r""
FEATURES_SCALAR = [
    "PupilSize",
    "IrisSize",
    "InnerEyedistance",
    "Eyeliddistance",
    "TheEyeOpening",
    "EarRatio",
]
CONDITION_COL = "label_video"     # 可改为 "label_judge"
MIN_LEN = 100                     # 最小有效长度
N_PERM = 1000                     # 置换次数
OUTPUT_DIR = r"D:\lie\xin\shiyan5\ceshi1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =========================================

def parse_labels_from_trial_name(trial_name: str):
    sfx = trial_name[-2:].lower()
    if not re.match(r"[tf]{2}$", sfx):
        return None, None
    m = {"t": 0, "f": 1}
    return m[sfx[0]], m[sfx[1]]

def robust_read_csv(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception:
            continue
    raise RuntimeError(f"读取失败：{path}")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 标准化列名：小写 + 下划线（只保留小写列）
    def norm(c):
        c = str(c).strip().replace("\t", " ").replace("-", "_")
        c = re.sub(r"\s+", "_", c)
        return c.lower()
    df = df.copy()
    df.columns = [norm(c) for c in df.columns]
    # 只保留小写开头（你的数据常有大写版重复）
    df = df[[c for c in df.columns if not re.match(r"^[A-Z]", c)]]
    return df

def preprocess_series(x) -> np.ndarray:
    """
    插补→去趋势→z-score；返回 1D ndarray。
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise TypeError(f"预期 1 列，但收到 {x.shape[1]} 列 DataFrame。列名={list(x.columns)[:5]} ...")
    elif isinstance(x, (list, tuple)):
        x = pd.Series(x)
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = pd.Series(x)
        elif x.ndim == 2 and x.shape[1] == 1:
            x = pd.Series(x[:, 0])
        else:
            raise TypeError(f"预期 1 维数组，但收到形状 {x.shape} 的 ndarray")
    elif not isinstance(x, pd.Series):
        raise TypeError(f"arg must be a 1-D Series/array/list/tuple or single-column DataFrame, got {type(x)}")

    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.interpolate(limit_direction="both").bfill().ffill()

    arr = x.to_numpy()
    with np.errstate(invalid="ignore"):
        arr = detrend(arr, type="linear")
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        return np.full_like(arr, np.nan, dtype=float)
    arr = (arr - np.nanmean(arr)) / std
    return arr

def align_and_corr(a: np.ndarray, b: np.ndarray):
    n = min(len(a), len(b))
    if n < MIN_LEN:
        return np.nan, n
    av, bv = a[:n], b[:n]
    if np.any(np.isnan(av)) or np.any(np.isnan(bv)):
        return np.nan, n
    r, _ = pearsonr(av, bv)
    return r, n

def snr_proxy(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = np.nanstd(a), np.nanstd(b)
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return 2 * sa * sb / (sa + sb)

def find_trial_folders(root: str):
    """
    递归查找“恰好 2 个 CSV”的文件夹，作为一个试次。
    主动从文件夹名末尾两位提取 label_video/label_judge。
    """
    trials = []
    for cur, _, files in os.walk(root):
        csvs = [os.path.join(cur, f) for f in files if f.lower().endswith(".csv")]
        if len(csvs) == 2:
            parts = os.path.normpath(cur).split(os.sep)
            subject = parts[-3] if len(parts) >= 3 else parts[-2] if len(parts) >= 2 else "S"
            trial = parts[-1]
            lv, lj = parse_labels_from_trial_name(trial)
            trials.append({
                "subject": subject,
                "trial": trial,
                "csv_paths": csvs,
                "label_video": lv, "label_judge": lj
            })
    return trials

def load_two_csv(csv_paths):
    df1 = clean_columns(robust_read_csv(csv_paths[0]))
    df2 = clean_columns(robust_read_csv(csv_paths[1]))
    return df1, df2

def decide_viewer_speaker(df1, df2):
    # 简单：列更多者视为观察者 viewer
    return (df1, df2) if df1.shape[1] >= df2.shape[1] else (df2, df1)

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def mixedlm_and_cluster_ols(df, condition_col):
    """
    返回：mixed_res, ols_res, r_bar, ols_type
    ols_type: "cluster" 或 "ols"（回退）
    """
    endog = df["z"]
    exog_cols = []
    if condition_col in df.columns: exog_cols.append(condition_col)
    if "Len" in df.columns: exog_cols.append("Len")
    if "SNR" in df.columns: exog_cols.append("SNR")
    X = sm.add_constant(df[exog_cols]) if exog_cols else sm.add_constant(pd.DataFrame({"ones": np.ones(len(df))}))
    model = MixedLM(endog, X, groups=df["Subject"])
    mixed_res = model.fit(reml=True, method="lbfgs", maxiter=200, disp=False)

    subj_mean = df.groupby("Subject").agg({
        "z": "mean",
        condition_col: "mean" if condition_col in df.columns else (lambda x: np.nan),
        "Len": "mean" if "Len" in df.columns else (lambda x: np.nan),
        "SNR": "mean" if "SNR" in df.columns else (lambda x: np.nan),
    }).reset_index()

    ols_cols = []
    if condition_col in df.columns: ols_cols.append(condition_col)
    if "Len" in df.columns: ols_cols.append("Len")
    if "SNR" in df.columns: ols_cols.append("SNR")
    Xs = sm.add_constant(subj_mean[ols_cols]) if ols_cols else sm.add_constant(pd.DataFrame({"ones": np.ones(len(subj_mean))}))
    try:
        ols_res = sm.OLS(subj_mean["z"], Xs).fit(cov_type="cluster", cov_kwds={"groups": subj_mean["Subject"]})
        ols_type = "cluster"
    except Exception:
        warnings.warn("cluster 协方差失败，回退普通 OLS。")
        ols_res = sm.OLS(subj_mean["z"], Xs).fit()
        ols_type = "ols"

    beta0 = ols_res.params["const"]
    r_bar = (np.exp(2 * beta0) - 1) / (np.exp(2 * beta0) + 1)
    return mixed_res, ols_res, r_bar, ols_type

def subject_within_circ_perm(cache, condition_col, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    beta0_null = []

    for _ in tqdm(range(n_perm), desc=f"被试内循环平移置换 B={n_perm}"):
        rows = []
        for c in cache:
            v = c["viewer"]; s = c["speaker"]; n = len(v)
            if n < MIN_LEN: continue
            k = rng.integers(0, n)
            s_roll = np.roll(s, k)
            r, _ = align_and_corr(v, s_roll)
            if np.isnan(r): continue
            rows.append({
                "Subject": c["Subject"],
                "z": fisher_z(r),
                "Len": c["Len"],
                "SNR": c["SNR"],
                "label_video": c["label_video"],
                "label_judge": c["label_judge"]
            })
        if not rows:
            beta0_null.append(np.nan); continue

        dfb = pd.DataFrame(rows)
        subj_mean = dfb.groupby("Subject").agg({
            "z": "mean",
            condition_col: "mean",
            "Len": "mean",
            "SNR": "mean",
        }).reset_index()
        Xs = sm.add_constant(subj_mean[[condition_col, "Len", "SNR"]])
        try:
            ols_b = sm.OLS(subj_mean["z"], Xs).fit(cov_type="cluster", cov_kwds={"groups": subj_mean["Subject"]})
            beta0_null.append(ols_b.params["const"])
        except Exception:
            beta0_null.append(sm.OLS(subj_mean["z"], Xs).fit().params["const"])

    return np.array(beta0_null, dtype=float)

def p_value_from_null(beta0_obs, beta0_null):
    beta0_null = beta0_null[~np.isnan(beta0_null)]
    if len(beta0_null) == 0:
        return np.nan
    more_extreme = np.sum(np.abs(beta0_null) >= np.abs(beta0_obs))
    return (more_extreme + 1) / (len(beta0_null) + 1)

def run_block_and_perm(df_block: pd.DataFrame, cache_block: list, label: str):
    """
    在一个数据块上（可为Overall或单个Feature）跑：MixedLM+OLS、置换，返回 dict。
    """
    out = {"label": label, "n_rows": len(df_block)}
    if df_block.empty:
        out.update({"r_bar": np.nan, "beta0_obs": np.nan, "perm_p_two_tailed": np.nan, "n_perm_effective": 0, "ols_type": "NA"})
        return out

    dfx = df_block.copy()
    dfx["z"] = fisher_z(dfx["r"])
    use_cols = ["Subject", "z", "Len", "SNR", CONDITION_COL]
    dfx = dfx[use_cols].dropna().reset_index(drop=True)
    if dfx.empty:
        out.update({"r_bar": np.nan, "beta0_obs": np.nan, "perm_p_two_tailed": np.nan, "n_perm_effective": 0, "ols_type": "NA"})
        return out

    mix_res, ols_res, r_bar, ols_type = mixedlm_and_cluster_ols(dfx, CONDITION_COL)
    beta0_obs = float(ols_res.params["const"])

    # 置换
    beta0_null = subject_within_circ_perm(cache_block, CONDITION_COL, n_perm=N_PERM, seed=2025)
    p_perm = p_value_from_null(beta0_obs, beta0_null)
    n_eff = int(np.sum(~np.isnan(beta0_null)))

    out.update({
        "r_bar": r_bar,
        "beta0_obs": beta0_obs,
        "perm_p_two_tailed": p_perm,
        "n_perm_effective": n_eff,
        "ols_type": ols_type
    })
    return out

def main():
    trials = find_trial_folders(ROOT)
    if not trials:
        print("没有找到“恰好 2 个 CSV”的试次文件夹。"); return

    rows, cache = [], []

    for meta in tqdm(trials, desc="逐试次：六个眼部特征相关"):
        sname, tname = meta["subject"], meta["trial"]
        lv, lj = meta["label_video"], meta["label_judge"]

        try:
            df1 = clean_columns(robust_read_csv(meta["csv_paths"][0]))
            df2 = clean_columns(robust_read_csv(meta["csv_paths"][1]))
        except Exception as e:
            warnings.warn(f"[跳过] 读取失败 {sname}/{tname}: {e}")
            continue

        viewer_df, speaker_df = decide_viewer_speaker(df1, df2)

        # ---------- 仅眼部六个标量特征 ----------
        vcols, scols = set(viewer_df.columns), set(speaker_df.columns)
        present_scalar = [f for f in FEATURES_SCALAR if f.lower() in vcols and f.lower() in scols]
        for feat in present_scalar:
            col = feat.lower()
            v_std = preprocess_series(viewer_df[col])
            s_std = preprocess_series(speaker_df[col])
            r, eff_len = align_and_corr(v_std, s_std)
            if np.isnan(r):
                continue
            rows.append({
                "Subject": sname, "Trial": tname, "Feature": feat,
                "r": r, "Len": eff_len, "SNR": snr_proxy(v_std[:eff_len], s_std[:eff_len]),
                "label_video": lv, "label_judge": lj
            })
            cache.append({
                "Subject": sname, "Trial": tname, "Feature": feat,
                "viewer": v_std[:eff_len].copy(), "speaker": s_std[:eff_len].copy(),
                "Len": eff_len, "SNR": snr_proxy(v_std[:eff_len], s_std[:eff_len]),
                "label_video": lv, "label_judge": lj
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("没有得到有效结果，请检查列名或 MIN_LEN。"); return

    # 保存逐试次结果
    per_trial_csv = os.path.join(OUTPUT_DIR, "per_trial_eye_features_corr.csv")
    df.to_csv(per_trial_csv, index=False)

    # ========= A) 总体（合并六特征） =========
    overall_res = run_block_and_perm(df_block=df, cache_block=cache, label="OVERALL_ALL_6_FEATURES")
    pd.Series(overall_res).to_csv(os.path.join(OUTPUT_DIR, "summary_overall.csv"))

    # ========= B) 逐特征（六个特征各自） =========
    summaries = []
    for feat, dfi in df.groupby("Feature"):
        cache_feat = [c for c in cache if c["Feature"] == feat]
        res = run_block_and_perm(df_block=dfi, cache_block=cache_feat, label=feat)
        res["Feature"] = feat
        res["n_trials_rows"] = len(dfi)
        summaries.append(res)

    summary_feat_df = pd.DataFrame(summaries)[[
        "Feature", "n_trials_rows", "r_bar", "beta0_obs", "perm_p_two_tailed", "n_perm_effective", "ols_type"
    ]].sort_values("Feature").reset_index(drop=True)
    summary_feat_csv = os.path.join(OUTPUT_DIR, "summary_by_feature.csv")
    summary_feat_df.to_csv(summary_feat_csv, index=False)

    # 友好提示
    print("\n=== 总体（六特征合并） ===")
    print(overall_res)
    print("\n=== 逐特征汇总（写入 summary_by_feature.csv） ===")
    print(summary_feat_df)

    print("\n结果已保存：")
    print(f"  - 逐试次结果（仅眼部特征）: {per_trial_csv}")
    print(f"  - 总体汇总: {os.path.join(OUTPUT_DIR, 'summary_overall.csv')}")
    print(f"  - 分特征汇总: {summary_feat_csv}")
    print("  - 置换次数 N_PERM =", N_PERM)

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============== 参数 ===============
ROOT = r""
OUT_DIR = r""
FPS = 30  # 帧率（秒↔帧转换）
SEC_BIN = 1  # 每多少秒聚合一次（1 或 5 常用）
FIXED_T_BINS = 200  # 统一相对时间长度（列数）
N_PERM = 500  # 置换次数（预跑 500；建议 >= 5000）
ALPHA = 0.05  # 显著性阈值（双尾）
FORCE_UNKNOWN_TO = None  # 'truth' / 'lie' / None
STANDARDIZE = "none"  # 'none' / 'z' / 'relative'
USE_FDR = True  # 是否对置换 p 值做 FDR(BH) 校正
SEED = 42  # 随机种子
# ——显示用白名单（原样大写，用在坐标轴标签/CSV展示）——
DISPLAY_WHITELIST = [
    "PupilSize", "IrisSize", "InnerEyeDistance",
    "EyelidDistance", "TheEyeOpening", "EarRatio"
]
# ——匹配用白名单（小写，用于读取/对齐数据）——
MATCH_WHITELIST = [w.lower() for w in DISPLAY_WHITELIST]
REF_FEATS = MATCH_WHITELIST[:]  # 聚合时的行顺序


# ==========================================


# --------- 工具函数 ---------
def ensure_dir(p): os.makedirs(p, exist_ok=True)


def norm_name(s: str) -> str:
    return str(s).strip().replace(" ", "").replace("\t", "").lower()


def read_csv_norm_cols(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding_errors="ignore")
    df.columns = [norm_name(c) for c in df.columns]
    return df


def assign_roles(csv_files):
    f1, f2 = sorted(csv_files, key=lambda x: x.lower())
    sd1, sa1 = f1[0].isdigit(), f1[0].isalpha()
    sd2, sa2 = f2[0].isdigit(), f2[0].isalpha()
    if sd1 and sa2:
        observer, observed = f1, f2
    elif sa1 and sd2:
        observer, observed = f2, f1
    else:
        observed, observer = f1, f2
    return observed, observer


def condition_from_observed_filename(observed_filename: str) -> str:
    if not observed_filename: return 'unknown'
    ch = observed_filename.strip()[0].lower()
    if ch == 't': return 'truth'
    if ch == 'l': return 'lie'
    return 'unknown'


def make_comparable(X_obs, X_obd, mode="none", eps=1e-8):
    if mode == "z":
        def z(x):
            m = np.nanmean(x, axis=0, keepdims=True)
            s = np.nanstd(x, axis=0, keepdims=True)
            return (x - m) / (s + eps)

        return z(X_obs), z(X_obd)
    return X_obs, X_obd


def diff_heatmap_matrix(X_obs, X_obd, fs, sec_bin, mode="none"):
    """
    输入 X_*: (T × F)，返回 times(nbins,), D_bins(F × nbins)
    """
    X_obs = np.asarray(X_obs);
    X_obd = np.asarray(X_obd)
    T = min(X_obs.shape[0], X_obd.shape[0])
    X_obs = X_obs[:T, :];
    X_obd = X_obd[:T, :]
    if X_obs.ndim == 1: X_obs = X_obs[:, None]
    if X_obd.ndim == 1: X_obd = X_obd[:, None]

    X_obs, X_obd = make_comparable(X_obs, X_obd, mode=STANDARDIZE)

    if mode == "relative":
        D_full = np.abs(X_obs - X_obd) / (np.abs(X_obs) + np.abs(X_obd) + 1e-8)  # (T×F)
    else:
        D_full = np.abs(X_obs - X_obd)  # (T×F)
    D_full = D_full.T  # -> (F×T)

    frames_per_bin = max(1, int(round(fs * sec_bin)))
    nbins = int(np.ceil(D_full.shape[1] / frames_per_bin))
    D_bins = np.empty((D_full.shape[0], nbins), dtype=float)
    for k in range(nbins):
        s = k * frames_per_bin
        e = min(D_full.shape[1], s + frames_per_bin)
        D_bins[:, k] = np.nanmean(D_full[:, s:e], axis=1)
    times = (np.arange(nbins) + 0.5) * sec_bin
    return times, D_bins


def resample_time_to_fixed(D, fixed_bins):
    if fixed_bins is None or D.shape[1] == fixed_bins:
        return D
    t_src = np.linspace(0, 1, num=D.shape[1])
    t_ref = np.linspace(0, 1, num=fixed_bins)
    out = np.empty((D.shape[0], fixed_bins), dtype=float)
    for i in range(D.shape[0]):
        out[i] = np.interp(t_ref, t_src, D[i])
    return out


def align_feat_time(D, common, ref_feats=None):
    """
    返回 Df(F×T, index=特征小写)；若缺行则补 NaN，多余截断。
    """
    D = np.asarray(D)
    F_expect = len(common)
    transposed = False

    if D.shape[0] == F_expect:
        pass
    elif D.shape[1] == F_expect:
        D = D.T;
        transposed = True
    else:
        if D.shape[1] >= D.shape[0]:
            D = D.T;
            transposed = True

    Df = pd.DataFrame(D)
    if Df.shape[0] > F_expect:
        Df = Df.iloc[:F_expect, :]
    elif Df.shape[0] < F_expect:
        add = F_expect - Df.shape[0]
        Df = pd.concat([Df, pd.DataFrame(np.nan, index=range(add), columns=Df.columns)], axis=0)

    Df.index = list(common)[:Df.shape[0]]
    if ref_feats is not None:
        Df = Df.reindex(index=ref_feats)
    return Df, transposed


def plot_diffmap(x, feat_names, D, title, save_path, vmin=None, vmax=None, sig_mask=None):
    """
    纵轴固定显示 DISPLAY_WHITELIST（与格子中心对齐）
    """
    ensure_dir(os.path.dirname(save_path))
    F = len(DISPLAY_WHITELIST)

    if D.shape[0] < F:
        pad = np.full((F - D.shape[0], D.shape[1]), np.nan, dtype=float)
        D = np.vstack([D, pad])
    elif D.shape[0] > F:
        D = D[:F, :]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(
        D, aspect='auto', origin='lower',
        extent=[x[0], x[-1], -0.5, F - 0.5],
        vmin=vmin, vmax=vmax
    )
    ax.set_ylim(-0.5, F - 0.5)
    ax.set_yticks(np.arange(F))
    ax.set_yticklabels(DISPLAY_WHITELIST)  # 显示原样大写

    ax.set_xlabel('Time (s)' if isinstance(x[0], (int, float)) and x[-1] > 1.1 else 'Relative time')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04);
    cb.set_label('|Observer - Observed|')

    if sig_mask is not None:
        try:
            ax.contour(
                sig_mask.astype(float), levels=[0.5], colors='k', linewidths=0.6,
                origin='lower', extent=[x[0], x[-1], -0.5, F - 0.5]
            )
        except Exception:
            pass

    plt.tight_layout()
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def fdr_bh_mask(pvals, alpha=0.05):
    pv = pvals.flatten()
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    if not np.any(passed):
        return np.zeros_like(pvals, dtype=bool)
    k_max = np.max(np.where(passed))
    crit = ranked[k_max]
    return pvals <= crit


def permutation_test_maps_online(truth_maps, lie_maps, n_perm=1000, alpha=0.05,
                                 two_tailed=True, seed=42, use_fdr=True):
    rng = np.random.default_rng(seed)
    A = np.stack(truth_maps, axis=0) if len(truth_maps) > 0 else None
    B = np.stack(lie_maps, axis=0) if len(lie_maps) > 0 else None
    if A is None or B is None:
        return None, None, None, None, None

    mean_A = np.nanmean(A, axis=0)
    mean_B = np.nanmean(B, axis=0)
    observed = mean_A - mean_B

    all_maps = np.concatenate([A, B], axis=0)
    nA, nB = A.shape[0], B.shape[0]

    greater = np.zeros_like(observed, dtype=float)
    smaller = np.zeros_like(observed, dtype=float)

    for _ in tqdm(range(n_perm), desc="Permutation test (diff maps)", unit="perm", dynamic_ncols=True):
        idx = rng.permutation(nA + nB)
        A_idx = all_maps[idx[:nA]]
        B_idx = all_maps[idx[nA:]]
        diff_p = np.nanmean(A_idx, axis=0) - np.nanmean(B_idx, axis=0)
        greater += (diff_p >= observed)
        smaller += (diff_p <= observed)

    pvals = (2 * np.minimum(greater, smaller) / n_perm) if two_tailed else (1 - greater / n_perm)
    pvals = np.clip(pvals, 0, 1)
    sig = fdr_bh_mask(pvals, alpha=alpha) if use_fdr else (pvals < alpha)
    return mean_A, mean_B, observed, pvals, sig


# ---------------- 主流程 ----------------
def main():
    ensure_dir(OUT_DIR)
    agg_dir = os.path.join(OUT_DIR, "_aggregate")
    ensure_dir(agg_dir)

    # 收集试次：每个子目录里“恰好 2 个 CSV”
    trials = []
    for dirpath, _, filenames in os.walk(ROOT):
        csvs = [f for f in filenames if f.lower().endswith(".csv")]
        if len(csvs) == 2:
            trials.append((dirpath, csvs))
    if not trials:
        print("⚠️ 没找到‘恰好 2 个 CSV’的试次。");
        return

    truth_maps, lie_maps, all_maps = [], [], []
    per_trial_meta = []

    for dirpath, csvs in tqdm(sorted(trials), desc="处理试次", unit="trial", total=len(trials), dynamic_ncols=True):
        rel = os.path.relpath(dirpath, ROOT)
        observed_name, observer_name = assign_roles(csvs)
        path_observed = os.path.join(dirpath, observed_name)
        path_observer = os.path.join(dirpath, observer_name)
        condition = condition_from_observed_filename(observed_name)

        try:
            df_obd = read_csv_norm_cols(path_observed)
            df_obs = read_csv_norm_cols(path_observer)

            feats_obd = [c for c in MATCH_WHITELIST if c in df_obd.columns]
            feats_obs = [c for c in MATCH_WHITELIST if c in df_obs.columns]
            common = [c for c in MATCH_WHITELIST if (c in feats_obd and c in feats_obs)]
            if len(common) == 0:
                tqdm.write(f"（跳过）{rel} 6个特征缺失过多（无共同列）。")
                per_trial_meta.append(
                    [rel, observed_name, observer_name, condition, 0, 0, 0, len(MATCH_WHITELIST), 'no_common'])
                continue

            df_obd_sel = df_obd[common].apply(pd.to_numeric, errors="coerce").interpolate(
                limit_direction="both").bfill().ffill()
            df_obs_sel = df_obs[common].apply(pd.to_numeric, errors="coerce").interpolate(
                limit_direction="both").bfill().ffill()
            X_obd = df_obd_sel.values
            X_obs = df_obs_sel.values

            times, D = diff_heatmap_matrix(X_obs, X_obd, fs=FPS, sec_bin=SEC_BIN, mode=STANDARDIZE)
            Df, transposed = align_feat_time(D, common, ref_feats=REF_FEATS)
            D_used = Df.values  # (6 × Tbins)

            # 单试次图（真实秒）
            vmin = np.nanpercentile(D_used, 5) if np.isfinite(D_used).any() else None
            vmax = np.nanpercentile(D_used, 95) if np.isfinite(D_used).any() else None
            plot_diffmap(times, REF_FEATS, D_used,
                         f"Difference |Observer-Observed|  ({condition})",
                         os.path.join(OUT_DIR, rel, "diffmap.png"),
                         vmin=vmin, vmax=vmax)

            # 聚合：重采样到固定长度
            D_rs = resample_time_to_fixed(D_used, FIXED_T_BINS)
            all_maps.append(D_rs)
            if condition == 'truth':
                truth_maps.append(D_rs);
                cond_used = 'truth'
            elif condition == 'lie':
                lie_maps.append(D_rs);
                cond_used = 'lie'
            else:
                if FORCE_UNKNOWN_TO == 'truth':
                    truth_maps.append(D_rs);
                    cond_used = 'unknown->truth'
                elif FORCE_UNKNOWN_TO == 'lie':
                    lie_maps.append(D_rs);
                    cond_used = 'unknown->lie'
                else:
                    cond_used = 'unknown(skip)'

            per_trial_meta.append([
                rel, observed_name, observer_name, condition,
                len(common), D_used.shape[1], int(transposed),
                len([f for f in REF_FEATS if f not in common]),
                cond_used
            ])

        except Exception as e:
            tqdm.write(f"❌ 试次 {rel} 失败：{e}")
            per_trial_meta.append(
                [rel, observed_name, observer_name, condition, 0, 0, 'err', len(MATCH_WHITELIST), str(e)])

    # 写 meta 日志
    ensure_dir(agg_dir)
    meta_cols = ["trial_relpath", "observed_csv", "observer_csv", "condition",
                 "n_common_feats", "n_time_bins", "transposed", "n_missing_ref_feats", "note"]
    pd.DataFrame(per_trial_meta, columns=meta_cols).to_csv(
        os.path.join(agg_dir, "per_trial_meta.csv"), index=False, encoding="utf-8-sig"
    )

    # 聚合与统计
    x_disp = np.linspace(0, 1, FIXED_T_BINS)

    def plot_agg(D, title, path, sig=None):
        vmin = np.nanpercentile(D, 5) if np.isfinite(D).any() else None
        vmax = np.nanpercentile(D, 95) if np.isfinite(D).any() else None
        plot_diffmap(x_disp, REF_FEATS, D, title, path, vmin=vmin, vmax=vmax, sig_mask=sig)

    if len(truth_maps) > 0 and len(lie_maps) > 0:
        mean_truth, mean_lie, diff_map, pvals, sig = permutation_test_maps_online(
            truth_maps, lie_maps, n_perm=N_PERM, alpha=ALPHA, two_tailed=True, seed=SEED, use_fdr=USE_FDR
        )
        plot_agg(mean_truth, "Mean Difference Map |O - V| (Truth)", os.path.join(agg_dir, "diff_truth_mean.png"))
        plot_agg(mean_lie, "Mean Difference Map |O - V| (Lie)", os.path.join(agg_dir, "diff_lie_mean.png"))
        plot_agg(diff_map, "Difference (Truth - Lie)", os.path.join(agg_dir, "diff_T_minus_L.png"), sig=None)
        plot_agg(diff_map, "Difference (Truth - Lie) with significant contours",
                 os.path.join(agg_dir, "diff_T_minus_L_with_sig.png"), sig=sig)

        # p 值热图
        fig, ax = plt.subplots(figsize=(10, 3.8))
        im = ax.imshow(pvals, aspect='auto', origin='lower',
                       extent=[x_disp[0], x_disp[-1], -0.5, len(DISPLAY_WHITELIST) - 0.5],
                       vmin=0, vmax=1)
        ax.set_ylim(-0.5, len(DISPLAY_WHITELIST) - 0.5)
        ax.set_yticks(np.arange(len(DISPLAY_WHITELIST)))
        ax.set_yticklabels(DISPLAY_WHITELIST)
        ax.set_xlabel('Relative time');
        ax.set_ylabel('Feature')
        ax.set_title(f'Permutation p-values (two-tailed), α={ALPHA}' + (' [FDR]' if USE_FDR else ''))
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04);
        cb.set_label('p')
        try:
            ax.contour(sig.astype(float), levels=[0.5], colors='k', linewidths=0.6,
                       origin='lower', extent=[x_disp[0], x_disp[-1], -0.5, len(DISPLAY_WHITELIST) - 0.5])
        except Exception:
            pass
        plt.tight_layout();
        fig.savefig(os.path.join(agg_dir, "diff_perm_pvalues.png"), dpi=240);
        plt.close(fig)

        # 显著置零图
        masked = diff_map.copy();
        masked[~sig] = 0.0
        plot_agg(masked, "Difference (Truth - Lie), significant cells only",
                 os.path.join(agg_dir, "diff_T_minus_L_significant.png"))

        # 导出 CSV（兼顾匹配名与展示名）
        pd.DataFrame({
            "feature_lower": np.repeat(REF_FEATS, FIXED_T_BINS),
            "feature_label": np.repeat(DISPLAY_WHITELIST, FIXED_T_BINS),
            "relative_time": np.tile(x_disp, len(REF_FEATS)),
            "diff_T_minus_L": diff_map.flatten(),
            "p_value": pvals.flatten(),
            "significant": sig.flatten().astype(int),
        }).to_csv(os.path.join(agg_dir, "diff_permutation_results.csv"), index=False, encoding="utf-8-sig")

        # 组内均值
        pd.DataFrame(mean_truth, index=REF_FEATS, columns=[f"t{i:03d}" for i in range(FIXED_T_BINS)]) \
            .to_csv(os.path.join(agg_dir, "mean_truth_matrix.csv"), encoding="utf-8-sig")
        pd.DataFrame(mean_lie, index=REF_FEATS, columns=[f"t{i:03d}" for i in range(FIXED_T_BINS)]) \
            .to_csv(os.path.join(agg_dir, "mean_lie_matrix.csv"), encoding="utf-8-sig")

    if len(all_maps) > 0:
        overall_mean = np.nanmean(np.stack(all_maps, axis=0), axis=0)
        plot_agg(overall_mean, "Mean Difference Map |O - V| (All trials)",
                 os.path.join(agg_dir, "diff_all_trials_mean.png"))

    print("\n 生成完成：查看输出目录：", OUT_DIR)
    print("显示特征（纵轴标签）：", DISPLAY_WHITELIST)
    print(f"FIXED_T_BINS={FIXED_T_BINS}, STANDARDIZE={STANDARDIZE}, FDR={USE_FDR}")


if __name__ == "__main__":
    main()

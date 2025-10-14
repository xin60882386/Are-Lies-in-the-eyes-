
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from scipy.signal import detrend
import pywt

# ================== 参数 ==================
ROOT    = r""   # 根目录（递归遍历）
OUT_DIR = r""          # 输出目录
FPS     = 30                         # 帧率（秒↔帧转换）
SEC_BIN = 1                          # 差值图的时间聚合间隔（秒），可改为 5
# WTC 设置
WAVELET = 'cmor1.5-1.0'              # 复 Morlet；格式 'cmorB-C'（带宽-中心频率）
NUM_SCALES = 64                      # 频率分辨率（越大越细）
SMOOTH_T = 2.0                       # WTC 平滑（时间向，高斯σ，单位=窗点数）
SMOOTH_S = 1.0                       # WTC 平滑（尺度向，高斯σ）
# 置换检验
N_PERM = 2000                        # 置换次数（>1000 更稳）
ALPHA  = 0.05                        # 显著性阈值（双尾）
# 条件关键字（从“文件夹名 或 CSV 文件名”里匹配）
LIE_KEYWORDS   = [r'lie', r'decep', r'false', r'fake', r'欺骗', r'谎']
TRUTH_KEYWORDS = [r'true', r'truth', r'honest', r'real', r'真实', r'真话']
# ====================================================

# === 列名候选（会统一：去空白+小写），命中多少用多少 ===
FEATURE_COL_CANDIDATES = [
    ["PupilSize","pupil","pupil_size","pupilsize","pupil_diameter"],
    ["IrisSize","iris","iris_size","irissize"],
    ["InnerEyedistance","inner_eye_distance","innereyedistance","inner_eye"],
    ["Eyeliddistance","eyelid_distance","eyeliddistance","eyelid_gap","eyelidwidth"],
    ["TheEyeOpening","eye_opening","eyeopening","theeyeopening"],
    ["EarRatio","EAR","ear","earratio"],
    # 中文
    ["瞳孔大小","瞳孔"], ["虹膜大小","虹膜"], ["内眼距","内眼距离"],
    ["眼睑距离","眼睑距"], ["眼裂开度","开度"], ["眼裂比","纵横比"]
]

# ---------------------- 工具函数 ----------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_csv_norm_cols(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding_errors="ignore")
    df.columns = [c.strip().replace(" ", "").replace("\t", "").lower() for c in df.columns]
    return df

def find_first_existing(df, candidates):
    cols = set(df.columns)
    for cand in candidates:
        if isinstance(cand, (list, tuple)):
            for cc in cand:
                name = str(cc).strip().replace(" ", "").lower()
                if name in cols: return name
        else:
            name = str(cand).strip().replace(" ", "").lower()
            if name in cols: return name
    return None

def assemble_feature_matrix(df, feature_groups):
    used = []
    for group in feature_groups:
        col = find_first_existing(df, group)
        if col is not None: used.append(col)
    if not used:
        raise ValueError("未匹配到任何眼部特征列，请在 FEATURE_COL_CANDIDATES 中补充列名。")
    X = df[used].apply(pd.to_numeric, errors="coerce").values
    # 插补 + 去趋势 + 标准化
    X = pd.DataFrame(X).interpolate(limit_direction="both").bfill().ffill().values
    X = detrend(X, axis=0, type='linear')
    X = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
    return X, used

def assign_roles(csv_files):
    """
    恰好2个CSV：
      数字开头 -> Observer\detectors（观察者）
      字母开头 -> Observed\deceivers（被观察者）
    若同类 -> 第一个=Observed，第二个=Observer
    """
    f1, f2 = sorted(csv_files, key=lambda x: x.lower())
    sd1, sa1 = f1[0].isdigit(), f1[0].isalpha()
    sd2, sa2 = f2[0].isdigit(), f2[0].isalpha()
    if sd1 and sa2: observer, observed = f1, f2
    elif sa1 and sd2: observer, observed = f2, f1
    else: observed, observer = f1, f2
    return observed, observer

def detect_condition(dirpath, csvs):
    """从目录名或文件名推断条件：'lie' or 'truth' or 'unknown'"""
    name_pool = [os.path.basename(dirpath).lower()] + [c.lower() for c in csvs]
    text = " ".join(name_pool)
    def has_any(keys): return any(re.search(k, text) for k in keys)
    if has_any(LIE_KEYWORDS): return 'lie'
    if has_any(TRUTH_KEYWORDS): return 'truth'
    return 'unknown'

# ---------------------- WTC 计算 ----------------------
def compute_wtc(x, y, fs, wavelet=WAVELET, num_scales=NUM_SCALES, smooth_t=SMOOTH_T, smooth_s=SMOOTH_S):
    """
    x,y: 1D 序列（长度相同，已标准化）
    返回：time, freq(Hz), WTC(时间×频率)  ∈ [0,1]
    """
    n = len(x)
    # 连续小波变换
    # scales 越大 -> 频率越低；取指数刻度更均匀
    scales = np.geomspace(2, n/4, num=num_scales)
    Wx, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0/fs)  # 形状: (scales, time)
    Wy, _ = pywt.cwt(y, scales, wavelet, sampling_period=1.0/fs)

    # 交叉谱与功率谱
    Wxy = Wx * np.conj(Wy)
    Sxx = np.abs(Wx)**2
    Syy = np.abs(Wy)**2

    # 平滑（时间、尺度双向高斯）
    def smooth2d(M):
        # 先时间向，再尺度向
        Ms = gaussian_filter(M.real, sigma=(smooth_s, smooth_t)) + 1j*gaussian_filter(M.imag, sigma=(smooth_s, smooth_t))
        return Ms
    S_Wxy = smooth2d(Wxy)
    S_Sxx = gaussian_filter(Sxx, sigma=(smooth_s, smooth_t))
    S_Syy = gaussian_filter(Syy, sigma=(smooth_s, smooth_t))

    wtc = np.abs(S_Wxy)**2 / (S_Sxx * S_Syy + 1e-12)
    wtc = np.clip(wtc, 0.0, 1.0)  # (scales, time)

    # 频率（Hz）
    freqs = pywt.scale2frequency(wavelet, scales) * fs
    t = np.arange(n) / fs
    return t, freqs, wtc

def plot_wtc(t, freqs, wtc, save_path):
    fig, ax = plt.subplots(figsize=(10,4))
    im = ax.imshow(wtc, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], freqs[0], freqs[-1]])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Wavelet Coherence (Observer vs Observed)')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('Coherence')
    plt.tight_layout(); fig.savefig(save_path, dpi=220); plt.close(fig)

# ---------------------- 差值图 & 置换检验 ----------------------
def diff_heatmap(X_obs, X_obd, fs, sec_bin=1, feature_names=None):
    """
    生成 |X_obs - X_obd| 的时间×特征热图（按 sec_bin 聚合成均值）
    返回：time_bins(中心秒), feature_names, D (F × T)
    """
    X_obs = np.asarray(X_obs); X_obd = np.asarray(X_obd)
    T = min(len(X_obs), len(X_obd))
    X_obs = X_obs[:T]; X_obd = X_obd[:T]
    # 按特征维求绝对差（F×T）
    if X_obs.ndim == 1: X_obs = X_obs[:, None]
    if X_obd.ndim == 1: X_obd = X_obd[:, None]
    D = np.abs(X_obs - X_obd).T  # 形状: (F, T)
    # 聚合到每 sec_bin 秒
    frames_per_bin = max(1, int(round(fs * sec_bin)))
    nbins = int(np.ceil(T / frames_per_bin))
    D_bins = np.zeros((D.shape[0], nbins))
    for k in range(nbins):
        s = k * frames_per_bin
        e = min(T, s + frames_per_bin)
        D_bins[:, k] = np.nanmean(D[:, s:e], axis=1)
    times = (np.arange(nbins) + 0.5) * sec_bin
    if feature_names is None:
        feature_names = [f"f{i+1}" for i in range(D_bins.shape[0])]
    return times, feature_names, D_bins

def plot_diffmap(times, feat_names, D, title, save_path):
    fig, ax = plt.subplots(figsize=(10,3.8))
    im = ax.imshow(D, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], 0, len(feat_names)])
    ax.set_yticks(np.arange(len(feat_names))+0.5)
    ax.set_yticklabels(feat_names)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Feature')
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('|Observer - Observed|')
    plt.tight_layout(); fig.savefig(save_path, dpi=220); plt.close(fig)

def permutation_test_maps(truth_maps, lie_maps, n_perm=1000, alpha=0.05, two_tailed=True, seed=42):
    """
    输入两组矩阵列表（每个元素形状相同：F×T）
    返回：mean_truth, mean_lie, diff (truth - lie), pvals(F×T), sig_mask(F×T)
    """
    rng = np.random.default_rng(seed)
    A = np.stack(truth_maps, axis=0) if len(truth_maps)>0 else None
    B = np.stack(lie_maps,   axis=0) if len(lie_maps)>0 else None
    if A is None or B is None:
        return None, None, None, None, None
    mean_A = np.nanmean(A, axis=0)
    mean_B = np.nanmean(B, axis=0)
    observed = mean_A - mean_B  # F×T

    # 置换：打乱 trial 标签
    all_maps = np.concatenate([A, B], axis=0)
    nA, nB = A.shape[0], B.shape[0]
    diffs = np.zeros((n_perm,)+observed.shape)
    for p in tqdm(range(n_perm), desc="Permutation test", unit="perm", dynamic_ncols=True):
        idx = rng.permutation(nA + nB)
        A_idx = all_maps[idx[:nA]]
        B_idx = all_maps[idx[nA:]]
        diffs[p] = np.nanmean(A_idx, axis=0) - np.nanmean(B_idx, axis=0)

    # 双尾 p 值
    greater = np.mean(diffs >= observed, axis=0)
    smaller = np.mean(diffs <= observed, axis=0)
    if two_tailed:
        pvals = 2*np.minimum(greater, smaller)
    else:
        pvals = 1 - greater
    pvals = np.clip(pvals, 0, 1)
    sig = pvals < alpha
    return mean_A, mean_B, observed, pvals, sig

# ---------------------- 主流程 ----------------------
def main():
    ensure_dir(OUT_DIR)
    wtc_stack = []            # 收集所有试次的 WTC（用时间重采样对齐）
    wtc_freq_ref = None
    wtc_time_ref = None

    # 差值图容器（按条件）
    diff_truth, diff_lie = [], []
    diff_meta = None  # (times, feat_names) 仅保存一次

    # 收集试次
    trials = []
    for dirpath, _, filenames in os.walk(ROOT):
        csvs = [f for f in filenames if f.lower().endswith(".csv")]
        if len(csvs) == 2:
            trials.append((dirpath, csvs))

    if not trials:
        print("⚠️ 没找到‘恰好 2 个 CSV’的试次。")
        return

    # 逐试次处理
    for dirpath, csvs in tqdm(sorted(trials), desc="处理试次", unit="trial", total=len(trials), dynamic_ncols=True):
        rel = os.path.relpath(dirpath, ROOT)
        condition = detect_condition(dirpath, csvs)  # 'truth'/'lie'/'unknown'

        observed_name, observer_name = assign_roles(csvs)
        path_observed = os.path.join(dirpath, observed_name)
        path_observer = os.path.join(dirpath, observer_name)

        try:
            dfObsrvd = read_csv_norm_cols(path_observed)  # 被观察者
            dfObsrvr = read_csv_norm_cols(path_observer)  # 观察者

            X_obd, feat_names = assemble_feature_matrix(dfObsrvd, FEATURE_COL_CANDIDATES)
            X_obs, _          = assemble_feature_matrix(dfObsrvr, FEATURE_COL_CANDIDATES)

            # 对齐长度
            T = min(len(X_obd), len(X_obs))
            X_obd = X_obd[:T]; X_obs = X_obs[:T]

            # ===== 1) WTC（用多特征均值的一维信号） =====
            s1 = X_obs.mean(axis=1)
            s2 = X_obd.mean(axis=1)
            t_sec, freqs, wtc = compute_wtc(s1, s2, fs=FPS)

            # 保存单个被试 WTC 图
            out_trial = os.path.join(OUT_DIR, rel)
            ensure_dir(out_trial)
            plot_wtc(t_sec, freqs, wtc, os.path.join(out_trial, "wtc.png"))

            # 为聚合对齐：统一时间维长度（取固定点数）
            WTC_T = 150
            tt = np.linspace(t_sec[0], t_sec[-1], WTC_T)
            wtc_rs = np.zeros((wtc.shape[0], WTC_T))
            for i in range(wtc.shape[0]):
                wtc_rs[i] = np.interp(tt, t_sec, wtc[i])
            wtc_stack.append(wtc_rs)
            if wtc_freq_ref is None:
                wtc_freq_ref = freqs
                wtc_time_ref = tt

            # ===== 2) 差值图（时间×特征；|O−V|，按秒聚合） =====
            times, feats, D = diff_heatmap(X_obs, X_obd, fs=FPS, sec_bin=SEC_BIN, feature_names=feat_names)
            plot_diffmap(times, feats, D, f"Difference |Observer-Observed|  ({condition})",
                         os.path.join(out_trial, "diffmap.png"))

            # 收集到条件池
            if diff_meta is None: diff_meta = (times, feats)
            if condition == 'truth':
                diff_truth.append(D)
            elif condition == 'lie':
                diff_lie.append(D)
            else:
                # 条件未知的不参加置换检验，但仍然会有单被试图
                pass

        except Exception as e:
            tqdm.write(f" 试次 {rel} 失败：{e}")

    # ===== A) 聚合 WTC =====
    if wtc_stack:
        W = np.stack(wtc_stack, axis=0)        # (N, scales, T)
        W_mean = np.nanmean(W, axis=0)
        # 绘制
        fig, ax = plt.subplots(figsize=(10,4))
        im = ax.imshow(W_mean, aspect='auto', origin='lower',
                       extent=[wtc_time_ref[0], wtc_time_ref[-1], wtc_freq_ref[0], wtc_freq_ref[-1]])
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Aggregated Wavelet Coherence (mean across trials)')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('Coherence')
        agg_dir = os.path.join(OUT_DIR, "_aggregate"); ensure_dir(agg_dir)
        plt.tight_layout(); fig.savefig(os.path.join(agg_dir, "wtc_aggregate.png"), dpi=240); plt.close(fig)

    # ===== B) 条件差值图 + 置换检验 =====
    if diff_meta is not None and (len(diff_truth) > 0) and (len(diff_lie) > 0):
        times, feats = diff_meta
        mean_truth, mean_lie, diff_map, pvals, sig = permutation_test_maps(
            diff_truth, diff_lie, n_perm=N_PERM, alpha=ALPHA, two_tailed=True
        )

        # 平均图
        plot_diffmap(times, feats, mean_truth, "Mean Difference Map |O - V| (Truth)", os.path.join(agg_dir, "diff_truth_mean.png"))
        plot_diffmap(times, feats, mean_lie,   "Mean Difference Map |O - V| (Lie)",   os.path.join(agg_dir, "diff_lie_mean.png"))

        # 差异图与显著性
        plot_diffmap(times, feats, diff_map, "Difference (Truth - Lie)", os.path.join(agg_dir, "diff_T_minus_L.png"))

        # p 值热图
        fig, ax = plt.subplots(figsize=(10,3.8))
        im = ax.imshow(pvals, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], 0, len(feats)], vmin=0, vmax=ALPHA)
        ax.set_yticks(np.arange(len(feats))+0.5)
        ax.set_yticklabels(feats)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Feature')
        ax.set_title(f'Permutation p-values (two-tailed), α={ALPHA}')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('p')
        plt.tight_layout(); fig.savefig(os.path.join(agg_dir, "diff_perm_pvalues.png"), dpi=240); plt.close(fig)

        # 显著掩码叠加在差异图上（非显著打淡）
        masked = diff_map.copy()
        masked[~sig] = 0.0
        fig, ax = plt.subplots(figsize=(10,3.8))
        im = ax.imshow(masked, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], 0, len(feats)])
        ax.set_yticks(np.arange(len(feats))+0.5)
        ax.set_yticklabels(feats)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Feature')
        ax.set_title('Difference (Truth - Lie), significant cells only')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label('Truth - Lie')
        plt.tight_layout(); fig.savefig(os.path.join(agg_dir, "diff_T_minus_L_significant.png"), dpi=240); plt.close(fig)

        # 导出表格
        pd.DataFrame({
            "feature": np.repeat(feats, len(times)),
            "time_s":  np.tile(times, len(feats)),
            "diff_T_minus_L": diff_map.flatten(),
            "p_value": pvals.flatten(),
            "significant": sig.flatten().astype(int)
        }).to_csv(os.path.join(agg_dir, "diff_permutation_results.csv"), index=False, encoding="utf-8-sig")

    print("\n 已完成：每试次 WTC+差值图、聚合 WTC、条件差值与置换检验。输出目录：", OUT_DIR)

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# === 自定义路径 ===
root_dir = r""        # 所有被试数据的总目录
output_dir = r""       # 输出结果目录
os.makedirs(output_dir, exist_ok=True)

# === 欧氏距离函数（同帧或跨帧） ===
def compute_distance(df, x1, y1, z1, x2, y2, z2, offset=1):
    result = []
    for i in range(len(df)):
        try:
            j = i + offset
            if j >= len(df):
                result.append(np.nan)
                continue
            d = np.sqrt(
                (df[x1].iloc[i] - df[x2].iloc[j]) ** 2 +
                (df[y1].iloc[i] - df[y2].iloc[j]) ** 2 +
                (df[z1].iloc[i] - df[z2].iloc[j]) ** 2
            )
            result.append(round(d, 2))
        except:
            result.append(np.nan)
    return result

# === EAR 特征计算 ===
def compute_ear(df):
    result = []
    for i in range(len(df)):
        try:
            L1 = np.sqrt((df['eye_lmk_X_11'][i] - df['eye_lmk_X_17'][i]) ** 2 + (df['eye_lmk_Y_11'][i] - df['eye_lmk_Y_17'][i]) ** 2)
            H1 = np.sqrt((df['eye_lmk_X_8'][i] - df['eye_lmk_X_14'][i]) ** 2 + (df['eye_lmk_Y_8'][i] - df['eye_lmk_Y_14'][i]) ** 2)
            if H1 != 0:
                result.append(round(L1 / H1, 4))
            else:
                result.append(np.nan)
        except:
            result.append(np.nan)
    return result

# === TheEyeOpening 特征计算 ===
def compute_eyeopening(df):
    result = []
    for i in range(len(df)-1):
        try:
            midX1 = (df['x_41'][i] + df['x_40'][i]) / 2
            midY1 = (df['y_41'][i] + df['y_40'][i]) / 2
            midX2 = (df['x_37'][i+1] + df['x_38'][i+1]) / 2
            midY2 = (df['y_37'][i+1] + df['y_38'][i+1]) / 2
            dist = np.sqrt((midX1 - midX2) ** 2 + (midY1 - midY2) ** 2)
            result.append(round(dist, 2))
        except:
            result.append(np.nan)
    result.append(np.nan)
    return result

# === 三个动态统计值 ===
def compute_statistics(series):
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    diff = round(clean.max() - clean.min(), 2)
    peak = round(clean.max(), 2)
    slope = round((clean.iloc[-1] - clean.iloc[0]) / len(clean), 4) if len(clean) > 1 else np.nan
    return diff, peak, slope

# === 开始处理 ===
dynamic_features = []
static_all = []
global_id = 1

for subject in tqdm(os.listdir(root_dir), desc="遍历被试"):
    subject_path = os.path.join(root_dir, subject)
    split_path = os.path.join(subject_path, "split")
    if not os.path.isdir(split_path):
        continue

    for trial in os.listdir(split_path):
        trial_path = os.path.join(split_path, trial)
        if not os.path.isdir(trial_path):
            continue

        label_suffix = trial[-2:]  # e.g., 'tf'
        label_video = label_suffix[0]  # 't'
        label_judge = label_suffix[1]  # 'f'

        for file in os.listdir(trial_path):
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(trial_path, file)
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()

                df['PupilSize'] = compute_distance(df, 'eye_lmk_X_21', 'eye_lmk_Y_21', 'eye_lmk_Z_21', 'eye_lmk_X_25', 'eye_lmk_Y_25', 'eye_lmk_Z_25')
                df['TheEyeOpening'] = compute_eyeopening(df)
                df['EarRatio'] = compute_ear(df)
                df['InnerEyedistance'] = compute_distance(df, 'eye_lmk_X_34', 'eye_lmk_Y_34', 'eye_lmk_Z_34', 'eye_lmk_X_36', 'eye_lmk_Y_36', 'eye_lmk_Z_36')
                df['Eyeliddistance'] = compute_distance(df, 'eye_lmk_X_11', 'eye_lmk_Y_11', 'eye_lmk_Z_11', 'eye_lmk_X_17', 'eye_lmk_Y_17', 'eye_lmk_Z_17')
                df['IrisSize'] = compute_distance(df, 'eye_lmk_X_2', 'eye_lmk_Y_2', 'eye_lmk_Z_2', 'eye_lmk_X_6', 'eye_lmk_Y_6', 'eye_lmk_Z_6')

                # === 每帧静态信息添加id标签等基础字段 ===
                df.insert(0, "id", global_id)
                df.insert(1, "label_video", label_video)
                df.insert(2, "label_judge", label_judge)
                df.insert(3, "subject", subject)
                df.insert(4, "trial", trial)
                static_all.append(df)

                # === 动态统计信息 ===
                record = {
                    "id": global_id,
                    "label_video": label_video,
                    "label_judge": label_judge,
                    "subject": subject,
                    "trial": trial
                }
                for feat in ["PupilSize", "TheEyeOpening", "EarRatio", "InnerEyedistance", "Eyeliddistance", "IrisSize"]:
                    diff, peak, slope = compute_statistics(df[feat])
                    record[f"{feat}_diff"] = diff
                    record[f"{feat}_peak"] = peak
                    record[f"{feat}_slope"] = slope
                dynamic_features.append(record)
                global_id += 1

            except Exception as e:
                print(f"❌ 错误文件: {file_path} - {str(e)}")

# === 保存整合后的静态数据 ===
static_df = pd.concat(static_all, ignore_index=True)
static_df.to_csv(os.path.join(output_dir, "static_all.csv"), index=False)

# === 保存动态统计结果表 ===
df_all = pd.DataFrame(dynamic_features)
df_all.to_csv(os.path.join(output_dir, "dynamic_features_summary.csv"), index=False)

print("✅ 所有特征提取完成（静态 + 动态）")

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)

# === Step 1: 读取数据，处理标签 ===
df = pd.read_csv(r"")
df["label_judge"] = df["label_judge"].replace({'t': 0, 'f': 1}).astype(int)

# === Step 2: 明确18个动态特征 ===
feature_cols = [
    'PupilSize_diff', 'PupilSize_peak', 'PupilSize_slope',
    'IrisSize_diff', 'IrisSize_peak', 'IrisSize_slope',
    'InnerEyedistance_diff', 'InnerEyedistance_peak', 'InnerEyedistance_slope',
    'Eyeliddistance_diff', 'Eyeliddistance_peak', 'Eyeliddistance_slope',
    'TheEyeOpening_diff', 'TheEyeOpening_peak', 'TheEyeOpening_slope',
    'EarRatio_diff', 'EarRatio_peak', 'EarRatio_slope'
]
X_full = df[feature_cols]
y_full = df["label_judge"]

# === Step 3: 下采样多数类（保留远离边界） ===
logit = LogisticRegression(max_iter=1000)
logit.fit(X_full, y_full)
df["proba"] = logit.predict_proba(X_full)[:, 1]

df_major = df[df["label_judge"] == 0].copy()
df_minor = df[df["label_judge"] == 1].copy()
n_minor = len(df_minor)
df_major["distance_to_0.5"] = np.abs(df_major["proba"] - 0.5)
df_major_sorted = df_major.sort_values(by="distance_to_0.5", ascending=False)
df_major_selected = df_major_sorted.head(n_minor)

df_balanced = pd.concat([df_major_selected, df_minor], ignore_index=True)
df_balanced = df_balanced.drop(columns=["proba", "distance_to_0.5"])
X = df_balanced[feature_cols]
y = df_balanced["label_judge"]

# === Step 4: 构建模型列表 ===
xgb_weight = Counter(y)[0] / Counter(y)[1]
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             scale_pos_weight=xgb_weight, verbosity=0)
}

# === Step 5: 模型训练 + 全指标输出 + 选最佳模型（AUC）===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
best_model = None
best_model_name = ""
best_auc = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===  模型 {name} ===")
    print(f" Accuracy  = {acc:.3f}")
    print(f" F1-score  = {f1:.3f}")
    print(f" AUC       = {auc:.3f}")
    print(f" Precision = {precision:.3f}")
    print(f" Recall    = {recall:.3f}")
    print(f" Confusion Matrix:\n{cm}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# === Step 6: 用最佳模型预测原始数据并保存 ===
df["judge_pred"] = best_model.predict(df[feature_cols])
df.to_csv(r"", index=False)

print(f"\n 最佳模型：{best_model_name}，AUC = {best_auc:.3f}")
print(" judge_pred 已保存！")

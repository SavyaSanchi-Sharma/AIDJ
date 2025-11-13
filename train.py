import os, json, joblib, numpy as np, pandas as pd
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt, seaborn as sns

DATASET = "datasets/features_dataset.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def normalize(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

def perceptual_energy(df):
    rms = normalize(df["rms"].astype(float))
    centroid = normalize(df["spectral_centroid"].astype(float))
    bandwidth = normalize(df["spectral_bandwidth"].astype(float))
    rolloff = normalize(df["spectral_rolloff"].astype(float))
    return 0.5*rms + 0.3*centroid + 0.2*(0.5*bandwidth + 0.5*rolloff)

def load_data():
    df = pd.read_csv(DATASET)
    X = df.drop(columns=["genre", "file_name", "start", "end"], errors="ignore")
    y_reg = perceptual_energy(df)
    y_cls = (y_reg > 0.6).astype(int)
    tempo = df["tempo"].fillna(df["tempo"].median())
    chroma = df["chroma_mean"].fillna(df["chroma_mean"].median())
    contrast = df["spectral_contrast"].fillna(df["spectral_contrast"].median())
    def mood_rule(i):
        e = y_reg.iloc[i]; t = tempo.iloc[i]; c = chroma.iloc[i]; s = contrast.iloc[i]
        if e > 0.6 and t > 110: return "energetic"
        if e < 0.2 and t < 90: return "calm"
        if c > 0.5 and s < 20: return "happy"
        return "sad"
    y_mood = pd.Series([mood_rule(i) for i in range(len(df))], index=df.index, name="y_mood")
    groups = pd.Series(df["file_name"].values, index=df.index, name="group")
    return X, y_reg, y_cls, y_mood, groups

def split_data(X, y_reg, y_cls, y_mood, groups):
    gss = GroupShuffleSplit(test_size=TEST_SIZE, n_splits=1, random_state=RANDOM_STATE)
    tr, ts = next(gss.split(X, y_reg, groups))
    return X.iloc[tr], X.iloc[ts], y_reg.iloc[tr], y_reg.iloc[ts], y_cls.iloc[tr], y_cls.iloc[ts], y_mood.iloc[tr], y_mood.iloc[ts], groups.iloc[tr], groups.iloc[ts]

def evaluate_reg(model, Xt, yt):
    yp = model.predict(Xt)
    return {"r2_score": float(r2_score(yt, yp)), "mae": float(mean_absolute_error(yt, yp)), "rmse": float(np.sqrt(np.mean((yt-yp)**2)))}

def evaluate_cls(model, Xt, yt):
    yp = model.predict(Xt)
    acc = accuracy_score(yt, yp)
    prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "report": classification_report(yt, yp, zero_division=0)}

def oversample(X, y):
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

def train_energy_regressor(X, y):
    m = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05, max_iter=600,
                                      l2_regularization=0.1, early_stopping=True,
                                      validation_fraction=0.1, random_state=RANDOM_STATE)
    m.fit(X, y)
    return m

def train_energy_classifier(X, y):
    m = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=600,
                                       early_stopping=True, validation_fraction=0.1,
                                       random_state=RANDOM_STATE)
    m.fit(X, y)
    return m

def train_mood_classifier(X, y):
    Xb, yb = oversample(X, y)
    m = HistGradientBoostingClassifier(max_depth=10, learning_rate=0.05, max_iter=700,
                                       early_stopping=True, validation_fraction=0.1,
                                       random_state=RANDOM_STATE)
    m.fit(Xb, yb)
    return m

def save_conf(y_true, y_pred, path, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("pred"); plt.ylabel("true"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def train_models():
    X, y_reg, y_cls, y_mood, groups = load_data()
    Xtr, Xte, yrtr, yrte, yctr, ycte, ymtr, ymte, gtr, gte = split_data(X, y_reg, y_cls, y_mood, groups)
    reg = train_energy_regressor(Xtr, yrtr)
    cls = train_energy_classifier(Xtr, yctr)
    mood = train_mood_classifier(Xtr, ymtr)
    r_reg = evaluate_reg(reg, Xte, yrte)
    r_cls = evaluate_cls(cls, Xte, ycte)
    r_mood = evaluate_cls(mood, Xte, ymte)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    save_conf(ycte, cls.predict(Xte), f"{REPORTS_DIR}/confusion_energy.png", labels=sorted(y_cls.unique()))
    save_conf(ymte, mood.predict(Xte), f"{REPORTS_DIR}/confusion_mood.png", labels=sorted(y_mood.unique()))
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(reg, f"{MODELS_DIR}/energy_regressor.joblib")
    joblib.dump(cls, f"{MODELS_DIR}/energy_classifier.joblib")
    joblib.dump(mood, f"{MODELS_DIR}/mood_classifier.joblib")
    report = {"energy_regression": r_reg, "energy_classification": r_cls, "mood_classification": r_mood,
              "meta": {"train_samples": len(Xtr), "test_samples": len(Xte)}}
    with open(f"{REPORTS_DIR}/training_report.json","w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    train_models()

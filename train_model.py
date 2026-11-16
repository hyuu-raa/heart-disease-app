"""Latih model Heart Disease (CRISP-DM) dan simpan ke folder models/.

Cara pakai:
    python train_model.py                      # unduh data dari UCI (butuh internet + ucimlrepo)
    python train_model.py --csv data/processed.cleveland.data   # pakai file lokal
"""
import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TARGET_RECALL = 0.90
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["gender", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path, header=None, names=COLUMNS, na_values="?")
    else:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        df = pd.concat([heart.data.features, heart.data.targets], axis=1)
    df["target"] = (df["num"] > 0).astype(int)   # 0 = sehat, 1 = ada penyakit
    df = df.drop(columns="num").rename(columns={"sex": "gender"})
    return df.drop_duplicates().reset_index(drop=True)


def main(csv_path=None):
    df = load_data(csv_path)
    X, y = df.drop(columns="target"), df["target"]
    print(f"Data: {df.shape[0]} baris | proporsi sakit: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUM_COLS),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    pipe = Pipeline([("prep", prep),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    inner = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE + 1)
    search = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=inner,
                          scoring="average_precision", n_jobs=-1).fit(X_train, y_train)
    model = search.best_estimator_
    print("Parameter terbaik:", search.best_params_)

    # Threshold dipilih dari prediksi out-of-fold pada data latih (bukan test set)
    oof = cross_val_predict(model, X_train, y_train, cv=outer, method="predict_proba")[:, 1]
    _, rec, thr = precision_recall_curve(y_train, oof)
    idx = np.where(rec[:-1] >= TARGET_RECALL)[0]
    threshold = float(thr[idx[-1]]) if len(idx) else 0.5
    print(f"Threshold terpilih: {threshold:.3f} (target recall >= {TARGET_RECALL})")

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"\nTest set ({len(y_test)} pasien): ROC-AUC={roc_auc_score(y_test, proba):.3f} "
          f"PR-AUC={average_precision_score(y_test, proba):.3f}")
    print(classification_report(y_test, pred, target_names=["sehat", "sakit"]))
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/heart_pipeline.joblib")
    with open("models/metadata.json", "w") as f:
        json.dump({"model": "Logistic Regression", "threshold": threshold,
                   "features": list(X.columns), "target_recall": TARGET_RECALL}, f, indent=2)
    print("\nTersimpan: models/heart_pipeline.joblib dan models/metadata.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="path ke processed.cleveland.data (opsional)")
    main(ap.parse_args().csv)

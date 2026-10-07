"""커스텀 제스처 분류기 학습 - collect.py로 모은 CSV로 MLP 분류기를 학습해 저장

실행:  python custom_gesture/train.py
       python custom_gesture/train.py --data my.csv --out my.joblib
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common

MIN_SAMPLES = 50  # 라벨당 이보다 적으면 경고


def load_data(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = [row for row in reader if row]
    labels = np.array([row[0] for row in rows])
    features = np.array([row[1:] for row in rows], dtype=np.float32)
    return features, labels


def build_model():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=0),
    )


def format_confusion(y_true, y_pred, labels):
    width = max(len(l) for l in labels) + 2
    lines = ["혼동 행렬 (행: 정답, 열: 예측)",
             " " * width + "".join(f"{l[:8]:>9}" for l in labels)]
    for label, row in zip(labels, confusion_matrix(y_true, y_pred, labels=labels)):
        lines.append(f"{label:<{width}}" + "".join(f"{v:>9}" for v in row))
    return "\n".join(lines)


def train_and_save(data_path, out_path, test_size=0.2, log=print):
    """학습 후 분류기를 저장. 문제가 있으면 ValueError"""
    data_path, out_path = Path(data_path), Path(out_path)
    if not data_path.exists():
        raise ValueError(f"데이터가 없습니다: {data_path}\n먼저 데이터를 수집하세요.")

    X, y = load_data(data_path)
    if len(y) == 0:
        raise ValueError("수집된 샘플이 없습니다.")
    if X.shape[1] != common.NUM_FEATURES:
        raise ValueError(f"특징 개수가 {X.shape[1]}개입니다 (기대값 {common.NUM_FEATURES}).")

    counts = Counter(y)
    log(f"샘플 {len(y)}개, 라벨 {len(counts)}개")
    for label, n in sorted(counts.items()):
        warn = f"  <- {MIN_SAMPLES}개 이상 권장" if n < MIN_SAMPLES else ""
        log(f"  {label:<15} {n:>5}{warn}")
    if len(counts) < 2:
        raise ValueError("라벨(제스처)이 2개 이상 있어야 학습할 수 있습니다.")

    labels = sorted(counts)
    n_test = int(len(y) * test_size)
    if n_test >= len(labels) and min(counts.values()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=0)
        pred = build_model().fit(X_train, y_train).predict(X_test)
        log(f"\n검증 정확도: {accuracy_score(y_test, pred):.3f} "
            f"(학습 {len(y_train)}개 / 검증 {len(y_test)}개)")
        log(classification_report(y_test, pred, labels=labels, zero_division=0))
        log(format_confusion(y_test, pred, labels))
    else:
        log("\n샘플이 적어 검증을 건너뜁니다.")

    # 검증이 끝났으면 전체 데이터로 다시 학습해서 저장
    model = build_model().fit(X, y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "num_features": common.NUM_FEATURES}, out_path)
    log(f"\n분류기 저장 -> {out_path}")
    return labels


def main():
    parser = argparse.ArgumentParser(description="커스텀 제스처 분류기 학습")
    parser.add_argument("--data", type=Path, default=common.DATA_PATH, help="collect.py로 모은 CSV")
    parser.add_argument("--out", type=Path, default=common.CLASSIFIER_PATH, help="저장할 분류기 경로")
    parser.add_argument("--test-size", type=float, default=0.2, help="검증용으로 떼어둘 비율")
    args = parser.parse_args()

    try:
        train_and_save(args.data, args.out, args.test_size)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()

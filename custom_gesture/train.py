"""커스텀 제스처 분류기 학습 - collect.py로 모은 CSV로 MLP 분류기를 학습해 저장

실행:  python custom_gesture/train.py
       python custom_gesture/train.py --data my.csv --out my.joblib
       python custom_gesture/train.py --fold-thumb nike   # nike에서 엄지만 접은 손을 none으로 합성
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


def fold_thumb(features, rng):
    """엄지를 편 손(예: 검지 위 + 엄지 옆 = nike)에서 엄지만 손바닥 쪽으로 접은 손 모양을 합성

    엄지 유무만 다른 손 모양(검지 위로 등)을 직접 찍지 않아도 none으로 학습시키기 위해 씀.
    """
    p = features.reshape(21, 3).copy()
    lerp = lambda a, b, k: p[a] + (p[b] - p[a]) * k
    depth = np.array([0, 0, -1.0]) * rng.uniform(0.05, 0.15)  # 접은 엄지는 다른 손가락보다 카메라 쪽
    thumb_mcp = lerp(2, 5, rng.uniform(0.2, 0.45))             # 엄지 뿌리를 검지 뿌리 쪽으로
    thumb_ip = lerp(5, 10, rng.uniform(0.3, 0.6)) + depth      # 검지 뿌리 ~ 중지 사이
    thumb_tip = lerp(10, 11, rng.uniform(0.2, 0.8)) + depth * 1.3  # 엄지 끝은 접힌 중지 위
    p[2], p[3], p[4] = thumb_mcp, thumb_ip, thumb_tip
    # common.landmarks_to_features와 같은 정규화 (손목 원점, 손목에서 가장 먼 점까지 거리 = 1)
    p -= p[0]
    return (p / max(np.linalg.norm(p[:, :2], axis=1).max(), 1e-6)).ravel().astype(np.float32)


def add_folded_thumb(X, y, source_label, seed=0):
    """source_label 샘플마다 엄지를 접은 버전을 none으로 추가"""
    source = X[y == source_label]
    if len(source) == 0:
        return X, y
    rng = np.random.default_rng(seed)
    synthetic = np.array([fold_thumb(f, rng) for f in source])
    return np.vstack([X, synthetic]), np.concatenate([y, [common.NONE_LABEL] * len(synthetic)])


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


def train_and_save(data_path, out_path, test_size=0.2, log=print, fold_thumb_from=None):
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
    if fold_thumb_from and fold_thumb_from not in counts:
        log(f"'{fold_thumb_from}' 샘플이 없어 엄지 접기 합성을 건너뜁니다.")
        fold_thumb_from = None
    if fold_thumb_from:
        log(f"'{fold_thumb_from}'에서 엄지를 접은 손 {counts[fold_thumb_from]}개를 '{common.NONE_LABEL}'으로 합성해 "
            "학습에 추가합니다 (검증 데이터에는 넣지 않음).")

    labels = sorted(counts)
    n_test = int(len(y) * test_size)
    if n_test >= len(labels) and min(counts.values()) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=0)
        if fold_thumb_from:
            X_train, y_train = add_folded_thumb(X_train, y_train, fold_thumb_from)
        pred = build_model().fit(X_train, y_train).predict(X_test)
        log(f"\n검증 정확도: {accuracy_score(y_test, pred):.3f} "
            f"(학습 {len(y_train)}개 / 검증 {len(y_test)}개)")
        log(classification_report(y_test, pred, labels=labels, zero_division=0))
        log(format_confusion(y_test, pred, labels))
    else:
        log("\n샘플이 적어 검증을 건너뜁니다.")

    # 검증이 끝났으면 전체 데이터로 다시 학습해서 저장
    if fold_thumb_from:
        X, y = add_folded_thumb(X, y, fold_thumb_from)
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
    parser.add_argument("--fold-thumb", metavar="LABEL",
                        help="이 제스처에서 엄지만 접은 손을 none으로 합성 (예: nike - 검지 위로를 nike로 착각할 때)")
    args = parser.parse_args()

    try:
        train_and_save(args.data, args.out, args.test_size, fold_thumb_from=args.fold_thumb)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()

"""커스텀 제스처 도구(app.py, collect.py, train.py, recognize.py)가 함께 쓰는 코드"""
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
# Gesture Recognizer는 Hand Landmarker와 똑같은 손 랜드마크를 주면서(직접 비교해 확인)
# 엄지척 같은 기본 제스처 7종도 함께 알려 줌
HAND_MODEL_PATH = HERE.parent / "models" / "gesture_recognizer.task"
NONE_LABEL = "none"  # 아무 제스처도 아닌 손

# 기본 제스처(Thumb_Up, Pointing_Up 등)로 보이는 손은 새 제스처가 아닌 것으로 처리
# - 수집: none이 아닌 라벨에서 그런 프레임을 저장하지 않음 (손 모양을 바꾸는 사이에 섞인 엄지척 등)
# - 인식: 기본 제스처로 보이면 새 제스처로 판단하지 않음
# 새 제스처가 기본 제스처와 같은 모양이라면 False로 바꾸세요 (web/app.js의 SKIP_BUILTIN도 같이)
SKIP_BUILTIN = True
BUILTIN_MIN_SCORE = 0.5
DATA_PATH = HERE / "data" / "gestures.csv"            # collect.py가 쌓는 학습 데이터
CLASSIFIER_PATH = HERE / "gesture_classifier.joblib"  # train.py가 만드는 분류기

NUM_FEATURES = 21 * 3  # 손 랜드마크 21개 x (x, y, z)
MIRROR_LEFT_HAND = True  # 왼손은 좌우 뒤집어 오른손 모양으로 통일 (한 손으로 모아도 양손 인식)

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIPS = (4, 8, 12, 16, 20)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class HandDetector:
    """손 랜드마크 + 기본 제스처 검출. 결과에는 hand_landmarks, handedness, gestures가 들어 있음"""

    def __init__(self, num_hands, video=True):
        if not HAND_MODEL_PATH.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {HAND_MODEL_PATH}")
        self.video = video
        self.recognizer = vision.GestureRecognizer.create_from_options(_detector_options(num_hands, video))

    def detect(self, image, timestamp_ms=0):
        if self.video:
            return self.recognizer.recognize_for_video(image, timestamp_ms)
        return self.recognizer.recognize(image)

    def close(self):
        self.recognizer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _detector_options(num_hands, video):
    return vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        # 서로 관계없는 사진들은 IMAGE 모드로 매번 새로 검출해야 이전 프레임 추적에 끌려가지 않음
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def builtin_gesture(result, index):
    """index번째 손이 기본 제스처로 보이면 그 이름, 아니면 None"""
    gestures = result.gestures[index] if index < len(result.gestures) else []
    top = gestures[0] if gestures else None
    if top and top.category_name != "None" and top.score >= BUILTIN_MIN_SCORE:
        return top.category_name
    return None


def classify_hand(classifier, result, index, frame):
    """(라벨, 확률, 기본 제스처 이름). 기본 제스처로 보이는 손은 분류기에 넣지 않고 none으로 처리"""
    builtin = builtin_gesture(result, index) if SKIP_BUILTIN else None
    if builtin:
        return NONE_LABEL, 1.0, builtin
    label, score = classifier.predict(hand_features(result, index, frame))
    return label, score, None


def skips_builtin(label):
    """이 라벨을 모을 때 기본 제스처 프레임을 거를지 ("none"은 오히려 그런 손 모양이 필요함)"""
    return SKIP_BUILTIN and label.strip().lower() != NONE_LABEL


def to_mp_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def landmarks_to_features(landmarks, handedness, width, height):
    """손 랜드마크 21개 -> 위치/크기에 무관한 63차원 특징 벡터

    - 정규화 좌표(0~1)를 픽셀 단위로 바꿔 화면 비율에 따른 찌그러짐을 없앰
    - 손목(0번)을 원점으로 옮기고, 손목에서 가장 먼 점까지 거리로 나눠 크기를 맞춤
    - 회전은 그대로 둠 (Thumb_Up / Thumb_Down처럼 방향이 다른 제스처를 구분해야 하므로)
    """
    pts = np.array([[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks],
                   dtype=np.float32)
    # handedness는 입력 이미지 그대로의 손 모양 기준이므로, 거울 모드 여부와 상관없이
    # "Left" 모양인 손을 뒤집으면 항상 오른손 모양으로 통일됨
    if MIRROR_LEFT_HAND and handedness == "Left":
        pts[:, 0] *= -1
    pts -= pts[0]
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    return (pts / max(scale, 1e-6)).flatten()


def hand_features(result, index, frame):
    h, w = frame.shape[:2]
    handedness = result.handedness[index][0].category_name
    return landmarks_to_features(result.hand_landmarks[index], handedness, w, h)


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], (255, 255, 255), 2)
    for i, p in enumerate(points):
        color = (0, 0, 255) if i in FINGERTIPS else (0, 255, 0)
        cv2.circle(frame, p, 5, color, -1)
    return points


def is_image_source(source):
    path = Path(source)
    return path.is_dir() or path.suffix.lower() in IMAGE_EXTS


def list_images(source):
    path = Path(source)
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [path]


def open_capture(source):
    if source.isdigit():
        index = int(source)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"웹캠({index})을 열 수 없습니다. 카메라 연결/권한을 확인하세요.")
        return cap, True

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"동영상을 열 수 없습니다: {source}")
    return cap, False


def open_writer(path):
    """학습 데이터 CSV를 이어 쓰기 모드로 열기 (새 파일이면 헤더 추가)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
    return f, writer


def count_labels(path):
    counts = {}
    if Path(path).exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    counts[row[0]] = counts.get(row[0], 0) + 1
    return counts


def delete_label(path, label):
    """CSV에서 해당 라벨 샘플을 모두 지우고, 지운 개수를 돌려줌"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    kept = [rows[0]] + [r for r in rows[1:] if r and r[0] != label]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(kept)
    return len(rows) - len(kept)


class VideoClock:
    """VIDEO 모드에 넣을 단조 증가 타임스탬프(ms)"""

    def __init__(self):
        self.start = time.monotonic()
        self.last = -1

    def now(self):
        ts = int((time.monotonic() - self.start) * 1000)
        self.last = max(ts, self.last + 1)
        return self.last


class GestureClassifier:
    def __init__(self, path=CLASSIFIER_PATH):
        import joblib  # 수집 단계에서는 scikit-learn이 없어도 되도록 여기서 import

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"분류기 파일이 없습니다: {path}\n먼저 train.py로 학습하세요.")
        self.model = joblib.load(path)["model"]
        self.labels = [str(c) for c in self.model.classes_]

    def predict(self, features):
        """(가장 확률 높은 라벨, 확률)"""
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        i = int(proba.argmax())
        return self.labels[i], float(proba[i])

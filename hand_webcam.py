"""MediaPipe Hand Landmarker - 웹캠 실시간 손 랜드마크 검출

실행:  python hand_webcam.py                 # 웹캠 (기본 0번)
       python hand_webcam.py --source 1      # 다른 웹캠 번호
       python hand_webcam.py --source a.mp4  # 동영상 파일
       python hand_webcam.py --source a.jpg  # 이미지 파일
종료:  q 또는 ESC
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
CAMERA_INDEX = 0
NUM_HANDS = 2

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
FINGERTIPS = (4, 8, 12, 16, 20)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_hands(frame, result, mirrored=True):
    h, w = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for conn in HAND_CONNECTIONS:
            cv2.line(frame, points[conn.start], points[conn.end], (255, 255, 255), 2)
        for i, p in enumerate(points):
            color = (0, 0, 255) if i in FINGERTIPS else (0, 255, 0)
            cv2.circle(frame, p, 5, color, -1)

        # 모델은 반전되지 않은 입력 기준으로 handedness를 판단하므로
        # 거울 모드(좌우 반전)로 넣은 프레임에서는 라벨을 뒤집어야 실제 손과 일치함
        name = handedness[0].category_name
        if mirrored:
            name = {"Left": "Right", "Right": "Left"}.get(name, name)
        label = f"{name} {handedness[0].score:.2f}"
        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        cv2.putText(frame, label, (x_min, max(y_min - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def create_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def to_mp_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def run_image(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")

    with create_landmarker() as landmarker:
        result = landmarker.detect_for_video(to_mp_image(frame), 0)

    draw_hands(frame, result, mirrored=False)
    print(f"검출된 손: {len(result.hand_landmarks)}개")
    cv2.imshow("MediaPipe Hand Landmarker", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def run_stream(source):
    cap, is_camera = open_capture(source)
    # 동영상 파일은 원래 속도에 맞춰 재생
    delay = 1 if is_camera else max(int(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30)), 1)

    start = time.monotonic()
    last_ts = -1
    prev = time.monotonic()

    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_camera:
                    print("프레임을 읽지 못했습니다.")
                break

            if is_camera:
                frame = cv2.flip(frame, 1)  # 거울 모드

            # VIDEO 모드는 단조 증가하는 타임스탬프(ms)가 필요
            ts = int((time.monotonic() - start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(to_mp_image(frame), ts)
            draw_hands(frame, result, mirrored=is_camera)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  hands {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow("MediaPipe Hand Landmarker", frame)
            if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hand Landmarker")
    parser.add_argument("--source", default=str(CAMERA_INDEX),
                        help="웹캠 번호(0, 1, ...) 또는 동영상/이미지 파일 경로")
    args = parser.parse_args()

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(args.source)
    else:
        run_stream(args.source)


if __name__ == "__main__":
    main()

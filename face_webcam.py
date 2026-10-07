"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크(478점) + 표정(blendshape) 검출

실행:  python face_webcam.py                 # 웹캠 (기본 0번)
       python face_webcam.py --source 1      # 다른 웹캠 번호
       python face_webcam.py --source a.mp4  # 동영상 파일
       python face_webcam.py --source a.jpg  # 이미지 파일
       python face_webcam.py --mesh          # 얼굴 전체 메쉬(삼각망)까지 표시
종료:  q 또는 ESC
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
CAMERA_INDEX = 0
NUM_FACES = 1  # 1일 때만 프레임 간 스무딩이 적용됨
TOP_BLENDSHAPES = 5  # 화면에 표시할 표정 점수 개수

FC = vision.FaceLandmarksConnections
CONTOUR_GROUPS = [
    (FC.FACE_LANDMARKS_FACE_OVAL, (220, 220, 220)),
    (FC.FACE_LANDMARKS_LEFT_EYEBROW, (0, 200, 0)),
    (FC.FACE_LANDMARKS_RIGHT_EYEBROW, (0, 200, 255)),
    (FC.FACE_LANDMARKS_LEFT_EYE, (0, 255, 0)),
    (FC.FACE_LANDMARKS_RIGHT_EYE, (0, 165, 255)),
    (FC.FACE_LANDMARKS_LEFT_IRIS, (255, 255, 0)),
    (FC.FACE_LANDMARKS_RIGHT_IRIS, (255, 0, 255)),
    (FC.FACE_LANDMARKS_NOSE, (200, 200, 200)),
    (FC.FACE_LANDMARKS_LIPS, (0, 0, 255)),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_connections(frame, points, connections, color, thickness):
    for conn in connections:
        cv2.line(frame, points[conn.start], points[conn.end], color, thickness, cv2.LINE_AA)


def draw_blendshapes(frame, blendshapes, x, y):
    top = sorted(blendshapes, key=lambda c: c.score, reverse=True)[:TOP_BLENDSHAPES]
    # 밝은 배경에서도 읽히도록 반투명 어두운 패널을 깔아줌
    x2, y2 = x + 290, y + len(top) * 22 - 6
    panel = frame[y - 20:y2, x - 6:x2]
    panel[:] = (panel * 0.4).astype(panel.dtype)
    for i, c in enumerate(top):
        yy = y + i * 22
        cv2.rectangle(frame, (x, yy - 12), (x + int(c.score * 100), yy + 2), (0, 200, 255), -1)
        cv2.putText(frame, f"{c.category_name} {c.score:.2f}", (x + 105, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def draw_result(frame, result, show_mesh=False):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.face_landmarks):
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        if show_mesh:
            draw_connections(frame, points, FC.FACE_LANDMARKS_TESSELATION, (90, 90, 90), 1)
        for connections, color in CONTOUR_GROUPS:
            draw_connections(frame, points, connections, color, 2)

        if result.face_blendshapes:
            draw_blendshapes(frame, result.face_blendshapes[i], 10, 60 + i * 130)


def create_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def to_mp_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def run_image(path, show_mesh):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")

    with create_landmarker() as landmarker:
        result = landmarker.detect_for_video(to_mp_image(frame), 0)

    draw_result(frame, result, show_mesh)
    print(f"검출된 얼굴: {len(result.face_landmarks)}개")
    cv2.imshow("MediaPipe Face Landmarker", frame)
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


def run_stream(source, show_mesh):
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
            draw_result(frame, result, show_mesh)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  faces {len(result.face_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow("MediaPipe Face Landmarker", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):  # m 키로 메쉬 표시 토글
                show_mesh = not show_mesh

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Face Landmarker")
    parser.add_argument("--source", default=str(CAMERA_INDEX),
                        help="웹캠 번호(0, 1, ...) 또는 동영상/이미지 파일 경로")
    parser.add_argument("--mesh", action="store_true", help="얼굴 전체 메쉬(삼각망) 표시")
    args = parser.parse_args()

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(args.source, args.mesh)
    else:
        run_stream(args.source, args.mesh)


if __name__ == "__main__":
    main()

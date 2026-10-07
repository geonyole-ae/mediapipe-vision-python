"""커스텀 제스처 인식 - train.py로 학습한 분류기로 손 제스처 인식

실행:  python custom_gesture/recognize.py                  # 웹캠 (기본 0번)
       python custom_gesture/recognize.py --source 1       # 다른 웹캠 번호
       python custom_gesture/recognize.py --source a.mp4   # 동영상 파일
       python custom_gesture/recognize.py --source a.jpg   # 이미지 파일 또는 폴더
       python custom_gesture/recognize.py --model my.joblib
종료:  q 또는 ESC (이미지는 아무 키)
"""
import argparse
from pathlib import Path

import cv2

import common

NUM_HANDS = 2
MIN_SCORE = 0.7  # 이보다 확률이 낮으면 "-"로 표시
WINDOW = "Custom Gesture Recognizer"


def recognize(frame, result, classifier):
    """손마다 (라벨, 확률). 확률이 낮으면 라벨은 None"""
    predictions = []
    for i in range(len(result.hand_landmarks)):
        label, score, builtin = common.classify_hand(classifier, result, i, frame)
        predictions.append((label if score >= MIN_SCORE else None, score, builtin))
    return predictions


def prediction_text(label, score, builtin):
    if builtin:  # 기본 제스처로 보이는 손은 새 제스처가 아님
        return f"({builtin})"
    return f"{label} {score:.2f}" if label else "-"


def draw_result(frame, result, predictions, mirrored):
    for landmarks, handedness, prediction in zip(
            result.hand_landmarks, result.handedness, predictions):
        points = common.draw_hand(frame, landmarks)

        # 거울 모드(좌우 반전)로 넣은 프레임에서는 라벨을 뒤집어야 실제 손과 일치함
        hand = handedness[0].category_name
        if mirrored:
            hand = {"Left": "Right", "Right": "Left"}.get(hand, hand)
        gesture = prediction_text(*prediction)

        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        cv2.putText(frame, hand, (x_min, max(y_min - 35, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, gesture, (x_min, max(y_min - 10, 45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)


def run_images(source, classifier):
    with common.HandDetector(NUM_HANDS, video=False) as detector:
        for path in common.list_images(source):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"이미지를 읽을 수 없습니다: {path}")
                continue

            result = detector.detect(common.to_mp_image(frame))
            predictions = recognize(frame, result, classifier)
            draw_result(frame, result, predictions, mirrored=False)
            print(f"{path.name}: " + (", ".join(prediction_text(*p) for p in predictions)
                                      or "손 없음"))
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


def run_stream(source, classifier):
    cap, is_camera = common.open_capture(source)
    # 동영상 파일은 원래 속도에 맞춰 재생
    delay = 1 if is_camera else max(int(1000 / (cap.get(cv2.CAP_PROP_FPS) or 30)), 1)
    clock = common.VideoClock()

    with common.HandDetector(NUM_HANDS) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_camera:
                    print("프레임을 읽지 못했습니다.")
                break

            if is_camera:
                frame = cv2.flip(frame, 1)  # 거울 모드

            result = detector.detect(common.to_mp_image(frame), clock.now())
            draw_result(frame, result, recognize(frame, result, classifier), mirrored=is_camera)
            cv2.putText(frame, f"hands {len(result.hand_landmarks)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="커스텀 제스처 인식")
    parser.add_argument("--source", default="0",
                        help="웹캠 번호(0, 1, ...) 또는 동영상/이미지 파일, 이미지 폴더 경로")
    parser.add_argument("--model", type=Path, default=common.CLASSIFIER_PATH,
                        help="train.py로 만든 분류기 경로")
    args = parser.parse_args()

    classifier = common.GestureClassifier(args.model)
    print("인식할 제스처:", ", ".join(classifier.labels))

    if common.is_image_source(args.source):
        run_images(args.source, classifier)
    else:
        run_stream(args.source, classifier)


if __name__ == "__main__":
    main()

"""커스텀 제스처 학습 데이터 수집 - 손 랜드마크를 특징 벡터로 바꿔 CSV에 한 줄씩 추가

실행:  python custom_gesture/collect.py --label heart                       # 웹캠, SPACE로 녹화 시작/정지
       python custom_gesture/collect.py --label heart --source 1            # 다른 웹캠 번호
       python custom_gesture/collect.py --label heart --source heart.mp4    # 동영상 (손이 잡힌 프레임 전부)
       python custom_gesture/collect.py --label heart --source photos/heart # 이미지 파일 또는 폴더
       python custom_gesture/collect.py --label heart --max 300             # 300개 모으면 자동 종료
종료:  q 또는 ESC
"""
import argparse
from pathlib import Path

import cv2

import common

WINDOW = "Collect custom gesture"


def draw_status(frame, label, saved, recording, has_hand, is_camera):
    if recording:
        cv2.circle(frame, (20, 25), 8, (0, 0, 255), -1)
        status = f"REC  {label}  saved {saved}"
    else:
        status = f"{label}  saved {saved}  [SPACE] start" if is_camera else f"{label}  saved {saved}"
    cv2.putText(frame, status, (35, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if not has_hand:
        cv2.putText(frame, "no hand", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


def collect_images(args, writer):
    saved = 0
    with common.create_landmarker(num_hands=1, video=False) as landmarker:
        for path in common.list_images(args.source):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  읽기 실패: {path}")
                continue

            result = landmarker.detect(common.to_mp_image(frame))
            if not result.hand_landmarks:
                print(f"  손 없음: {path.name}")
                continue

            writer.writerow([args.label, *common.hand_features(result, 0, frame)])
            saved += 1

            if not args.no_show:
                common.draw_hand(frame, result.hand_landmarks[0])
                draw_status(frame, args.label, saved, True, True, False)
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max and saved >= args.max:
                break
    cv2.destroyAllWindows()
    return saved


def collect_stream(args, writer):
    cap, is_camera = common.open_capture(args.source)
    # 파일은 처음부터 저장, 웹캠은 SPACE를 눌러 손 모양을 잡은 뒤 녹화 시작
    recording = not is_camera or args.no_show
    clock = common.VideoClock()
    saved = frame_idx = 0

    with common.create_landmarker(num_hands=1) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_camera:
                    print("프레임을 읽지 못했습니다.")
                break

            if is_camera:
                frame = cv2.flip(frame, 1)  # 거울 모드

            result = landmarker.detect_for_video(common.to_mp_image(frame), clock.now())
            has_hand = bool(result.hand_landmarks)
            frame_idx += 1

            if recording and has_hand and frame_idx % args.every == 0:
                writer.writerow([args.label, *common.hand_features(result, 0, frame)])
                saved += 1
                if args.max and saved >= args.max:
                    break

            if args.no_show:
                continue
            if has_hand:
                common.draw_hand(frame, result.hand_landmarks[0])
            draw_status(frame, args.label, saved, recording, has_hand, is_camera)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and is_camera:
                recording = not recording

    cap.release()
    cv2.destroyAllWindows()
    return saved


def main():
    parser = argparse.ArgumentParser(description="커스텀 제스처 학습 데이터 수집")
    parser.add_argument("--label", required=True, help="제스처 이름 (예: heart, ok, none)")
    parser.add_argument("--source", default="0",
                        help="웹캠 번호(0, 1, ...) 또는 동영상/이미지 파일, 이미지 폴더 경로")
    parser.add_argument("--data", type=Path, default=common.DATA_PATH, help="저장할 CSV 경로")
    parser.add_argument("--max", type=int, default=0, help="이만큼 모으면 종료 (0: 제한 없음)")
    parser.add_argument("--every", type=int, default=1,
                        help="N프레임마다 1개 저장 (웹캠/동영상, 비슷한 샘플이 너무 많을 때)")
    parser.add_argument("--no-show", action="store_true", help="화면 표시 없이 수집")
    args = parser.parse_args()
    args.every = max(args.every, 1)

    f, writer = common.open_writer(args.data)
    with f:
        if common.is_image_source(args.source):
            saved = collect_images(args, writer)
        else:
            saved = collect_stream(args, writer)

    print(f"'{args.label}' 샘플 {saved}개 저장 -> {args.data}")
    print("현재 데이터:", ", ".join(f"{k} {v}개" for k, v in common.count_labels(args.data).items()))


if __name__ == "__main__":
    main()

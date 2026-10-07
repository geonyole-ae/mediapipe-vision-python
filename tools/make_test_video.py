"""예제 사진으로 테스트용 동영상을 만드는 스크립트 (웹캠이 없을 때 테스트용)

장면: 한 손 엄지척 -> 양손 엄지척(웹 데모에서 폭죽) -> 브이 -> 검지 위로 -> 엄지 아래
- 사진을 천천히 확대·회전·흔들어 동영상처럼 만들고, 아래쪽에 장면 설명 자막을 넣음
- 장면 사이에 짧은 검은 화면을 넣어, 앞 장면의 손 추적이 다음 장면으로 이어지지 않게 함

실행:  python tools/make_test_video.py
       -> samples/gestures_test.mp4 (Python/OpenCV용), samples/gestures_test.webm (브라우저용)
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = SAMPLES / "gestures_test"
SIZE = 480          # 사진이 들어가는 정사각형 영역
CAPTION_H = 64      # 아래쪽 자막 영역
FPS = 30
GAP_FRAMES = 8      # 장면 사이 검은 화면
FONT_PATHS = ["C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
              "/System/Library/Fonts/AppleSDGothicNeo.ttc",
              "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]


def load(name):
    img = cv2.imread(str(SAMPLES / name))
    if img is None:
        raise FileNotFoundError(SAMPLES / name)
    return img


def two_thumbs():
    """엄지척 사진(손 쪽만 자름)과 좌우 반전본을 나란히 붙여 양손 엄지척 사진을 만듦"""
    img = load("thumbs_up.jpg")
    w = img.shape[1]
    crop = img[:, int(w * 0.3):int(w * 0.85)]
    return np.hstack([crop, cv2.flip(crop, 1)])


SCENES = [  # (사진, 길이(초), 자막)
    (load("thumbs_up.jpg"), 3.0, "한 손 엄지척 → (Thumb_Up), 효과 없음"),
    (two_thumbs(), 5.0, "양손 엄지척 → 폭죽"),
    (load("victory.jpg"), 3.0, "브이 → (Victory)"),
    (load("pointing_up.jpg"), 3.0, "검지 위로 → (Pointing_Up)"),
    (load("thumbs_down.jpg"), 3.0, "엄지 아래 → (Thumb_Down)"),
]


def caption_bar(text):
    bar = Image.new("RGB", (SIZE, CAPTION_H), (20, 20, 20))
    font_path = next((p for p in FONT_PATHS if Path(p).exists()), None)
    if font_path:  # 한글 글꼴이 없으면 자막 없이 만듦
        font = ImageFont.truetype(font_path, 24)
        draw = ImageDraw.Draw(bar)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((SIZE - (box[2] - box[0])) / 2, (CAPTION_H - (box[3] - box[1])) / 2 - box[1]),
                  text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(bar), cv2.COLOR_RGB2BGR)


def scene_frames(img, secs):
    s = SIZE / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    n = int(FPS * secs)
    for i in range(n):
        t = i / n
        zoom = 1.0 + 0.12 * t                       # 천천히 확대
        angle = 4 * np.sin(2 * np.pi * t)           # 살짝 기울임
        dx = 20 * np.sin(2 * np.pi * t)             # 좌우로 흔들림
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
        M[0, 2] += dx + (SIZE - w) / 2
        M[1, 2] += (SIZE - h) / 2
        yield cv2.warpAffine(img, M, (SIZE, SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))


def main():
    frame_size = (SIZE, SIZE + CAPTION_H)
    writers = [
        cv2.VideoWriter(str(OUT.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), FPS, frame_size),
        # 브라우저는 OpenCV 기본 mp4 코덱(mp4v)을 재생하지 못해서 WebM(VP8)도 만듦
        cv2.VideoWriter(str(OUT.with_suffix(".webm")), cv2.VideoWriter_fourcc(*"VP80"), FPS, frame_size),
    ]
    black = np.zeros((SIZE + CAPTION_H, SIZE, 3), np.uint8)
    total = 0
    for k, (img, secs, text) in enumerate(SCENES):
        bar = caption_bar(text)
        for frame in scene_frames(img, secs):
            out = np.vstack([frame, bar])
            for w in writers:
                w.write(out)
            total += 1
        if k < len(SCENES) - 1:
            for _ in range(GAP_FRAMES):
                for w in writers:
                    w.write(black)
                total += 1
    for w in writers:
        w.release()
    print(f"{total}프레임 ({total / FPS:.1f}초) -> {OUT.with_suffix('.mp4')}, {OUT.with_suffix('.webm')}")


if __name__ == "__main__":
    main()

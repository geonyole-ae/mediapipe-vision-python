"""제스처 사진 4장을 이어 붙여 테스트용 동영상을 만드는 스크립트 (웹캠이 없을 때 테스트용)

실행:  python tools/make_test_video.py samples samples/gestures_test.mp4
"""
import sys
import cv2
import numpy as np

samples, out = sys.argv[1], sys.argv[2]
names = ["thumbs_up", "victory", "pointing_up", "thumbs_down"]
size, fps, secs = 640, 30, 2.5
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
for n in names:
    img = cv2.imread(f"{samples}/{n}.jpg")
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    frames = int(fps * secs)
    for i in range(frames):
        t = i / frames
        zoom = 1.0 + 0.15 * t                     # 천천히 확대
        dx = int(30 * np.sin(2 * np.pi * t))      # 좌우로 살짝 흔들기
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 5 * np.sin(2 * np.pi * t), zoom)
        M[0, 2] += dx + (size - img.shape[1]) / 2
        M[1, 2] += (size - img.shape[0]) / 2
        writer.write(cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE))
writer.release()
print("saved", out)

# MediaPipe Vision (Python)

[MediaPipe Tasks](https://developers.google.com/edge/mediapipe/solutions/guide) API로 만든 손 랜드마크, 손 제스처 인식, 얼굴 랜드마크 예제입니다.
웹캠, 동영상, 이미지 파일을 입력으로 쓸 수 있습니다.

| 스크립트 | 기능 | 모델 |
|---|---|---|
| `hand_webcam.py` | 손 랜드마크 21점, 왼손/오른손 구분 | `models/hand_landmarker.task` |
| `gesture_webcam.py` | 손 제스처 인식 (7종) + 손 랜드마크 | `models/gesture_recognizer.task` |
| `face_webcam.py` | 얼굴 랜드마크 478점 + 표정 점수(blendshape) 52종 | `models/face_landmarker.task` |

## 설치

```bash
pip install -r requirements.txt
```

테스트 환경: Windows 11, Python 3.14, mediapipe 1.1.0 (OpenCV는 mediapipe와 함께 설치됨)

## 실행

세 스크립트 모두 사용법이 같습니다.

```bash
python gesture_webcam.py                                  # 웹캠 (기본 0번)
python gesture_webcam.py --source 1                       # 다른 웹캠 번호
python gesture_webcam.py --source samples/gestures_test.mp4   # 동영상
python gesture_webcam.py --source samples/victory.jpg     # 이미지
```

- 종료: `q` 또는 `ESC` (이미지는 아무 키)
- 웹캠 입력은 거울 모드(좌우 반전)로 표시됩니다.
- `face_webcam.py`는 `--mesh` 옵션으로 얼굴 전체 메쉬를 표시할 수 있고, 실행 중에는 `m` 키로 켜고 끕니다.

## 결과 예시

| 손 랜드마크 | 제스처 인식 | 얼굴 랜드마크 |
|---|---|---|
| ![hand](samples/woman_hands_result.jpg) | ![gesture](samples/victory_gesture.jpg) | ![face](samples/business-person_face.jpg) |

## 인식하는 제스처

`Closed_Fist`(주먹), `Open_Palm`(손바닥), `Pointing_Up`(검지 위로), `Thumb_Up`, `Thumb_Down`, `Victory`(브이), `ILoveYou`(🤟)

해당 없거나 점수가 `MIN_GESTURE_SCORE`(기본 0.5)보다 낮으면 `-`로 표시합니다.

## 참고 사항

- **왼손/오른손:** 모델은 좌우 반전하지 않은 이미지를 기준으로 왼손/오른손을 판단합니다. 그래서 거울 모드로 표시하는 웹캠 입력에서만 라벨을 뒤집어 실제 손과 맞춥니다.
- **실행 모드:** 모든 입력을 `VIDEO` 실행 모드로 처리하므로 프레임마다 증가하는 타임스탬프를 넣습니다.
- **웹캠이 없을 때:** `samples/gestures_test.mp4`는 제스처 사진 4장으로 만든 테스트 영상입니다 (`tools/make_test_video.py`로 다시 만들 수 있음).

## 폴더 구조

```
├── hand_webcam.py
├── gesture_webcam.py
├── face_webcam.py
├── models/            # MediaPipe 모델 번들 (.task)
├── samples/           # 예제 이미지·영상과 결과 이미지
└── tools/
    └── make_test_video.py
```

## 모델 출처

모델은 Google MediaPipe 공식 문서에서 받았습니다 (Apache License 2.0).

- [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker): `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- [Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer): `https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task`
- [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker): `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`

예제 이미지는 MediaPipe 공식 예제에서 쓰는 이미지입니다 (`storage.googleapis.com/mediapipe-tasks`, `mediapipe-assets`).

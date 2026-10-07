# ffmpeg.wasm (@ffmpeg/ffmpeg 0.12.15)

브라우저가 바로 재생하지 못하는 동영상(휴대폰 HEVC 등)을 변환하는 데 씁니다.

- 출처: https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/esm/ (MIT, `LICENSE` 참고)
- 브라우저는 다른 사이트의 스크립트로 Worker를 만들 수 없어서, 이 작은 래퍼 파일만 사이트에 함께 둡니다.
- 실제 변환 엔진(@ffmpeg/core, 약 30MB)은 변환이 필요할 때만 jsDelivr에서 받습니다.

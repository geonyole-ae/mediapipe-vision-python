"""제스처 이펙트 웹 데모 서버

학습한 분류기(custom_gesture/gesture_classifier.joblib)를 브라우저가 읽을 수 있는 JSON으로
바꿔 주고, 저장소 폴더를 로컬 웹 서버로 띄운 뒤 브라우저를 엽니다.
분류기는 요청할 때마다 새로 읽으므로, 다시 학습한 뒤에는 페이지만 새로고침하면 됩니다.
분류기 파일이 없으면 web/model/gesture_classifier.json(GitHub Pages용으로 내보낸 파일)을 씁니다.

실행:  python web/serve.py                      # http://localhost:8000/web/ 이 열림
       python web/serve.py --port 8080
       python web/serve.py --model my.joblib
       python web/serve.py --export             # GitHub Pages용 JSON 파일만 만들고 종료
종료:  Ctrl+C
"""
import argparse
import functools
import json
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "custom_gesture"))
import common  # noqa: E402

CLASSIFIER_URL = "/web/model/gesture_classifier.json"
STATIC_CLASSIFIER = ROOT / "web" / "model" / "gesture_classifier.json"


def export_classifier(path, note=None):
    """scikit-learn 파이프라인(StandardScaler + MLPClassifier) -> 브라우저에서 계산할 숫자들"""
    import joblib

    pipeline = joblib.load(path)["model"]
    scaler, mlp = pipeline[0], pipeline[-1]
    return {
        "labels": [str(c) for c in mlp.classes_],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "activation": mlp.activation,
        "outActivation": mlp.out_activation_,
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in zip(mlp.coefs_, mlp.intercepts_)],
        "mirrorLeftHand": common.MIRROR_LEFT_HAND,
        "note": note,  # 페이지에 보여 줄 설명 (예: 데모용 임시 모델)
    }


class Handler(SimpleHTTPRequestHandler):
    model_path = common.CLASSIFIER_PATH
    # Windows는 레지스트리 설정에 따라 .js를 text/plain으로 보내 모듈 로딩이 막히는 경우가 있음
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".svg": "image/svg+xml",
        ".mp4": "video/mp4",
        ".task": "application/octet-stream",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # 재학습한 분류기가 바로 반영되도록
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == CLASSIFIER_URL:
            self.send_classifier()
        else:
            super().do_GET()

    def send_classifier(self):
        if self.model_path.exists():
            status, data = 200, export_classifier(self.model_path)
        elif STATIC_CLASSIFIER.exists():
            status, data = 200, json.loads(STATIC_CLASSIFIER.read_text(encoding="utf-8"))
        else:
            status, data = 404, {"error": f"학습한 분류기가 없습니다: {self.model_path}\n"
                                          "custom_gesture/app.py에서 먼저 학습하세요."}
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code="-", size="-"):
        if str(code).isdigit() and int(code) >= 400:  # 오류만 출력
            super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description="제스처 이펙트 웹 데모 서버")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=common.CLASSIFIER_PATH, help="분류기 경로")
    parser.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않음")
    parser.add_argument("--export", action="store_true",
                        help=f"GitHub Pages용으로 분류기를 {STATIC_CLASSIFIER.relative_to(ROOT)}에 저장하고 종료")
    parser.add_argument("--note", help="--export 때 페이지에 보여 줄 설명")
    args = parser.parse_args()

    if args.export:
        if not args.model.exists():
            raise SystemExit(f"분류기가 없습니다: {args.model}")
        STATIC_CLASSIFIER.parent.mkdir(parents=True, exist_ok=True)
        data = export_classifier(args.model, args.note)
        STATIC_CLASSIFIER.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"저장 -> {STATIC_CLASSIFIER} (제스처: {', '.join(data['labels'])})")
        return

    Handler.model_path = args.model.resolve()
    handler = functools.partial(Handler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/web/"
    print(f"웹 데모: {url}  (종료: Ctrl+C)")
    print(f"분류기: {Handler.model_path}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

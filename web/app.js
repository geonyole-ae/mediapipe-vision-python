// 제스처 이펙트 - 브라우저에서 손 랜드마크를 찾고, 학습한 분류기로 제스처를 인식해 효과를 띄움
import { FilesetResolver, GestureRecognizer } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const TASKS_VISION_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
// Gesture Recognizer: Hand Landmarker와 같은 손 랜드마크 + 엄지척 같은 기본 제스처 7종
const HAND_MODEL = "../models/gesture_recognizer.task";
const CLASSIFIER_URL = "model/gesture_classifier.json";  // serve.py가 joblib을 JSON으로 바꿔 줌
const SAMPLE_VIDEO = "../samples/gestures_test.webm";  // 브라우저는 OpenCV 기본 mp4(mp4v)를 못 읽어서 WebM 사용

// 제스처 이름(수집할 때 붙인 라벨, 대소문자 무시) -> 화면에 띄울 효과
// 새 효과를 추가하려면 여기에 한 줄 추가: 이미지는 { image: "경로" }, 이모지는 { emoji: "🎉" }
const EFFECTS = {
  nike: { image: "assets/nike.svg", name: "나이키 로고" },
  ok: { emoji: "👌", name: "OK 이모지" },
};

// 기본 제스처(Thumb_Up, Pointing_Up 등)로 보이는 손은 새 제스처가 아닌 것으로 처리
// (custom_gesture/common.py의 SKIP_BUILTIN과 같게 맞출 것)
const SKIP_BUILTIN = true;
const BUILTIN_MIN_SCORE = 0.5;

const SHOW_AFTER_MS = 150;  // 이만큼 연속으로 인식돼야 효과 표시 (깜빡임 방지)
const HOLD_MS = 400;        // 인식이 잠깐 끊겨도 이만큼은 효과 유지
const POP_MS = 350;         // 효과가 튀어나오는 애니메이션 시간
const HAND_KO = { Left: "왼손", Right: "오른손" };
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);
const EMOJI_FONT = '"Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif';

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");

let detector = null;
let classifier = null;
let source = null;        // { kind: "camera" | "video" | "image", el, mirrored, stream, lastTime, detected }
let result = null;
let predictions = [];     // 손마다 { hand, label, score, box }
let lastTimestamp = 0;
let runningMode = "VIDEO";
let switchingMode = false;
let minScore = 0.7;
let activeKey = "";
const effectState = {};
const effectImages = {};

// ---------------------------------------------------------------- 분류기
// train.py가 만든 StandardScaler + MLPClassifier를 그대로 계산
class GestureClassifier {
  constructor(m) {
    this.m = m;
    this.labels = m.labels;
    const acts = {
      relu: (v) => Math.max(v, 0),
      tanh: Math.tanh,
      logistic: (v) => 1 / (1 + Math.exp(-v)),
      identity: (v) => v,
    };
    this.act = acts[m.activation];
    if (!this.act) throw new Error(`지원하지 않는 활성화 함수: ${m.activation}`);
  }

  predict(features) {
    const { mean, scale, layers, outActivation } = this.m;
    let x = features.map((v, i) => (v - mean[i]) / scale[i]);
    layers.forEach(({ W, b }, li) => {
      const out = b.slice();
      for (let i = 0; i < x.length; i++) {
        const row = W[i];
        for (let j = 0; j < out.length; j++) out[j] += x[i] * row[j];
      }
      x = li < layers.length - 1 ? out.map(this.act) : out;
    });

    let probs;
    if (outActivation === "softmax") {
      const max = Math.max(...x);
      const exps = x.map((v) => Math.exp(v - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      probs = exps.map((v) => v / sum);
    } else {  // 라벨이 2개면 출력 1개(logistic) = 두 번째 라벨일 확률
      const p = 1 / (1 + Math.exp(-x[0]));
      probs = [1 - p, p];
    }
    const best = probs.indexOf(Math.max(...probs));
    return { label: this.labels[best], score: probs[best] };
  }
}

// custom_gesture/common.py의 landmarks_to_features와 같은 계산
function toFeatures(landmarks, handedness, width, height) {
  const pts = landmarks.map((l) => [l.x * width, l.y * height, l.z * width]);
  if (classifier.m.mirrorLeftHand && handedness === "Left") {
    for (const p of pts) p[0] = -p[0];
  }
  const [x0, y0, z0] = pts[0];
  let scale = 0;
  for (const p of pts) {
    p[0] -= x0; p[1] -= y0; p[2] -= z0;
    scale = Math.max(scale, Math.hypot(p[0], p[1]));
  }
  scale = Math.max(scale, 1e-6);
  return pts.flatMap((p) => p.map((v) => v / scale));
}

// ---------------------------------------------------------------- 초기화
async function init() {
  renderLegend();
  for (const [key, fx] of Object.entries(EFFECTS)) {
    if (fx.image) {
      const img = new Image();
      img.src = fx.image;
      effectImages[key] = img;
    }
  }

  setStatus("손 인식 모델을 불러오는 중...");
  const [fileset] = await Promise.all([
    FilesetResolver.forVisionTasks(TASKS_VISION_WASM),
    loadClassifier(),
  ]);
  detector = await GestureRecognizer.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  if (classifier) setStatus("준비 완료 - 웹캠을 켜거나 동영상/사진을 여세요.");

  // ?src=경로 로 바로 열기 (예: ?src=../samples/victory.jpg)
  const src = new URLSearchParams(location.search).get("src");
  if (src) openUrl(src, /\.(jpe?g|png|bmp|webp)$/i.test(src));
  requestAnimationFrame(frame);
}

async function loadClassifier() {
  try {
    const res = await fetch(CLASSIFIER_URL, { cache: "no-store" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    classifier = new GestureClassifier(data);
  } catch (e) {
    classifier = null;
    setStatus(`분류기를 불러오지 못했습니다.\n${e.message}\n(python web/serve.py로 실행했는지 확인하세요)`, true);
  }
  renderLegend();
}

function renderLegend() {
  const labels = new Set((classifier?.labels || []).map((l) => l.toLowerCase()));
  $("legend").innerHTML = "";
  for (const [key, fx] of Object.entries(EFFECTS)) {
    const li = document.createElement("li");
    const icon = document.createElement("span");
    icon.className = "icon";
    icon.append(effectNode(fx));
    const text = document.createElement("span");
    text.textContent = `${key} → ${fx.name}`;
    li.append(icon, text);
    if (classifier && !labels.has(key)) {
      const miss = document.createElement("span");
      miss.className = "missing";
      miss.textContent = "학습 안 됨";
      li.append(miss);
    }
    $("legend").append(li);
  }
  $("modelLabels").textContent = classifier
    ? `학습된 제스처: ${classifier.labels.join(", ")}` + (classifier.m.note ? `\n\n${classifier.m.note}` : "")
    : "";
}

function effectNode(fx) {
  if (fx.image) {
    const img = document.createElement("img");
    img.src = fx.image;
    img.alt = fx.name;
    return img;
  }
  return document.createTextNode(fx.emoji);
}

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

// ---------------------------------------------------------------- 입력
async function useCamera() {
  stopSource();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (e) {
    setStatus(`웹캠을 열 수 없습니다: ${e.message}\n동영상/사진 열기나 예제 영상을 써 보세요.`, true);
    return;
  }
  const video = makeVideo();
  video.srcObject = stream;
  await video.play();
  setSource({ kind: "camera", el: video, mirrored: true, stream });
  setStatus("웹캠 인식 중");
}

function openUrl(url, isImage, name = url) {
  stopSource();
  if (isImage) {
    const img = new Image();
    img.onload = () => setSource({ kind: "image", el: img, mirrored: false });
    img.onerror = () => {
      stopSource();
      setStatus(`사진을 열 수 없습니다: ${name}`, true);
    };
    img.src = url;
  } else {
    const video = makeVideo();
    video.loop = true;
    video.onerror = () => {
      stopSource();
      setStatus(`동영상을 열 수 없습니다: ${name}
브라우저가 지원하는 형식(H.264 mp4, WebM)인지 확인하세요.\n휴대폰으로 찍은 HEVC(H.265) 영상은 브라우저에 따라 열리지 않습니다.`, true);
    };
    video.src = url;
    video.play().catch(() => {});
    setSource({ kind: "video", el: video, mirrored: false });
  }
  setStatus(`${name} 인식 중`);
}

function openFile(file) {
  if (!file) return;
  openUrl(URL.createObjectURL(file), file.type.startsWith("image/"), file.name);
}

function makeVideo() {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  return video;
}

function setSource(s) {
  source = { lastTime: -1, detected: false, ...s };
  result = null;
  predictions = [];
  $("placeholder").classList.add("hidden");
}

function stopSource() {
  if (!source) return;
  source.stream?.getTracks().forEach((t) => t.stop());
  if (source.el instanceof HTMLVideoElement) {
    source.el.pause();
    if (source.el.src.startsWith("blob:")) URL.revokeObjectURL(source.el.src);
  }
  source = null;
  result = null;
  predictions = [];
  updatePanel([]);
  $("placeholder").classList.remove("hidden");
}

// ---------------------------------------------------------------- 매 프레임
function frame(now) {
  requestAnimationFrame(frame);
  if (!source || !detector) return;
  const el = source.el;
  const w = el.videoWidth || el.naturalWidth;
  const h = el.videoHeight || el.naturalHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  // 사진은 IMAGE 모드로 매번 새로 검출해야 이전 입력의 손 추적에 끌려가지 않음
  const wantMode = source.kind === "image" ? "IMAGE" : "VIDEO";
  if (wantMode !== runningMode) {
    if (!switchingMode) {
      switchingMode = true;
      detector.setOptions({ runningMode: wantMode }).then(() => {
        runningMode = wantMode;
        switchingMode = false;
      });
    }
    return;
  }

  // 새 프레임이 들어왔을 때만 인식 (사진은 한 번만)
  const isNew = source.kind === "image"
    ? !source.detected
    : el.readyState >= 2 && el.currentTime !== source.lastTime;
  if (isNew) {
    source.lastTime = el.currentTime;
    source.detected = true;
    if (runningMode === "IMAGE") {
      result = detector.recognize(el);
    } else {
      lastTimestamp = Math.max(performance.now(), lastTimestamp + 1);  // VIDEO 모드는 증가하는 시각 필요
      result = detector.recognizeForVideo(el, lastTimestamp);
    }
    predictions = recognize(result, w, h);
  }

  draw(now, w, h);
}

function recognize(res, w, h) {
  return res.landmarks.map((landmarks, i) => {
    const hand = res.handedness[i][0].categoryName;
    const builtin = builtinGesture(res, i);
    let pred = { label: null, score: 0 };
    if (builtin) pred = { label: "none", score: 1 };  // 기본 제스처로 보이는 손은 분류기에 넣지 않음
    else if (classifier) pred = classifier.predict(toFeatures(landmarks, hand, w, h));
    // 화면에 그릴 위치 (웹캠은 좌우 반전해서 보여 주므로 x도 뒤집음)
    const xs = landmarks.map((l) => (source.mirrored ? 1 - l.x : l.x));
    const ys = landmarks.map((l) => l.y);
    const box = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
    return { hand, ...pred, builtin, box };
  });
}

function builtinGesture(res, i) {
  const top = res.gestures?.[i]?.[0];
  if (SKIP_BUILTIN && top && top.categoryName !== "None" && top.score >= BUILTIN_MIN_SCORE) {
    return top.categoryName;
  }
  return null;
}

function predictionText(p) {
  if (p.builtin) return `(${p.builtin})`;
  return p.label && p.score >= minScore ? `${p.label} ${p.score.toFixed(2)}` : "-";
}

function draw(now, w, h) {
  // 작은 사진을 크게 늘려 보여 줄 때도 글자와 점이 너무 작아지지 않도록 화면 크기 기준 최솟값
  const cssScale = w / (canvas.clientWidth || w);
  const unit = Math.max(Math.max(w, h) / 200, 3 * cssScale);

  ctx.save();
  if (source.mirrored) {  // 웹캠은 거울처럼 보여 줌 (인식은 원본으로)
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(source.el, 0, 0, w, h);
  if (result) {
    for (const landmarks of result.landmarks) drawHand(landmarks, w, h, unit);
  }
  ctx.restore();

  const shown = [];
  for (const p of predictions) {
    drawText(predictionText(p), p.box.x0 * w, p.box.y0 * h - unit * 3, unit * 5, "#ff4dff");
  }
  for (const key of Object.keys(EFFECTS)) {
    if (updateEffect(key, now)) {
      drawEffect(key, now, w, h);
      shown.push(key);
    }
  }
  updatePanel(shown);
}

function drawHand(landmarks, w, h, unit) {
  ctx.lineWidth = Math.max(unit * 0.8, 2);
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }
  landmarks.forEach((l, i) => {
    ctx.fillStyle = FINGERTIPS.has(i) ? "#ff3b3b" : "#2fd36b";
    ctx.beginPath();
    ctx.arc(l.x * w, l.y * h, Math.max(unit * 1.2, 3), 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawText(text, x, y, size, color) {
  ctx.font = `bold ${size}px "Malgun Gothic", sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.lineWidth = size / 6;
  ctx.strokeStyle = "rgba(0,0,0,0.8)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

// ---------------------------------------------------------------- 효과
// 효과마다 언제부터 인식됐는지 추적해서, 잠깐 튀는 오인식이나 깜빡임 없이 보여 줌
function updateEffect(key, now) {
  const st = (effectState[key] ??= { since: null, lastSeen: 0, shownAt: null, box: null });
  const hit = predictions.find((p) => p.label && p.score >= minScore && p.label.toLowerCase() === key);
  if (hit) {
    st.since ??= now;
    st.lastSeen = now;
    st.box = st.box ? smoothBox(st.box, hit.box) : { ...hit.box };
  } else if (now - st.lastSeen > HOLD_MS) {
    st.since = null;
    st.shownAt = null;
    st.box = null;
  }
  const visible = st.since !== null && now - st.since >= SHOW_AFTER_MS;
  if (visible && st.shownAt === null) st.shownAt = now;
  return visible;
}

function smoothBox(a, b, k = 0.35) {
  const out = {};
  for (const key of Object.keys(a)) out[key] = a[key] + (b[key] - a[key]) * k;
  return out;
}

function easeOutBack(t) {
  const c = 1.7;
  return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2;
}

function drawEffect(key, now, w, h) {
  const fx = EFFECTS[key];
  const st = effectState[key];
  const { x0, x1, y0, y1 } = st.box;
  const handSize = Math.max((x1 - x0) * w, (y1 - y0) * h);
  const size = Math.max(handSize * 1.1, Math.min(w, h) * 0.2);
  const pop = easeOutBack(Math.min((now - st.shownAt) / POP_MS, 1));
  const bob = Math.sin(now / 250) * size * 0.03;

  // 손 바로 위에 띄우고, 화면 밖으로 나가지 않게 맞춤
  const cx = Math.min(Math.max(((x0 + x1) / 2) * w, size / 2), w - size / 2);
  const cy = Math.max(y0 * h - size * 0.6, size / 2) + bob;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(pop, pop);
  if (fx.emoji) {
    ctx.font = `${size}px ${EMOJI_FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.shadowColor = "rgba(0,0,0,0.45)";
    ctx.shadowBlur = size * 0.08;
    ctx.fillText(fx.emoji, 0, 0);
  } else {
    const img = effectImages[key];
    if (img?.complete && img.naturalWidth) {
      const dw = size * 1.6;
      const dh = dw * (img.naturalHeight / img.naturalWidth);
      ctx.shadowColor = "rgba(255,255,255,0.95)";  // 어두운 배경에서도 검은 로고가 보이도록
      ctx.shadowBlur = size * 0.12;
      ctx.drawImage(img, -dw / 2, -dh / 2, dw, dh);
      ctx.drawImage(img, -dw / 2, -dh / 2, dw, dh);
    }
  }
  ctx.restore();
}

function updatePanel(shown) {
  const key = shown.join(",");
  if (key !== activeKey) {
    activeKey = key;
    const preview = $("effectPreview");
    preview.innerHTML = "";
    if (shown.length === 0) {
      preview.innerHTML = '<span class="empty">인식된 효과 없음</span>';
    } else {
      for (const k of shown) {
        const span = document.createElement("span");
        span.className = "pop";
        span.append(effectNode(EFFECTS[k]));
        preview.append(span);
      }
    }
  }
  const lines = predictions.map((p) => {
    return `${HAND_KO[p.hand] || p.hand}: ${predictionText(p)}`;
  });
  $("resultText").textContent = source ? (lines.join("\n") || "손이 보이지 않습니다") : "-";
}

// ---------------------------------------------------------------- 버튼
$("cameraBtn").onclick = useCamera;
$("fileBtn").onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => { openFile(e.target.files[0]); e.target.value = ""; };
$("sampleBtn").onclick = () => {
  openUrl(SAMPLE_VIDEO, false, "예제 영상");
  setStatus("예제 영상은 엄지척·브이·검지 위로·엄지 아래 사진으로 만든 영상이라 nike·ok 동작이 없습니다.\n"
            + "효과가 뜨지 않고 (Thumb_Up)처럼 기본 제스처 이름이 나오는 것이 정상입니다.");
};
$("stopBtn").onclick = () => { stopSource(); setStatus("정지했습니다."); };
$("scoreRange").oninput = (e) => {
  minScore = Number(e.target.value);
  $("scoreText").textContent = minScore.toFixed(2);
};

// 브라우저 개발자 도구에서 인식 결과를 확인할 수 있도록
window.gestureDemo = {
  get result() { return result; },
  get predictions() { return predictions; },
  features: (i = 0) => result && toFeatures(result.landmarks[i], result.handedness[i][0].categoryName,
                                             canvas.width, canvas.height),
};

init().catch((e) => setStatus(`초기화 실패: ${e.message}`, true));

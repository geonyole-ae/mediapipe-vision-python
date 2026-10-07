"""커스텀 제스처 학습 도구 (GUI) - 데이터 수집, 학습, 인식을 한 화면에서

실행:  python custom_gesture/app.py
"""
import argparse
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import cv2
from PIL import Image, ImageTk

import common
import train

PREVIEW_W, PREVIEW_H = 720, 540
COUNTDOWN_SEC = 3   # 웹캠 녹화 시작 전 손 모양을 잡을 시간
SLIDE_MS = 1500     # 인식 탭에서 사진 한 장을 보여주는 시간
RECOGNIZE_HANDS = 2
FONT = "Malgun Gothic"
HAND_KO = {"Left": "왼손", "Right": "오른손"}
MEDIA_TYPES = [("동영상/사진", "*.mp4 *.avi *.mov *.mkv *.wmv *.jpg *.jpeg *.png *.bmp *.webp"),
               ("모든 파일", "*.*")]


class FrameSource:
    """웹캠 / 동영상 / 사진(파일 또는 폴더)을 같은 방식으로 한 장씩 읽음"""

    def __init__(self, source, is_camera):
        self.is_camera = is_camera
        self.is_images = not is_camera and common.is_image_source(source)
        self.cap = None
        self.fps = 0
        if self.is_images:
            self.images = common.list_images(source)
            self.index = 0
            if not self.images:
                raise RuntimeError(f"사진이 없습니다: {source}")
        else:
            self.cap, _ = common.open_capture(str(source))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

    def read(self):
        """다음 프레임 (끝나면 None)"""
        if self.is_images:
            while self.index < len(self.images):
                frame = cv2.imread(str(self.images[self.index]))
                self.index += 1
                if frame is not None:
                    return frame
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.flip(frame, 1) if self.is_camera else frame  # 웹캠은 거울 모드

    def close(self):
        if self.cap is not None:
            self.cap.release()


class App:
    def __init__(self, root, data_path=common.DATA_PATH, model_path=common.CLASSIFIER_PATH):
        self.root = root
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)

        # 실행 중인 입력 처리 상태
        self.mode = None          # "collect" | "recognize" | None
        self.source = None
        self.landmarker = None
        self.classifier = None
        self.clock = None
        self.job = None           # root.after 예약 id
        self.frame_idx = 0

        # 녹화 상태
        self.rec_state = "off"    # "off" | "countdown" | "on"
        self.rec_label = ""
        self.countdown_end = 0.0
        self.saved = 0
        self.data_file = None
        self.writer = None

        self.train_queue = queue.Queue()
        self.photo = None

        root.title("커스텀 제스처 학습 도구")
        root.resizable(False, False)
        self.build_ui()
        self.refresh_counts()
        self.update_controls()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI 구성
    def build_ui(self):
        style = ttk.Style()
        style.configure("Big.TButton", font=(FONT, 11, "bold"), padding=6)

        # 입력 선택
        top = ttk.LabelFrame(self.root, text="입력", padding=8)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        self.src_kind = tk.StringVar(value="camera")
        self.cam_index = tk.IntVar(value=0)
        self.file_path = tk.StringVar()
        self.cam_radio = ttk.Radiobutton(top, text="웹캠", value="camera", variable=self.src_kind)
        self.cam_radio.grid(row=0, column=0)
        self.cam_spin = ttk.Spinbox(top, from_=0, to=9, width=3, textvariable=self.cam_index)
        self.cam_spin.grid(row=0, column=1, padx=(2, 16))
        self.file_radio = ttk.Radiobutton(top, text="파일", value="file", variable=self.src_kind)
        self.file_radio.grid(row=0, column=2)
        ttk.Entry(top, textvariable=self.file_path, width=62, state="readonly").grid(
            row=0, column=3, padx=4)
        self.file_btn = ttk.Button(top, text="동영상/사진...", command=self.browse_file)
        self.file_btn.grid(row=0, column=4, padx=2)
        self.dir_btn = ttk.Button(top, text="사진 폴더...", command=self.browse_dir)
        self.dir_btn.grid(row=0, column=5, padx=2)

        # 미리보기
        self.canvas = tk.Canvas(self.root, width=PREVIEW_W, height=PREVIEW_H, bg="#202020",
                                highlightthickness=0)
        self.canvas.grid(row=1, column=0, padx=(8, 4), pady=4)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.create_text(PREVIEW_W / 2, PREVIEW_H / 2, text="입력을 고르고 시작을 누르세요",
                                fill="#aaaaaa", font=(FONT, 14), tags="hint")

        # 탭
        self.tabs = ttk.Notebook(self.root, width=440)
        self.tabs.grid(row=1, column=1, sticky="ns", padx=(4, 8), pady=4)
        self.tabs.add(self.build_collect_tab(), text="  1. 데이터 수집  ")
        self.tabs.add(self.build_train_tab(), text="  2. 학습  ")
        self.tabs.add(self.build_recognize_tab(), text="  3. 인식  ")
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.stop())

        self.status = tk.StringVar(value="준비")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w",
                  padding=(6, 2)).grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

    def build_collect_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="제스처 이름").grid(row=0, column=0, sticky="w")
        self.label_var = tk.StringVar()
        self.label_combo = ttk.Combobox(tab, textvariable=self.label_var)
        self.label_combo.grid(row=0, column=1, sticky="ew", pady=3)

        ttk.Label(tab, text="최대 개수").grid(row=1, column=0, sticky="w")
        self.max_var = tk.IntVar(value=300)
        ttk.Spinbox(tab, from_=0, to=10000, increment=50, width=8, textvariable=self.max_var).grid(
            row=1, column=1, sticky="w", pady=3)
        ttk.Label(tab, text="(0 = 제한 없음)", foreground="gray").grid(row=1, column=1, sticky="e")

        ttk.Label(tab, text="N프레임마다 저장").grid(row=2, column=0, sticky="w")
        self.every_var = tk.IntVar(value=1)
        ttk.Spinbox(tab, from_=1, to=30, width=8, textvariable=self.every_var).grid(
            row=2, column=1, sticky="w", pady=3)

        buttons = ttk.Frame(tab)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        buttons.columnconfigure((0, 1, 2), weight=1)
        self.preview_btn = ttk.Button(buttons, text="▶ 미리보기", command=self.start_preview)
        self.preview_btn.grid(row=0, column=0, sticky="ew", padx=2)
        self.record_btn = ttk.Button(buttons, text="● 녹화 시작", style="Big.TButton",
                                     command=self.toggle_record)
        self.record_btn.grid(row=0, column=1, sticky="ew", padx=2)
        self.collect_stop_btn = ttk.Button(buttons, text="■ 정지", command=self.stop)
        self.collect_stop_btn.grid(row=0, column=2, sticky="ew", padx=2)

        ttk.Label(tab, foreground="gray", wraplength=410, justify="left", text=(
            f"웹캠: 미리보기로 손 위치를 확인하고 녹화 시작을 누르면 {COUNTDOWN_SEC}초 뒤부터 "
            "저장합니다. 손 각도·거리·위치를 바꿔 가며 모으세요.\n"
            "동영상/사진: 녹화 시작을 누르면 손이 보이는 프레임을 모두 저장합니다.\n"
            "제스처당 200~500개를 권장하고, 아무 제스처도 아닌 손을 'none'으로 모아 두면 "
            "오인식이 줄어듭니다.")).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 10))

        box = ttk.LabelFrame(tab, text="모은 데이터", padding=6)
        box.grid(row=5, column=0, columnspan=2, sticky="nsew")
        tab.rowconfigure(5, weight=1)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.count_tree = ttk.Treeview(box, columns=("count",), height=8, selectmode="browse")
        self.count_tree.heading("#0", text="제스처")
        self.count_tree.heading("count", text="샘플 수")
        self.count_tree.column("#0", width=200)
        self.count_tree.column("count", width=90, anchor="e")
        self.count_tree.grid(row=0, column=0, sticky="nsew")
        self.count_tree.bind("<<TreeviewSelect>>", self.on_select_label)
        self.delete_btn = ttk.Button(box, text="선택한 제스처 삭제", command=self.delete_selected)
        self.delete_btn.grid(row=1, column=0, sticky="e", pady=(6, 0))
        self.total_var = tk.StringVar()
        ttk.Label(box, textvariable=self.total_var).grid(row=1, column=0, sticky="w", pady=(6, 0))
        return tab

    def build_train_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        self.train_btn = ttk.Button(tab, text="학습 시작", style="Big.TButton",
                                    command=self.start_training)
        self.train_btn.grid(row=0, column=0, sticky="ew")
        self.progress = ttk.Progressbar(tab, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=6)
        self.train_log = ScrolledText(tab, width=56, height=26, font=("Consolas", 9), wrap="none")
        self.train_log.grid(row=2, column=0, sticky="nsew")
        self.write_log("수집한 데이터로 분류기를 학습합니다.\n"
                       "20%를 떼어 검증한 뒤, 전체 데이터로 다시 학습해서 저장합니다.\n"
                       f"\n데이터: {self.data_path}\n저장 위치: {self.model_path}\n")
        return tab

    def build_recognize_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        tab.columnconfigure((0, 1), weight=1)
        self.recog_start_btn = ttk.Button(tab, text="▶ 인식 시작", style="Big.TButton",
                                          command=self.start_recognize)
        self.recog_start_btn.grid(row=0, column=0, sticky="ew", padx=2)
        self.recog_stop_btn = ttk.Button(tab, text="■ 정지", command=self.stop)
        self.recog_stop_btn.grid(row=0, column=1, sticky="ew", padx=2)

        ttk.Label(tab, text="최소 확률").grid(row=1, column=0, sticky="w", pady=(14, 0))
        self.min_score = tk.DoubleVar(value=0.7)
        score_text = tk.StringVar(value="0.70")
        self.min_score.trace_add("write", lambda *_: score_text.set(f"{self.min_score.get():.2f}"))
        ttk.Label(tab, textvariable=score_text).grid(row=1, column=1, sticky="e", pady=(14, 0))
        ttk.Scale(tab, from_=0.3, to=0.99, variable=self.min_score).grid(
            row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(tab, text="이보다 확률이 낮으면 '-'로 표시합니다.", foreground="gray").grid(
            row=3, column=0, columnspan=2, sticky="w")

        result_box = ttk.LabelFrame(tab, text="인식 결과", padding=10)
        result_box.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(16, 8))
        self.result_var = tk.StringVar(value="-")
        ttk.Label(result_box, textvariable=self.result_var, font=(FONT, 20, "bold"),
                  foreground="#b0008a", wraplength=400).pack(anchor="w")

        self.model_info = tk.StringVar()
        ttk.Label(tab, textvariable=self.model_info, foreground="gray", wraplength=410,
                  justify="left").grid(row=5, column=0, columnspan=2, sticky="w")
        self.update_model_info()
        return tab

    # ------------------------------------------------------------------ 입력
    def browse_file(self):
        path = filedialog.askopenfilename(title="동영상 또는 사진 선택", filetypes=MEDIA_TYPES)
        if path:
            self.file_path.set(path)
            self.src_kind.set("file")

    def browse_dir(self):
        path = filedialog.askdirectory(title="사진 폴더 선택")
        if path:
            self.file_path.set(path)
            self.src_kind.set("file")

    def make_source(self):
        if self.src_kind.get() == "camera":
            return FrameSource(str(self.cam_index.get()), is_camera=True)
        path = self.file_path.get()
        if not path:
            raise RuntimeError("동영상/사진 파일이나 폴더를 먼저 선택하세요.")
        return FrameSource(path, is_camera=False)

    # ------------------------------------------------------------------ 실행/정지
    def start(self, mode):
        self.stop()
        try:
            source = self.make_source()
        except Exception as e:
            messagebox.showerror("입력 오류", str(e))
            return False
        try:
            if mode == "recognize":
                self.classifier = common.GestureClassifier(self.model_path)  # 재학습 결과 반영
            self.landmarker = common.create_landmarker(
                num_hands=1 if mode == "collect" else RECOGNIZE_HANDS,
                video=not source.is_images)
        except Exception as e:
            source.close()
            messagebox.showerror("오류", str(e))
            return False

        self.source = source
        self.mode = mode
        self.clock = common.VideoClock()
        self.frame_idx = 0
        self.canvas.delete("hint")
        if mode == "collect":
            self.data_file, self.writer = common.open_writer(self.data_path)
        self.update_controls()
        self.tick()
        return True

    def stop(self):
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.job = None
        if self.rec_state != "off":
            self.stop_recording()
        if self.data_file is not None:
            self.data_file.close()
            self.data_file = self.writer = None
        if self.source is not None:
            self.source.close()
            self.source = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        self.mode = None
        self.update_controls()

    def finish(self, message):
        """입력이 끝났을 때 (마지막 화면은 그대로 둠)"""
        self.stop()
        self.status.set(message)

    def on_close(self):
        self.stop()
        self.root.destroy()

    # ------------------------------------------------------------------ 수집
    def start_preview(self):
        if self.start("collect"):
            self.status.set("미리보기 중 - 녹화 시작을 누르면 저장합니다.")

    def toggle_record(self):
        if self.rec_state != "off":
            self.stop_recording()
            if self.source is not None and not self.source.is_camera:
                self.stop()
            return

        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning("제스처 이름", "저장할 제스처 이름을 입력하세요. (예: heart, ok, none)")
            self.label_combo.focus_set()
            return
        if self.mode != "collect" and not self.start("collect"):
            return

        self.rec_label = label
        self.saved = 0
        if self.source.is_camera:
            self.rec_state = "countdown"
            self.countdown_end = time.monotonic() + COUNTDOWN_SEC
        else:
            self.rec_state = "on"
        self.status.set(f"'{label}' 녹화 중...")
        self.update_controls()

    def stop_recording(self):
        was_recording = self.rec_state == "on"
        self.rec_state = "off"
        if self.data_file is not None:
            self.data_file.flush()
        self.refresh_counts()
        if was_recording:
            self.status.set(f"'{self.rec_label}' 샘플 {self.saved}개 저장했습니다.")
        else:
            self.status.set("녹화를 취소했습니다.")
        self.update_controls()

    def process_collect(self, frame, result, hands, texts):
        has_hand = bool(result.hand_landmarks)
        if has_hand:
            hands.append(result.hand_landmarks[0])

        if self.rec_state == "countdown":
            remain = self.countdown_end - time.monotonic()
            if remain > 0:
                texts.append((0.5, 0.5, str(int(remain) + 1), "#ffdd00", 72, "center"))
            else:
                self.rec_state = "on"

        if self.rec_state == "on":
            if has_hand and self.frame_idx % max(get_int(self.every_var, 1), 1) == 0:
                self.writer.writerow([self.rec_label, *common.hand_features(result, 0, frame)])
                self.saved += 1
            texts.append((0.02, 0.07, f"● REC  {self.rec_label}  {self.saved}개", "#ff3030", 16, "w"))
            limit = get_int(self.max_var, 0)
            if limit and self.saved >= limit:
                self.stop_recording()
                if not self.source.is_camera:
                    self.finish(f"'{self.rec_label}' 샘플 {self.saved}개 저장했습니다.")
        elif self.rec_state == "off":
            texts.append((0.02, 0.07, "미리보기", "#ffffff", 14, "w"))

        if not has_hand:
            texts.append((0.02, 0.14, "손이 보이지 않습니다", "#ffa500", 13, "w"))

    def on_select_label(self, _event):
        selected = self.count_tree.selection()
        if selected and self.rec_state == "off":
            self.label_var.set(self.count_tree.item(selected[0], "text"))

    def delete_selected(self):
        selected = self.count_tree.selection()
        if not selected:
            messagebox.showinfo("삭제", "삭제할 제스처를 목록에서 선택하세요.")
            return
        label = self.count_tree.item(selected[0], "text")
        if not messagebox.askyesno("삭제", f"'{label}' 샘플을 모두 삭제할까요?"):
            return
        self.stop()
        removed = common.delete_label(self.data_path, label)
        self.refresh_counts()
        self.status.set(f"'{label}' 샘플 {removed}개를 삭제했습니다.")

    def refresh_counts(self):
        counts = common.count_labels(self.data_path)
        self.count_tree.delete(*self.count_tree.get_children())
        for label, n in sorted(counts.items()):
            self.count_tree.insert("", "end", text=label, values=(n,))
        self.label_combo["values"] = sorted(counts)
        self.total_var.set(f"합계 {sum(counts.values())}개")

    # ------------------------------------------------------------------ 학습
    def write_log(self, text):
        self.train_log.insert("end", text + "\n")
        self.train_log.see("end")

    def start_training(self):
        if self.rec_state != "off":
            self.stop_recording()
        self.train_log.delete("1.0", "end")
        self.train_btn.state(["disabled"])
        self.progress.start(12)
        self.status.set("학습 중...")
        threading.Thread(target=self.train_worker, daemon=True).start()
        self.root.after(100, self.poll_training)

    def train_worker(self):
        try:
            labels = train.train_and_save(self.data_path, self.model_path,
                                          log=lambda s: self.train_queue.put(("log", s)))
            self.train_queue.put(("done", f"학습 완료 - 제스처 {len(labels)}개: {', '.join(labels)}"))
        except Exception as e:
            self.train_queue.put(("error", str(e)))

    def poll_training(self):
        finished = None
        while not self.train_queue.empty():
            kind, text = self.train_queue.get()
            if kind == "log":
                self.write_log(text)
            else:
                finished = (kind, text)
        if finished is None:
            self.root.after(100, self.poll_training)
            return

        self.progress.stop()
        self.train_btn.state(["!disabled"])
        kind, text = finished
        if kind == "done":
            self.write_log("\n" + text + "\n'3. 인식' 탭에서 확인해 보세요.")
            self.status.set(text)
            self.update_model_info()
        else:
            self.write_log("\n[오류] " + text)
            self.status.set("학습 실패")
            messagebox.showerror("학습 실패", text)

    # ------------------------------------------------------------------ 인식
    def start_recognize(self):
        if self.start("recognize"):
            self.status.set("인식 중")

    def update_model_info(self):
        if not self.model_path.exists():
            self.model_info.set("아직 학습한 분류기가 없습니다. '2. 학습' 탭에서 먼저 학습하세요.")
            return
        try:
            labels = common.GestureClassifier(self.model_path).labels
            self.model_info.set("인식할 제스처: " + ", ".join(labels))
        except Exception as e:
            self.model_info.set(f"분류기를 읽을 수 없습니다: {e}")

    def process_recognize(self, frame, result, hands, texts):
        lines = []
        for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            hands.append(landmarks)
            label, score = self.classifier.predict(common.hand_features(result, i, frame))
            text = f"{label} {score:.2f}" if score >= self.min_score.get() else "-"

            # 거울 모드(좌우 반전)로 넣은 프레임에서는 라벨을 뒤집어야 실제 손과 일치함
            hand = handedness[0].category_name
            if self.source.is_camera:
                hand = {"Left": "Right", "Right": "Left"}.get(hand, hand)

            x = min(lm.x for lm in landmarks)
            y = min(lm.y for lm in landmarks)
            texts.append((x, max(y - 0.02, 0.06), text, "#ff4dff", 18, "sw"))
            lines.append(f"{HAND_KO.get(hand, hand)}: {text}")
        self.result_var.set("\n".join(lines) if lines else "손이 보이지 않습니다")

    # ------------------------------------------------------------------ 프레임 처리
    def tick(self):
        self.job = None
        if self.source is None:
            return
        started = time.monotonic()

        frame = self.source.read()
        if frame is None:
            if self.source.is_camera:
                self.finish("웹캠 프레임을 읽지 못했습니다.")
            elif self.rec_state == "on":
                self.finish(f"입력이 끝났습니다. '{self.rec_label}' 샘플 {self.saved}개 저장했습니다.")
            else:
                self.finish("입력이 끝났습니다.")
            return

        image = common.to_mp_image(frame)
        if self.source.is_images:
            result = self.landmarker.detect(image)
        else:
            result = self.landmarker.detect_for_video(image, self.clock.now())
        self.frame_idx += 1

        hands, texts = [], []
        if self.mode == "collect":
            self.process_collect(frame, result, hands, texts)
        else:
            self.process_recognize(frame, result, hands, texts)
        self.show(frame, hands, texts)

        if self.source is None:  # 처리 중에 정지됨 (최대 개수 도달 등)
            return
        self.job = self.root.after(self.next_delay(started), self.tick)

    def next_delay(self, started):
        if self.mode == "collect" or self.source.is_camera:
            return 1
        if self.source.is_images:
            return SLIDE_MS
        # 인식 탭의 동영상은 원래 속도로 재생
        elapsed = (time.monotonic() - started) * 1000
        return max(int(1000 / self.source.fps - elapsed), 1)

    def show(self, frame, hands, texts):
        """프레임을 미리보기 크기로 줄이고 손 랜드마크와 글자를 그림"""
        h, w = frame.shape[:2]
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        dw, dh = int(w * scale), int(h * scale)
        disp = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        for landmarks in hands:
            common.draw_hand(disp, landmarks)

        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        ox, oy = (PREVIEW_W - dw) // 2, (PREVIEW_H - dh) // 2
        self.canvas.itemconfig(self.image_item, image=self.photo)
        self.canvas.coords(self.image_item, ox, oy)

        # 한글이 깨지지 않도록 글자는 OpenCV가 아니라 캔버스에 그림 (그림자 + 본문)
        self.canvas.delete("overlay")
        for nx, ny, text, color, size, anchor in texts:
            x, y = ox + nx * dw, oy + ny * dh
            for dx, dy, fill in ((2, 2, "black"), (0, 0, color)):
                self.canvas.create_text(x + dx, y + dy, text=text, fill=fill, anchor=anchor,
                                        font=(FONT, size, "bold"), tags="overlay")

    # ------------------------------------------------------------------ 버튼 상태
    def update_controls(self):
        running = self.mode is not None
        for widget in (self.cam_radio, self.cam_spin, self.file_radio, self.file_btn, self.dir_btn):
            widget.state(["disabled"] if running else ["!disabled"])

        recording = self.rec_state != "off"
        self.record_btn.configure(text="■ 녹화 정지" if recording else "● 녹화 시작")
        self.preview_btn.state(["disabled"] if self.mode == "collect" else ["!disabled"])
        self.collect_stop_btn.state(["!disabled"] if self.mode == "collect" else ["disabled"])
        self.label_combo.state(["disabled"] if recording else ["!disabled"])
        self.delete_btn.state(["disabled"] if recording else ["!disabled"])

        self.recog_start_btn.state(["disabled"] if self.mode == "recognize" else ["!disabled"])
        self.recog_stop_btn.state(["!disabled"] if self.mode == "recognize" else ["disabled"])


def get_int(var, default):
    """스핀박스에 숫자가 아닌 값이 들어 있으면 기본값"""
    try:
        return int(var.get())
    except (tk.TclError, ValueError):
        return default


def enable_high_dpi():
    """Windows 화면 배율에서 글자가 흐리게 보이지 않도록"""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="커스텀 제스처 학습 도구 (GUI)")
    parser.add_argument("--data", type=Path, default=common.DATA_PATH, help="학습 데이터 CSV 경로")
    parser.add_argument("--model", type=Path, default=common.CLASSIFIER_PATH, help="분류기 저장 경로")
    args = parser.parse_args()

    enable_high_dpi()
    root = tk.Tk()
    App(root, args.data, args.model)
    root.mainloop()


if __name__ == "__main__":
    main()

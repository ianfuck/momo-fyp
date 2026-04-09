# Momo MVP

Momo 是一個單機互動裝置 MVP：用 webcam 追蹤觀眾，透過 Ollama 生成文本與 TTS 情緒標記，交給 Fish Speech V1.5、Fish Audio S1 Mini、Qwen3-TTS、Kokoro-82M 或 MeloTTS 做朗讀，再透過 ESP32 控制雙眼 SG90 伺服馬達。

## 架構

- `backend/`: Python 長駐程式，負責狀態機、prompt、Ollama、TTS、serial、telemetry。
- `frontend/`: Vite + TypeScript 控制台，調參、監看 pipeline、記憶體、servo 角度。
- `esp32/sg90/`: Arduino firmware，接收左右眼角度。
- `resource/`: system prompt、examples、voice clone 素材。

## 視覺與控制規則

- `YOLO11n` 用於 `person bbox` 與 threshold。
- 馬達追視優先看 face/eye tracking；若眼睛失效，回退到 person center。
- person bbox 決定 lock/unlock，眼睛定位只決定 servo aiming。
- Track mode LLM 依句序 `1..10` 對齊對應 example stage。
- UI 高亮顯示當前流程：`LLM > TTS > PLAYBACK`。

## 安裝

### Python

```bash
uv sync
```

`uv sync` 會依作業系統自動選 PyTorch wheel：
- macOS: `torch==2.4.1` `torchaudio==2.4.1` `torchvision==0.19.1`
- Windows: `torch==2.4.1+cu118` `torchaudio==2.4.1+cu118` `torchvision==0.19.1+cu118`

Python 建議使用 `3.11` 或 `3.12`；`torch 2.4.1` 不適合 `3.13`。

### Node

前端固定使用 Node 22。

```bash
source ~/.nvm/nvm.sh
nvm install 22
nvm use 22
```

macOS Apple Silicon:
- `uv sync` 會安裝對應的 macOS wheel，可用 `MPS`。

Windows + NVIDIA:
- `uv sync` 會自動安裝 `cu118` wheel；仍需 NVIDIA driver 支援 CUDA 11.8。

如要補齊完整 vision/TTS 執行依賴，可再加裝：
- `ultralytics`
- `opencv-python`
- `mediapipe`
- `fish-speech`
- `torch`

### Hugging Face gated model

Fish Speech V1.5 與 Fish Audio S1 Mini 都是 gated model。第一次使用前要先：

```bash
hf auth login
```

但光 login 不夠；你還必須先在瀏覽器開啟你要用的模型頁面並同意條款：
- `Fish Speech V1.5`: <https://huggingface.co/fishaudio/fish-speech-1.5>
- `Fish Audio S1 Mini`: <https://huggingface.co/fishaudio/s1-mini>

也可以改用 `HF_TOKEN=...` 啟動後端。

也支援專案根目錄 `.env`：

```bash
cp .env.example .env
```

### Frontend

```bash
cd frontend
npm install
```

### Ollama

```bash
ollama serve
ollama pull llama3.1
```

## 啟動

後端：

```bash
uv run python -m backend.app --reload
```

如要跳過 TTS 啟動 benchmark：

```bash
uv run python -m backend.app --skip-tts-benchmark
```

如只要跑 YOLO / vision，完全不要在啟動時載入 TTS 與 Ollama：

```bash
uv run python -m backend.app --yolo-only
```

前端：

```bash
cd frontend
npm install
npm run dev
```

ESP32:
- 用 Arduino IDE 開啟 [esp32/sg90/sg90.ino](/Users/ian/Desktop/work/job/momo/esp32/sg90/sg90.ino)
- 安裝 `ESP32Servo`
- 燒錄後把 serial port 填進 UI

## API

- `GET /api/status`
- `GET /api/config`
- `POST /api/config`
- `GET /api/cameras`
- `GET /api/serial/ports`
- `GET /api/ollama/models`
- `POST /api/control/recenter-servos`
- `POST /api/control/simulate-track`
- `POST /api/control/simulate-pipeline`

## 測試

```bash
uv run pytest
```

前端 build:

```bash
cd frontend
npm run build
```

## 注意

- 瀏覽器相機是目前建議路徑：由前端取得 camera 權限，持續把 JPEG frame 上傳到後端做 YOLO/face/eye tracking。
- 若要用 backend OpenCV 直接開相機，macOS 需要對啟動後端的終端或 IDE 單獨授權 Camera。
- Windows 預設會把 YOLO 與 Fish Audio TTS 放到 `cuda:0`，並把本程序的 CUDA 記憶體上限設成 `72%`，保留剩餘 VRAM 給 Ollama；可用 `MOMO_CUDA_MEMORY_FRACTION` 覆寫。
- macOS 會讓 YOLO 走 `cpu`，Fish Audio TTS 走 `MPS`。
- 預設 TTS model path 是 `model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-0.6B-Base`。
- 也支援 `model/huggingface/hf_snapshots/fishaudio__s1-mini`；切換 model path 後，Ollama 的情緒分類清單會自動跟著目前模型切換：
  - `Fish Speech V1.5`: 只用該模型穩定支援的 basic emotion tags
  - `Fish Audio S1 Mini`: 用 S1 的完整固定 emotion tag 清單
- 也支援 `model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-0.6B-Base` 與 `model/huggingface/hf_snapshots/Qwen__Qwen3-TTS-12Hz-1.7B-Base`。Qwen3-TTS 會直接走 `qwen-tts` runtime，不使用 Fish 的括號 emotion tag。
- 也支援 `model/huggingface/hf_snapshots/hexgrad__Kokoro-82M-v1.1-zh` 與 `model/huggingface/hf_snapshots/myshell-ai__MeloTTS-Chinese`。
  - `Kokoro-82M` 這個 option 會實際下載官方中文模型 `hexgrad/Kokoro-82M-v1.1-zh`，固定使用 `zf_001` 中文 voice。
  - `MeloTTS` 會使用 `myshell-ai/MeloTTS-Chinese` 的中文 speaker。
  - 這兩個 provider 都會直接生成中文語音，但不支援 reference voice clone，也不使用 Fish 的 structured emotion tag。
- Fish TTS 情緒 tag 會固定包成 `({emotion})中文句子`，括號 tag 永遠放在最前面，避免只讀出 tag 本身。
- 若 `TTS Device` 在 UI 或 config 是 `auto`，後端啟動時會 benchmark 三個候選：
  - 加速器單裝置：Windows `gpu` / macOS `mps`
  - `semantic-auto-*`：只把 Fish semantic transformer 交給 `accelerate.load_checkpoint_and_dispatch(..., device_map="auto")`，decoder 仍留在單一裝置
  - `cpu`
- `Kokoro-82M` 與 `MeloTTS` 也會走同一套啟動 benchmark，但它們只測 `float32` candidate，不會跑 Fish/Qwen 的 `float16` / `semantic-auto-*` 變體。
- benchmark 順序會優先試加速器，再試 `semantic-auto-*`，最後才試 `cpu`，避免在 Windows 先卡住很慢的 CPU 路線。
- 每個 TTS benchmark candidate 都會用獨立 subprocess 跑，避免前一個 candidate 的 CUDA OOM 或殘留 VRAM 汙染下一個 candidate。
- 若在 Windows/macOS 明確指定 `gpu` 或 `mps`，但 TTS preload 因 OOM 失敗，後端會自動退回 `auto benchmark` 路線，而不是直接把程式炸掉。
- 單個 benchmark candidate 預設最長 `120` 秒；可用 `MOMO_TTS_BENCHMARK_TIMEOUT_SEC` 覆寫。
- 啟動 benchmark 的優先序是：
  - 1. 使用者在 UI 明確選的 device
  - 2. `auto` benchmark 選出的最快 device
  - 3. 程式原始 default
- 當 `auto` benchmark 選出結果後，runtime 會把目前生效的 `TTS Device` 視為 benchmark 選中的 device，所以 UI 顯示的會是實際生效值，不會一直停在 `auto`。
- `--skip-tts-benchmark` 只會跳過這段啟動 benchmark，直接使用目前 config 的 `TTS Device`。
- `--yolo-only` 會讓後端從啟動開始就跳過 TTS model bootstrap、TTS preload、Ollama warmup、Ollama runtime status refresh，僅保留 vision / YOLO / servo 路徑。
- UI 頂部的 YOLO / TTS / Ollama device 與 RAM / VRAM 會即時顯示目前生效裝置。
  - Ollama 的數字來自它自己的 runtime 回報。
  - YOLO / TTS 的數字是本程序在模型 warmup / preload 時量到的 component footprint，屬於近似值。
- `GET /api/audio/devices` 會列出本機 output devices，UI 可直接切換播放輸出。
- 前端 production build 已在 Node 22 驗證通過；Node 25 不建議使用。

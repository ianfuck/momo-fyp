# Momo — 鎖定 / 閒置模式、雙眼 SG90、Ollama、Qwen3-TTS

Python（FastAPI）後端負責 webcam、YOLO 人形偵測、狀態機、Ollama HTTP、**Qwen3-TTS**（`torch` + `qwen-tts` 已列在預設依賴，`uv sync` 即裝）、跨平台播音、Serial→Arduino。前端為 **Vite + 原生 JS**。

## 需求

- Python **3.11+**（TTS 建議 3.12）
- [uv](https://docs.astral.sh/uv/) 建議
- [Ollama](https://ollama.com) 本機運行；預設模型為 **qwen3.5:0.8b**。若本機尚無該模型，首次對話時會自動 **`ollama pull`**（可於 UI 直接輸入其他模型名，例如 `llama3.2`）。
- Node.js **18+**（僅建置前端）

### PyTorch 與 TTS

- **`uv sync`** 會安裝 **CPU 版 PyTorch**（全平台可用）。程式會依環境選 **CUDA → MPS（Apple Silicon）→ CPU** 載入 Qwen3-TTS。
- **NVIDIA GPU**：若要用 CUDA，請到 [pytorch.org](https://pytorch.org) 複製對應 **cu12x** 的 `pip/uv` 指令，在同一 venv 內**覆蓋安裝** `torch`（與官方 wheel 一致即可）。
- 僅在 **Qwen 模型或參考音載入失敗**（例如缺 `HF_TOKEN`、路徑錯誤）時，才會退回極短**雜訊 stub**，此時請看後端 log 與 **`tmp/log.csv`** 的 `tts/synthesize_stub`。
- 參考音為 **`.m4a`** 時，建議安裝 **ffmpeg**，啟動時會自動轉成 `tmp/ref_voice_for_tts.wav` 再餵給克隆（否則可能只有雜音或載入失敗）。
- Qwen3-TTS 會下載到 **`model/huggingface/hf_snapshots/<模型 id 安全化>/`**，並以**本機路徑**載入（避開 `qwen_tts` 只把 `cache_dir` 傳給 `AutoModel`、processor 卻讀 `~/.cache` 導致半成品缺 `speech_tokenizer/preprocessor_config.json` 的問題）。若載入仍失敗，可刪除該快照目錄後重啟讓它重新 `snapshot_download`。
- 儀表板 **`tts_backend`** 會顯示 **`qwen3(weights=mps:0)`** 或 **`cpu`** 等，代表**權重實際落在哪**；若你以為在用 GPU 卻看到 `cpu`，代表 MPS 載入失敗已回退。Apple Silicon 上 **YOLO** 會明確使用 **`device=mps`**（若可用）。

## 安裝與執行

```bash
cd /path/to/momo
uv sync
uv run python -c "from src.main import app"  # 快速檢查匯入

# 建置前端
cd frontend && npm install && npm run build && cd ..

# 啟動（請先開啟 Ollama）
uv run python -m src.main
# 或
uv run momo-server
```

瀏覽器開啟 **http://127.0.0.1:8000**（靜態檔來自 `frontend/dist`）。

開發前端時可另開終端：

```bash
cd frontend && npm run dev
# Vite 會 proxy /api 與 /ws 到 8000，後端仍須先跑 src.main
```

## 目錄

| 路徑 | 說明 |
|------|------|
| `src/` | Python 套件（匯入前綴 `src.`） |
| `resource/md/` | Persona |
| `resource/example/` | Few-shot CSV（UI 勾選） |
| `resource/voice/` | 克隆參考音與講稿 |
| `model/` | YOLO / HF 權重快取（預設 gitignore） |
| `arduino/sg90/` | 雙 SG90 韌體 |
| `tmp/audience.csv` | **僅**鎖定中觀眾的特徵快照（覆寫寫入）；離開 LOCK／LOST／重啟後清空只剩表頭 |
| `tmp/log.csv` | 其餘事件（狀態轉移、LLM 結果／錯誤、TTS qwen/stub 等）append 紀錄 |

## 環境變數（可選）

- `HF_TOKEN`：下載 Hugging Face 私有或限流模型時使用。

## 狀態機（WebSocket `state` 欄位）

| 顯示值 | 意義 |
|--------|------|
| `IDLE` | 無鎖定或一般閒置 |
| `LOCK` | 已鎖定觀眾並追蹤 |
| `RECONNECT_PENDING` | 鎖定中但暫時看不到人，仍在 `reconnect_window_ms` 內可重連 |
| `LOST` | 已確認離場（超過重連窗）後短暫顯示，再變回 `IDLE` |
| `PURGE` | 第十句播畢後的驅逐：馬達 `FRENZY` 階段 |
| `DEAD` | 裝死：`purge_dead_ms` 內忽略追蹤與一般語音 |
| `SLEEP` | 長時間閒置後靜音休眠 |

一輪結束（`DEAD` 結束回到 `IDLE`）時後端會送 WebSocket `type: "round_reset"`，前端應清空該輪 History。

## Arduino

將 `arduino/sg90/sg90.ino` 燒入 Uno 相容板，預設 **115200**，左 **D9**、右 **D10**。在 UI 填 `serial_port`（例如 macOS `/dev/cu.usbserial-*`、Windows `COM3`）。

## 授權

專案資源與程式碼依你專案需求自行標註。

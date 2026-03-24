const API = "";

let schemaFields = [];
let exampleFiles = [];
let defaults = {};

async function loadExamples() {
  const r = await fetch(`${API}/api/resource/examples`);
  exampleFiles = await r.json();
}

async function loadSchema() {
  const r = await fetch(`${API}/api/config/schema`);
  const data = await r.json();
  schemaFields = data.fields || [];
}

async function loadConfig() {
  const r = await fetch(`${API}/api/config`);
  return r.json();
}

function buildForm(cfg) {
  const form = document.getElementById("cfg-form");
  form.innerHTML = "";
  for (const f of schemaFields) {
    const key = f.key;
    const val = cfg[key] !== undefined ? cfg[key] : f.default;
    defaults[key] = f.default;

    const wrap = document.createElement("div");
    wrap.className = "field";
    const id = `f-${key}`;

    if (f.type === "path_multiselect") {
      const lab = document.createElement("label");
      lab.textContent = key;
      wrap.appendChild(lab);
      const help = document.createElement("p");
      help.className = "help";
      help.textContent = f.help_zh || "";
      wrap.appendChild(help);
      const cg = document.createElement("div");
      cg.className = "check-grid";
      const isTrack = key.includes("tracking");
      const filtered = exampleFiles.filter((ex) => {
        if (ex.suggested_mode === "any") return true;
        if (isTrack) return ex.suggested_mode === "tracking" || ex.suggested_mode === "any";
        return ex.suggested_mode === "idle" || ex.suggested_mode === "any";
      });
      const selected = new Set(Array.isArray(val) ? val : []);
      for (const ex of filtered) {
        const row = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.dataset.path = ex.path;
        cb.checked = selected.has(ex.path);
        row.appendChild(cb);
        const span = document.createElement("span");
        span.textContent = `${ex.basename} (${ex.suggested_mode})`;
        row.appendChild(span);
        cg.appendChild(row);
      }
      cg.dataset.fieldKey = key;
      cg.dataset.multiselect = "1";
      wrap.appendChild(cg);
    } else if (f.type === "enum") {
      const lab = document.createElement("label");
      lab.setAttribute("for", id);
      lab.textContent = key;
      wrap.appendChild(lab);
      const help = document.createElement("p");
      help.className = "help";
      help.textContent = f.help_zh || "";
      wrap.appendChild(help);
      const sel = document.createElement("select");
      sel.id = id;
      sel.dataset.key = key;
      for (const opt of f.enum_options || []) {
        const o = document.createElement("option");
        o.value = opt.value;
        o.textContent = opt.label + (opt.help ? ` — ${opt.help}` : "");
        if (String(opt.value) === String(val)) o.selected = true;
        sel.appendChild(o);
      }
      wrap.appendChild(sel);
    } else if (f.type === "boolean") {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.id = id;
      cb.dataset.key = key;
      cb.checked = !!val;
      lab.appendChild(cb);
      lab.append(` ${key}`);
      wrap.appendChild(lab);
      const help = document.createElement("p");
      help.className = "help";
      help.textContent = f.help_zh || "";
      wrap.appendChild(help);
    } else {
      const lab = document.createElement("label");
      lab.setAttribute("for", id);
      lab.textContent = key;
      wrap.appendChild(lab);
      const help = document.createElement("p");
      help.className = "help";
      help.textContent = f.help_zh || "";
      wrap.appendChild(help);
      const inp = document.createElement("input");
      inp.id = id;
      inp.dataset.key = key;
      inp.type = f.type === "number" ? "number" : "text";
      inp.value = val === null || val === undefined ? "" : val;
      if (f.placeholder) inp.placeholder = f.placeholder;
      if (Array.isArray(f.suggestions) && f.suggestions.length && f.type !== "number") {
        const dl = document.createElement("datalist");
        dl.id = `dl-${key}`;
        for (const s of f.suggestions) {
          const opt = document.createElement("option");
          opt.value = s;
          dl.appendChild(opt);
        }
        inp.setAttribute("list", dl.id);
        wrap.appendChild(dl);
      }
      wrap.appendChild(inp);
    }
    form.appendChild(wrap);
  }
}

function collectForm() {
  const form = document.getElementById("cfg-form");
  const body = {};
  for (const f of schemaFields) {
    const key = f.key;
    if (f.type === "path_multiselect") {
      const cg = form.querySelector(`[data-field-key="${key}"]`);
      if (!cg) continue;
      const paths = [];
      cg.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
        if (cb.checked) paths.push(cb.dataset.path);
      });
      body[key] = paths;
    } else if (f.type === "boolean") {
      const cb = form.querySelector(`input[data-key="${key}"]`);
      if (cb) body[key] = cb.checked;
    } else if (f.type === "enum") {
      const sel = form.querySelector(`select[data-key="${key}"]`);
      if (sel) {
        body[key] = sel.value;
      }
    } else {
      const inp = form.querySelector(`input[data-key="${key}"]`);
      if (inp) {
        let v = inp.value;
        if (f.type === "number") v = v === "" ? 0 : Number(v);
        body[key] = v;
      }
    }
  }
  return body;
}

function renderGeneratingBanner(payload) {
  const ban = document.getElementById("generating-banner");
  if (!ban) return;
  const llm = !!payload.llm_generating;
  const tts = !!payload.tts_generating;
  const ph = payload.speech_phase || "";
  const busy = llm || tts || ph === "queued";
  if (!busy) {
    ban.hidden = true;
    ban.textContent = "";
    return;
  }
  ban.hidden = false;
  const parts = [];
  if (ph === "queued") parts.push("佇列等待處理");
  if (llm) parts.push("Ollama LLM 生成中");
  if (tts) parts.push("TTS 語音合成中");
  ban.textContent = `進行中：${parts.join(" · ")}`;
}

function formatGpuMetrics(g) {
  if (!g || typeof g !== "object") return "";
  const rows = [];
  const nv = g.nvidia_smi;
  if (nv) {
    const bits = [];
    if (nv.utilization_gpu_pct != null) bits.push(`GPU 利用率 ${nv.utilization_gpu_pct}%`);
    if (nv.memory_used_mb != null && nv.memory_total_mb != null) {
      bits.push(`顯存 ${nv.memory_used_mb} / ${nv.memory_total_mb} MiB`);
    }
    if (nv.power_draw_w != null) bits.push(`功耗約 ${nv.power_draw_w} W`);
    if (nv.power_limit_w != null) bits.push(`上限 ${nv.power_limit_w} W`);
    if (nv.temperature_c != null) bits.push(`溫度 ${nv.temperature_c}°C`);
    rows.push(`<span class="gpu-kv"><strong>NVIDIA</strong>：${bits.join(" · ")}</span>`);
  }
  const mps = g.torch_mps;
  if (mps) {
    const bits = [`MPS 已配置記憶體 ~${mps.current_allocated_mb} MiB`];
    if (mps.driver_allocated_mb != null) bits.push(`driver ~${mps.driver_allocated_mb} MiB`);
    rows.push(`<span class="gpu-kv"><strong>Apple MPS（本行程）</strong>：${bits.join(" · ")}</span>`);
  }
  const cuda = g.torch_cuda;
  if (cuda) {
    rows.push(
      `<span class="gpu-kv"><strong>CUDA</strong>：${cuda.device_name} · alloc ${cuda.allocated_mb} MiB · reserved ${cuda.reserved_mb} MiB</span>`,
    );
  }
  const src = (g.sources || []).join(", ") || "—";
  let note = g.note || "";
  if (g.error) note = (note ? `${note} ` : "") + String(g.error);
  return { rows, src, note };
}

function renderGpuMetrics(payload) {
  const el = document.getElementById("gpu-metrics");
  if (!el) return;
  const g = payload.gpu_metrics || {};
  const vd = payload.vision_device || "—";
  const { rows, src, note } = formatGpuMetrics(g);
  const inner =
    rows.length > 0
      ? `<div class="gpu-row">${rows.join("")}</div>`
      : `<div class="gpu-row"><span class="gpu-kv">尚無 GPU 計量（見下方說明）</span></div>`;
  const noteHtml = note ? `<p class="gpu-note">${note}</p>` : "";
  el.innerHTML = `<div class="gpu-title">運算 / GPU</div>
    <div class="gpu-row"><span class="gpu-kv"><strong>YOLO 裝置</strong>：${vd}</span>
    <span class="gpu-kv"><strong>資料來源</strong>：${src}</span></div>
    ${inner}
    ${noteHtml}`;
}

function renderPipeline(payload) {
  const el = document.getElementById("pipeline-status");
  if (!el) return;
  const ph = payload.speech_phase || "idle";
  const mode = payload.speech_mode || "none";
  const sn = Number(payload.speech_sentence) || 0;

  const llmLabel =
    mode === "tracking" && sn > 0
      ? `LLM（第 ${sn} 句）`
      : mode === "idle"
        ? "LLM（閒置）"
        : "LLM";

  const phaseOrder = { queued: 0, llm: 1, tts: 2, playing: 3 };
  const cur = phaseOrder[ph] ?? -1;
  const stepClass = (stepName) => {
    const i = phaseOrder[stepName];
    if (i === undefined || cur < 0) return "wait";
    if (i < cur) return "done";
    if (i === cur) return "active";
    return "wait";
  };

  if (ph === "idle" && mode === "none") {
    el.innerHTML = `<strong>生成流程</strong>：<span class="step wait">待機</span>
      <p class="pipeline-hint">尚未排隊。須為 LOCK、滿足 post_lock_tts_delay_ms、畫面清晰（未 suppress_tts）才會排入；若曾中斷，請確認狀態表 sentence_index 有變化。</p>`;
    return;
  }

  const qCls = stepClass("queued");
  const lCls = stepClass("llm");
  const tCls = stepClass("tts");
  const pCls = stepClass("playing");

  let hint = "";
  if (ph === "queued") {
    hint =
      mode === "tracking"
        ? `佇列中：即將處理 ${llmLabel}`
        : "佇列中：即將產出閒置台詞";
  } else if (ph === "llm") {
    hint = "正在呼叫 Ollama（可能含首次 pull 模型，請稍候）…";
  } else if (ph === "tts") {
    hint = "正在合成語音…";
  } else if (ph === "playing") {
    hint = "播音中；結束後才會排下一句。";
  }

  el.innerHTML = `<strong>生成流程</strong>：
    <span class="step ${qCls}">排隊</span><span class="arrow">→</span>
    <span class="step ${lCls}">${llmLabel}</span><span class="arrow">→</span>
    <span class="step ${tCls}">TTS</span><span class="arrow">→</span>
    <span class="step ${pCls}">播放</span>
    <p class="pipeline-hint">${hint}</p>`;
}

function renderState(payload) {
  renderGeneratingBanner(payload);
  renderGpuMetrics(payload);
  renderPipeline(payload);
  const tbody = document.querySelector("#state-table tbody");
  tbody.innerHTML = "";
  const rows = [
    ["state", payload.state],
    ["bbox_value", payload.bbox_value?.toFixed?.(1) ?? payload.bbox_value],
    ["bbox", JSON.stringify(payload.bbox)],
    ["audience_features", payload.audience_features],
    ["sentence_index", payload.sentence_index],
    ["speech_phase / mode / 句", `${payload.speech_phase} / ${payload.speech_mode} / ${payload.speech_sentence}`],
    ["llm_generating", payload.llm_generating],
    ["tts_generating", payload.tts_generating],
    ["last_llm_ms", payload.last_llm_ms],
    ["last_tts_ms", payload.last_tts_ms],
    ["angle_left_deg", payload.angle_left_deg?.toFixed?.(1)],
    ["angle_right_deg", payload.angle_right_deg?.toFixed?.(1)],
    ["mode_servo", payload.mode_servo],
    ["audio_backend_active", payload.audio_backend_active],
    ["tts_backend", payload.tts_backend],
    ["behavior_tags", (payload.behavior_tags || []).join(", ")],
    ["vision_device", payload.vision_device],
    ["gpu_metrics", JSON.stringify(payload.gpu_metrics || {})],
  ];
  for (const [k, v] of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${k}</td><td>${v}</td>`;
    tbody.appendChild(tr);
  }
}

const historyEl = document.getElementById("history");
let historyLines = [];

function pushHistory(line) {
  historyLines.push(`${new Date().toLocaleTimeString()} — ${line}`);
  historyEl.innerHTML = historyLines.map((l) => `<li>${l}</li>`).join("");
}

function clearHistory() {
  historyLines = [];
  historyEl.innerHTML = "";
}

async function init() {
  await loadExamples();
  await loadSchema();
  const cfg = await loadConfig();
  buildForm(cfg);

  document.getElementById("btn-update").addEventListener("click", async () => {
    const body = collectForm();
    const r = await fetch(`${API}/api/config`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!j.ok) alert(j.error || "設定失敗");
    else pushHistory("設定已套用");
  });

  document.getElementById("btn-defaults").addEventListener("click", () => {
    buildForm(defaults);
  });

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${location.host}/ws`;
  const ws = new WebSocket(wsUrl);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "round_reset") {
      clearHistory();
      pushHistory("round_reset");
      return;
    }
    if (msg.type === "state") renderState(msg.payload);
  };

  const img = document.getElementById("mjpeg");
  img.src = `${API}/api/stream/mjpeg?t=${Date.now()}`;
}

init().catch(console.error);

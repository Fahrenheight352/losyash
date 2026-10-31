#!/usr/bin/env python3
# ============================================================
#  Лосяш-Тьютор — голосовой контур (этап 1)
#  Микрофон → faster-whisper → LLM (Ollama/Kimi) → Piper TTS
#  Управление: Enter — говорить, "q" — выход
# ============================================================
import os, json, time, subprocess, tempfile
from datetime import datetime
from pathlib import Path

# ============ НАСТРОЙКИ (меняй здесь или через переменные окружения) ============
BACKEND   = os.getenv("LOS_BACKEND", "ollama")   # "ollama" (бесплатно) или "kimi"
KIMI_API_KEY  = os.getenv("KIMI_API_KEY", "")    # ключ с platform.moonshot.ai
KIMI_MODEL    = os.getenv("KIMI_MODEL", "kimi-k3-pro")  # актуальный ID смотри в консоли
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
OLLAMA_MODEL  = os.getenv("LOS_OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

WHISPER_MODEL = os.getenv("LOS_WHISPER", "small")   # tiny/base/small/medium
DEVICE = "cuda" if os.getenv("LOS_CUDA", "1") == "1" else "cpu"  # 0 — если без NVIDIA

PIPER_VOICE = os.getenv("LOS_PIPER_VOICE", "ru_RU-dmitri-medium.onnx")  # полный путь к .onnx
USE_TTS = os.getenv("LOS_TTS", "1") == "1"

SILENCE_SEC = 1.2      # столько тишины = фраза закончена
MAX_REC_SEC = 20       # лимит записи
ENERGY = 0.02          # порог громкости RMS (подбери: 0.005 тихая комната, 0.05 шумно)
HISTORY_TURNS = 12     # память диалога (пар реплик)
# ================================================================================

BASE = Path(__file__).parent
LOG = BASE / "losyash_log.jsonl"      # журнал для родителя — читай, чему учат
PROMPT = (BASE / "prompt_losyash.txt").read_text(encoding="utf-8")

history: list[dict] = []

# ---------- STT: faster-whisper ----------
from faster_whisper import WhisperModel
print(f"[1/4] Загружаю Whisper ({WHISPER_MODEL}, {DEVICE})...")
whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")

# ---------- LLM: OpenAI-совместимый клиент ----------
from openai import OpenAI
if BACKEND == "kimi":
    assert KIMI_API_KEY, "Задай KIMI_API_KEY (переменная окружения)"
    client = OpenAI(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL)
    MODEL = KIMI_MODEL
else:
    client = OpenAI(api_key="ollama", base_url=OLLAMA_URL)
    MODEL = OLLAMA_MODEL
print(f"[2/4] LLM: {BACKEND} / {MODEL}")

# ---------- Запись с микрофона (push-to-talk, VAD по энергии) ----------
import numpy as np, sounddevice as sd, soundfile as sf

def record():
    print("🎤 Слушаю... (говори)")
    frames, started, quiet_since = [], False, None
    t0 = time.time()
    # Используем проверенный DirectSound (device=6), который успешно стартует
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=512, device=6) as stream:
        # Даем аудио-движку Windows 100 мс на стабилизацию потока перед чтением
        time.sleep(0.1) 
        
        while True:
            try:
                block, overflowed = stream.read(512)
                # Если буфер пустой или не инициализировался, пропускаем шаг
                if block is None or len(block) == 0:
                    time.sleep(0.01)
                    continue
            except Exception:
                # Мягко пропускаем системные сбои чтения буфера
                time.sleep(0.01)
                continue

            rms = float(np.sqrt(np.mean(block**2)))
            print(f"Громкость: {rms:.5f} | Порог (ENERGY): {ENERGY}", end="\r")
            now = time.time()
            
            if rms > ENERGY:
                started, quiet_since = True, None
            elif started:
                quiet_since = quiet_since or now
                if now - quiet_since > SILENCE_SEC or now - t0 > MAX_REC_SEC:
                    break
                    
            if now - t0 > MAX_REC_SEC: 
                break
                
            if started: 
                frames.append(block)
            if not started: 
                t0 = now  # не считаем тишину в лимит

# ---------- LLM-ответ ----------
def ask(user_text):
    history.append({"role": "user", "content": user_text})
    msgs = [{"role": "system", "content": PROMPT}] + history[-HISTORY_TURNS*2:]
    r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0.7, max_tokens=400)
    ans = r.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": ans})
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": datetime.now().isoformat(), "user": user_text, "losyash": ans}, ensure_ascii=False) + "\n")
    return ans

# ---------- TTS: Piper ----------
def speak(text):
    if not USE_TTS:
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav = tf.name
    subprocess.run(["piper", "--model", PIPER_VOICE, "--output_file", wav],
                   input=text.encode("utf-8"), check=True)
    data, sr = sf.read(wav)
    sd.play(data, sr); sd.wait()
    os.unlink(wav)

print("[3/4] Готов. Нажми Enter и говори. \"q\" — выход.\n")
while True:
    cmd = input("── Enter: говорить │ q: выход ──\n")
    if cmd.strip().lower() == "q": break
    audio = record()
    if audio is None or len(audio) < 16000:
        print("(не расслышал)\n"); continue
    print("[4/4] Думаю...")
    text = " ".join(seg.text for seg in whisper.transcribe(audio, language="ru")[0]).strip()
    print(f"Ребёнок: {text}")
    answer = ask(text)
    print(f"Лосяш:   {answer}\n")
    speak(answer)
print("Пока! Загляни в losyash_log.jsonl — там вся беседа.")

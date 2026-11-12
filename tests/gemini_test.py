"""Uji coba Gemini 3.5 Transcribe dan Transcribe Live untuk aplikasi transkrip.

Dijalankan oleh GitHub Actions (.github/workflows/gemini-test.yml).
Input:
  samples/*.m4a|mp3|wav|ogg|aac|opus  -> rekaman uji (pakai 1 file pertama)
  samples/glossary.txt                -> satu istilah per baris (opsional)
Output:
  results/<waktu>/report.md + file JSON mentah
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

from google import genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".ogg", ".aac", ".opus", ".flac", ".webm"}
LANGS = ["id-ID", "jv-ID", "en-US", "ja-JP"]
UNARY_MODEL = "gemini-3.5-transcribe"
LIVE_MODEL = "gemini-3.5-transcribe-live"
UNARY_MAX_SEC = 360  # 10K TPM / 25 token per detik = ~6,6 menit
UNARY_GAP_SEC = 25   # batas 3 request per menit
LIVE_MAX_SEC = 540   # sesi live maksimal 10 menit

stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%Y%m%d-%H%M%S")
OUT = ROOT / "results" / stamp
OUT.mkdir(parents=True, exist_ok=True)
report: list[str] = [f"# Hasil uji Gemini Transcribe ({stamp} WIB)", ""]


def log(line: str = "") -> None:
    print(line, flush=True)
    report.append(line)


def save_json(name: str, data) -> None:
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def to_dict(obj):
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn(mode="json") if attr == "model_dump" else fn()
            except TypeError:
                return fn()
    return obj


def ffmpeg(src: pathlib.Path, dst: pathlib.Path, seconds: int, fmt: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(seconds),
         "-ac", "1", "-ar", "16000", *fmt, str(dst)],
        check=True,
    )


def duration(src: pathlib.Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(src)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def finish(code: int = 0) -> None:
    (OUT / "report.md").write_text("\n".join(report) + "\n")
    sys.exit(code)


# ---------- persiapan ----------
key = os.environ.get("GEMINI_API_KEY", "").strip()
if not key:
    log("**Gagal:** secret `GEMINI_API_KEY` belum diisi di Settings → Secrets and variables → Actions.")
    finish(0)

audio_files = sorted(p for p in SAMPLES.glob("*") if p.suffix.lower() in AUDIO_EXT)
if not audio_files:
    log("**Belum ada rekaman uji** di folder `samples/`.")
    finish(0)

src = audio_files[0]
glossary_file = SAMPLES / "glossary.txt"
glossary = [l.strip() for l in glossary_file.read_text().splitlines() if l.strip()] if glossary_file.exists() else []
total = duration(src)
log(f"- Rekaman: `{src.name}` ({total:.0f} detik)")
log(f"- Glosarium: {len(glossary)} istilah" + (f" ({', '.join(glossary[:15])})" if glossary else ""))
log(f"- SDK google-genai: {getattr(genai, '__version__', '?')}")
log("")

wav = OUT / "unary.wav"
ffmpeg(src, wav, UNARY_MAX_SEC, ["-c:a", "pcm_s16le"])
pcm = OUT / "live.pcm"
ffmpeg(src, pcm, LIVE_MAX_SEC, ["-f", "s16le", "-c:a", "pcm_s16le"])
unary_sec = min(total, UNARY_MAX_SEC)
if total > UNARY_MAX_SEC:
    log(f"> Rekaman dipotong ke {UNARY_MAX_SEC} detik untuk tes non-live (batas 10K token per menit).")
    log("")

client = genai.Client(api_key=key)
audio_input = [{
    "type": "audio",
    "data": base64.b64encode(wav.read_bytes()).decode("ascii"),
    "mime_type": "audio/wav",
}]


# ---------- tes non-live ----------
def unary(label: str, tconfig: dict) -> None:
    log(f"## Non-live: {label}")
    log(f"Config: `{json.dumps(tconfig)}`")
    t0 = time.time()
    try:
        resp = client.interactions.create(
            model=UNARY_MODEL,
            input=audio_input,
            generation_config={"transcription_config": tconfig},
        )
    except Exception as e:  # noqa: BLE001
        log(f"**Error:** `{type(e).__name__}: {str(e)[:800]}`")
        log("")
        return
    took = time.time() - t0
    data = to_dict(resp)
    save_json(f"unary-{label}.json", data)
    text = getattr(resp, "output_text", None) or ""
    log(f"Waktu proses: {took:.1f} detik untuk {unary_sec:.0f} detik audio")
    usage = data.get("usage") if isinstance(data, dict) else None
    if usage:
        log(f"Usage: `{json.dumps(usage)}`")
    # susun per pembicara dari anotasi kata
    turns: list[list] = []
    try:
        for step in resp.steps or []:
            for content in getattr(step, "content", None) or []:
                for ann in getattr(content, "annotations", None) or []:
                    spk = getattr(ann, "speaker", None)
                    word = getattr(ann, "text", "") or ""
                    start = getattr(ann, "start_offset", None)
                    if spk is None:
                        continue
                    if turns and turns[-1][0] == spk:
                        turns[-1][2].append(word)
                    else:
                        turns.append([spk, start, [word]])
    except Exception:  # noqa: BLE001
        log(f"(gagal membaca anotasi: {traceback.format_exc(limit=1)})")
    if turns:
        log("")
        log("Per pembicara:")
        log("")
        for spk, start, words in turns:
            log(f"- **{spk}** [{start}]: {' '.join(words)}")
    log("")
    log("Teks lengkap:")
    log("")
    log("> " + text.replace("\n", "\n> "))
    log("")


unary_tests = [
    ("diarization", {"mode": {"type": "verbatim", "diarization_mode": "speaker"}, "language_codes": LANGS}),
    ("glosarium", {"mode": {"type": "verbatim"}, "language_codes": LANGS, "custom_vocabulary": glossary}),
    ("auto-bahasa", {"mode": {"type": "verbatim"}}),
]
if not glossary:
    unary_tests = [t for t in unary_tests if t[0] != "glosarium"]

for i, (label, cfg) in enumerate(unary_tests):
    if i:
        time.sleep(UNARY_GAP_SEC)
    unary(label, cfg)


# ---------- tes live ----------
CHUNK = 3200  # 100 ms PCM 16 kHz mono 16-bit


async def live(speed: float, seconds: float) -> None:
    label = f"{speed:g}x"
    log(f"## Live: kecepatan {label} ({seconds:.0f} detik audio)")
    data = pcm.read_bytes()[: int(seconds * 16000 * 2)]
    tcfg = {"language_codes": LANGS}
    if glossary:
        tcfg["custom_vocabulary"] = glossary
    variants = [
        {"input_audio_transcription": tcfg},
        {"input_audio_transcription": tcfg, "response_modalities": ["TEXT"]},
    ]
    last_err = None
    for cfg in variants:
        finals: list[tuple[float, str]] = []
        interim_count = 0
        first_at = None
        msgs_seen: list = []
        try:
            async with client.aio.live.connect(model=LIVE_MODEL, config=cfg) as session:
                t0 = time.time()

                async def sender():
                    for off in range(0, len(data), CHUNK):
                        await session.send_realtime_input(
                            audio={"data": data[off:off + CHUNK], "mime_type": "audio/pcm;rate=16000"}
                        )
                        await asyncio.sleep(0.1 / speed)
                    await session.send_realtime_input(audio_stream_end=True)

                async def receiver():
                    nonlocal interim_count, first_at
                    async for msg in session.receive():
                        if len(msgs_seen) < 15:
                            msgs_seen.append(to_dict(msg))
                        sc = getattr(msg, "server_content", None)
                        if not sc:
                            continue
                        it = getattr(sc, "input_transcription", None)
                        if it and getattr(it, "text", None):
                            first_at = first_at or time.time() - t0
                            finals.append((time.time() - t0, it.text))
                        iit = getattr(sc, "interim_input_transcription", None)
                        if iit and getattr(iit, "text", None):
                            interim_count += 1
                            first_at = first_at or time.time() - t0

                send_task = asyncio.create_task(sender())
                recv_task = asyncio.create_task(receiver())
                await send_task
                sent_done = time.time() - t0
                try:
                    await asyncio.wait_for(recv_task, timeout=20)
                except asyncio.TimeoutError:
                    recv_task.cancel()
                total_t = time.time() - t0
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:600]}"
            save_json(f"live-{label}-error-{variants.index(cfg)}.json", {"config": cfg, "error": last_err, "msgs": msgs_seen})
            continue
        save_json(f"live-{label}.json", {"config": cfg, "finals": finals, "msgs_sample": msgs_seen})
        log(f"Config yang berhasil: `{json.dumps(cfg)}`")
        log(f"Teks pertama muncul: {first_at if first_at is None else round(first_at, 1)} detik · "
            f"kirim audio selesai: {sent_done:.1f} detik · total: {total_t:.1f} detik · "
            f"update sementara: {interim_count} · potongan final: {len(finals)}")
        text = " ".join(t for _, t in finals).strip()
        log("")
        log("> " + (text or "(tidak ada teks)"))
        log("")
        return
    log(f"**Error di semua variasi config:** `{last_err}`")
    log("")


live_sec = min(total, LIVE_MAX_SEC)


async def run_live():
    await live(1, min(live_sec, 60))
    for s in (2, 5, 10):
        await live(s, live_sec)


try:
    asyncio.run(run_live())
except Exception:  # noqa: BLE001
    log(f"**Error live:** ```{traceback.format_exc(limit=3)}```")

finish(0)

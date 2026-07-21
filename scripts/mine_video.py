#!/usr/bin/env python
"""Phase-2 data engine (skeleton): curated video -> paired (text, SMPL-X motion).

Pipeline per URL:
  1. yt-dlp downloads video + audio
  2. GVHMR (third_party/GVHMR, own setup — see below) extracts world-grounded
     SMPL-X motion from the video
  3. faster-whisper transcribes the audio (word timestamps)
  4. Aligned (transcript, poses, trans) shards written to data/mined/

Setup (once):
  git clone https://github.com/zju3dv/GVHMR third_party/GVHMR
  # follow its INSTALL.md in a dedicated venv (heavy: pytorch3d etc.)

Status: SKELETON — download+transcribe implemented; the GVHMR invocation is a
stub until its environment is set up (phase 2).
"""
import argparse
import subprocess
import sys
from pathlib import Path

OUT = Path("data/mined")
GVHMR_DIR = Path("third_party/GVHMR")


def download(url: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(workdir / "%(id)s.%(ext)s")
    subprocess.run([sys.executable, "-m", "yt_dlp", "-f", "mp4", "-o", out_tpl, url], check=True)
    vids = sorted(workdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    return vids[-1]


def transcribe(video: Path) -> list:
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="auto")
    segments, _ = model.transcribe(str(video), word_timestamps=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append((w.start, w.end, w.word.strip()))
    return words


def run_gvhmr(video: Path) -> Path:
    if not GVHMR_DIR.is_dir():
        raise SystemExit(
            f"GVHMR not found at {GVHMR_DIR}.\n"
            "Phase 2 setup: git clone https://github.com/zju3dv/GVHMR third_party/GVHMR "
            "and follow its INSTALL.md (dedicated venv).")
    raise NotImplementedError("GVHMR invocation lands in phase 2 once its env is set up.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--workdir", default="data/mined/raw")
    args = p.parse_args()

    video = download(args.url, Path(args.workdir))
    print(f"downloaded: {video}")
    words = transcribe(video)
    print(f"transcribed {len(words)} words")
    run_gvhmr(video)  # -> will produce poses/trans; alignment + shard writing then


if __name__ == "__main__":
    main()

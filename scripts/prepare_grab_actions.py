#!/usr/bin/env python
"""Turn converted GRAB sequences into a captioned action dataset for the flow
model's ACTION slot (finger-articulated hand actions).

GRAB filenames encode object + action:
    GRAB_s4_camera_takepicture_1_stageii.npz -> "takes a picture with a camera"
    GRAB_s10_mug_drink_1_stageii.npz         -> "drinks from a mug"

Output (same layout HumanML3DDataset reads):
    data/grab_actions/motions/{id}.npz   (copied poses/trans)
    data/grab_actions/texts/{id}.txt     (one caption per line)
"""
import re
import shutil
from pathlib import Path

import numpy as np

SRC = Path("data/amass/motions")
OUT = Path("data/grab_actions")

# action verb -> caption template ({o} = object)
VERB = {
    "pass": "passes a {o} with the hand",
    "pick": "picks up a {o}",
    "lift": "lifts a {o}",
    "drink": "drinks from a {o}",
    "use": "uses a {o} with the hand",
    "eat": "eats a {o}",
    "pour": "pours from a {o}",
    "inspect": "holds and inspects a {o}",
    "see": "holds up a {o} to look at it",
    "peel": "peels a {o} with both hands",
    "on": "turns on a {o}",
    "off": "turns off a {o}",
    "staple": "staples with a stapler",
    "stamp": "stamps with a stamp",
    "squeeze": "squeezes a {o}",
    "set": "sets a {o} with the fingers",
    "fly": "flies a toy {o} in the hand",
    "cook": "cooks with a {o}",
    "takepicture": "takes a picture with a camera",
    "call": "makes a call on a {o}",
    "play": "plays with a {o}",
    "open": "opens a {o}",
    "browse": "browses on a {o}",
    "clean": "wipes a {o} with the hand",
    "switch": "flips a switch on a {o}",
    "shake": "shakes a {o} in the hand",
    "shuffle": "shuffles a {o} in the hands",
    "shout": "gestures while holding a {o}",
    "brush": "brushes with a {o}",
    "wear": "puts on a {o}",
    "toast": "raises a {o} in a toast",
    "cut": "cuts with a {o}",
    "screw": "screws a {o} with the fingers",
    "spray": "sprays a {o}",
    "chop": "chops with a {o}",
    "drive": "grips and turns a {o}",
    "handover": "hands over a {o}",
    "counting": "counts on the fingers",
    "point": "points with a finger",
}
STRIP = {"all", "retake", "1", "2", "stageii"}


def caption(stem: str) -> str:
    # GRAB_s10_camera_takepicture_1 -> tokens after speaker id
    body = re.sub(r"^GRAB_s\d+_", "", stem)
    toks = [t for t in body.split("_") if t.lower() not in STRIP and not t.isdigit()]
    if not toks:
        return "handles an object with the hands"
    action = toks[-1].lower()
    obj = " ".join(toks[:-1]).lower() or "object"
    tmpl = VERB.get(action, "handles a {o} with the hands")
    return tmpl.format(o=obj)


def main():
    (OUT / "motions").mkdir(parents=True, exist_ok=True)
    (OUT / "texts").mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("GRAB_*.npz"))
    n = 0
    verbs_seen = {}
    for p in files:
        stem = p.stem.replace("_stageii", "")
        cap = caption(stem)
        cid = stem
        # copy motion (poses/trans) — keep as-is
        with np.load(p) as d:
            if "poses" not in d or d["poses"].shape[0] < 16:
                continue
            np.savez(OUT / "motions" / f"{cid}.npz",
                     poses=d["poses"].astype(np.float32),
                     trans=d["trans"].astype(np.float32),
                     mocap_frame_rate=30)
        (OUT / "texts" / f"{cid}.txt").write_text(cap + "\n")
        verbs_seen[cap.split()[0]] = verbs_seen.get(cap.split()[0], 0) + 1
        n += 1
    print(f"wrote {n} GRAB action clips -> {OUT}")
    print("sample captions:")
    for t in sorted((OUT / "texts").glob("*.txt"))[:6]:
        print(f"  {t.stem}: {t.read_text().strip()}")


if __name__ == "__main__":
    main()

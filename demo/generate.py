#!/usr/bin/env python
"""Generate motion from text and/or an action prompt, render to mp4.

  .venv/bin/python demo/generate.py --text "hola, ¿cómo estás?" --duration 3
  .venv/bin/python demo/generate.py --action "waves hand excitedly" --intensity 1.5
"""
import argparse

from motionlab.pipeline import MotionPipeline
from motionlab.viz.render import render_motion
from motionlab.data.beat2 import EMOTIONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="", help="transcript slot (what she is saying)")
    p.add_argument("--action", default="", help="action slot (what she should do)")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--emotion", default="neutral", choices=EMOTIONS)
    p.add_argument("--intensity", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--vae", default="runs/vae/latest.pt")
    p.add_argument("--flow", default="runs/flow/latest.pt")
    p.add_argument("--out", default="demo_out.mp4")
    args = p.parse_args()

    if not args.text and not args.action:
        p.error("provide --text and/or --action")

    pipe = MotionPipeline(args.vae, args.flow)
    result, _ = pipe.generate(
        text=args.text, action=args.action, duration_s=args.duration,
        emotion=EMOTIONS.index(args.emotion), intensity=args.intensity, steps=args.steps)
    print(f"generated {result.poses.shape[0]} frames @ {result.fps}fps")
    render_motion(result.poses, result.trans, args.out, fps=result.fps)
    print(f"rendered: {args.out}")


if __name__ == "__main__":
    main()

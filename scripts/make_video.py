"""Compose the side-by-side demo: one episode, three execution modes, the same mid-episode cube move.

Reads the per-mode episode videos written by chunk_eval.py (--videos-per-condition 1) and the
episode records next to them, adds a header per panel, a time counter, a "cube moved" marker and
the outcome, and encodes an MP4 (and a GIF for the README) with the ffmpeg CLI.

Usage:
    python scripts/make_video.py --run outputs/video_1018 --seed 1018 --out results/video/seed1018_6cm
"""

import argparse
import json
import subprocess
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANELS = (
    ("full_d00_m6", "Full chunk", "2 s open loop"),
    ("replan25_d00_m6", "Replan every 25 steps", "0.5 s"),
    ("te_d00_m6", "Temporal ensembling", "every step, averaged"),
)
FPS_IN, SLOWDOWN, HOLD_S = 50, 2, 2.0
BAR, TEXT, TEXT_2, ACCENT = (26, 26, 25), (255, 255, 255), (195, 194, 183), (250, 178, 25)
FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--title", default="ACT, our checkpoint (100k steps): the same episode, cube moved 6 cm")
    parser.add_argument("--out", type=Path, required=True, help="output path without extension")
    args = parser.parse_args()

    panels = []
    for cond, name, detail in PANELS:
        record = json.loads(next((args.run / cond).glob(f"seeds_{args.seed}-*.json")).read_text())
        episode = record["episodes"][0]
        clip = frames(Path(record["video_paths"][0]))
        success_step = episode["first_step_at_reward"].get("4")
        outcome = f"SUCCESS: handover done at {success_step / FPS_IN:.1f} s" if episode["success"] else (
            "FAILED: never touched the cube" if not episode["first_step_at_reward"] else "FAILED")
        panels.append({"name": name, "detail": detail, "frames": clip, "t_move": episode["t_move"],
                       "moved_cm": 100 * episode["moved_m"], "outcome": outcome, "success": episode["success"]})

    h, w = panels[0]["frames"][0].shape[:2]
    n = max(len(p["frames"]) for p in panels) + int(HOLD_S * FPS_IN)
    title_h, head_h, foot_h = 56, 64, 44
    size = (w * len(panels), title_h + head_h + h + foot_h)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mp4 = args.out.with_suffix(".mp4")
    fps_out = FPS_IN / SLOWDOWN
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
         "-r", f"{fps_out:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)],
        stdin=subprocess.PIPE)
    f_title, f_head, f_small, f_badge = font(26), font(24), font(18), font(22)
    for i in range(n):
        canvas = Image.new("RGB", size, BAR)
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 14), args.title, font=f_title, fill=TEXT)
        t = min(i, max(len(p["frames"]) for p in panels) - 1) / FPS_IN
        draw.text((size[0] - 240, 18), f"t = {t:4.2f} s  (0.5x speed)", font=f_small, fill=TEXT_2)
        for j, p in enumerate(panels):
            x0 = j * w
            draw.text((x0 + 16, title_h + 8), p["name"], font=f_head, fill=TEXT)
            draw.text((x0 + 16, title_h + 38), p["detail"], font=f_small, fill=TEXT_2)
            frame = p["frames"][min(i, len(p["frames"]) - 1)]
            canvas.paste(Image.fromarray(frame), (x0, title_h + head_h))
            if p["t_move"] is not None and p["t_move"] <= i < p["t_move"] + FPS_IN:  # 1 s marker after the move
                draw.rectangle([x0 + 12, title_h + head_h + 12, x0 + 262, title_h + head_h + 46], fill=BAR)
                draw.text((x0 + 22, title_h + head_h + 17), f"cube moved {p['moved_cm']:.0f} cm", font=f_badge, fill=ACCENT)
            ended = i >= len(p["frames"]) - 1
            if ended or i >= n - int(HOLD_S * FPS_IN):
                draw.text((x0 + 16, title_h + head_h + h + 10), p["outcome"], font=f_badge, fill=TEXT)
        enc.stdin.write(np.asarray(canvas).tobytes())
    enc.stdin.close()
    enc.wait()
    gif = args.out.with_suffix(".gif")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=12,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer",
                    str(gif)], check=True)
    print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB) and {gif} ({gif.stat().st_size / 1e6:.1f} MB), {n} frames")


if __name__ == "__main__":
    main()

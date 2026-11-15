"""Render the 30-second demo video from real data products only (no generated imagery).

Segments (owner's script): reveal slider 0–6 s; raw vs corrected blink 6–13 s; object page with
check-it 13–20 s; measured numbers 20–26 s; name and address 26–30 s. English subtitles are burned in
and also written as an .srt file. Every number shown is read from site/data/findings.json.
Usage: .venv/bin/python scripts/make_video.py --url "https://<github-user>.github.io/orrery/"
       [--field ecliptic_control] [--object-shot path.png]
Outputs: dist/orrery_demo.mp4, dist/orrery_demo.srt (length verified with ffprobe).
"""
import argparse, json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
BG = (8, 12, 24)
FG = (238, 242, 250)
ACC = (86, 180, 233)
ORG = (230, 159, 0)
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
SITE = "site/data"


def font(size, bold=True):
    return ImageFont.truetype(BOLD if bold else REG, size)


def canvas():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def subtitle(d, text):
    f = font(40)
    w = d.textlength(text, font=f)
    x, y = (W - w) / 2, H - 92
    d.rounded_rectangle([x - 22, y - 12, x + w + 22, y + 54], 12, fill=(0, 0, 0))
    d.text((x, y), text, font=f, fill=FG)


def tag(d, x, y, text, col):
    f = font(18)
    w = d.textlength(text, font=f)
    d.rounded_rectangle([x, y, x + w + 16, y + 28], 6, fill=col)
    d.text((x + 8, y + 4), text, font=f, fill=(0, 0, 0))


def load(path, size):
    return Image.open(os.path.join(SITE, path)).convert("RGB").resize(size, Image.NEAREST)


def seg_reveal(rev, frames_out, n_frames):
    singles = rev["frames"]
    stacks = rev["stack_pngs"]
    N = rev["n_frames"]
    for k in range(n_frames):
        t = k / n_frames
        im, d = canvas()
        si = singles[int(t * (len(singles) - 1))]
        n_show = max(1, int(round(1 + t ** 1.6 * (N - 1))))
        st = [s for s in stacks if s["n"] <= n_show][-1]
        im.paste(load(si["single_png"], (420, 420)), (130, 110))
        im.paste(load(st["png"], (420, 420)), (730, 110))
        f = font(30)
        d.text((130, 62), "One picture", font=f, fill=FG)
        d.text((730, 62), f"{st['n']} pictures combined", font=f, fill=FG)
        tag(d, 130, 540, "DATA", (0, 158, 115))
        tag(d, 730, 540, "DERIVED", (0, 158, 115))
        subtitle(d, "Too faint for any single picture" if t < 0.5 else "Combine them along its path: there it is")
        frames_out.append(im)


def seg_blink(field, frames_out, n_frames):
    fr = [f for f in field["frames"] if f.get("corr_png")]
    lab = [f for f in fr if f.get("labels")]
    seq = (lab if len(lab) >= 6 else fr)[:24]
    for k in range(n_frames):
        f = seq[(k // 8) % len(seq)]
        im, d = canvas()
        im.paste(load(f["img_png"], (420, 420)), (130, 110))
        im.paste(load(f["corr_png"], (420, 420)), (730, 110))
        sc = 420 / field["size_px"]
        for L in f.get("labels", []):
            if L.get("label") == "known asteroid":
                x, y = 730 + (L["x"] + 0.5) * sc, 110 + (field["size_px"] - L["y"] - 0.5) * sc
                d.ellipse([x - 18, y - 18, x + 18, y + 18], outline=ACC, width=3)
        f30 = font(30)
        d.text((130, 62), "Plain blink", font=f30, fill=FG)
        d.text((730, 62), "Colour-corrected", font=f30, fill=FG)
        d.text((130, 540), f"{f['date']}   {f['wl_um']:.2f} µm", font=font(22, False), fill=FG)
        tag(d, 730, 540, "MODEL / RESIDUAL", (204, 121, 167))
        subtitle(d, "Raw: colour shifts look like change" if k < n_frames / 2 else "Corrected: only real movers stay")
        frames_out.append(im)


def seg_object(shot, frames_out, n_frames, line):
    base = Image.open(shot).convert("RGB")
    s = min(W / base.width, (H - 120) / base.height)
    base = base.resize((int(base.width * s), int(base.height * s)))
    for k in range(n_frames):
        im, d = canvas()
        im.paste(base, ((W - base.width) // 2, 10))
        subtitle(d, line)
        frames_out.append(im)


def seg_numbers(fnd, frames_out, n_frames):
    lines = []
    if "f2" in fnd:
        g = fnd["f2"]["detail"]
        lines.append((f"{10 ** (0.4 * g['depth_gain_mag']):.0f}×", "fainter than one picture can see"))
    if "f1" in fnd:
        lines.append((f"{fnd['f1']['number']}%", "of plain-blink 'changes' were only colour"))
    if "f3" in fnd:
        lines.append((f"{fnd['f3']['detail']['n']}", f"asteroid sizes, typical {fnd['f3']['number']}% from NEOWISE"))
    if "f4" in fnd:
        lines.append((f"V {float(fnd['f4']['number']):.1f}", "faintest planted test objects recovered (half the time)"))
    for k in range(n_frames):
        im, d = canvas()
        shown = min(len(lines), 1 + int(k / n_frames * len(lines) * 1.3))
        for i, (big, small) in enumerate(lines[:shown]):
            y = 70 + i * 135
            d.text((150, y), big, font=font(84), fill=ORG)
            d.text((520, y + 26), small, font=font(36, False), fill=FG)
        subtitle(d, "Measured, and checked against independent data")
        frames_out.append(im)


def seg_name(url, frames_out, n_frames):
    for k in range(n_frames):
        im, d = canvas()
        t = "Orrery"
        d.text(((W - d.textlength(t, font=font(120))) / 2, 170), t, font=font(120), fill=FG)
        s = "everything that moves, in 102 colors"
        d.text(((W - d.textlength(s, font=font(40, False))) / 2, 320), s, font=font(40, False), fill=ACC)
        d.text(((W - d.textlength(url, font=font(34))) / 2, 420), url, font=font(34), fill=FG)
        s2 = "NASA SPHEREx public data · Space Apps 2026"
        d.text(((W - d.textlength(s2, font=font(26, False))) / 2, 480), s2, font=font(26, False), fill=FG)
        subtitle(d, "Try any patch of sky")
        frames_out.append(im)


def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main(a):
    rev = json.load(open(f"{SITE}/reveal.json"))
    field = json.load(open(f"{SITE}/timemachine/{a.field}.json"))
    fnd = json.load(open(f"{SITE}/findings.json"))
    os.makedirs("dist/frames", exist_ok=True)
    frames = []
    seg_reveal(rev, frames, 6 * FPS)
    seg_blink(field, frames, 7 * FPS)
    seg_object(a.object_shot, frames, 7 * FPS, "Every number: check it against NEOWISE and Gaia")
    seg_numbers(fnd, frames, 6 * FPS)
    seg_name(a.url, frames, 4 * FPS - 6)  # finish safely under 30.0 s
    for i, im in enumerate(frames):
        im.save(f"dist/frames/{i:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", "dist/frames/%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "dist/orrery_demo.mp4"], check=True)
    subs = [(0, 3, "Too faint for any single picture"), (3, 6, "Combine them along its path: there it is"),
            (6, 9.5, "Raw: colour shifts look like change"), (9.5, 13, "Corrected: only real movers stay"),
            (13, 20, "Every number: check it against NEOWISE and Gaia"), (20, 26, "Measured, and checked against independent data"),
            (26, 29.7, "Try any patch of sky")]
    with open("dist/orrery_demo.srt", "w") as f:
        for i, (s, e, t) in enumerate(subs, 1):
            f.write(f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{t}\n\n")
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "dist/orrery_demo.mp4"]).decode())
    print("duration_s", dur)
    assert dur < 30.0, "video exceeds 30 s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="Orrery · SPHEREx Space Apps 2026")
    ap.add_argument("--field", default="ecliptic_control")
    ap.add_argument("--object-shot", default="dist/object_shot.png")
    main(ap.parse_args())

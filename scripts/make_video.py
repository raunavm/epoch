"""Render the 30-second demo video from real data products only (no generated imagery).

Segments: title card 0–2 s; plain blink vs one-detector change maps 2–7 s; WISE 0855 combined along its path
7–13 s; Barnard's Star, earlier minus later pictures 13–19 s; measured numbers, one at a time, 19–26 s; name and
address 26–29.8 s. English subtitles (SUBS, one list) are burned in and also written as an .srt file. Numbers are
read from site/data/headline.json, data/findings/e4.json, data/findings/motion_epsindib.json and
site/data/index/dr1/meta.json.
Usage: .venv/bin/python scripts/make_video.py --url "<github-user>.github.io/orrery" --team "<team-name>"
       [--field ecliptic_control]
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
MARGIN = 60
AI_NOTE = "Made from real SPHEREx data"
# the only subtitle list: burned into the frames and written to the .srt, so the two cannot differ
SUBS = [(0, 3.5, "NASA's SPHEREx shoots each spot in a new colour"),
        (3.5, 7, "So a plain blink flickers; one detector at a time removes most of it"),
        (7, 9.7, "The coldest known brown dwarf, picture by picture"), (9.7, 13, "Combine pictures along its path: there it is"),
        (13, 19, "About seven months apart: the star has moved"), (19, 26, "Measured, and checked against independent data"),
        (26, 29.7, "Try any spot in the sky")]


def font(size, bold=True):
    return ImageFont.truetype(BOLD if bold else REG, size)


def canvas():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def fit(d, text, size, bold=True, max_w=W - 2 * MARGIN):
    """Largest font up to `size` at which the text fits inside the frame margins."""
    while d.textlength(text, font=font(size, bold)) > max_w:
        size -= 2
    return font(size, bold)


def centred(d, y, text, size, fill, bold=True):
    f = fit(d, text, size, bold)
    w = d.textlength(text, font=f)
    assert MARGIN <= (W - w) / 2 and (W + w) / 2 <= W - MARGIN, text
    d.text(((W - w) / 2, y), text, font=f, fill=fill)


def subtitle(d, text):
    f = fit(d, text, 40, max_w=W - 2 * MARGIN - 44)
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


def seg_blink_vs_det(field_name, frames_out, n_frames):
    """Change maps from the frozen change module on the cached field (E2 rule, k = 5): left, each picture minus the
    previous picture in time (plain blink); right, minus the previous picture from the SAME detector. Flagged changes
    (≥ 2 pixels beyond 5σ) are circled. Diverging colours, clipped at ±5σ."""
    import sys as _s
    _s.path.insert(0, "src")
    from orrery.change import change_model, find_changes
    z = np.load(f"data/cache/fields/{field_name}_S48.npz", allow_pickle=True)
    f = json.load(open("data/findings/f1_calibration.json"))["f"]
    out = change_model(z["v"], z["lam"], z["detector"], z["mjd"], f)
    order = np.argsort(z["mjd"])
    pairs = [(out["naive_all"][e], out["naive_det"][e]) for e in order if out["naive_all"][e] is not None and out["naive_det"][e] is not None]
    def render(m):
        rgb = np.zeros(m.shape + (3,))
        x = np.clip(np.nan_to_num(m) / 5, -1, 1)
        pos, neg = np.clip(x, 0, 1), np.clip(-x, 0, 1)
        rgb += pos[..., None] * np.array([230, 159, 0]) + neg[..., None] * np.array([86, 180, 233])
        im = Image.fromarray(rgb.astype(np.uint8)[::-1]).resize((420, 420), Image.NEAREST)
        dd = ImageDraw.Draw(im)
        for c in find_changes(m):
            yx = np.array(c["pixels"] if isinstance(c, dict) and "pixels" in c else c, float).reshape(-1, 2) if not isinstance(c, dict) else None
            if isinstance(c, dict):
                cx, cy = c.get("x", c.get("cx")), c.get("y", c.get("cy"))
            else:
                cy, cx = yx[:, 0].mean(), yx[:, 1].mean()
            X, Y = (cx + 0.5) * 420 / 48, (48 - cy - 0.5) * 420 / 48
            dd.ellipse([X - 16, Y - 16, X + 16, Y + 16], outline=(255, 255, 255), width=2)
        return im, len(find_changes(m))
    tot_a = tot_d = 0
    for k in range(n_frames):
        ma, md = pairs[(k // 4) % len(pairs)]
        ia, na = render(ma); idd, nd = render(md)
        if k % 4 == 0:
            tot_a += na; tot_d += nd
        im, d = canvas()
        im.paste(ia, (130, 110)); im.paste(idd, (730, 110))
        d.text((130, 62), "Plain blink: what changed?", font=font(30), fill=FG)
        d.text((730, 62), "One detector at a time", font=font(30), fill=FG)
        d.text((130, 540), f"flagged so far: {tot_a}", font=font(24), fill=FG)
        d.text((730, 540), f"flagged so far: {tot_d}", font=font(24), fill=FG)
        tag(d, 400, 540, "DERIVED", (0, 158, 115)); tag(d, 1000, 540, "DERIVED", (0, 158, 115))
        frames_out.append(im)


def seg_title(team, frames_out, n_frames):
    for k in range(n_frames):
        im, d = canvas()
        centred(d, 250, f"Orrery · NASA SPHEREx · {team}", 64, FG)
        frames_out.append(im)


def seg_reveal(rev, frames_out, n_frames):
    singles, stacks, N = rev["frames"], rev["stack_pngs"], rev["n_frames"]
    for k in range(n_frames):
        t = k / n_frames
        im, d = canvas()
        si = singles[int(t * (len(singles) - 1))]
        n_show = max(1, int(round(1 + t ** 1.4 * (N - 1))))
        st = [s for s in stacks if s["n"] <= n_show][-1]
        im.paste(load(si["single_png"], (420, 420)), (130, 110))
        im.paste(load(st["png"], (420, 420)), (730, 110))
        d.text((130, 62), "One picture", font=font(30), fill=FG)
        d.text((730, 62), f"{st['n']} pictures along its path", font=font(30), fill=FG)
        tag(d, 130, 540, "DATA", (0, 158, 115)); tag(d, 730, 540, "DERIVED", (0, 158, 115))
        frames_out.append(im)


def seg_shift(shot, crop, frames_out, n_frames):
    im0 = Image.open(shot).convert("RGB").crop(crop).resize((460, 460), Image.NEAREST)
    for k in range(n_frames):
        im, d = canvas()
        im.paste(im0, (410, 70))
        d.text((410, 30), "Barnard's Star: earlier minus later pictures", font=font(28), fill=FG)
        tag(d, 410, 540, "MODEL / RESIDUAL", (204, 121, 167))
        frames_out.append(im)


def seg_numbers(nums, frames_out, n_frames):
    """One number at a time (equal time each), centred, with its short label underneath."""
    for k in range(n_frames):
        im, d = canvas()
        big, small = nums[k * len(nums) // n_frames]
        assert len(small.split()) <= 7, small
        centred(d, 190, big, 130, ORG)
        centred(d, 370, small, 40, FG, bold=False)
        frames_out.append(im)


def seg_name(url, frames_out, n_frames):
    for k in range(n_frames):
        im, d = canvas()
        centred(d, 150, "Orrery", 120, FG)
        centred(d, 300, "At this spot in the sky, what really changed?", 38, ACC, bold=False)
        centred(d, 395, url, 34, FG)
        centred(d, 455, "NASA SPHEREx public data · Space Apps 2026", 26, FG, bold=False)
        centred(d, 505, AI_NOTE, 20, (170, 178, 195), bold=False)
        frames_out.append(im)


def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main(a):
    field = json.load(open(f"{SITE}/timemachine/{a.field}.json"))
    e4r = json.load(open(f"{SITE}/e4/reveal.json"))
    hl = json.load(open(f"{SITE}/headline.json"))
    e4 = json.load(open("data/findings/e4.json"))
    mo = json.load(open("data/findings/motion_epsindib.json"))
    meta = json.load(open(f"{SITE}/index/dr1/meta.json"))
    pm = mo["proper_motion"]
    nums = [(f"{hl['number']}%", "of a plain blink's changes removed (median)"),
            (f"{e4['passes'][1]['significance']:.1f}σ", "coldest known brown dwarf, against empty paths"),
            (f"{pm['pmra']['spherex_mas_yr']/1000:+.2f} ± {pm['pmra']['err_mas_yr']/1000:.2f}″/yr", f"Eps Indi B east–west; Gaia {pm['pmra']['gaia_mas_yr']/1000:+.2f}"),
            (f"{meta['n_frames']:,}", "SPHEREx pictures indexed")]
    os.makedirs("dist/frames", exist_ok=True)
    for f in os.listdir("dist/frames"):
        os.remove(os.path.join("dist/frames", f))
    frames = []
    seg_title(a.team, frames, 2 * FPS)
    seg_blink_vs_det(a.field, frames, 5 * FPS)
    seg_reveal(e4r, frames, 6 * FPS)
    seg_shift(a.shift_shot, tuple(int(x) for x in a.shift_crop.split(",")), frames, 6 * FPS)
    seg_numbers(nums, frames, 7 * FPS)
    seg_name(a.url, frames, 4 * FPS - 6)  # finish safely under 30.0 s
    for i, im in enumerate(frames):
        text = next((t for s_, e_, t in SUBS if s_ <= i / FPS < e_), None)
        if text:
            subtitle(ImageDraw.Draw(im), text)
        im.save(f"dist/frames/{i:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", "dist/frames/%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-metadata", "title=Orrery: at this spot in the sky, what really changed?",
                    "-metadata", f"comment={AI_NOTE} (DR1); no generated imagery. See README.",
                    "dist/orrery_demo.mp4"], check=True)
    with open("dist/orrery_demo.srt", "w") as f:
        for i, (s_, e_, t) in enumerate(SUBS, 1):
            f.write(f"{i}\n{srt_time(s_)} --> {srt_time(e_)}\n{t}\n\n")
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", "dist/orrery_demo.mp4"]).decode())
    print("duration_s", dur)
    assert dur <= 29.9, "video must stay at or under 29.9 s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="<github-user>.github.io/orrery")
    ap.add_argument("--team", default="<team-name>", help="registered team name for the title card")
    ap.add_argument("--field", default="ecliptic_control")
    ap.add_argument("--shift-shot", default="dist/barnard_sub.png")
    ap.add_argument("--shift-crop", default="123,936,543,1356", help="x0,y0,x1,y1 of the subtract image in the screenshot")
    main(ap.parse_args())

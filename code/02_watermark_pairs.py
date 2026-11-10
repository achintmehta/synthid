"""
02_watermark_pairs.py -- what an invisible image watermark looks like, with and without.
Needs the invisible-watermark and trustmark pachages.

Google does not publish the SynthID encoder, and its API is built so that you cannot
obtain the same Imagen picture with and without the mark (Vertex AI only honours a
seed when addWatermark is false). So this script makes a real before / after pair
with open schemes instead, to show what "invisible" means and what "fragile" means:

  dwtDct     the DWT-DCT watermark Stable Diffusion's reference pipeline ships with
             (pip install invisible-watermark), a fixed-pattern frequency-domain mark
  TrustMark  Adobe's learned encoder/decoder, one of the methods the SynthID-Image
             paper benchmarks against (pip install trustmark; downloads weights once)

For each scheme it writes the original, the watermarked copy, the difference between
them amplified 20x, and a table of whether the payload still decodes after the edits
that happen on the way to a group chat.

Run:
  python 02_watermark_pairs.py                       # uses the jellyfish photo next to the script
  python 02_watermark_pairs.py path/to/image.jpg --payload "lab!"
  python 02_watermark_pairs.py --amplify 20          # fix the difference-panel gain instead of auto
  python 02_watermark_pairs.py --channels            # also write the difference per colour channel
Writes: out/pairs/*.png, out/pairs/results.md, out/pairs/results.json, out/pairs/watermark_pairs.jpg
        (with --channels, also <scheme>_diff_x<gain>_R.png, _G.png and _B.png)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "pairs")
os.makedirs(OUT, exist_ok=True)
DEFAULT_IMAGE = os.path.join(HERE, "Jellyfish_in_Blue_Water_1.jpg")   # my own photo, shipped next to the script


# ----------------------------------------------------------------------------- edits
def jpeg(img: Image.Image, q: int) -> Image.Image:
    """Re-encode the image as a JPEG of quality q (0 to 95) and decode it again, in memory.

    JPEG throws away fine detail to save space, more so at lower quality, so this is the
    gentlest edit a shared image goes through and the first test any watermark must pass.
    Encoding into a BytesIO buffer and reading it back avoids touching the disk.
    """
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")

def resize(img: Image.Image, factor: float) -> Image.Image:
    """Scale both dimensions by `factor` (0.9 means 90% of the original width and height).

    Uses Lanczos resampling, the high-quality filter most editors use, so the result is what
    a real resize produces rather than a crude nearest-neighbour one.
    """
    w, h = img.size
    return img.resize((round(w * factor), round(h * factor)), Image.LANCZOS)

def crop(img: Image.Image, frac: float) -> Image.Image:
    """Trim `frac` of the width and of the height off the picture, split equally between the edges.

    crop(im, 0.1) removes 5% from each side, keeping the central 90% by 90%. A fixed-pattern
    watermark is laid out relative to the image corners, so even a small crop shifts it out of
    alignment with the decoder, which is the fragility this edit is here to show.
    """
    w, h = img.size; dx, dy = round(w * frac / 2), round(h * frac / 2)
    return img.crop((dx, dy, w - dx, h - dy))

# The edits each watermarked image is pushed through before decoding, in the order they appear in
# results.md. Each value is a function from image to image. "none (PNG)" is the untouched control,
# and "screenshot" simulates taking a screenshot of the image on a smaller screen and saving it.
EDITS = {
    "none (PNG)":            lambda im: im,
    "JPEG quality 75":       lambda im: jpeg(im, 75),
    "JPEG quality 40":       lambda im: jpeg(im, 40),
    "resize to 90%":         lambda im: resize(im, 0.9),
    "crop 10% off the edges": lambda im: crop(im, 0.1),
    "screenshot (80% + JPEG 85)": lambda im: jpeg(resize(im, 0.8), 85),
}


# ----------------------------------------------------------------------------- schemes
class DwtDct:
    """The invisible-watermark package. Stable Diffusion's pipeline uses method 'dwtDct'; some
    OpenCV builds break its decoder, in which case we fall back to 'dwtDctSvd', the same family."""
    def __init__(self, payload: bytes, img: Image.Image):
        from imwatermark import WatermarkEncoder, WatermarkDecoder
        self.payload = payload
        self.enc = WatermarkEncoder(); self.enc.set_watermark("bytes", payload)
        self.dec = WatermarkDecoder("bytes", len(payload) * 8)
        self.method = "dwtDct"
        if not self.decodes(self.encode(img)):           # self-test on the untouched image
            self.method = "dwtDctSvd"
            print("dwtDct decoder failed its own round trip on this install; using dwtDctSvd")
        self.name = f"{'DWT-DCT' if self.method == 'dwtDct' else 'DWT-DCT-SVD'} (invisible-watermark)"
    def encode(self, img: Image.Image) -> Image.Image:
        """Return a copy of the image with the 32-bit payload hidden in it.

        The library expects OpenCV's channel order (blue, green, red), so the array is reversed
        on the way in and back on the way out. The library converts to YUV and, with its default
        strengths ([0, 36, 0] for dwtDctSvd, [0, 36, 36] for dwtDct), leaves the brightness channel
        Y alone and writes the bits into the chroma channel U (blue-difference; dwtDct uses V as
        well), as small adjustments to frequency coefficients in 4x4 blocks after a Haar wavelet
        transform. That is why the change shows up almost entirely in the blue channel.
        """
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        return Image.fromarray(self.enc.encode(bgr, self.method)[:, :, ::-1])
    def decodes(self, img: Image.Image) -> bool:
        """True if the decoder reads back exactly the payload we embedded, False otherwise.

        A wrong payload counts as a failure, not a partial success: a provenance mark that comes
        back as different bits tells you nothing. Decoder exceptions (odd sizes, modes) also count
        as failures rather than crashing the run.
        """
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        try:
            return bytes(self.dec.decode(bgr, self.method)) == self.payload
        except Exception:
            return False

class TrustMarkScheme:
    """Adobe's TrustMark: a learned encoder/decoder pair, one of the methods the SynthID-Image paper
    benchmarks against. Needs the `trustmark` package and a one-time download of its weights; if the
    download fails (some networks block it) the constructor raises and the script carries on without it.
    """
    name = "TrustMark (Adobe)"
    def __init__(self, payload: bytes):
        from trustmark import TrustMark
        self.text = payload.decode("ascii", "ignore")
        self.tm = TrustMark(verbose=False, model_type="Q")
        if getattr(self.tm, "encoder", None) is None or getattr(self.tm, "decoder", None) is None:
            raise RuntimeError("TrustMark weights could not be downloaded")
    def encode(self, img: Image.Image) -> Image.Image:
        """Hide the payload text in the image with TrustMark's neural encoder (model Q, the quality variant)."""
        return self.tm.encode(img.convert("RGB"), self.text)
    def decodes(self, img: Image.Image) -> bool:
        """True if TrustMark both reports a mark present and reads back our exact text."""
        try:
            text, present, _ = self.tm.decode(img.convert("RGB"))
            return bool(present) and text == self.text
        except Exception:
            return False


# ----------------------------------------------------------------------------- helpers
def psnr(a: Image.Image, b: Image.Image) -> float:
    """Peak signal-to-noise ratio between two same-sized images, in decibels.

    The standard single-number measure of how much one image differs from another. It takes the
    mean of the squared per-pixel differences (the mean squared error), compares it with the
    largest possible difference (255 for 8-bit channels), and puts the ratio on a log scale:

        PSNR = 10 * log10(255**2 / MSE)

    Higher means closer. Identical images give infinity. Above about 40 dB the change is
    invisible in normal viewing, 30 to 40 dB is visible if you look for it, and below 30 dB
    is obvious. The SynthID paper distrusts this number as a measure of invisibility, with
    reason, because it treats a change in a flat sky the same as the same change in a busy
    texture even though the eye does not. It is still the number everyone quotes, so it is
    printed under each watermarked panel.
    """
    x = np.asarray(a, dtype=np.float64); y = np.asarray(b, dtype=np.float64)
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

def amplified_diff(a: Image.Image, b: Image.Image, gain: int | None = None) -> tuple[Image.Image, int]:
    """Make the invisible change visible: a grayscale map of how much each pixel moved, amplified.

    For every pixel it takes the largest absolute change across the three colour channels, so a
    change in any channel shows up. Raw, those values are tiny (a few levels out of 255), so the
    map is multiplied by a gain. With `gain` left as None it is chosen automatically: the
    99th-percentile change is scaled to about 230, near white, which makes the pattern legible
    without letting a few outliers dominate. Pass a number (the --amplify option) to fix it
    instead, for example to use the same gain for two schemes so their panels are comparable.
    Returns the map as an RGB image plus the gain actually used, so the caption can say
    "amplified Nx" truthfully.
    """
    d = np.abs(np.asarray(a, dtype=np.int32) - np.asarray(b, dtype=np.int32)).max(axis=2)
    if gain is None:
        p99 = max(1, int(np.percentile(d, 99)))
        k = max(1, round(230 / p99))
    else:
        k = max(1, int(gain))
    return Image.fromarray(np.clip(d * k, 0, 255).astype(np.uint8)).convert("RGB"), k

def channel_diffs(a: Image.Image, b: Image.Image, gain: int) -> dict[str, Image.Image]:
    """The same difference map as amplified_diff, but one per colour channel instead of the maximum.

    Returns {"R": image, "G": image, "B": image}, each a grayscale map of how much that channel
    alone changed at every pixel, multiplied by the given gain and clipped at white. Using the
    gain that amplified_diff chose for the combined map keeps the four pictures on one scale, so
    a channel that looks darker really did change less. Useful for seeing where a scheme hides
    its mark: invisible-watermark writes into the blue-difference chroma channel, so its map
    lights up almost only in blue, while a learned scheme like TrustMark spreads its change
    across all three.
    """
    da = np.asarray(a, dtype=np.int32); db = np.asarray(b, dtype=np.int32)
    out = {}
    for i, name in enumerate("RGB"):
        d = np.abs(da[:, :, i] - db[:, :, i])
        out[name] = Image.fromarray(np.clip(d * gain, 0, 255).astype(np.uint8)).convert("RGB")
    return out

def font(size: int):
    """Pick a readable font for the panel labels, trying the site's Inter first, then DejaVu (Linux),
    then Segoe UI and Arial (Windows), and falling back to Pillow's built-in font if none exists."""
    for p in ("/usr/share/fonts/opentype/inter/Inter-Medium.otf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)

def composite(rows: list[tuple[str, list[tuple[str, Image.Image]]]], path: str, panel_w: int = 480):
    """Lay the panels out as a labelled grid and save it as the figure for the post.

    `rows` is a list of (row title, panels), where each panel is (label, image); one row per
    watermarking scheme, three panels per row (original, watermarked, amplified difference).
    Every panel is resized to `panel_w` pixels wide, keeping the aspect ratio of the first image,
    and gets its label printed underneath (the second line of a label, after a newline, is set
    smaller and muted). The canvas uses the site's ivory background and is saved as a JPEG at
    quality 88, which is fine for a display figure since the decoding tests use the PNGs.
    """
    bg, ink, muted = (250, 247, 241), (28, 26, 23), (95, 90, 78)
    f_lab, f_small = font(22), font(18)
    gap, pad, label_h = 18, 24, 58
    n = max(len(r[1]) for r in rows)
    sample = rows[0][1][0][1]; panel_h = round(panel_w * sample.size[1] / sample.size[0])
    row_h = panel_h + label_h + gap
    W = pad * 2 + n * panel_w + (n - 1) * gap
    H = pad * 2 + len(rows) * row_h - gap
    canvas = Image.new("RGB", (W, H), bg); draw = ImageDraw.Draw(canvas)
    y = pad
    for title, panels in rows:
        x = pad
        for label, im in panels:
            canvas.paste(im.resize((panel_w, panel_h), Image.LANCZOS), (x, y))
            lines = label.split("\n")
            draw.text((x, y + panel_h + 10), lines[0], fill=ink, font=f_lab)
            if len(lines) > 1:
                draw.text((x, y + panel_h + 36), lines[1], fill=muted, font=f_small)
            x += panel_w + gap
        y += row_h
    canvas.save(path, "JPEG", quality=88, optimize=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    ap.add_argument("--payload", default="lab!", help="4 ASCII characters = 32 bits")
    ap.add_argument("--max-width", type=int, default=1200)
    ap.add_argument("--amplify", type=int, default=None,
                    help="fixed gain for the difference panel (e.g. 20); default picks one automatically per scheme")
    ap.add_argument("--channels", action="store_true",
                    help="also save the difference map per colour channel (R, G, B), at the same gain")
    args = ap.parse_args()
    payload = args.payload.encode("ascii")
    if len(payload) != 4:
        sys.exit("payload must be exactly 4 ASCII characters (32 bits)")

    img = Image.open(args.image).convert("RGB")
    if img.size[0] > args.max_width:
        img = resize(img, args.max_width / img.size[0])
    img.save(os.path.join(OUT, "original.png"))
    print(f"image: {args.image} -> {img.size[0]}x{img.size[1]}")

    schemes = [DwtDct(payload, img)]
    try:
        schemes.append(TrustMarkScheme(payload))
    except Exception as e:  # weights not downloadable, package missing, etc.
        print(f"TrustMark unavailable ({type(e).__name__}: {str(e)[:80]}), continuing with DWT-DCT only")

    results, rows = {}, []
    for s in schemes:
        key = s.name.split()[0].lower()
        wm = s.encode(img)
        wm.save(os.path.join(OUT, f"{key}.png"))
        diff, gain = amplified_diff(img, wm, args.amplify)
        diff.save(os.path.join(OUT, f"{key}_diff_x{gain}.png"))
        if args.channels:
            for ch, im_ch in channel_diffs(img, wm, gain).items():
                im_ch.save(os.path.join(OUT, f"{key}_diff_x{gain}_{ch}.png"))
        p = psnr(img, wm)
        table = {name: s.decodes(edit(wm)) for name, edit in EDITS.items()}
        results[s.name] = {"psnr_db": round(p, 1), "max_pixel_change": int(np.abs(np.asarray(img, int) - np.asarray(wm, int)).max()), "decodes_after": table}
        print(f"\n{s.name}: PSNR {p:.1f} dB")
        for name, ok in table.items():
            print(f"  {name:28s} {'decodes' if ok else 'lost'}")
        rows.append((s.name, [("Original", img),
                              (f"With {s.name.split(' (')[0]} watermark\n32-bit payload, PSNR {p:.1f} dB", wm),
                              (f"Difference, amplified {gain}x\nthe only way to see it", diff)]))

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write("| Edit | " + " | ".join(results) + " |\n|---|" + "---|" * len(results) + "\n")
        for name in EDITS:
            f.write(f"| {name} | " + " | ".join("decodes" if results[s]["decodes_after"][name] else "lost" for s in results) + " |\n")
    composite(rows, os.path.join(OUT, "watermark_pairs.jpg"))
    print(f"\nwrote {OUT}/watermark_pairs.jpg, results.md, results.json")

if __name__ == "__main__":
    main()

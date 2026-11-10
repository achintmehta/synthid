# SynthID lab

The scripts used to generate the figures in [*Understanding SynthID: What can AI watermarks do and what they can't*](https://www.achintmehta.com/ai-arena/detecting-ai-content-synthid/). 

| Script | What it shows | Needs |
|---|---|---|
| `00_toy_tournament.py` | Tournament sampling on a ten-word vocabulary: the model's distribution, nine watermark layers by the closed-form formula, the same nine layers measured over 20,000 simulated tournaments, and the average over random keys that lands back on the original. Produces `out/toy_tournament.png`. | matplotlib only, no model |
| `01_generate.py` | The same prompt answered twice by a real open-weight model, plain and watermarked, from the same seed. Prints both answers with the detector's score for each (plain near 0.5, watermarked above it, watermarked-with-the-wrong-keys back at 0.5) and writes `out/text_pair_<n>.html`, the two answers side by side with every token shaded by its g-values. | torch + transformers, a 3 GB model download |
| `02_watermark_pairs.py` | What an invisible image watermark looks like: an original, the same image carrying a 32-bit DWT-DCT mark (and Adobe's TrustMark if its weights download), the difference amplified twenty times, optionally per colour channel, and a table of which everyday edits each payload survives. Produces `out/pairs/watermark_pairs.jpg`. | pillow, numpy, invisible-watermark (trustmark optional) |

`out/` holds the output of my runs, so you can see what each script produces without having to running anything.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # bash: source .venv/bin/activate
pip install -r requirements.txt
```

For the GPU, install the torch build that matches your card before the rest.
The CPU-only wheel is what `pip install torch` gives you on Windows, so:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

`01_generate.py` checks that the wheel has kernels for your GPU and tells you what
to install if not (a Blackwell card needs the cu130 wheel), then falls back to the
CPU. On the CPU a 1.5B model takes a few minutes per prompt.

## Running

```
python 00_toy_tournament.py
python 01_generate.py                                           # three built-in prompts
python 01_generate.py --prompt "Write a limerick about a lighthouse keeper."
python 02_watermark_pairs.py --channels                         # the jellyfish photo in this folder
python 02_watermark_pairs.py path\to\your\photo.jpg --payload "hi!!"
```

The default text model is `Qwen/Qwen2.5-1.5B-Instruct` (ungated). To use the
model family Google's own examples use:

```
huggingface-cli login            # after accepting the Gemma licence on huggingface.co
$env:SYNTHID_MODEL = "google/gemma-2-2b-it"
```

## What a run looks like

`01_generate.py` with Qwen2.5-1.5B-Instruct, 16 keys, seed 0, 200 new tokens:

| Prompt | Plain | Watermarked | Watermarked, wrong keys | Shared opening | Mean entropy (plain / wm) |
|---|---|---|---|---|---|
| story | 0.508 | 0.621 | 0.503 | 0 tokens | 2.12 / 1.87 |
| fact | 0.514 | 0.535 | 0.487 | 15 tokens | 0.84 / 0.47 |
| code | 0.492 | 0.565 | 0.516 | 2 tokens | 0.18 / 0.66 |

The score is the mean g-value over the answer's tokens, and 0.5 means no watermark.
The story, where the model had the most choice, moved furthest. The two-sentence
fact shared its whole first sentence with the plain answer, because there was
nothing there for the tournament to choose between, and ended up only 0.02 above
it. The full answers are in `out/text_pairs.md` and the shaded versions in
`out/text_pair_1.html` to `_3.html`.

`02_watermark_pairs.py` on the jellyfish photo:

| Edit | DWT-DCT-SVD (invisible-watermark) | TrustMark (Adobe) |
|---|---|---|
| none (PNG) | decodes | decodes |
| JPEG quality 75 | lost | decodes |
| JPEG quality 40 | lost | decodes |
| resize to 90% | lost | decodes |
| crop 10% off the edges | lost | decodes |
| screenshot (80% + JPEG 85) | lost | decodes |

## Two things worth knowing

**The keys are the secret.** `KEYS` in `01_generate.py` is public here on purpose.
In a real deployment whoever holds the keys can detect the mark and can forge it,
which is why no vendor publishes theirs and why Google's detector cannot read
Anthropic's text.

**Build the detector on the device you generated on.** The watermark processor
fills its hash lookup table from a seeded random generator, and PyTorch's CPU and
CUDA generators produce different numbers from the same seed. Text watermarked on
a GPU and scored on the CPU looks unwatermarked (0.5). `01_generate.py` keeps both
on one device, but if you move token ids to another machine, build the detector
on the same kind of device.

## Copyright of the test photo

`Jellyfish_in_Blue_Water_1.jpg` is my own photograph and I hold the copyright. It is
in this repository so that `02_watermark_pairs.py` runs out of the box and so that the
figure in the post can be reproduced. That is the only use it is here for: it may not be
reposted, redistributed or used elsewhere, in the original or in any of the derived
images under `out/pairs/` (the watermarked copies and the difference maps), without my
permission. To try the script on something else, pass your own image as the first argument.

## Files

```
00_toy_tournament.py         tournament sampling in pure Python, no model
01_generate.py               plain vs watermarked answers from a real model, with scores and token shading
02_watermark_pairs.py        invisible image watermark, with and without, and what edits it survives
Jellyfish_in_Blue_Water_1.jpg  the test photo, my own and copyright, see above
requirements.txt
out/                         what the scripts write; the committed copies are from my runs
```

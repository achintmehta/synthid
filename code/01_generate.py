"""
01_generate.py -- the same prompt answered twice, once plain and once watermarked.

What this shows
---------------
SynthID-Text adds nothing to the text. No hidden characters, no odd spacing, no
secret words. The only difference between the two answers this script prints is
one argument to model.generate(): watermarking_config=. Same model, same prompt,
same sampling settings, same random seed. Read the two answers and you will not
be able to tell which one carries the mark. Then look at the score the detector
gives each of them, and open the HTML page to see where the mark lives.

For every prompt the script:

  1. generates a plain answer and a watermarked answer from the same seed
  2. does what a detector does: recomputes the g-values for both answers with
     the real keys, and once more for the watermarked answer with a deliberately
     wrong set of keys
  3. prints both answers with their scores, and writes

        out/text_pairs.md         every pair with its scores, ready to paste
        out/text_pair_<n>.html    the two answers side by side, each token shaded
                                  by its g-values: green means its coin flips came
                                  up 1 (evidence for the mark), red means 0

How to read the scores
----------------------
The score is the mean g-value over the answer's tokens. For every token the
detector hashes the previous four tokens plus the token itself with each of the
keys in KEYS (16 of them here, so 16 tournament layers), which gives one bit of 0 or 1
per key, and averages them. On text that was not
watermarked those bits are coin flips, so the mean is 0.5. The watermarked
generator preferred tokens whose bits came up 1, so its answer scores above 0.5.
How far above depends on how much choice the model had: a story gives the
tournament room to work, a one-line fact or a few lines of code gives it almost
none ("no entropy, no watermark"). Scoring the watermarked answer with the wrong
keys lands back at 0.5, which is why Google's detector cannot read Anthropic's
text and the other way round: same algorithm, different keys.

Why the two answers start the same way
--------------------------------------
Both runs use the same seed, so the sampler draws the same random numbers. The
watermark changes the probabilities those numbers are compared against, so the
two answers stay identical until the first token where the reweighting tips a
draw the other way, and diverge from there. The script prints how many tokens
they share.

The one thing to get right: build the detector where you generated
-----------------------------------------------------------------
The watermark processor turns its hash into g-values through a lookup table that
it fills from a seeded random generator (see SynthIDTextWatermarkLogitsProcessor
in transformers/generation/logits_process.py). PyTorch's CPU and CUDA generators
produce different numbers from the same seed. generate() builds the processor on
the device the model is on, so if you then score on a different kind of device
the tables disagree and watermarked text scores 0.5, as if it were plain. This
script builds its detector on the same device it generated on, so it cannot go
wrong here, but keep it in mind if you take the token ids elsewhere.

What a run looks like (Qwen2.5-1.5B-Instruct, 16 keys, seed 0, 200 new tokens)
----------------------------------------------------------------------------
  Prompt   Plain   Watermarked   Watermarked, wrong keys   Shared opening   Mean entropy (plain / wm)
  story    0.508   0.621         0.503                     0 tokens         2.12 / 1.87
  fact     0.514   0.535         0.487                     15 tokens        0.84 / 0.47
  code     0.492   0.565         0.516                     2 tokens         0.18 / 0.66

The story, where the model had the most choice (about 2 nats of entropy per token),
moved furthest. The two-sentence fact shared its whole first sentence, "The Berlin
Wall fell on November 9, 1989.", with the plain answer, because there was nothing
there for the tournament to choose between, and ended up only 0.02 above it. Entropy
is measured on each answer's own tokens, so once the two answers diverge they can
have different entropies for the same prompt. The full answers are in
out/text_pairs.md and the shaded versions in out/text_pair_1.html to _3.html.

Run
---
  python 01_generate.py                                   # three built-in prompts: a story, a fact, some code
  python 01_generate.py --prompt "Write a limerick about a lighthouse keeper."
  python 01_generate.py --max-new-tokens 300 --seed 7     # longer answers, different draws
  python 01_generate.py --device cpu                      # force the CPU (slow, a few minutes per prompt)

The default model is Qwen/Qwen2.5-1.5B-Instruct (ungated, about 3 GB on first
run). Set SYNTHID_MODEL or pass --model to use another, for example
google/gemma-2-2b-it after accepting its licence and running huggingface-cli login.
"""
from __future__ import annotations

import argparse
import html
import os
import shutil
import sys
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# ---------------------------------------------------------------------------
# The watermark configuration. In production these keys are the secret: whoever
# holds them can detect the mark and can forge it. Google recommends 20 to 30
# keys (the Nature paper's main configuration used 30 tournament layers). 16
# keeps a CPU run quick and is plenty to see the effect.
# ---------------------------------------------------------------------------
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990]
NGRAM_LEN = 5                       # four previous tokens + the candidate token are hashed
CONFIG = SynthIDTextWatermarkingConfig(keys=KEYS, ngram_len=NGRAM_LEN)
WRONG_KEYS = [k + 1 for k in KEYS]  # a detector holding these sees nothing

DEFAULT_MODEL = os.environ.get("SYNTHID_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# One prompt per entropy regime, so the scores show where the watermark has room.
PROMPTS = [
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("fact", "What year did the Berlin Wall fall, and why did it matter? Answer in two sentences."),
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively."),
]


def eprint(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device(forced: str | None) -> torch.device:
    """CUDA if it is usable, then Apple MPS, then CPU. A forced device is used as given.

    "Usable" includes a check that this torch build has kernels for the GPU: a wheel is
    compiled for a fixed list of GPU generations (torch.cuda.get_arch_list()), and a newer
    card is detected fine but every kernel launch then fails with "no kernel image is
    available". Checking up front turns that crash into a message that says what to install.
    """
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        major, minor = torch.cuda.get_device_capability(dev)
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return dev
        eprint(f"NOTE: {torch.cuda.get_device_name(dev)} is compute capability {major}.{minor} (sm_{major}{minor}) "
               f"but torch {torch.__version__} only has kernels for {', '.join(torch.cuda.get_arch_list())}.\n"
               "      Install a torch build for a newer CUDA (nvidia-smi must report at least that version):\n"
               "        pip uninstall -y torch\n"
               f"        pip install torch=={torch.__version__.split('+')[0]} --index-url https://download.pytorch.org/whl/cu130\n"
               "      Falling back to the CPU for this run.")
        return torch.device("cpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if shutil.which("nvidia-smi") and torch.version.cuda is None:
        eprint(f"NOTE: an NVIDIA driver is present but this torch build is CPU-only (torch {torch.__version__}). "
               "Install the CUDA build to use the GPU:\n"
               "        pip uninstall -y torch\n"
               "        pip install torch --index-url https://download.pytorch.org/whl/cu130")
    else:
        eprint("NOTE: no GPU found, running on the CPU. Expect a few minutes per prompt for a 1.5B model.")
    return torch.device("cpu")


def load_model(name: str, dev: torch.device):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if dev.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
    return tok, model


# ---------------------------------------------------------------------------
# The detector side
# ---------------------------------------------------------------------------
def detector(keys: list[int], dev: torch.device) -> SynthIDTextWatermarkLogitsProcessor:
    """The same object generate() builds internally when it is given watermarking_config.

    We only use it to recompute g-values. It must live on the same kind of device as the
    one generation ran on (see the header).
    """
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=CONFIG.ngram_len,
        keys=keys,
        sampling_table_size=CONFIG.sampling_table_size,
        sampling_table_seed=CONFIG.sampling_table_seed,
        context_history_size=CONFIG.context_history_size,
        device=dev,
    )


@torch.no_grad()
def score(proc: SynthIDTextWatermarkLogitsProcessor, token_ids: torch.Tensor, eos_id: int | None) -> dict:
    """The mean-g detector, with no training.

    Receives the generated token ids (1-D tensor). Returns the mean g-value over the tokens
    that count, plus the per-token values and a mask saying which ones counted.

    Two kinds of token are left out, exactly as the generator left them out: a token whose
    four-token context already appeared earlier in the text (the generator does not
    watermark those, so they carry no signal), and anything after the end-of-sequence
    token. The first four tokens have no complete context and get no g-value at all, so
    per_token_g[j] belongs to token j + 4.
    """
    ids = token_ids.view(1, -1).to(proc.device)
    if ids.shape[1] < proc.ngram_len:
        return {"score": float("nan"), "n_scored": 0, "per_token_g": [], "mask": []}
    g = proc.compute_g_values(ids).float()                   # [1, T-4, len(KEYS)]
    mask = proc.compute_context_repetition_mask(ids)         # [1, T-4]
    if eos_id is not None:
        mask = mask * proc.compute_eos_token_mask(ids, eos_id)[:, proc.ngram_len - 1:]
    per_token = g.mean(dim=-1)[0]                            # average over the keys -> [T-4]
    m = mask[0].bool()
    n = int(m.sum())
    return {
        "score": float(per_token[m].mean()) if n else float("nan"),
        "n_scored": n,
        "per_token_g": per_token.tolist(),
        "mask": m.int().tolist(),
    }


@torch.no_grad()
def token_entropies(model, ids: torch.Tensor, prompt_len: int) -> list[float]:
    """Entropy (nats) of the model's next-token distribution at every generated position.

    Entropy is one number for how spread out the model's probabilities were, which is the
    same as how much choice it had: H = -sum p ln p. Zero when one token has all the
    probability, ln 2 = 0.69 for two equally likely tokens, ln 10 = 2.3 for ten. The
    watermark can only live where H is high, so this is the number that explains why the
    story scores higher than the code.

    The model is run once over prompt + answer with no sampling. Its logits at position i
    are its prediction for position i + 1, so the slice [prompt_len-1 : -1] is exactly the
    distribution the sampler was choosing from at each generated step. The watermark is not
    applied here, so this measures the room the model left, not what the watermark did.
    """
    logits = model(ids.view(1, -1)).logits[0, prompt_len - 1:-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return (-(logp.exp() * logp).sum(-1)).tolist()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def chat_prompt(tok, user_text: str) -> str:
    """Wrap the prompt in the model's chat template, if it has one."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                       tokenize=False, add_generation_prompt=True)
    return user_text


def generate_pair(model, tok, dev, prompt: str, seed: int, max_new_tokens: int) -> dict:
    """Answer `prompt` twice from the same seed, with and without the watermark.

    Returns, for each mode, the generated token ids (prompt excluded), the decoded text,
    and the per-token entropies.
    """
    enc = tok(chat_prompt(tok, prompt), return_tensors="pt").to(dev)
    prompt_len = enc.input_ids.shape[1]
    out = {}
    for mode in ("plain", "wm"):
        torch.manual_seed(seed)                      # same draws for both modes
        kwargs = dict(do_sample=True, temperature=1.0, top_k=50, top_p=1.0,
                      max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        if mode == "wm":
            kwargs["watermarking_config"] = CONFIG    # the only difference
        full = model.generate(**enc, **kwargs)[0]
        gen = full[prompt_len:]
        out[mode] = {
            "ids": gen,
            "text": tok.decode(gen, skip_special_tokens=True),
            "entropy": token_entropies(model, full, prompt_len),
        }
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def shared_prefix(a: torch.Tensor, b: torch.Tensor) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if int(a[i]) != int(b[i]):
            return i
    return n


def highlight_html(tok, label: str, prompt: str, pair: dict, scores: dict, path: str) -> None:
    """Two columns, one per answer, every token shaded by its mean g-value."""
    def column(mode: str, title: str) -> str:
        ids, s, ent = pair[mode]["ids"], scores[mode], pair[mode]["entropy"]
        off = NGRAM_LEN - 1
        spans = []
        for i, tid in enumerate(ids.tolist()):
            piece = html.escape(tok.decode([tid]))
            j = i - off
            if j < 0 or j >= len(s["per_token_g"]):
                style, tip = "background:#eee", "not scored (first four tokens)"
            elif not s["mask"][j]:
                style, tip = "background:#ddd;text-decoration:underline dotted", "masked: context seen before"
            else:
                g = s["per_token_g"][j]
                a = (g - 0.5) * 2
                col = f"rgba(0,160,80,{abs(a):.2f})" if a >= 0 else f"rgba(220,60,60,{abs(a):.2f})"
                style, tip = f"background:{col}", f"mean g = {g:.2f}, entropy = {ent[i]:.2f} nats"
            spans.append(f'<span title="{tip}" style="{style}">{piece}</span>')
        body = "".join(spans).replace("\n", "<br>")
        extra = f" &middot; wrong keys {scores['wm_wrong']['score']:.3f}" if mode == "wm" else ""
        return (f"<section><h2>{title}</h2><p class=meta>score {s['score']:.3f} over {s['n_scored']} tokens{extra}"
                f" &middot; mean entropy {sum(ent)/max(len(ent),1):.2f} nats</p><p>{body}</p></section>")
    doc = (f"<!doctype html><meta charset=utf-8><title>{html.escape(label)}: plain vs watermarked</title>"
           "<style>body{font:15px/1.7 system-ui;max-width:72em;margin:2em auto;padding:0 1em}"
           "main{display:grid;grid-template-columns:1fr 1fr;gap:2em}h2{font-size:1.1em;margin:0}"
           ".meta{color:#666;font-size:.9em}span{padding:1px 0}@media(max-width:50em){main{grid-template-columns:1fr}}</style>"
           f"<h1 style='font-size:1.3em'>{html.escape(prompt)}</h1>"
           f"<p>Each token is shaded by the average of its {len(KEYS)} g-values, one per key. Green: mostly 1, evidence for the watermark. "
           "Red: mostly 0. Light grey: the first four tokens, which have no complete context. Dotted grey: a context "
           "seen earlier in the text, skipped by the generator and the detector alike. Hover any token for its numbers.</p>"
           f"<main>{column('plain', 'Plain')}{column('wm', 'Watermarked')}</main>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", action="append", help="your own prompt (repeatable); default: three built-in ones")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda, cuda:1, mps or cpu (default: best available)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    dev = pick_device(args.device)
    tok, model = load_model(args.model, dev)
    eprint(f"model={args.model} device={dev} dtype={next(model.parameters()).dtype}"
           + (f" gpu={torch.cuda.get_device_name(dev)}" if dev.type == "cuda" else ""))
    real, wrong = detector(KEYS, dev), detector(WRONG_KEYS, dev)
    prompts = [(f"prompt {i+1}", p) for i, p in enumerate(args.prompt)] if args.prompt else PROMPTS

    md = ["| Prompt | Plain | Watermarked | Watermarked, wrong keys | Shared opening | Mean entropy (plain / wm) |",
          "|---|---|---|---|---|---|"]
    bodies, means = [], {"plain": [], "wm": []}
    t0 = time.time()
    for n, (label, prompt) in enumerate(prompts, 1):
        pair = generate_pair(model, tok, dev, prompt, args.seed + n, args.max_new_tokens)
        scores = {
            "plain": score(real, pair["plain"]["ids"], tok.eos_token_id),
            "wm": score(real, pair["wm"]["ids"], tok.eos_token_id),
            "wm_wrong": score(wrong, pair["wm"]["ids"], tok.eos_token_id),
        }
        same = shared_prefix(pair["plain"]["ids"], pair["wm"]["ids"])
        ent = {m: sum(pair[m]["entropy"]) / max(len(pair[m]["entropy"]), 1) for m in ("plain", "wm")}
        for m in ("plain", "wm"):
            means[m].append(scores[m]["score"])

        rule = "-" * 78
        print(f"\n{'=' * 78}\n[{label}] {prompt}\n{rule}")
        print(f"PLAIN         score {scores['plain']['score']:.3f}  ({scores['plain']['n_scored']} tokens scored, "
              f"mean entropy {ent['plain']:.2f} nats)\n{pair['plain']['text'].strip()}\n{rule}")
        print(f"WATERMARKED   score {scores['wm']['score']:.3f}  ({scores['wm']['n_scored']} tokens scored, "
              f"mean entropy {ent['wm']:.2f} nats)   with the wrong keys: {scores['wm_wrong']['score']:.3f}\n"
              f"{pair['wm']['text'].strip()}\n{rule}")
        print(f"The two answers are identical for the first {same} tokens, then diverge.   ({time.time() - t0:.0f}s)")

        md.append(f"| {label} | {scores['plain']['score']:.3f} | {scores['wm']['score']:.3f} | "
                  f"{scores['wm_wrong']['score']:.3f} | {same} tokens | {ent['plain']:.2f} / {ent['wm']:.2f} |")
        bodies.append(f"\n### {label}: {prompt}\n\n**Plain** (score {scores['plain']['score']:.3f})\n\n"
                      f"> {pair['plain']['text'].strip().replace(chr(10), chr(10) + '> ')}\n\n"
                      f"**Watermarked** (score {scores['wm']['score']:.3f}, wrong keys {scores['wm_wrong']['score']:.3f})\n\n"
                      f"> {pair['wm']['text'].strip().replace(chr(10), chr(10) + '> ')}\n")
        highlight_html(tok, label, prompt, pair, scores, os.path.join(OUT, f"text_pair_{n}.html"))

    with open(os.path.join(OUT, "text_pairs.md"), "w", encoding="utf-8") as f:
        f.write(f"# Plain vs watermarked: {args.model}, {len(KEYS)} keys, seed {args.seed}\n\n"
                "Score = mean g-value over the scored tokens. 0.5 means no watermark.\n\n"
                + "\n".join(md) + "\n" + "".join(bodies))
    print(f"\nwrote {OUT}/text_pairs.md and {len(prompts)} text_pair_<n>.html page(s)")

    mp, mw = sum(means["plain"]) / len(means["plain"]), sum(means["wm"]) / len(means["wm"])
    print(f"mean score over all prompts: plain {mp:.3f}, watermarked {mw:.3f}")
    if mw - mp < 0.02:
        eprint("NOTE: the watermarked answers barely score above the plain ones. Short or low-entropy answers "
               "give the tournament little room; try --max-new-tokens 300 or a more open-ended --prompt.")


if __name__ == "__main__":
    main()

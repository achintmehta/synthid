"""
00_toy_tournament.py -- SynthID-Text tournament sampling from scratch, no torch.

Implements the core idea of the Nature paper on a toy vocabulary so you can see
exactly what the watermark does to a probability distribution:

  * a keyed pseudo-random g-function  g(context, token, key_layer) -> {0, 1}
  * tournament sampling with 2 candidates per match and `depth` layers
  * the closed-form distribution the reference implementation applies:
        p'(x) = p(x) * (1 + g(x) - E_p[g])      (applied once per layer)
  * the detection statistic: mean g-value of the chosen tokens

Vocabulary used below (see the blog post for the longer version):

  token        the unit the model emits; here just ten English words
  p(x)         the model's probability for each possible next token
  g-value      a pseudo-random coin flip, 0 or 1, fixed for a given
               (context, token, key); the watermark favours tokens whose coin
               came up 1
  layer        one round of the knockout tournament; each layer has its own
               secret key, so each round consults a different coin
  context      the previous tokens; they are hashed into the coin so the same
               word gets different flips in different sentences

The tournament, drawn for depth = 3 (this script uses 9 layers, so 512
candidates per emitted token). The number after each token is its coin under
that round's key; coins are illustrative, not the real hash values.

    draw 2**3 = 8 candidates      round 1          round 2           round 3
    from the model's p(x)         key[0]           key[1]            key[2]

    mat      1 --+
                 +-- mat     0 --+
    the      0 --+               |
                                 +-- quietly 1 --+
    quietly  1 --+               |               |
                 +-- quietly 1 --+               |
    mat      1 --+  tie: coin                    |
                                                 +-- quietly   ==> emitted
    mat      1 --+                               |
                 +-- mat     0 --+               |
    mat      1 --+  tie: same    |               |
                    token        +-- slow    0 --+
    slow     0 --+               |
                 +-- slow    1 --+
    dog      0 --+  tie: coin

    In each match the higher coin wins; equal coins, or the same token twice,
    are a tie settled at random and carry no signal. The winner's coin is
    re-drawn under the next round's key, so a token that was lucky in round 1
    is no more likely to be lucky in round 2. The emitted token has, on
    average, won one coin-biased contest per layer, and that bias (coins of 1
    showing up more often than half the time) is what the detector counts.

What each function does:

    g_value(context, token, key)   the coin: a keyed hash of the previous
                                   tokens, the candidate and the layer key,
                                   reduced to 0 or 1
    tournament_sample(...)         the bracket above, run for real: draw the
                                   candidates, play the rounds, return the winner
    closed_form(...)               the formula the production code uses instead
                                   of running the bracket,
                                   p'(x) = p(x) * (1 + g(x) - E_p[g]), once per
                                   layer; gives the same distribution exactly
    main()                         builds the toy model, runs the bracket 20,000
                                   times and checks it against the formula,
                                   averages over random keys to show
                                   "non-distortionary", computes the detection
                                   statistic for a normal and a near-certain
                                   distribution, prints a table, saves the plot

Why implement both tournament_sample and closed_form when they give the same
result? tournament_sample is the definition: the algorithm exactly as the paper
states it, with real draws and real matches. It is slow (512 draws per token
here, 2**30 for Google's 30 layers) but obviously correct, because it does
literally what the paper describes. closed_form is the derived shortcut: a
formula that claims to produce the same distribution with no randomness and a
few multiplications per layer, and it is what production code uses. That the
formula equals the bracket is a mathematical result, and this script puts the
two side by side so you can watch it hold (20,000 real tournaments land on the
formula to three decimals) instead of taking it on trust. They differ in kind,
not in outcome: one is a random procedure that returns a token, the other an
exact calculation that returns the ten probabilities that procedure samples from.

Run:  python 00_toy_tournament.py
Writes: out/toy_tournament.png, out/toy_tournament.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# The whole "language" of the toy model: ten possible next tokens.
VOCAB = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "slow", "quietly"]

# One secret integer per tournament layer. Nine keys means a nine-round
# tournament with 2**9 = 512 candidates drawn per emitted token. These are the
# example keys from the Hugging Face documentation; any integers would do.
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]  # one key per tournament layer


def g_value(context: tuple[str, ...], token: str, key: int) -> int:
    """Keyed hash of (previous tokens, candidate token, layer key) -> 0 or 1.

    This is the "coin flip" of the scheme. For a fixed context and key, every
    token in the vocabulary gets a fixed 0 or 1 that looks random and has
    nothing to do with the token's meaning or probability. Change the key and
    every token's bit is reshuffled, which is why a detector without the key
    sees nothing.

    The real implementation uses a linear-congruential accumulate_hash over
    integer token ids plus a pre-computed Bernoulli(0.5) sampling table. The
    exact hash does not matter for the idea; any keyed PRF works.
    """
    h = hashlib.sha256(f"{context}|{token}|{key}".encode()).digest()
    return h[0] & 1  # lowest bit of the first hash byte: a fair coin


def tournament_sample(probs: dict[str, float], context: tuple[str, ...], keys: list[int], rng: random.Random) -> str:
    """Pick the next token by running the whole tournament, all layers, once.

    Receives the model's next-token distribution `probs`, the preceding tokens
    `context`, and one key per layer (nine in this script, so nine rounds).
    Draws 2**len(keys) candidates from `probs`, plays one knockout round per
    key, and returns the single token left standing. Call it once per token
    you want to emit.

    This is the literal algorithm from the paper, run for real:

      1. Draw 2**depth candidate tokens from the model's distribution, with
         replacement. Likely tokens show up many times, unlikely ones rarely.
      2. Round 1: pair the candidates off. In each pair, the token whose coin
         under key[0] is 1 beats the token whose coin is 0. Equal coins (or
         the same token twice) are a tie, settled at random, and carry no
         watermark signal.
      3. Round 2 does the same with the survivors under key[1], and so on.
      4. The last token standing is emitted.

    Nothing about the model is changed. Only the randomness of *which* token
    gets picked is steered, towards tokens whose coins came up 1.
    """
    depth = len(keys)
    tokens = list(probs)
    weights = [probs[t] for t in tokens]
    # Step 1: the candidate pool, e.g. 512 draws for a 9-layer tournament.
    candidates = rng.choices(tokens, weights=weights, k=2 ** depth)
    # Steps 2 and 3: one knockout round per layer, halving the pool each time.
    for layer, key in enumerate(keys):
        nxt = []
        for a, b in zip(candidates[0::2], candidates[1::2]):  # adjacent pairs
            ga, gb = g_value(context, a, key), g_value(context, b, key)
            if ga > gb:
                nxt.append(a)            # a's coin was 1, b's was 0
            elif gb > ga:
                nxt.append(b)            # the reverse
            else:
                nxt.append(rng.choice([a, b]))  # tie: same coin, or the same token twice
        candidates = nxt
    # Step 4: after `depth` rounds exactly one candidate is left.
    return candidates[0]


def closed_form(probs: dict[str, float], context: tuple[str, ...], keys: list[int]) -> dict[str, float]:
    """The production shortcut: the distribution the tournament samples from, by formula.

    "Production code" here means what actually runs when the watermark is
    deployed: the open-source SynthIDTextWatermarkLogitsProcessor in the
    Hugging Face Transformers library (this function mirrors its update_scores
    method), and, per the paper, Google's own implementation behind Gemini,
    which uses the same closed form. Nobody runs a bracket in production.
    Running 2**depth draws per token is far too slow for a real model. Because every match is
    between two independent draws from the same distribution, the probability
    that a token comes out of a round as the winner has a formula:

        p'(x) = p(x) * (1 + g(x) - E_p[g])

    where E_p[g] is the share of the probability mass whose coin is 1 under
    this round's key. A token with g = 1 gains, one with g = 0 loses, and the
    gain is biggest when the mass is split evenly. Applying the formula once
    per layer, each time to the distribution the previous layer produced,
    gives exactly the distribution the real tournament would sample from.
    `main()` checks that claim against 20,000 real tournaments.

    The layers are the same concept in Transformers. You configure the
    watermark with SynthIDTextWatermarkingConfig(keys=[...]), one integer per
    layer, and the logits processor loops over those keys applying this update
    once for each. The number of keys is the number of layers. KEYS above is
    the nine-key example from the Hugging Face documentation; Google's
    production setting uses 30. "Layer" here is that loop over keys, inherited
    from the paper's tournament rounds, not a layer of the neural network.
    """
    p = dict(probs)
    for key in keys:
        g = {t: g_value(context, t, key) for t in p}          # this layer's coins
        g_mass = sum(p[t] * g[t] for t in p)                   # E_p[g] under the current distribution
        p = {t: p[t] * (1 + g[t] - g_mass) for t in p}         # the update, one layer
    z = sum(p.values())
    return {t: v / z for t, v in p.items()}  # renormalise (guards against float drift; the formula preserves mass)


def main() -> None:
    # ------------------------------------------------------------------------
    # Overall flow
    #
    # We invent a tiny "model": a probability distribution over ten possible
    # next tokens after the context "the cat sat on", heavily favouring "mat".
    # Then we do four things with it:
    #
    #   1.  Watermark it both ways, tournament sample and closed form,
    #       and check they agree. Run 20,000 real tournaments (the literal bracket)
    #       and compare the frequencies with the closed-form formula the production
    #       code uses.
    #
    #   1b. Show what "non-distortionary" means. Under one fixed key the
    #       distribution is heavily distorted. Averaged over many random keys
    #       it comes back to the original. The claim is "no bias in
    #       expectation over the key", not "no change to this sentence".
    #
    #   2.  Compute the detection statistic. For tokens drawn without the
    #       watermark, the average g-value is 0.5 (a fair coin). For tokens
    #       drawn with the tournament, it is noticeably above 0.5. That shift,
    #       accumulated over hundreds of tokens, is the entire watermark.
    #
    #   3.  Repeat step 2 with a near-deterministic distribution, where the
    #       model is 99% sure the next token is "mat". The watermark can only
    #       live where the model has a choice, so here the statistic barely
    #       moves. This is the "no entropy, no watermark" limitation.
    #
    # Everything is written to out/toy_tournament.json, printed as a table,
    # and plotted as out/toy_tournament.png (the grouped bar chart in the post).
    # The random seed is fixed so the numbers are reproducible.
    # ------------------------------------------------------------------------
    rng = random.Random(0)

    # The toy model: a peaky next-token distribution after "the cat sat on".
    # "mat" takes 45% of the mass and "quietly" 25%; the other eight share 30%.
    #
    # "Distribution" throughout this script means exactly this shape: one
    # probability for each of the ten tokens in VOCAB, summing to 1, read as
    # "how likely is this token to be the next one after the context". The
    # ten tokens are the whole range; they are the x-axis of the bar chart.
    # `base`, `empirical`, `analytic` and `avg` below are all
    # distributions in this sense, computed in different ways, and every
    # comparison in the script is a comparison of those ten numbers.
    base = {"the": 0.02, "cat": 0.03, "sat": 0.02, "on": 0.01, "mat": 0.45, "dog": 0.05,
            "ran": 0.06, "fast": 0.04, "slow": 0.07, "quietly": 0.25}
    context = ("the", "cat", "sat", "on")

    # 1. Empirical tournament distribution vs closed form, all nine layers
    #
    #    Why 20,000 runs: one call to tournament_sample emits one token. To see
    #    the *distribution* the watermarked model samples from, we have to emit
    #    many tokens for the same context and count how often each word wins.
    #    20,000 runs pins each token's frequency to within about half a percent
    #    (sampling noise shrinks as 1/sqrt(n)). `empirical` is that count,
    #    divided by n: a Monte Carlo estimate of what the bracket really does.
    #
    #    Why compare with closed_form: closed_form is a shortcut for calulating the
    #    watermarked distribution but claims to produce the same
    #    distribution from a formula, p'(x) = p(x) * (1 + g(x) - E_p[g]) applied
    #    once per layer, with no sampling at all. That formula is what the
    #    production code in the Hugging Face Transformers library (the software,
    #    not the network architecture) runs instead of a bracket: it loops over
    #    the layers reshaping the probability vector, draws nothing inside the
    #    loop, and only then lets the ordinary sampler pick one token from the
    #    reshaped probabilities. One random draw instead of 512. If the formula
    #    is right, the 20,000-run frequencies should land on it to within the
    #    noise, and they do (to three decimals). So the comparison is the check
    #    that the production shortcut is exact, not an approximation.
    #
    #    `analytic` is the formula applied for all nine layers.
    n = 20000
    counts = Counter(tournament_sample(base, context, KEYS, rng) for _ in range(n))
    empirical = {t: counts[t] / n for t in base}
    analytic = closed_form(base, context, KEYS)

    # 1b. "Non-distortionary" means: averaged over the secret key, nothing
    # changes. Average the closed form over many random key sets.
    #
    #     An example from this script's own numbers. Under our nine KEYS,
    #     "mat" collapses from 0.450 to 0.004 and "quietly" jumps from 0.250 to
    #     0.839, because those keys happened to give "quietly" coins of 1 in
    #     most rounds and "mat" coins of 0. That is a heavy distortion. Now
    #     pick nine different keys: the coins are reshuffled, and this time it
    #     might be "mat" that is favoured and "quietly" that is punished. No
    #     single key set is neutral, but no direction is preferred across key
    #     sets, so when we average the watermarked distribution over 2000
    #     random sets, "mat" comes back to about 0.43 and "quietly" to about
    #     0.27, the original 0.45 and 0.25 within sampling noise.
    #
    #     Each iteration below invents a fresh set of nine keys, watermarks the
    #     same distribution with them, and accumulates the result. The claim
    #     is about this average, not about any one sentence: a particular
    #     response under a particular key is skewed, by design, since a
    #     detector needs a skew to detect.
    avg = {t: 0.0 for t in base}
    n_keys = 2000
    for i in range(n_keys):
        ks = [rng.randrange(1, 10**6) for _ in KEYS]
        cf = closed_form(base, context, ks)
        for t in base:
            avg[t] += cf[t] / n_keys

    # 2. Mean g-value of sampled tokens (the detection statistic), layer 0
    #    This is what a detector does: for each emitted token, recompute its
    #    coin under every layer's key (it holds the keys), average the coins,
    #    and average again over tokens. Plain text gives 0.5. Watermarked text
    #    gives more, because the tournament preferred tokens with coins of 1.
    def mean_g(sampler, trials=5000):
        tot = 0.0
        for _ in range(trials):
            tok = sampler()
            tot += sum(g_value(context, tok, k) for k in KEYS) / len(KEYS)  # this token's mean coin over the 9 keys
        return tot / trials

    # Plain sampling: draw straight from the model, no watermark.
    mean_g_plain = mean_g(lambda: rng.choices(list(base), weights=list(base.values()))[0])
    # Watermarked sampling: draw through the tournament.
    mean_g_wm = mean_g(lambda: tournament_sample(base, context, KEYS, rng))

    # 3. Same experiment with a near-deterministic distribution (low entropy)
    #    99.1% of the mass on "mat", 0.1% on each of the other nine. Almost every
    #    match in the tournament is "mat" against "mat", which is a tie, so the
    #    watermark has nothing to bias and the statistic hardly moves.
    peaky = {t: 0.001 for t in base}
    peaky["mat"] = 1 - 0.009
    mean_g_peaky_plain = mean_g(lambda: rng.choices(list(peaky), weights=list(peaky.values()))[0])
    mean_g_peaky_wm = mean_g(lambda: tournament_sample(peaky, context, KEYS, rng))

    # Save everything, so the figures can be checked against the numbers.
    result = {
        "base_distribution": base,
        "empirical_tournament": empirical,
        "closed_form": analytic,
        "closed_form_averaged_over_random_keys": avg,
        "mean_g_plain": mean_g_plain,
        "mean_g_watermarked": mean_g_wm,
        "mean_g_lowentropy_plain": mean_g_peaky_plain,
        "mean_g_lowentropy_watermarked": mean_g_peaky_wm,
    }
    with open(os.path.join(OUT, "toy_tournament.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Console table: one row per token, one column per way of computing the distribution.
    print(f"{'token':9s} {'base':>6s} {'closed form':>12s} {'empirical':>10s} {'avg/keys':>9s}")
    for t in base:
        print(f"{t:9s} {base[t]:6.3f} {analytic[t]:12.3f} {empirical[t]:10.3f} {avg[t]:9.3f}")
    print("\n'empirical' = 20k real tournaments; it matches the closed form.")
    print("'avg/keys'  = closed form averaged over 2000 random key sets; it matches 'base'.")
    print("That is what 'non-distortionary' means: no change *in expectation over the key*.")
    print(f"\nmean g, plain sampling      : {mean_g_plain:.3f}   (expect ~0.5)")
    print(f"mean g, tournament sampling : {mean_g_wm:.3f}   (expect > 0.5)")
    print(f"mean g, low-entropy plain   : {mean_g_peaky_plain:.3f}   (one fixed context: the dominant token's own g-bits;"
          f" across many contexts this averages to 0.5)")
    print(f"mean g, low-entropy tourney : {mean_g_peaky_wm:.3f}   (barely moves: nothing to choose between)")

    # The figure: four bars per token.
    #   blue   the model's own distribution
    #   orange nine layers by the closed-form formula (heavily skewed under this key)
    #   green  nine layers, measured over 20k real tournaments (lands on the orange bars)
    #   red    nine layers averaged over 2000 random keys (back to the original)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(base))
    w = 0.2  # bar width; four bars per token, offset by -1.5w, -0.5w, +0.5w, +1.5w
    ax.bar([i - 1.5 * w for i in x], [base[t] for t in base], w, label="model p(x) - no watermark")
    ax.bar([i - 0.5 * w for i in x], [analytic[t] for t in base], w, label=f"{len(KEYS)} layers, closed form")
    ax.bar([i + 0.5 * w for i in x], [empirical[t] for t in base], w, label=f"{len(KEYS)} layers (20k tournaments with same key)")
    ax.bar([i + 1.5 * w for i in x], [avg[t] for t in base], w, label=f"{len(KEYS)} layers, averaged over 2000 tournaments with different keys")
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(base), rotation=30)
    ax.set_ylabel("probability")
    ax.set_title(f"Tournament sampling after {' '.join(context)!r}: one key distorts, the average key does not")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "toy_tournament.png"), dpi=150)
    print(f"\nwrote {OUT}/toy_tournament.png and toy_tournament.json")


if __name__ == "__main__":
    main()

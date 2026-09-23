"""Jev-style typed decisions on Apple Silicon with mlx-lm: one prefill pass, no text generation.

    pip install mlx-lm
    python jev_style_mlx.py                 # downloads the model from the Hub on first run

The hidden state at the last prompt position is projected onto the option-letter rows of the
(tied) embedding matrix only, and a softmax over the declared options gives the decision
distribution. The calibration temperature is already folded into the weights.
"""

import sys

import mlx.core as mx
from mlx_lm import load

REPO = "chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER = ("You are a decision function. Read the state, then answer the question "
          "by choosing exactly one option.\n\n")


class JevStyle:
    def __init__(self, path=REPO):
        self.model, self.tok = load(path)
        self.model.eval()
        self.label_ids = mx.array([self.tok.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS])

    def decide(self, state, question, options):
        """Choice: -> list of (option, probability), most likely first."""
        assert 2 <= len(options) <= len(LETTERS)
        lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        prompt = f"{HEADER}[State]\n{state}\n\n[Question]\n{question}\n\n[Options]\n{lines}\n\nAnswer:"
        ids = mx.array([self.tok.encode(prompt, add_special_tokens=False)])
        inner = self.model.language_model.model
        last = inner(ids)[0, -1]  # final norm applied
        rows = inner.embed_tokens.weight[self.label_ids[: len(options)]]
        probs = mx.softmax((rows @ last).astype(mx.float32)).tolist()
        return sorted(zip(options, probs), key=lambda t: -t[1])

    def decide_bool(self, state, proposition):
        """Bool: -> probability that the proposition is true."""
        return dict(self.decide(state, proposition, ["yes", "no"]))["yes"]

    def decide_score(self, state, question, levels):
        """Score: ordered levels -> (expected level index, distribution)."""
        dist = dict(self.decide(state, question, levels))
        return sum(i * dist[l] for i, l in enumerate(levels)), [(l, dist[l]) for l in levels]


if __name__ == "__main__":
    jev = JevStyle(sys.argv[1] if len(sys.argv) > 1 else REPO)
    review = "The plot is thin, but the two leads are so charming that I left the cinema smiling."
    print("Choice:", jev.decide(review, "What is the sentiment of this review?", ["negative", "positive"]))
    print("Score :", jev.decide_score(review, "Rate the sentiment of this review on an ordered scale.",
                                      ["very negative", "negative", "neutral", "positive", "very positive"]))
    print("Bool  :", jev.decide_bool(
        "Premise: A man is playing a guitar on stage.\nHypothesis: Someone is performing music.",
        "Does the premise entail the hypothesis?"))
    news = "Shares of the chipmaker jumped 8% after it raised its full-year revenue forecast."
    print("Choice:", jev.decide(news, "Which news section does this article belong to?",
                                ["World", "Sports", "Business", "Science/Technology"]))

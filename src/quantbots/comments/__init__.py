"""Adversarial + consensus comment engine (the Bridgewater-supervisor pattern).

The clone hosts a live bot society (~1M comments, ~12k/day: the tal persona fleet
plus this repo's own bots), and trade-justification comments embed the author's
bet inline — so a bad argument comes with a fadeable position attached.

Pipeline (comments/cycle.py, dry-run by default):
  1. read   — fetch new comments on covered markets (reader.py; store-deduped)
  2. judge  — evidence-anchored LLM verdict (judge.py). Bridgewater's key negative
              result: an LLM critiquing reasoning BY READING ALONE is worse than
              nothing (it overweights outliers). So a comment is `unsound` ONLY
              when its factual claims contradict our ingested feed data, and only
              HIGH-confidence unsound verdicts are actionable.
  3. reply  — drafted (stored) always; posted only when replies go live
  4. fade   — strategies/comment_fade.py turns actionable verdicts into
              counter-bets through the normal runner (sizing/caps/resolvability)
  5. consensus — consensus.py pools the bettor crowd's implied probabilities and
              Platt-extremizes σ(√3·logit(p̄)); strategies/comment_consensus.py
              trades toward it (AIA Forecaster, arXiv:2511.07678)
"""

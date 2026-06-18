# equity_options — Safety contract (READ BEFORE TOUCHING EXECUTION)

This package trades **real listed options with real broker accounts**. The rest of
`quantbots` is hard-wired to the play-money Manifold clone and is forbidden from
ever pointing at real money (`CLAUDE.md` → "Trade on the clone ONLY"). This package
is the **sanctioned, fenced carve-out** for real options — modeled on the existing
`mercury_ensemble` hosted-inference exception (`docs/mercury-ensemble-calibration.md`
§0). The carve-out applies to **this package only**.

## The fence (non-negotiable invariants)

1. **No clone client.** Nothing under `equity_options/` may import
   `quantbots.manifold`. Enforced by `tests/test_eo_safety.py` (AST import scan).
2. **Not wired into the clone.** `equity_options` is absent from `runner.py`,
   `cli.py` (the `quantbots` entry point), `strategies/__init__._REGISTRY`, and
   `scripts/daily_cycle.sh`. It has its own `eo` CLI, its own SQLite DB
   (`EQUITY_OPTIONS_DB`, default `data/equity_options.sqlite`), and its own ops loop
   (`scripts/equity_options_cycle.sh`).
3. **Staged execution.** Broker mode is one of `dry` | `paper` | `live`:
   - `dry`   — no broker call; prints the order it *would* place. The default.
   - `paper` — Alpaca **paper** account (`paper-api.alpaca.markets`). No real money.
   - `live`  — **refused** by `execution/live.py` unless BOTH hold:
       - env `EQUITY_OPTIONS_OWNER_APPROVAL=1`, AND
       - a committed risk-limits file exists at the configured path.
     This build ships **paper as the ceiling**. `live.py` is a refusing stub.
4. **Default to dry.** Every entry point defaults to `dry`. `--paper` / `--live`
   are explicit opt-ins. A live run additionally requires the two gates above.

## Execution ladder & go/no-go gates

| Phase | What runs | Gate to advance |
|---|---|---|
| 0 plumbing | sources + chain/history into the store | ≥N underlyings have clean chains+history |
| 1 analytics | `eo recommend` + `eo backtest` (NO orders) | Brier skill > 0 vs implied baseline AND positive risk-adj backtest PnL net of bid/ask |
| 2 paper | `eo trade --paper` via Alpaca | multi-month paper PnL tracks backtest; greeks within caps |
| 3 live | `execution/live.py` (refusing stub here) | explicit owner sign-off + risk-limits file + tested kill-switch; start tiny |

## Kill switch

`eo flatten --paper` cancels all open orders and closes all option positions on the
configured account. Always available; never gated.

## If you are an automated agent

Do **not** flip broker mode to `live`, do **not** set `EQUITY_OPTIONS_OWNER_APPROVAL`,
and do **not** weaken `execution/live.py`'s refusal. Those require a human owner.

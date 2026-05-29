# Design System — quantbots Dashboard

The dashboard is the operator's view of an autonomous bot fleet. Every visual
decision serves one feeling: **the bots are alive and competing**.

---

## Product Context

- **What this is:** a single-operator command center for an autonomous quant-bot
  fleet trading the private Manifold clone.
- **Who it's for:** the operator (you). One user, deep technical sophistication,
  long sessions, runs daily.
- **Space:** trading desk / ops console. Bloomberg lineage; Linear / Vercel
  modern reference; Jane Street internal tools as the unstated aspiration.
- **Project type:** local web app. Polling-grade live (SSE push). Dark-only.

---

## Aesthetic Direction

**Direction:** Mission Control / Ops Console.
**Decoration level:** minimal. Type, hairlines, and one accent do the work.
**Mood:** alive, technical, low-stakes-but-treated-serious. The interface signals
that something is actively running. Status dots pulse. Numbers update. The trade
tape scrolls. Nothing is purely decorative.

**Three rules the visuals enforce:**

1. **Liveness is visible.** A live bot looks different from a paused one — pulse,
   recent activity, real-time numbers. Stale state is muted to the edge of
   readable.
2. **The leaderboard is the spine.** Bots compete; rank order is the story. PnL
   is the headline number on every surface.
3. **No decoration without data.** No purple gradients, no decorative blobs, no
   3-column icon grids, no centered hero stock copy. Every pixel earns its place.

---

## Typography

Family: **Geist** — Vercel's typeface, used as a single family across sans
and mono. Modern, tight letterforms; tabular numerals by default; minimal
character without ornament. Free from Google Fonts.

| Role          | Font       | Weight  | Notes                                              |
| ------------- | ---------- | ------- | -------------------------------------------------- |
| Display       | Geist      | 600/700 | Page titles, section headlines.                    |
| Body / UI     | Geist      | 400/500 | Labels, descriptions, controls.                    |
| Numbers / IDs | Geist Mono | 500/700 | All numeric data. Tabular nums always on.          |
| Code / Params | Geist Mono | 400     | Strategy params, JSON blocks, IDs.                 |

**Loading:** Google Fonts in `index.html`. Subsets: `latin`. Display + Sans +
Mono only.

**Why Geist over Plex (2026-05-28):** Plex reads as "institutional / IBM-coded"
and felt too obvious. Geist is tighter, more contemporary, and lets the data
do the work without the typeface itself making a statement. Same-family
sans+mono pairing keeps weight relationships consistent across UI and data.

**Scale (rem; 16 px base):**

| Token  | rem    | px   | Use                                          |
| ------ | ------ | ---- | -------------------------------------------- |
| 2xs    | 0.625  | 10   | Labels in ALL-CAPS, kbd badges.              |
| xs     | 0.75   | 12   | Table cells (default), captions.             |
| sm     | 0.8125 | 13   | Body small, secondary descriptions.          |
| base   | 0.875  | 14   | Body. Default.                               |
| md     | 1      | 16   | Larger body, section intros.                 |
| lg     | 1.25   | 20   | Section headings.                            |
| xl     | 1.625  | 26   | Page titles.                                 |
| 2xl    | 2.25   | 36   | Hero number on bot detail (mono).            |
| 3xl    | 3.5    | 56   | Portfolio PnL hero (mono).                   |

Line-height: 1.15 for display+hero, 1.4 for body, 1 for tabular rows.
Letter-spacing: -0.01em on display, +0.08em uppercase on labels, 0 on mono.

**Numerics:** every number uses `font-feature-settings: 'tnum', 'ss01'` so
columns align. Never proportional digits in tables.

---

## Color

**Mode:** dark-only. No light counterpart in v1.

### Surface scale (near-black, very slightly cool)

| Token            | Hex        | Use                                              |
| ---------------- | ---------- | ------------------------------------------------ |
| `bg`             | `#08090C`  | Page background. Just barely off pure black.     |
| `surface`        | `#0E1015`  | Cards, table rows.                               |
| `surface-2`      | `#14171E`  | Hover, raised surface.                           |
| `surface-3`      | `#1B1F28`  | Code blocks, inset wells.                        |
| `border`         | `#1E222B`  | Hairlines.                                       |
| `border-strong`  | `#2A2F3A`  | Emphasis hairlines, input borders.               |

### Text scale

| Token          | Hex        | Use                                              |
| -------------- | ---------- | ------------------------------------------------ |
| `text`         | `#E8ECEF`  | Primary text. Numbers, headlines.                |
| `text-2`       | `#B8BEC8`  | Secondary text. Descriptions.                    |
| `text-3`       | `#7A828F`  | Tertiary. Metadata.                              |
| `text-muted`   | `#4A5263`  | Disabled, faded captions.                        |

### Signal accent (one)

| Token        | Hex        | Use                                              |
| ------------ | ---------- | ------------------------------------------------ |
| `signal`     | `#00D9FF`  | Live status, interactive states, focus rings.    |
| `signal-dim` | `#0099B8`  | Hover-darker, sparkline.                         |
| `signal-bg`  | `#003848`  | Faint tint behind signal pills.                  |

### Semantic — PnL and state

| Token       | Hex        | Use                                              |
| ----------- | ---------- | ------------------------------------------------ |
| `positive`  | `#00C896`  | Profit. Up arrows. Wins.                         |
| `negative`  | `#FF5C5C`  | Loss. Down arrows. Failures.                     |
| `warn`      | `#FFB740`  | Attention. Drift. Cancellation-prone.            |
| `neutral`   | `#7A828F`  | Refund. Flat. No-change.                         |

Color usage rules:
- **PnL is the only place green/red appear.** Never use them for borders, icons,
  or backgrounds outside of PnL or success/failure semantics.
- **Cyan is the only accent.** It marks aliveness, focus, interactive states.
- **Amber is for attention only.** Never decorative.
- **Status dots:** cyan = LIVE, gray = PAUSED, dim = DISABLED. No green dots.

---

## Spacing

Base unit: **4 px** (Tailwind default).

| Token | px | Use                                                     |
| ----- | -- | ------------------------------------------------------- |
| `1`   | 4  | Inline gaps, sparkline padding.                         |
| `2`   | 8  | Tight grids, badge padding.                             |
| `3`   | 12 | Table cell padding (compact).                           |
| `4`   | 16 | Default row gap, card padding small.                    |
| `5`   | 20 | KPI card internal spacing.                              |
| `6`   | 24 | Section internal gap.                                   |
| `8`   | 32 | Card padding large, section margins.                    |
| `12`  | 48 | Section vertical gap.                                   |
| `16`  | 64 | Page top margin, hero spacing.                          |

**Density rules:**
- Table cells: 12 px vertical, 16 px horizontal padding.
- KPI strip cards: 20 px vertical, 20 px horizontal.
- Page gutter: 32 px on first viewport, can tighten to 24 px on narrow screens.
- Max content width: 1640 px. Beyond that, gutters grow.

---

## Layout

**Approach:** disciplined grid. One column of full-width sections on home;
12-column grid on bot detail; full-bleed table on /feed and /markets.

**Navigation:** persistent left sidebar (60 px collapsed icon-only, 220 px
expanded). Routes: Fleet (home), Feed, Strategies, Markets. Operator badge
+ system status at the bottom of the sidebar.

**Border radius:**

| Token | px | Use                                              |
| ----- | -- | ------------------------------------------------ |
| `sm`  | 4  | Badges, input borders, kbd.                      |
| `md`  | 6  | Buttons, inline pills.                           |
| `lg`  | 10 | Cards, panels.                                   |
| `xl`  | 14 | Hero cards (KPI strip).                          |
| `full`| 9999 | Status dots, avatars.                          |

No bubble-radius (no uniform 20 px+ corners). Cards are crisp; corners hint
at material without softening it.

---

## Motion

**Approach:** functional. Motion exists only where it communicates state change.

| Token       | ms     | Easing                | Use                                       |
| ----------- | ------ | --------------------- | ----------------------------------------- |
| `instant`   | 0      | linear                | Number flickers on data push.             |
| `micro`     | 90     | ease-out              | Hover state changes.                      |
| `short`     | 160    | ease-out              | Page-route transitions, tab swaps.        |
| `pulse`     | 2400   | custom                | Live status dot pulse.                    |

**Number updates:** when a PnL or trade count changes via SSE, the cell briefly
brightens (text → text + 1 luminance step for 150 ms, then back). No slides, no
fades. The number is the message; motion just signals "this changed."

**Live status dot:** subtle box-shadow pulse on cyan. 2.4 s cycle. Only on LIVE
bots — never on PAUSED.

---

## Components — canonical patterns

### KPI card
- 20 px padding.
- Label: 10 px uppercase, `text-3`, +0.08 em tracking.
- Value: 36 px mono, weight 700, `text`. Tabular nums.
- Sub-row: 12 px sans + mono delta. Arrow + percent.

### Leaderboard row
- Rank (mono, `text-3`) · Bot name (sans 500, `text`) · Strategy badge (mono
  micro caps in a hairlined pill) · Status dot · PnL (mono 700 with arrow +
  color) · Sparkline (40 × 14 SVG, signal color) · Trades / Win / Last-trade.
- Hover: `surface-2` background.
- Click: navigate to `/bots/<name>`.

### Status dot
- 6 px circle. Cyan for LIVE (pulsing box-shadow), `text-3` for PAUSED, `text-muted` for DISABLED. Label adjacent in mono micro caps.

### Trade-tape row
- Time (mono 11 px, `text-3`) · Bot (mono 11 px, signal-dim) · YES/NO chip ·
  Question (sans, truncate) · Size (mono) · Price before→after (mono small,
  `text-2`).

### Equity curve
- Recharts area chart, signal-dim stroke, signal-bg fill at 25% opacity, no
  grid, single hairline x-axis, tooltip in `surface-3` with mono numbers.

---

## Anti-slop checklist

The dashboard will NEVER contain:

- Purple gradients.
- Centered hero with "Built for X" / "Designed for Y" copy.
- 3-column SaaS feature grid with icons in colored circles.
- Generic stock-photo hero.
- Inter / Roboto / Space Grotesk / Arial as primary fonts.
- Bubble-radius uniformity (every corner radius the same).
- Gradient CTA buttons.
- Rainbow-colored chart series.
- Decorative blobs, sparkles, or glow auras around content.
- Marketing language ("Supercharge your...", "Powerful insights").

---

## Decisions Log

| Date       | Decision                                  | Rationale                                             |
| ---------- | ----------------------------------------- | ----------------------------------------------------- |
| 2026-05-28 | Mission Control aesthetic                 | Bot fleet + ops console; user wants "alive" feeling.  |
| 2026-05-28 | IBM Plex (Mono + Sans)                    | Technical heritage; coordinated family; tabular nums. |
| 2026-05-28 | Switched to Geist + Geist Mono            | Plex felt too institutional/obvious. Geist is tighter, more contemporary. |
| 2026-05-28 | Cyan signal + green/red PnL               | Cyan = aliveness; finance convention for P&L sign.    |
| 2026-05-28 | Dark-only (no light mode)                 | Operator confirmed dark-only.                         |
| 2026-05-28 | Vite + React + TS + Tailwind v4 + shadcn  | Static-built bundle served by Flask.                  |
| 2026-05-28 | SSE for live updates (5 s cadence)        | Push UX without WebSocket complexity.                 |
| 2026-05-28 | Routes: /, /bots/[name], /feed, /strategies/[name], /markets | Operator chose 4 routes for v1.    |

#!/usr/bin/env python3
"""
claude_carbon.py - portable (Linux/macOS) reimplementation of ClaudeCarbon's
energy/carbon analysis.

Reads the same source of truth as the macOS app: the assistant messages with
`usage` blocks in ~/.claude/projects/<encoded-project-path>/<session-id>.jsonl

Two methodologies are available (--method):

"token" (default) mirrors ClaudeCarbon/Resources/Methodology.json and
Services/EnergyCalculator.swift, see METHODOLOGY.md. Every token costs the
same, with cache tokens weighted by Anthropic's pricing ratios:

    effective_input = input_tokens
                    + int(cache_read_input_tokens  * 0.10)   # cheap: cache lookup
                    + int(cache_creation_input_tokens * 1.25) # extra: cache write
    total_tokens    = effective_input + output_tokens
    energy_J        = total_tokens * joules_per_token[model] * PUE

"physical" costs the three things the hardware actually does. Decoding an
output token reads the whole model once per step (shared across the batch)
plus the sequence's own KV cache, so its cost grows with context length.
Prefilling new input (input + cache creation) is compute-bound and cheap
per token. Reading a cached token is only a memory transfer, and is almost
free; the real cost of a long context is paid per output token instead.

    context         = input + cache_read + cache_creation      # tokens attended
    decode_J        = output_tokens * (out_base + out_ctx * context)
    prefill_J       = (input_tokens + cache_creation_tokens) * prefill
    load_J          = cache_read_tokens * cache_load
    energy_J        = (decode_J + prefill_J + load_J) * PUE

The physical constants (PHYSICAL) are first-principles estimates for a
frontier model served in batches of a few dozen sequences on an 8-GPU
node, scaled per model with the same ratios as the token method. Cache
reads are 97% of a typical Claude Code history's raw tokens, so the two
methods differ by roughly 5x in total; see METHODOLOGY.md.

Either way:

    energy_Wh       = energy_J / 3600
    carbon_gCO2e    = energy_Wh * carbon_intensity_gCO2e_per_kWh / 1000

Results are reported as totals (kWh, kgCO2e) and as average daily rates
(kWh/day, kgCO2e/day) over the period the data covers.

Active time is derived, not logged: the transcripts carry only a timestamp
per event (user prompt, assistant message, tool result...), so a session's
active time is the sum of gaps between consecutive events, ignoring any gap
longer than --idle-gap minutes. Average power (W) is energy / active time.

These are educational approximations, good to roughly an order of magnitude.

Usage:
    ./claude_carbon.py                          # all history, summary
    ./claude_carbon.py --days 7                 # last 7 days
    ./claude_carbon.py --by-day --by-project    # breakdowns
    ./claude_carbon.py --csv out.csv            # per-message rows
    ./claude_carbon.py --carbon-intensity 162   # UK 2024 grid instead of US
    ./claude_carbon.py --period-days 30         # fix the /day denominator
    ./claude_carbon.py --by-project --idle-gap 10  # count gaps up to 10 min
    ./claude_carbon.py --method physical        # decode/prefill cost model
    ./claude_carbon.py --method physical --physical-param out_base=8
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# --- Methodology constants (Methodology.json) --------------------------------

JOULES_PER_TOKEN = {
    "haiku": 0.3,   # low confidence  - inferred from pricing ratio vs sonnet
    "sonnet": 1.0,  # medium          - middle of 0.4-4 J/token research range
    "opus": 2.0,    # low confidence  - inferred from pricing ratio vs sonnet
    "fable": 4.0,   # low confidence  - Fable 5.1 lists at 2x Opus per token
                    #                   ($10/$50 vs $5/$25 per MTok), and it
                    #                   always reasons (thinking cannot be
                    #                   disabled), so 2x the Opus figure.
}
DEFAULT_MODEL = "opus"

PUE = 1.2                     # data centre power usage effectiveness
CARBON_INTENSITY = 384.0      # gCO2e/kWh, US grid average 2024 (EPA)

CACHE_READ_WEIGHT = 0.10      # cache_read_input_tokens energy multiplier
CACHE_CREATE_WEIGHT = 1.25    # cache_creation_input_tokens energy multiplier

# --- "physical" method constants --------------------------------------------
#
# IT-level joules for the reference (opus) tier; PUE is applied on top.
# Derived from node power x time / batch size for an ~15 kW 8-GPU node
# serving a few dozen sequences, and 2 x active parameters FLOPs for prefill.
# Each is an order-of-magnitude estimate; the plausible range is in brackets.
PHYSICAL = {
    "out_base": 4.0,     # J per output token: weight-read share of a decode
                         # step, amortised over the batch          [3 - 10]
    "out_ctx": 10e-6,    # J per output token per context token: reading the
                         # sequence's KV cache plus attention      [5 - 40 uJ]
    "prefill": 0.5,      # J per input or cache-creation token: compute-bound
                         # forward pass, 2 x active params FLOPs   [0.3 - 1.5]
    "cache_load": 1e-4,  # J per cache-read token per message: moving the KV
                         # entry back into accelerator memory. Generous upper
                         # bound; rounds to zero in practice.      [0 - 1e-4]
}
# Per-model multipliers on the reference constants, same ratios as
# JOULES_PER_TOKEN (larger models read more weights, more layers of KV).
PHYSICAL_SCALE = {
    "haiku": 0.15,
    "sonnet": 0.5,
    "opus": 1.0,
    "fable": 2.0,
}
METHODS = ("token", "physical")

IDLE_GAP_MINUTES = 5.0        # gaps longer than this do not count as active time

# Household comparisons (EnergyEstimate.swift): 10W LED bulb, 20Wh phone
# battery, 60W laptop, 1000W microwave.
def household_comparison(wh: float) -> str:
    if wh < 0.01:
        return "less than 1 second of a 10W LED bulb"
    if wh < 0.1:
        return f"10W LED bulb for {wh / 0.01:.0f} second(s)"
    if wh < 1.0:
        return f"10W LED bulb for {wh * 10:.0f} seconds"
    if wh < 10.0:
        return f"charging a phone {wh * 0.05:.2f}%"
    if wh < 100.0:
        return f"60W laptop for {wh / 10.0 * 30:.0f} seconds"
    if wh < 1000.0:
        return f"charging a phone {wh * 0.05:.1f}%"
    return f"1000W microwave for {wh / 1000.0 * 60:.1f} minutes"


def parse_model_name(full_model_id):
    """'claude-opus-4-5-20250514' -> 'opus' (DataStore.parseModelName).

    Fable and Mythos share a tier (same pricing, same capabilities), so both
    map to 'fable'."""
    if not full_model_id:
        return DEFAULT_MODEL
    m = full_model_id.lower()
    if "fable" in m or "mythos" in m:
        return "fable"
    if "opus" in m:
        return "opus"
    if "haiku" in m:
        return "haiku"
    if "sonnet" in m:
        return "sonnet"
    return DEFAULT_MODEL


def parse_timestamp(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def decode_project_path(dirname):
    """'-home-trystan-claude-carbon' -> '/home/trystan/claude-carbon' (lossy:
    real path separators and literal dashes are indistinguishable)."""
    return "/" + dirname.lstrip("-").replace("-", "/")


# --- Extraction --------------------------------------------------------------

def iter_transcripts(projects_dir):
    """Yield (project_dir_name, path) for every transcript under projects_dir.

    Two layouts exist: main sessions at <project>/<session-id>.jsonl and
    subagent sessions one level deeper, at
    <project>/<session-id>/subagents/agent-*.jsonl. Both are billed, so both
    are counted."""
    patterns = (
        os.path.join(projects_dir, "*", "*.jsonl"),
        os.path.join(projects_dir, "*", "*", "subagents", "*.jsonl"),
    )
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            rel = os.path.relpath(path, projects_dir)
            yield rel.split(os.sep)[0], path


def iter_messages(projects_dir, events=None):
    """Yield one dict per assistant message carrying usage data.

    If `events` (a dict) is given, every timestamped record is also appended
    to events[(project_dir, session_id)] so active time can be derived."""
    for project, path in iter_transcripts(projects_dir):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if events is not None:
                    ts = parse_timestamp(rec.get("timestamp"))
                    if ts and rec.get("sessionId"):
                        events[(project, rec["sessionId"])].append(ts)
                if rec.get("type") != "assistant":
                    continue
                msg = rec.get("message")
                if not isinstance(msg, dict):
                    continue
                usage = msg.get("usage")
                if not isinstance(usage, dict):
                    continue

                yield {
                    "message_id": msg.get("id"),
                    "session_id": rec.get("sessionId"),
                    "project_dir": project,
                    "project_path": decode_project_path(project),
                    "file": path,
                    "timestamp": parse_timestamp(rec.get("timestamp")),
                    "raw_model": msg.get("model") or "unknown",
                    "model": parse_model_name(msg.get("model")),
                    "input_tokens": usage.get("input_tokens") or 0,
                    "output_tokens": usage.get("output_tokens") or 0,
                    "cache_read_tokens": usage.get("cache_read_input_tokens") or 0,
                    "cache_create_tokens": usage.get("cache_creation_input_tokens") or 0,
                }


def dedupe(messages):
    """Streaming produces several JSONL entries per API message; the app keeps
    the one with the highest output_tokens (the final value). Messages without
    an id are kept as-is."""
    best = {}
    order = []
    out = []
    for m in messages:
        mid = m["message_id"]
        if not mid:
            out.append(m)
            continue
        if mid not in best:
            best[mid] = m
            order.append(mid)
        elif m["output_tokens"] > best[mid]["output_tokens"]:
            best[mid] = m
    out.extend(best[mid] for mid in order)
    return out


def active_gaps(timestamps, idle_gap_s):
    """Yield (end_timestamp, seconds) for each gap between consecutive events
    that is no longer than idle_gap_s. Longer gaps mean the user walked away
    and are not counted."""
    prev = None
    for ts in sorted(timestamps):
        if prev is not None:
            gap = (ts - prev).total_seconds()
            if 0 < gap <= idle_gap_s:
                yield ts, gap
        prev = ts


# --- Calculation -------------------------------------------------------------

def compute(msg, joules_per_token, pue, carbon_intensity,
            method="token", physical=PHYSICAL, physical_scale=PHYSICAL_SCALE):
    effective_input = (
        msg["input_tokens"]
        + int(msg["cache_read_tokens"] * CACHE_READ_WEIGHT)
        + int(msg["cache_create_tokens"] * CACHE_CREATE_WEIGHT)
    )
    total = effective_input + msg["output_tokens"]
    jpt = joules_per_token.get(msg["model"], joules_per_token[DEFAULT_MODEL])
    decode_j = prefill_j = load_j = 0.0
    if method == "token":
        energy_j = total * jpt * pue
    elif method == "physical":
        scale = physical_scale.get(msg["model"], physical_scale[DEFAULT_MODEL])
        context = (msg["input_tokens"] + msg["cache_read_tokens"]
                   + msg["cache_create_tokens"])
        decode_j = msg["output_tokens"] * scale * (
            physical["out_base"] + physical["out_ctx"] * context)
        prefill_j = ((msg["input_tokens"] + msg["cache_create_tokens"])
                     * scale * physical["prefill"])
        load_j = msg["cache_read_tokens"] * physical["cache_load"]
        energy_j = (decode_j + prefill_j + load_j) * pue
    else:
        raise ValueError(f"unknown method: {method}")
    energy_wh = energy_j / 3600.0
    carbon_g = energy_wh * carbon_intensity / 1000.0
    msg = dict(msg)
    msg.update(
        effective_input_tokens=effective_input,
        billable_tokens=total,
        raw_tokens=(
            msg["input_tokens"] + msg["output_tokens"]
            + msg["cache_read_tokens"] + msg["cache_create_tokens"]
        ),
        joules_per_token=jpt,
        decode_wh=decode_j * pue / 3600.0,
        prefill_wh=prefill_j * pue / 3600.0,
        load_wh=load_j * pue / 3600.0,
        energy_wh=energy_wh,
        carbon_g=carbon_g,
    )
    return msg


class Totals:
    __slots__ = ("messages", "raw_tokens", "tokens", "energy_wh", "carbon_g",
                 "active_s")

    def __init__(self):
        self.messages = 0
        self.raw_tokens = 0
        self.tokens = 0
        self.energy_wh = 0.0
        self.carbon_g = 0.0
        self.active_s = 0.0

    @property
    def avg_watts(self):
        return self.energy_wh * 3600.0 / self.active_s if self.active_s else 0.0

    tokens_field = "billable_tokens"   # what the 'tokens' column reports

    def add(self, r):
        self.messages += 1
        self.raw_tokens += r["raw_tokens"]
        self.tokens += r[self.tokens_field]
        self.energy_wh += r["energy_wh"]
        self.carbon_g += r["carbon_g"]


TOKENS_LABEL = "eff. tokens"


def fmt_row(label, t, width, period_days, rates=True, time=True):
    row = (f"  {label:<{width}} {t.messages:>7,}  {t.tokens:>15,}  "
           f"{t.energy_wh / 1000:>10.4f}  {t.carbon_g / 1000:>10.4f}")
    if rates:
        row += (f"  {t.energy_wh / 1000 / period_days:>10.4f}"
                f"  {t.carbon_g / 1000 / period_days:>11.4f}")
    if time:
        row += f"  {t.active_s / 3600:>8.2f}  {t.avg_watts:>7.1f}"
    return row


def header(width, rates=True, time=True):
    row = (f"  {'':<{width}} {'msgs':>7}  {TOKENS_LABEL:>15}  "
           f"{'kWh':>10}  {'kgCO2e':>10}")
    if rates:
        row += f"  {'kWh/day':>10}  {'kgCO2e/day':>11}"
    if time:
        row += f"  {'hours':>8}  {'avg W':>7}"
    return row


def main():
    p = argparse.ArgumentParser(
        description="Estimate Claude Code energy use and carbon emissions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--projects-dir",
                   default=os.path.expanduser("~/.claude/projects"),
                   help="Claude Code projects directory")
    p.add_argument("--days", type=int, default=None,
                   help="only include the last N days")
    p.add_argument("--by-day", action="store_true", help="daily breakdown")
    p.add_argument("--by-project", action="store_true", help="per-project breakdown")
    p.add_argument("--by-session", action="store_true", help="per-session breakdown")
    p.add_argument("--csv", metavar="FILE", help="write per-message rows to CSV")
    p.add_argument("--carbon-intensity", type=float, default=CARBON_INTENSITY,
                   metavar="G", help="grid carbon intensity, gCO2e/kWh")
    p.add_argument("--pue", type=float, default=PUE, help="data centre PUE")
    p.add_argument("--period-days", type=float, default=None, metavar="D",
                   help="days to divide totals by for the /day rates "
                        "(default: the span actually covered by the data)")
    p.add_argument("--method", choices=METHODS, default="token",
                   help="'token': every token costs J/token, cache tokens "
                        "weighted by pricing ratios (the app's method). "
                        "'physical': decode cost grows with context, prefill "
                        "is cheap, cache reads are nearly free")
    p.add_argument("--joules-per-token", metavar="M=J", action="append", default=[],
                   help="[token] override J/token, e.g. --joules-per-token opus=4.0")
    p.add_argument("--physical-param", metavar="K=V", action="append", default=[],
                   help="[physical] override a reference constant: "
                        + ", ".join(f"{k}={v:g}" for k, v in PHYSICAL.items()))
    p.add_argument("--physical-scale", metavar="M=X", action="append", default=[],
                   help="[physical] override a per-model multiplier, "
                        "e.g. --physical-scale fable=1.5")
    p.add_argument("--idle-gap", type=float, default=IDLE_GAP_MINUTES,
                   metavar="MIN", help="gaps between events longer than this "
                                       "(minutes) are not counted as active time")
    p.add_argument("--no-dedupe", action="store_true",
                   help="do not collapse streaming duplicates by message id")
    args = p.parse_args()

    if not os.path.isdir(args.projects_dir):
        sys.exit(f"error: no such directory: {args.projects_dir}")

    jpt = dict(JOULES_PER_TOKEN)
    for override in args.joules_per_token:
        try:
            name, value = override.split("=", 1)
            jpt[name.strip().lower()] = float(value)
        except ValueError:
            sys.exit(f"error: bad --joules-per-token value: {override!r}")
    physical = dict(PHYSICAL)
    for override in args.physical_param:
        try:
            name, value = override.split("=", 1)
            name = name.strip().lower()
            if name not in physical:
                raise ValueError
            physical[name] = float(value)
        except ValueError:
            sys.exit(f"error: bad --physical-param value: {override!r} "
                     f"(expected one of {', '.join(PHYSICAL)})")
    physical_scale = dict(PHYSICAL_SCALE)
    for override in args.physical_scale:
        try:
            name, value = override.split("=", 1)
            physical_scale[name.strip().lower()] = float(value)
        except ValueError:
            sys.exit(f"error: bad --physical-scale value: {override!r}")

    global TOKENS_LABEL
    if args.method == "physical":
        # Weighted token counts are not the energy driver here; show raw.
        TOKENS_LABEL = "raw tokens"
        Totals.tokens_field = "raw_tokens"

    events = defaultdict(list)
    messages = list(iter_messages(args.projects_dir, events))
    if not args.no_dedupe:
        messages = dedupe(messages)

    cutoff = None
    if args.days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    rows = []
    undated = 0
    for m in messages:
        if cutoff is not None:
            if m["timestamp"] is None:
                undated += 1
                continue
            if m["timestamp"] < cutoff:
                continue
        rows.append(compute(m, jpt, args.pue, args.carbon_intensity,
                            args.method, physical, physical_scale))

    if not rows:
        print("No assistant messages with usage data found.")
        return

    # Active time, attributed to session / project / day by the gap's end.
    idle_gap_s = args.idle_gap * 60.0
    active_by_session = defaultdict(float)
    active_by_project = defaultdict(float)
    active_by_day = defaultdict(float)
    for (project, sid), stamps in events.items():
        if cutoff is not None:
            stamps = [t for t in stamps if t >= cutoff]
        for ts, secs in active_gaps(stamps, idle_gap_s):
            active_by_session[sid] += secs
            active_by_project[decode_project_path(project)] += secs
            active_by_day[ts.astimezone().strftime("%Y-%m-%d")] += secs

    grand = Totals()
    by_model = defaultdict(Totals)
    by_day = defaultdict(Totals)
    by_project = defaultdict(Totals)
    by_session = defaultdict(Totals)
    raw_input = raw_output = raw_cache_read = raw_cache_create = 0
    first_ts = last_ts = None

    for r in rows:
        grand.add(r)
        by_model[r["model"]].add(r)
        by_project[r["project_path"]].add(r)
        by_session[r["session_id"] or "unknown"].add(r)
        raw_input += r["input_tokens"]
        raw_output += r["output_tokens"]
        raw_cache_read += r["cache_read_tokens"]
        raw_cache_create += r["cache_create_tokens"]
        ts = r["timestamp"]
        if ts:
            by_day[ts.astimezone().strftime("%Y-%m-%d")].add(r)
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts = ts if last_ts is None or ts > last_ts else last_ts

    for k, secs in active_by_session.items():
        if k in by_session:
            by_session[k].active_s = secs
    for k, secs in active_by_project.items():
        if k in by_project:
            by_project[k].active_s = secs
    for k, secs in active_by_day.items():
        if k in by_day:
            by_day[k].active_s = secs
    grand.active_s = sum(t.active_s for t in by_project.values())

    print(f"\nClaude Code energy estimate  ({args.projects_dir})")
    if first_ts and last_ts:
        print(f"Period: {first_ts.astimezone():%Y-%m-%d %H:%M} "
              f"to {last_ts.astimezone():%Y-%m-%d %H:%M}")
    print(f"Method: {args.method}")
    if args.method == "token":
        print(f"Assumptions: PUE {args.pue}, grid {args.carbon_intensity:g} gCO2e/kWh, "
              f"J/token " + ", ".join(f"{k} {v:g}" for k, v in sorted(jpt.items())))
    else:
        print(f"Assumptions: PUE {args.pue}, grid {args.carbon_intensity:g} gCO2e/kWh")
        print(f"  per output token     {physical['out_base']:g} J + "
              f"{physical['out_ctx'] * 1e6:g} uJ x context tokens")
        print(f"  per prefilled token  {physical['prefill']:g} J "
              f"(input + cache creation)")
        print(f"  per cache-read token {physical['cache_load']:g} J")
        print(f"  model scale          " + ", ".join(
            f"{k} {v:g}x" for k, v in sorted(physical_scale.items())))
    if undated:
        print(f"Note: {undated} message(s) without a timestamp excluded by --days")

    print("\nTokens reported by the API")
    print(f"  input                {raw_input:>18,}")
    print(f"  output               {raw_output:>18,}")
    weights = args.method == "token"
    print(f"  cache read           {raw_cache_read:>18,}"
          + (f"  (weighted {CACHE_READ_WEIGHT:g}x)" if weights else ""))
    print(f"  cache creation       {raw_cache_create:>18,}"
          + (f"  (weighted {CACHE_CREATE_WEIGHT:g}x)" if weights else ""))
    print(f"  total raw            {grand.raw_tokens:>18,}")
    if args.method == "token":
        print(f"  energy-effective     {grand.tokens:>18,}")

    # Denominator for the /day rates: the span the data actually covers,
    # floored at an hour so one short session cannot blow the rate up.
    if args.period_days is not None:
        period_days = args.period_days
        period_note = "from --period-days"
    elif first_ts and last_ts:
        period_days = max((last_ts - first_ts).total_seconds() / 86400.0, 1 / 24.0)
        period_note = "span covered by the data"
    else:
        period_days = 1.0
        period_note = "no timestamps, assumed 1 day"

    print("\nTotal")
    print(f"  API messages         {grand.messages:>18,}")
    print(f"  energy               {grand.energy_wh / 1000:>18,.4f} kWh "
          f"({grand.energy_wh:,.2f} Wh)")
    print(f"  carbon               {grand.carbon_g / 1000:>18,.4f} kgCO2e "
          f"({grand.carbon_g:,.2f} g)")
    if args.method == "physical":
        decode = sum(r["decode_wh"] for r in rows)
        prefill = sum(r["prefill_wh"] for r in rows)
        load = sum(r["load_wh"] for r in rows)
        print(f"    decoding output    {decode / 1000:>18,.4f} kWh "
              f"({decode / grand.energy_wh:.0%})")
        print(f"    prefilling input   {prefill / 1000:>18,.4f} kWh "
              f"({prefill / grand.energy_wh:.0%})")
        print(f"    loading cache      {load / 1000:>18,.4f} kWh "
              f"({load / grand.energy_wh:.0%})")
    print(f"  equivalent to        {household_comparison(grand.energy_wh)}")
    print(f"  active time          {grand.active_s / 3600:>18,.2f} hours "
          f"(gaps over {args.idle_gap:g} min not counted)")
    print(f"  average power        {grand.avg_watts:>18,.1f} W while active")

    print(f"\nAverage rate over {period_days:,.2f} days ({period_note})")
    print(f"  energy               "
          f"{grand.energy_wh / 1000 / period_days:>18,.4f} kWh/day")
    print(f"  carbon               "
          f"{grand.carbon_g / 1000 / period_days:>18,.4f} kgCO2e/day")
    print(f"  annualised           "
          f"{grand.carbon_g / 1000 / period_days * 365:>18,.2f} kgCO2e/year")

    def section(title, mapping, sort_key=None, limit=None, rates=True, time=True):
        keys = sorted(mapping, key=sort_key) if sort_key else sorted(mapping)
        if limit:
            keys = keys[:limit]
        width = max(len(str(k)) for k in keys)
        print(f"\n{title}")
        print(header(width, rates, time))
        for k in keys:
            print(fmt_row(str(k), mapping[k], width, period_days, rates, time))

    # Active time is per session, not per message, so it cannot be split by model.
    section("By model", by_model, sort_key=lambda k: -by_model[k].energy_wh,
            time=False)
    if args.by_day:
        # Each row already covers exactly one day, so a /day column is noise.
        section("By day", by_day, rates=False)
    if args.by_project:
        section("By project", by_project,
                sort_key=lambda k: -by_project[k].energy_wh)
    if args.by_session:
        section("By session (top 20)", by_session,
                sort_key=lambda k: -by_session[k].energy_wh, limit=20)

    print("\nEstimates are educational approximations; treat them as reliable to"
          "\nan order of magnitude, not as measurements. See METHODOLOGY.md.\n")

    if args.csv:
        fields = ["timestamp", "session_id", "project_path", "raw_model", "model",
                  "input_tokens", "output_tokens", "cache_read_tokens",
                  "cache_create_tokens", "effective_input_tokens",
                  "billable_tokens", "joules_per_token", "decode_wh",
                  "prefill_wh", "load_wh", "energy_wh", "carbon_g"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=lambda r: r["timestamp"] or datetime.min.replace(tzinfo=timezone.utc)):
                out = dict(r)
                out["timestamp"] = r["timestamp"].isoformat() if r["timestamp"] else ""
                w.writerow(out)
        print(f"Wrote {len(rows)} rows to {args.csv}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
claude_carbon.py - portable (Linux/macOS) reimplementation of ClaudeCarbon's
energy/carbon analysis.

Reads the same source of truth as the macOS app: the assistant messages with
`usage` blocks in ~/.claude/projects/<encoded-project-path>/<session-id>.jsonl

Methodology (mirrors ClaudeCarbon/Resources/Methodology.json and
Services/EnergyCalculator.swift, see METHODOLOGY.md):

    effective_input = input_tokens
                    + int(cache_read_input_tokens  * 0.10)   # cheap: cache lookup
                    + int(cache_creation_input_tokens * 1.25) # extra: cache write
    total_tokens    = effective_input + output_tokens
    energy_J        = total_tokens * joules_per_token[model] * PUE
    energy_Wh       = energy_J / 3600
    carbon_gCO2e    = energy_Wh * carbon_intensity_gCO2e_per_kWh / 1000

Results are reported as totals (kWh, kgCO2e) and as average daily rates
(kWh/day, kgCO2e/day) over the period the data covers.

These are educational approximations, good to roughly an order of magnitude.

Usage:
    ./claude_carbon.py                          # all history, summary
    ./claude_carbon.py --days 7                 # last 7 days
    ./claude_carbon.py --by-day --by-project    # breakdowns
    ./claude_carbon.py --csv out.csv            # per-message rows
    ./claude_carbon.py --carbon-intensity 162   # UK 2024 grid instead of US
    ./claude_carbon.py --period-days 30         # fix the /day denominator
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
DEFAULT_MODEL = "sonnet"

PUE = 1.2                     # data centre power usage effectiveness
CARBON_INTENSITY = 384.0      # gCO2e/kWh, US grid average 2024 (EPA)

CACHE_READ_WEIGHT = 0.10      # cache_read_input_tokens energy multiplier
CACHE_CREATE_WEIGHT = 1.25    # cache_creation_input_tokens energy multiplier

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


def iter_messages(projects_dir):
    """Yield one dict per assistant message carrying usage data."""
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
                if not isinstance(rec, dict) or rec.get("type") != "assistant":
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


# --- Calculation -------------------------------------------------------------

def compute(msg, joules_per_token, pue, carbon_intensity):
    effective_input = (
        msg["input_tokens"]
        + int(msg["cache_read_tokens"] * CACHE_READ_WEIGHT)
        + int(msg["cache_create_tokens"] * CACHE_CREATE_WEIGHT)
    )
    total = effective_input + msg["output_tokens"]
    jpt = joules_per_token.get(msg["model"], joules_per_token[DEFAULT_MODEL])
    energy_wh = total * jpt * pue / 3600.0
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
        energy_wh=energy_wh,
        carbon_g=carbon_g,
    )
    return msg


class Totals:
    __slots__ = ("messages", "raw_tokens", "tokens", "energy_wh", "carbon_g")

    def __init__(self):
        self.messages = 0
        self.raw_tokens = 0
        self.tokens = 0
        self.energy_wh = 0.0
        self.carbon_g = 0.0

    def add(self, r):
        self.messages += 1
        self.raw_tokens += r["raw_tokens"]
        self.tokens += r["billable_tokens"]
        self.energy_wh += r["energy_wh"]
        self.carbon_g += r["carbon_g"]


def fmt_row(label, t, width, period_days, rates=True):
    row = (f"  {label:<{width}} {t.messages:>7,}  {t.tokens:>15,}  "
           f"{t.energy_wh / 1000:>10.4f}  {t.carbon_g / 1000:>10.4f}")
    if rates:
        row += (f"  {t.energy_wh / 1000 / period_days:>10.4f}"
                f"  {t.carbon_g / 1000 / period_days:>11.4f}")
    return row


def header(width, rates=True):
    row = (f"  {'':<{width}} {'msgs':>7}  {'eff. tokens':>15}  "
           f"{'kWh':>10}  {'kgCO2e':>10}")
    if rates:
        row += f"  {'kWh/day':>10}  {'kgCO2e/day':>11}"
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
    p.add_argument("--joules-per-token", metavar="M=J", action="append", default=[],
                   help="override J/token, e.g. --joules-per-token opus=4.0")
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

    messages = list(iter_messages(args.projects_dir))
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
        rows.append(compute(m, jpt, args.pue, args.carbon_intensity))

    if not rows:
        print("No assistant messages with usage data found.")
        return

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

    print(f"\nClaude Code energy estimate  ({args.projects_dir})")
    if first_ts and last_ts:
        print(f"Period: {first_ts.astimezone():%Y-%m-%d %H:%M} "
              f"to {last_ts.astimezone():%Y-%m-%d %H:%M}")
    print(f"Assumptions: PUE {args.pue}, grid {args.carbon_intensity:g} gCO2e/kWh, "
          f"J/token " + ", ".join(f"{k} {v:g}" for k, v in sorted(jpt.items())))
    if undated:
        print(f"Note: {undated} message(s) without a timestamp excluded by --days")

    print("\nTokens reported by the API")
    print(f"  input                {raw_input:>18,}")
    print(f"  output               {raw_output:>18,}")
    print(f"  cache read           {raw_cache_read:>18,}  (weighted {CACHE_READ_WEIGHT:g}x)")
    print(f"  cache creation       {raw_cache_create:>18,}  (weighted {CACHE_CREATE_WEIGHT:g}x)")
    print(f"  total raw            {grand.raw_tokens:>18,}")
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
    print(f"  equivalent to        {household_comparison(grand.energy_wh)}")

    print(f"\nAverage rate over {period_days:,.2f} days ({period_note})")
    print(f"  energy               "
          f"{grand.energy_wh / 1000 / period_days:>18,.4f} kWh/day")
    print(f"  carbon               "
          f"{grand.carbon_g / 1000 / period_days:>18,.4f} kgCO2e/day")
    print(f"  annualised           "
          f"{grand.carbon_g / 1000 / period_days * 365:>18,.2f} kgCO2e/year")

    def section(title, mapping, sort_key=None, limit=None, rates=True):
        keys = sorted(mapping, key=sort_key) if sort_key else sorted(mapping)
        if limit:
            keys = keys[:limit]
        width = max(len(str(k)) for k in keys)
        print(f"\n{title}")
        print(header(width, rates))
        for k in keys:
            print(fmt_row(str(k), mapping[k], width, period_days, rates))

    section("By model", by_model, sort_key=lambda k: -by_model[k].energy_wh)
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
                  "billable_tokens", "joules_per_token", "energy_wh", "carbon_g"]
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

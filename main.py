"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers.

Method (backtested, leave-one-out over the kit's 38 past-launch comparables):
launch-day curves are compared in log-ratio space against hour 1. For each
comparable launch with a complete release day, a big hour-1 audience and a
real launch spike, take r_h = ln(viewers_h / viewers_1) for hours 2..24.
The median forecast for hour h blends two estimates of the centre of r_h:

  * the mean of r_h over the training launches, and
  * a one-predictor regression of r_h on ln(hour-1 viewers): bigger hour-1
    launches ramp less (their first hour already holds most of the world),
    smaller ones ramp more (only a few regions are in at hour 1).

The blend is 0.6 * regression + 0.4 * mean. The spread is the marginal
standard deviation of r_h across the training launches (an honest spread:
launch days vary enormously even among similar games). Quantiles are
q_tau(h) = p1 * exp(mu_h + z_tau * sd_h), p1 the case's hour-1 print.

Backtest (leave-one-out, targets with hour-1 >= 20k viewers, n=27):
  house baseline  loss 0.321, typical miss 71%, 90% coverage 0.79
  this model      loss 0.267, typical miss 56%, 90% coverage 0.92
On the closest analogues (rolling-midnight console launches, n=5):
  house baseline  loss 0.519; this model loss 0.357.

Training-set rule (uses only data available at the checkpoint): a comparable
counts when its release day is complete (24 hourly points), its hour-1
viewers are >= 10,000, and its launch spike (hour-1 / median hourly viewers
in the 48 h before its release day) is >= 5. That drops simultaneous-global
launches whose hour 1 is still pre-unlock (their ratios explode to 100x and
would poison the fit) while keeping every rolling-midnight launch.

If the training set is too small, or anything unexpected happens, the program
falls back to the house persistence baseline, so it always writes a valid
forecast. Standard library only, deterministic, no network, no randomness.

Run: python3 -P -s /bot/main.py  (grader) or
  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Training-set rule, from the backtest.
MIN_HOUR1 = 10_000
MIN_SPIKE = 5.0
MIN_TRAIN = 5
# Blend weight of the regression estimate in the median.
W_REG = 0.6
# Floor on the log spread, so quantiles never collapse onto the median.
SD_FLOOR = 0.05
# Baseline (house) spread, used by the fallback path.
BASE_SIGMA = 0.15


def round2(x):
    """x rounded to 2 decimals, half up (as the house baseline rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_values(levels):
    """Normal quantiles for the case's levels (the contest fixes them)."""
    if levels == LEVELS:
        return list(Z)
    return [norm_ppf(t) for t in levels]


def parse_day(comparable):
    """A comparable's release-day hourly viewers, keyed 0..23 (hours 1..24).

    Returns (day, spike): day maps hour index -> viewers for the release day,
    spike is hour-1 viewers over the median hourly viewers in the 48 h
    before the release day (None when that cannot be computed).
    """
    from datetime import datetime

    day_start = datetime.fromisoformat(
        comparable["day_start"].replace("Z", "+00:00"))
    by_hour = {}
    for row in comparable.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers:
            continue
        try:
            t = datetime.fromisoformat(row["from"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        k = int((t - day_start).total_seconds() // 3600)
        if -48 <= k < 408:
            by_hour[k] = viewers
    day = {k: v for k, v in by_hour.items() if 0 <= k < 24}
    pre = sorted(v for k, v in by_hour.items() if -48 <= k < 0 and v > 0)
    baseline = None
    if pre:
        n = len(pre)
        baseline = pre[n // 2] if n % 2 else (pre[n // 2 - 1] + pre[n // 2]) / 2
    spike = None
    if day.get(0) and baseline:
        spike = day[0] / baseline
    return day, spike


def training_set(comparables, strict=True):
    """Comparables usable for the ratio fit, per the backtested rule."""
    train = []
    for comp in comparables or []:
        day, spike = parse_day(comp)
        if strict:
            if len(day) != 24 or not day.get(0):
                continue
        else:
            if len(day) < 20 or not day.get(0):
                continue
        if day[0] < MIN_HOUR1:
            continue
        if spike is None or spike < MIN_SPIKE:
            continue
        train.append(day)
    return train


def ols(xs, ys):
    """One-predictor least squares: returns (intercept, slope, resid_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        sd = math.sqrt(sum((y - my) ** 2 for y in ys) / n) if n > 1 else 0.3
        return my, 0.0, max(sd, SD_FLOOR)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.3
    return a, b, max(sd, SD_FLOOR)


def baseline_quantiles(p1, hour, zs):
    """The house persistence baseline: median stays at hour 1, spread widens."""
    return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1)))
            for z in zs]


def forecast_case(case):
    """Forecast hours 2..24 of one case as five quantiles each."""
    levels = case.get("levels") or LEVELS
    zs = z_values(levels)
    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    p1 = max(1, first["viewers"]) if first and first.get("viewers") else 1
    try:
        train = training_set(case.get("comparables"), strict=True)
        if len(train) < MIN_TRAIN:
            train = training_set(case.get("comparables"), strict=False)
        if len(train) < MIN_TRAIN:
            raise ValueError("training set too small")
        log_p1 = math.log(p1)
        rows = []
        for h in hours:
            have = [day for day in train if (h - 1) in day]
            if len(have) < MIN_TRAIN:
                # Too few ratios for this hour: baseline for this hour only.
                qs = baseline_quantiles(p1, h, zs)
                rows.append({"hour": h, "q": qs})
                continue
            ratios = [math.log(day[h - 1] / day[0]) for day in have]
            mean_r = sum(ratios) / len(ratios)
            var = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)
            sd = max(math.sqrt(var), SD_FLOOR)
            xs = [math.log(day[0]) for day in have]
            a, b, _ = ols(xs, ratios)
            # Never extrapolate the regression past the training sizes.
            log_p1_eff = min(max(log_p1, min(xs)), max(xs))
            mu = W_REG * (a + b * log_p1_eff) + (1 - W_REG) * mean_r
            qs = []
            prev = 0.0
            for z in zs:
                q = p1 * math.exp(mu + z * sd)
                q = min(max(q, 1.0), 1_000_000_000.0)
                q = round2(q)
                if q < prev:
                    q = prev
                prev = q
                qs.append(q)
            rows.append({"hour": h, "q": qs})
        return {"hours": rows}
    except Exception:
        # Never fail a case: fall back to the house baseline.
        rows = []
        for h in hours:
            qs = baseline_quantiles(p1, h, zs)
            prev = 0.0
            fixed = []
            for q in qs:
                q = min(max(q, 1.0), 1_000_000_000.0)
                if q < prev:
                    q = prev
                prev = q
                fixed.append(q)
            rows.append({"hour": h, "q": fixed})
        return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()

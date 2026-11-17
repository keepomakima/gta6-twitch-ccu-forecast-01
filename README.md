# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/keepomakima/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/keepomakima/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for make.fun's "Forecast GTA VI's launch-day Twitch CCU" bounty. It predicts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), hours 2–24, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per hour, scored by mean pinball loss on the log scale.

Method: launch-day curves are compared in log-ratio space against hour 1. For each comparable past launch, r_h = ln(viewers_h / viewers_1). The median forecast blends two estimates of the centre of r_h — the mean over training launches and a one-predictor regression of r_h on ln(hour-1 viewers) (bigger hour-1 launches ramp less, smaller ones ramp more) — with weight 0.6 on the regression. The spread is the marginal standard deviation of r_h across training launches. A comparable counts only when its release day is complete (24 hourly points), its hour-1 audience is at least 10,000, and its launch spike (hour 1 over the median hourly viewers in the 48 h before) is at least 5. If the training set is too small, the program falls back to the house persistence baseline, so it always writes a valid forecast.

Backtest (leave-one-out over the kit's past-launch comparables, targets with hour-1 >= 20,000 viewers, n=27): house baseline loss 0.321, this model loss 0.267 (typical miss 56% vs 71%, 90% coverage 0.92 vs 0.79). On the closest analogues (rolling-midnight console launches, n=5): house baseline 0.519, this model 0.357. Standard library only, deterministic, no network.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case file in the input directory and writes one `<id>.out` forecast file per case to the output directory.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

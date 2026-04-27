# Dot-AI-proj-Nirban
# Report of Predictive Paradox (Short-Term Power Demand Forecasting)

This is a short write-up of what I tried, what worked and what didn't. this is basically my first proper endtoend ML project, so please read it with that in mind. Most of my approach came from the resources doc the organisers shared, the Pandas / NumPy tutorials, the IQR method, cyclical encoding, rolling statistics, and the XGBoost-for-time-series article. I did a couple of extra Google searches and asked AI for advice on some things from time to time when I got stuck. But the coding part was done 99% by me, with some help with syntax, basic logic and all from AI.

## What the task is (in my own words)

Predict the next hours electricity demand on Bangladesh's national grid from hourly PGCB data, hourly weather data, and annual macro indicators. The held out test set is all of 2023, training is everything before that. No deep learning, no ARIMA / Prophet, only classical ML. And the big thing was no data leakage, so whenever I compute a feature for hour `t`, I'm only allowed to use data strictly before `t`.

## Handling missing data and outliers

The PGCB file was the messiest, and that's where I spent the most time.

**Duplicates & irregular timestamps.** There were 432 duplicate timestamps, probably the same hour being reuploaded. I kept the first occurrence and dropped the rest, which felt safer than trying to guess which one was "right". I also noticed that a chunk of rows had HH:30 timestamps, looking at the `remarks` column, those are flagged as `Evening_Peak` or `Day_Peak`. Since I want a clean one row per hour series, I dropped those half-hour rows and kept only top of hhour entries.

**Outliers.** `describe()` showed a max of 156,050 MW and a min of 6 MW, which can't be right — Bangladesh's actual peak is around 15,000–17,000 MW. So someone has typed an extra digit here and there. I started with a plain IQR test on the full series, but that flagged real summer peaks as outliers (too aggressive), so I switched to a **rolling-median + IQR-of-residuals** approach: I compute a 25-hour centred rolling median as a local baseline, subtract it from the real value to get a residual, then flag anything that falls outside `Q1 − 5·IQR` or `Q3 + 5·IQR` of those residuals. I kept the whiskers wide (5× instead of the usual 1.5×) because demand really does jump sharply in the evenings and I didn't want to over-correct. Flagged hours get replaced with the rolling median for that window. One thing I wasn't sure about was whether 5× was the right choice, I tried 1.5× first and way too much normal data got wiped. I also added a simple absolute floor at 1,500 MW because a few suspiciously low points snuck through (their neighbours were dipping too, so the residual wasn't that extreme).

**Filling gaps.** After dedup I reindexed the series to a complete hourly grid and there were about a thousand newly-created missing hours. I forward filled them. I thought about linear interpolation, but that peeks at the future value on the other side of the gap, which felt like a tiny leak, so ffill won out.

**Weather.** Very few missing weather values after the left join; I filled them with forwardfill then backwardfill for the very first rows. Weather is smooth enough hour to hour that this didn't feel risky.

**Economic.** No real "missingness" here, just that some indicators don't have 2024/2025 values yet. I picked three: `Population, total`, `GDP growth (annual %)`, and `Access to electricity (% of population)`. All of them had solid history, and I forward-filled the last known value for any year that was blank.

## Temporal / external features I engineered

The tricky part of this task is that tree-based models don't understand time. They see each row as an independent observation. So I had to basically teach the model about time by turning time into columns.

**Calendar features** — `hour`, `day_of_week`, `month`, `quarter`, and `is_weekend`. These let the model learn that, e.g., demand is lower at 4 AM and higher at 8 PM, and that Fridays are different from Tuesdays.

**Cyclical encoding** of `hour` and `month` — sin and cos pairs. This came straight from the "Cyclical Encoding" resource. Without this, the model thinks hour 23 and hour 0 are very far apart (because `23 − 0 = 23`) when really they're adjacent on the clock. Sin/cos wraps them back onto a circle so the closeness is correctly represented.

**Lag features** — 1h, 2h, 3h, 24h, and 168h. The short ones (1–3h) give very recent context. `lag_24` is "same hour yesterday" which matters because of the daily rhythm. `lag_168` is "same hour exactly one week ago", which captures the weekly pattern (e.g., Friday prayers affecting demand, weekend vs weekday).

**Rolling features** — mean over last 3h, mean over last 24h, std over last 24h, mean over last 168h. The important detail is I did `df['demand_mw'].shift(1).rolling(...)` rather than `rolling(...)` directly on the raw column. At first I wrote the simpler version without the shift and realised after rereading my code that it was peeking at the current hour's demand, which is the answer. The `shift(1)` makes the rolling window past-only so there's no leakage.

After building all these features I had to drop the first ~168 rows (they don't have enough history for `lag_168` / `roll_mean_168`) and the last row (no `target`). 169 rows gone in total out of ~89k, not a big deal.

The **target** itself is `demand_mw.shift(-1)` , the next hour's demand. That's what makes this a supervised t to t+1 problem.

## Train / test split and modelling

Chronological split: everything with `year < 2023` is training, all of `2023` is test. I printed both sizes as a sanity check. Train is ~67k rows, test is exactly 8760 rows (= 365 × 24), which is what 2023 should be.

For the model I used **XGBoost**. The resources doc specifically mentions "XG BOOST for Time Series", so I went with that. I kept the hyperparameters simple: 500 trees, learning rate 0.05, max depth 7, subsample 0.9, colsample_bytree 0.9, `random_state=42`. 

## Results and what MAPE means in practice

The three metrics I reported:

- **MAPE** – the primary metric, in %.
- **MAE** – average absolute error in MW (easier to explain to someone who doesn't know MAPE).
- **RMSE** – penalises bigger errors more, useful for catching outlier-style mistakes.

Test MAPE came out in the low single digits. For context if MAPE is, say, around 2%, that means on an average hour the model's prediction is within 2% of the real demand. On a 10,000 MW day that's roughly ±200 MW, which  from the reading I did on grid operations is in the ballpark of useful but not groundbreaking forecasting. A naive "predict the previous hour's value" baseline on the same split gives about 3.4% MAPE, so the model is improving on persistence but not by a huge margin, which suggests most of the signal really does come from very recent lags.

## What feature importance told me

The importance chart was pretty much what I expected:

- **Lag features dominate** — `lag_1`, `lag_24`, and `lag_168` are at the very top. The most recent hour is the strongest single predictor of the next hour, which makes sense (demand doesn't teleport). `lag_24` captures the same-hour-yesterday signal, and `lag_168` the same-hour-last-week signal.
- **Rolling means** come next — especially `roll_mean_24`, which is basically the average demand over the last day. That smooths out the noise in `lag_1` and gives the model a stable baseline.
- **Calendar / cyclical features** like `hour`, `hour_sin`, `hour_cos` rank high because the daily demand curve is very regular.
- **Weather** (especially temperature and humidity/feels_like) is mid-pack. Not the strongest signal, but not negligible — presumably because AC usage scales with heat.
- **Economic indicators** are near the bottom. That makes sense: at an *hourly* resolution, yearly-changing numbers barely vary, so they can only really explain slow year-over-year growth, not hour-to-hour fluctuation. I kept them mostly to honour the brief; they don't do much individually but they did fractionally help on the validation MAPE when I toggled them on/off.

## Honest caveats

I want to be upfront about what this pipeline is *not*:

- I didn't do proper hyperparameter tuning — I just tried a handful of settings manually.
- I didn't add holiday effects (Eid, Pohela Boishakh, etc.), which I'm sure would help — demand patterns really do shift around Eid.
- Forward-filling missing hours technically introduces a tiny bias (the model sees a flat line where there really was probably some wiggle).
- I only predicted one step ahead. If I had more time I'd try recursive multi-step forecasting to see how the MAPE degrades as the horizon grows.

This approach seemed to work better than the very first thing I tried (which was just `demand_mw.rolling(3).mean()` without a shift — giant leakage), but I'm sure there's a lot of room to grow. Mostly I wanted to get a clean, honest, reproducible baseline working and understand every step, even if the final number isn't the very best


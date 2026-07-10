# Assessment answer key

# Part A (70 points)

1. Origin: information cutoff/issue time. Horizon: future lead(s) predicted.
2. It trains on observations later than validation, leaks regimes/feature fitting, and estimates an IID interpolation task rather than future forecasting.
3. Seasonality repeats at a stable known frequency; cycles have variable duration.
4. Constant mean and finite variance, with covariance depending only on lag.
5. The series has a unit root (is nonstationary in that sense).
6. First-difference once, then model differences with two AR terms and one MA error term.
7. Its MAE is 80% of the in-sample seasonal-naive scale, a 20% improvement on that scale.
8. Conditional median.
9. A prediction interval covers a future observation and includes irreducible noise; a mean confidence interval describes uncertainty in an estimated conditional mean/parameter and is narrower.
10. Residual autocorrelation remains jointly significant: dynamics or uncertainty are misspecified.

11. Any four with explanation: future-to-past leakage; preprocessing on all rows; regime shift; deployment horizon mismatch; unavailable covariates; overlapping labels; entity leakage; changed cadence. Redesign with chronological rolling origins matching horizon/cadence, all transforms fitted inside folds, an availability gap if needed, and an untouched final test.

12. The rolling window contains (y_t), directly leaking the label. Use `demand.shift(1).rolling(28).mean()`. At a fixed 14-day origin, even shifted features for later dates can contain post-origin actuals; freeze origin features, construct direct horizon rows, or recursively update using predictions.

13. Recursive: one model, data-efficient; accumulation/exposure bias. Direct: horizon-specific and no feedback accumulation; many models and possible cross-horizon incoherence. Multi-output: shares horizon structure; requires fixed horizon/more data and a model/loss that handles joint outputs.

14. Seasonal naive first; Ridge with planned inputs and safe lags; ETS as a strong component-based model; SARIMAX if residual dynamics/covariates justify it; boosted trees for nonlinear interactions. DL is last and unlikely justified by one short series. Compare identical rolling folds, stability and operational cost.

15. The interval is too narrow/miscalibrated and upper-tail risk is underestimated, possibly due to skew, heteroskedasticity, shift, or wrong distribution. Remedies include log/distributional modeling, quantile loss, residual/bootstrap intervals, conformal calibration on future-like folds, regime features, and wider tail-aware intervals. Credit any three concrete valid actions.

16. Realized weather contains information unavailable when the historical forecast was issued and makes results optimistic. Use archived/vintage weather forecasts exactly as available at each origin; otherwise exclude weather or forecast it and propagate its uncertainty.

17. Errors (forecast − actual) are `[2,-2,5,-5]`. MAE = (14/4=3.5). RMSE = \(\sqrt{(4+4+25+25)/4}=\sqrt{14.5}\approx3.81\). Bias = 0. MAPE divides by the first actual zero and overweights near-zero actuals. MASE = 3.5/5 = 0.7. Award method credit.

18. Repeat the weekly vector: `[10,12,11,13,18,20,15,10,12,11]`. At Sunday origin all observations through Sunday are legal. For all leads, lag 10 or older is safe if defined relative to target date only when it lands at/before origin; shorter target-relative lags may point after origin. The clean answer is to align each feature timestamp explicitly: a true lag is legal iff `target_date - lag <= origin`. Thus lag 1 is legal only at h=1, lag 7 through h=7, and lag 10 through h=10.

# Part B rubric (30 points)

1. **5:** Final 60 untouched (1); four ordered origins (1); 14-day blocks (1); no overlap violation/leakage (1); deployment-like spacing/justification (1).
2. **5:** Correct lag-7 alignment (2); MAE/MASE/bias (2); lead-time aggregation across origins (1).
3. **7:** shifted target features (2); fixed-origin legality for all leads (2); planned price/promo/holiday aligned to target date (1); realized temperature excluded (1); transformations per fold (1).
4. **7:** identical folds (2); sensible preprocessing/models (2); fold distribution plus metric comparison (2); selection before test (1).
5. **6:** one test evaluation (1); deployment-faithful multi-step simulation (1); diagnostic (1); recommendation covers champion and fallback (1), monitoring (1), limitation/next experiment (1).

Use [CAPSTONE.md](CAPSTONE.md)’s worked outline as the reference architecture. Exact numerical scores vary with implementation and candidate settings; grade correctness of the forecasting simulation above marginal score differences.

# Score interpretation

- **90–100:** Can lead a solid interview solution and challenge leakage/operational assumptions.
- **75–89:** Interview-ready; review any missed critical concepts.
- **60–74:** Rework Modules 3–5 and repeat the practical test.
- **Below 60:** Repeat Modules 1–5 with solutions hidden, then retest.

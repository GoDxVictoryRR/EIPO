# Results Summary

Generated: 2026-10-07T19:48:14.197698+00:00
Config hash: `e4dfa0d6`  Data end: `2026-09-29`  Tickers: 41

## Strategy Metrics

|                 |   ann_return |   ann_vol |   sharpe |   excess_return |   realized_TE |       IR |   max_drawdown |   beta |   ann_turnover |   hit_rate |   ex_ante_TE_mean |   cost_drag |   gross_excess_return |
|:----------------|-------------:|----------:|---------:|----------------:|--------------:|---------:|---------------:|-------:|---------------:|-----------:|------------------:|------------:|----------------------:|
| benchmark       |       0.1451 |    0.1558 |   0.5145 |          0      |        0      | nan      |        -0.3593 | 1      |         0.0959 |     0      |            0      |      0.0001 |                0.0001 |
| enhanced_lw     |       0.1536 |    0.1575 |   0.5624 |          0.0084 |        0.0238 |   0.3191 |        -0.3597 | 0.9994 |         4.6609 |     0.5391 |            0.0221 |      0.0047 |                0.0131 |
| enhanced_sample |       0.1549 |    0.1575 |   0.5705 |          0.0097 |        0.0244 |   0.358  |        -0.3591 | 0.9991 |         4.8497 |     0.5312 |            0.0217 |      0.0048 |                0.0146 |
| mv              |       0.1513 |    0.1338 |   0.6449 |          0.0062 |        0.0787 |   0.0271 |        -0.2551 | 0.7414 |         4.5404 |     0.4453 |            0.0774 |      0.0045 |                0.0107 |

## Statistical Significance (enhanced_lw vs benchmark)

| Metric | Value |
|--------|-------|
| IR | 0.3191 |
| Annual excess return | 0.0084 (0.84%) |
| Sample length (years) | 10.4 |
| **IR t-stat** (IR x sqrt(years)) | **1.031** |
| Bootstrap 95% CI - IR | [-0.2498, 0.8936] |
| Bootstrap 95% CI - Ann. excess return | [-0.0060, 0.0208] |
| Placebo seeds | 100 |
| Noise placebo mean IR | -0.2134 |
| Noise placebo std IR | 0.2654 |
| **Noise placebo empirical p-value** (share >= real IR) | **0.020** |
| Label perm placebo mean IR | 0.0222 |
| Label perm placebo std IR | 0.2535 |
| **Label perm empirical p-value** (share >= real IR) | **0.130** |

> **Verdict: Not statistically significant at 5%** (t-stat = 1.03, bootstrap 95% CI includes 0 [-0.250, 0.894], permutation placebo p = 0.130).

## Cost Sensitivity (enhanced_lw vs benchmark)

|   tc_bps |     IR |   excess_return |   ann_turnover |   cost_drag |   gross_excess_return |   ann_return |   realized_TE |
|---------:|-------:|----------------:|---------------:|------------:|----------------------:|-------------:|--------------:|
|        0 | 0.5161 |          0.0141 |         5.5735 |      0      |                0.0141 |       0.1593 |        0.0241 |
|        5 | 0.4145 |          0.0111 |         5.0966 |      0.0025 |                0.0137 |       0.1563 |        0.024  |
|       10 | 0.3191 |          0.0084 |         4.6609 |      0.0047 |                0.0131 |       0.1536 |        0.0238 |
|       20 | 0.2188 |          0.0056 |         3.9083 |      0.0078 |                0.0134 |       0.1506 |        0.0235 |
|       50 | 0.0312 |          0.0005 |         2.3281 |      0.0116 |                0.0122 |       0.1452 |        0.0224 |

## Sub-Period Analysis

| period      | start      | end        |   n_days |   excess |      IR |   realized_TE |
|:------------|:-----------|:-----------|---------:|---------:|--------:|--------------:|
| first_half  | 2016-02-01 | 2021-06-08 |     1315 |   0.0139 |  0.465  |        0.0253 |
| second_half | 2021-06-09 | 2026-09-29 |     1313 |   0.0034 |  0.1545 |        0.0223 |
| 2016        | 2016-02-01 | 2016-12-30 |      225 |  -0.0212 | -0.7263 |        0.027  |
| 2017        | 2017-01-02 | 2017-12-29 |      248 |   0.072  |  2.1659 |        0.0235 |
| 2018        | 2018-01-02 | 2018-12-31 |      245 |   0.0259 |  1.0034 |        0.0241 |
| 2019        | 2019-01-02 | 2019-12-31 |      241 |   0.0124 |  0.4528 |        0.0224 |
| 2020        | 2020-01-01 | 2020-12-31 |      250 |  -0.0179 | -0.4858 |        0.0289 |
| 2021        | 2021-01-01 | 2021-12-31 |      248 |   0.0277 |  0.9238 |        0.0235 |
| 2022        | 2022-01-03 | 2022-12-30 |      248 |  -0.0129 | -0.5554 |        0.0226 |
| 2023        | 2023-01-02 | 2023-12-29 |      245 |   0.0396 |  1.7793 |        0.018  |
| 2024        | 2024-01-01 | 2024-12-31 |      246 |   0.0079 |  0.3271 |        0.0243 |
| 2025        | 2025-01-01 | 2025-12-31 |      249 |  -0.0114 | -0.5159 |        0.0194 |
| 2026        | 2026-01-01 | 2026-09-29 |      183 |  -0.0124 | -0.5217 |        0.0277 |

## Limitations

1. **Survivorship Bias**: Asset universe consists of current large-cap constituents; delisted or demoted companies over 2015-2026 are excluded.
2. **Static Shares Outstanding**: Market-cap weights are constructed using static shares outstanding.
3. **Execution Cost Model**: Linear 10 bps model does not model variable bid-ask spreads, market impact, or statutory transaction taxes.
4. **Dividend Reinvestment**: Adjusted closes reflect gross dividend reinvestment without tax frictions.
5. **Placebo Interpretation**: The noise-alpha placebo is biased in favor of the real strategy because random alphas trade more and pay more costs; the label-permutation placebo is the primary test.

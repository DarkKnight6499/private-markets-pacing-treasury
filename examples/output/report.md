# Synthetic results

All values below are computed by this example. Monetary units are synthetic millions.

## Fund performance

| fund | irr | ks_pme | tvpi | dpi | rvpi |
| - | - | - | - | - | - |
| buyout_A | 0.120000 | 1.200848 | 1.795889 | 1.795889 | 0.000000 |
| venture_B | 0.150000 | 1.576410 | 2.528627 | 2.528627 | 0.000000 |
| credit_C | 0.080000 | 0.971706 | 1.233668 | 1.233668 | 0.000000 |
| real_assets_D | 0.090000 | 1.067452 | 1.563541 | 1.563541 | 0.000000 |
| secondaries_E | 0.110000 | 1.012297 | 1.234238 | 1.234238 | 0.000000 |
| co_invest_F | 0.120000 | 1.153939 | 1.501967 | 1.501967 | 0.000000 |

## Treasury

| scenario | lp_funding | interest | cash_income | peak_line | final_line | minimum_coverage | peak_funding_shortfall |
| - | - | - | - | - | - | - | - |
| base | 140.205072 | 7.817362 | 77.517638 | 30.000000 | 0.000000 | 0.185327 | 0.992808 |
| distributions_halved | 181.840601 | 7.817362 | 31.882962 | 30.000000 | 0.000000 | 0.179705 | 2.562686 |
| calls_accelerated | 143.088886 | 7.745437 | 78.473899 | 30.000000 | 0.000000 | 0.180976 | 4.094011 |
| combined_stress | 183.388800 | 7.745437 | 32.316262 | 30.000000 | 0.000000 | 0.178183 | 4.094011 |

## Unsmoothing

| asset | true_lag_weight | estimated_lag_weight | reported_ar1_slope | reported_volatility | unsmoothed_volatility | reported_autocorrelation | unsmoothed_autocorrelation | known_weight_max_error |
| - | - | - | - | - | - | - | - | - |
| private_equity | 0.300000 | 0.305202 | 0.367702 | 0.164826 | 0.216704 | 0.368218 | 0.000765 | 0.000000 |
| private_credit | 0.250000 | 0.253780 | 0.305550 | 0.053791 | 0.068943 | 0.304831 | -0.032320 | 0.000000 |
| real_assets | 0.350000 | 0.359791 | 0.426178 | 0.083054 | 0.113171 | 0.427099 | -0.000873 | 0.000000 |

## Allocation

| cma | expected_return | modeled_volatility | volatility_on_unsmoothed_covariance | illiquid_share | liquidity_surplus | stationarity_residual | public_equity | public_bonds | private_equity | private_credit | real_assets |
| - | - | - | - | - | - | - | - | - | - | - | - |
| reported | 0.071907 | 0.075263 | 0.095113 | 0.450000 | 0.087508 | 0.000000 | 0.127822 | 0.422178 | 0.332914 | 0.000000 | 0.117086 |
| unsmoothed | 0.066298 | 0.071391 | 0.071391 | 0.450000 | 0.094804 | 0.000000 | 0.176876 | 0.373124 | 0.163387 | 0.270723 | 0.015891 |

The subscription line shifts LP funding dates and adds interest. LP top-ups are explicit funding requirements. The standard buyout ramp has unavoidable overshoot from scheduled future calls. The faster distribution example demonstrates a feasible target ramp. Appraisal filtering uses an invertible MA(1) specialization of Getmansky-Lo-Makarov, verified in Section 4, equations (21) to (23). Lag weights are stylized; estimation uses a method-of-moments equation derived for iid true returns.

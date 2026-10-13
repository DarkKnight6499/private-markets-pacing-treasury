# Methodology and conventions

All example amounts, returns, index paths, exposures, and policy limits are synthetic. Amounts use synthetic millions. Calculations run offline. The return panel and public index use `numpy.random.default_rng`, with seeds defined in `data.py`. Fund schedules and investment policies are deterministic stylized inputs. No parameter is asserted to be a regulatory requirement.

## Fund cash flows

The implemented annual structure follows the original Takahashi and Alexander working paper, later published as *Illiquid Alternative Asset Fund Modeling*, Journal of Portfolio Management, 2002. Capital contributions, distributions and NAV equations were verified directly in the authors' January 2001 working paper, sections "Capital Contributions", "Distributions" and "Net Asset Value", printed pages 5 to 7. The [original working paper](https://allocatortraining.com/wp-content/uploads/2023/06/Illiquid-Alternative-Asset-Fund-Modeling-Takahashi-Alexander.pdf) is a mirrored primary document, rather than a third-party summary.

For annual fund age `a`, commitment `K`, unfunded `U`, growth `G`, distribution yield `Y`, life `L` and bow `B`:

```text
C[a] = RC[a] * U[a-1]
RD[a] = max(Y, (a/L)^B)
D[a] = RD[a] * NAV[a-1] * (1+G)
NAV[a] = NAV[a-1] * (1+G) + C[a] - D[a]
U[a] = U[a-1] - C[a]
```

Growth applies to opening NAV. Calls enter at period end. The final supplied call rate repeats. No recycling or overcalling is allowed. A larger bow delays distributions. Yield supplies a minimum distribution rate.

The exact published equations leave final-period calls in NAV. The default finite-life extension distributes those calls immediately and cancels residual unfunded commitments. Both extensions are exposed separately in output columns. With `liquidate_terminal=False`, the original equations apply throughout and residual closing NAV and unfunded remain. The final unfunded identity under the extension is `closing = opening - calls - canceled`.

Each vintage starts its first call year at its `year` value. Program tables aggregate on calendar years, including empty years. Aggregate opening unfunded includes new commitments. Thus `closing_unfunded = previous_closing_unfunded + new_commitments - calls - canceled`. NAV transitions reconcile both by fund and by program.

`STYLIZED_PARAMETERS` supplies buyout, venture, private credit, real assets, secondaries and co-invest assumptions. These are illustrations, not calibrated estimates or parameters claimed from the paper. Secondaries use a shorter horizon, a larger initial call rate and an earlier distribution curve. They are modeled as an accelerated fund, not as a priced purchase of existing NAV. Co-invest capital is called immediately, and its assumed net growth has no fee or carry deduction. Other sleeves assume fees and carry are already embedded in the chosen net growth rate, without a waterfall calculation.

## Annual commitment pacing

Each year's unit fund projection supplies a linear response to a new commitment. A bisection solver chooses a nonnegative commitment to minimize the current year's allocation error against a linear ramp to the target. It respects an absolute annual cap and reserves public funding for every projected remaining call of all committed vintages, assuming no further commitments during each reservation check. It revisits this reservation every year. This ensures future obligations under the model do not require a negative public balance.

The public sleeve grows at its stylized annual rate, pays calls and receives distributions. These transfers cancel in total wealth. Total NAV changes only by public and private growth P&L. A new commitment changes the allocation numerator through first-year calls; it does not create wealth.

Exact tracking is conditional on feasibility. Older funds may create unavoidable overshoot through scheduled calls even with zero new commitments. Output distinguishes `on_target`, `above_target`, `capped` and `funding_limited`. The standard buyout example shows such overshoot. A separately labeled faster distribution fund demonstrates an exact feasible ramp and subsequent target hold. The solver is a greedy annual controller with funding reservations, not a global optimum over all future allocation errors.

The exposure overcommitment ratio is `(private_NAV + unfunded) / total_NAV`. The separately reported active commitment ratio is `active_original_commitments / total_NAV`. Neither is leverage. Sensitivities change bow, growth and call rates separately, then re-solve the entire path. Higher growth can reduce commitments per unit of NAV while increasing nominal commitments because total portfolio wealth grows faster.

## Quarterly treasury

Annual program cash flows split equally across quarters. Inputs are sorted by calendar year; missing years receive zero flows, and duplicate years are rejected. This is a deterministic timing convention, not a calibrated intra-year call pattern. Calls represent investments funded by the facility or LP funding. They are not already LP contributions.

At each quarter opening, facility tranches reaching their repayment horizon repay in full. Available facility capacity then finances a policy fraction of current calls. Each new tranche has a fixed repayment quarter. Interest is simple annual rate divided by four, applied to principal outstanding after opening repayments and new draws. Consequently, a two-quarter tranche incurs two quarters of interest. Cash yield is simple annual rate divided by four on opening cash, a stylized convention that ignores intra-quarter cash balances.

LP funding is the positive amount required to meet current outflows and leave the prospective buffer for the next quarter. This is explicit required external funding, not evidence that funding is available. `pre_funding_cash` and `funding_shortfall` expose the cash position without the top-up. No LP payout sweep is assumed; excess cash remains invested at the cash yield. A facility runoff tail ensures final principal repayment appears even when projected investments have ended.

The buffer is a policy fraction of gross projected calls over the rolling window. It includes only calls expected from the provided unfunded schedule, not the entire remaining commitment balance. Distribution stress halves proceeds. Call acceleration shifts a fraction of each next-quarter call into the preceding quarter, preserving cumulative contributions and preventing fictitious overcalling. The combined case applies both.

Liquidity coverage is opening liquid cash divided by the maximum positive cumulative stressed net outflow over the current and next three quarters. Net outflows include calls, reduced distributions, existing facility maturities and projected interest on those existing tranches. Future borrowing and LP funding are excluded as liquidity resources. New current-quarter borrowing is also excluded from the opening snapshot. Zero positive outflows yield infinite coverage. At the forecast end, `coverage_forecast_quarters` identifies shortened windows. Coverage is a synthetic internal measure, not a banking liquidity coverage ratio.

## Pre-trade compliance

A rule configuration defines strategy, vintage and geography bands; manager, fund and issuer concentrations; sector and jurisdiction exclusions; and an unfunded limit. It is serialized to JSON by the example. Duplicate rule ids, unknown rule types and incomplete issuer weights are rejected.

The denominator is current total portfolio NAV. Commitments reserve economic exposure `NAV + unfunded` and do not inflate the NAV denominator. A funded co-invest reallocates public NAV to private NAV without creating total NAV. This conservative policy basis differs from conventional invested-NAV allocation bands, and is displayed in every result row. Policy users should select the appropriate basis before adopting these illustrative rules.

Concentrations aggregate matching labels across sleeves. Issuer exposure uses each complete look-through weight. Side-letter exclusions test the entire proposed post-trade portfolio. Limits are inclusive with a small floating-point tolerance. Warning thresholds lie inside permitted bands. Below a band minimum or above its maximum fails; at or above the warning threshold warns; otherwise passes. Exclusion exposure above zero fails. No rule result is inferred from missing policy data.

## Appraisal returns and performance

The chosen approach is an invertible MA(1) specialization of Getmansky-Lo-Makarov. Its weighted-return structure and weight normalization were verified in the [original paper hosted by its author, Section 4, equations (21) to (23), printed page 546](https://www.mit.edu/~alo/Papers/JFE2004Pub.pdf). Applied to appraisal-based private returns here, it is a synthetic model rather than a calibrated private market estimator.

```text
reported[t] = (1-weight) * true[t] + weight * true[t-1]
recovered[t] = (reported[t] - weight * recovered[t-1]) / (1-weight)
```

The implementation restricts the lag weight to `[0, 0.5)` for stable inversion. An initial true-return lag is required and defaults explicitly to zero in the synthetic generator. An initial lag error decays with factor `-weight/(1-weight)` each period. The example omits the initial recovered observation and aligns every covariance panel on the same remaining dates. Return exports include the original quarter index.

The supplied-weight simulation tests recover the true series to floating-point precision. Estimated weights solve our method-of-moments equation by bisection:

```text
lag_correlation = weight * (1-weight) / ((1-weight)^2 + weight^2)
```

This equation follows from the covariance of an iid true-return sequence after the stated MA(1) filter. The estimator is our specialization, not a replication of the paper's likelihood estimation. Sample correlations outside `[0, 0.5)` are rejected. AR(1) slopes and lag correlations serve as diagnostics; an AR(1) slope is not treated as the MA(1) weight. Higher-order lag structure, true market persistence, measurement noise and uncertain initial values require a richer model.

The previously implemented recursive filter remains available in `cma.smooth`, `cma.unsmooth` and `data.synthetic_returns`:

```text
reported[t] = (1-alpha) * true[t] + alpha * reported[t-1]
```

It is a separate stylized model. Its precise original Geltner-paper equation attribution remains unverified: the publisher PDF, full-text endpoint and legacy journal PDF were inaccessible through the permitted web tool. It is marked in `UNSMOOTHING_SOURCE_STATUS` and is not used in the main results. Runtime calculations and algebraic recovery tests do not require that attribution. No shell network request was attempted because the environment explicitly prohibits it.

Annual CMAs use arithmetic quarterly mean times four and sample covariance times four. This assumes serially independent returns for risk scaling. On the smoothed panel it deliberately produces the naive volatility comparison. It does not estimate long-run autocovariance risk or claim that unsmoothing creates risk-free diversification. The synthetic true returns use shared factors and independent shocks, with sample means centered to stylized annual expected returns.

Kaplan-Schoar PME is `sum(distributions/index) / sum(calls/index)`, adding residual NAV divided by the terminal index level to the numerator when requested. This ratio was verified in the original [Kaplan and Schoar paper, Table VI description](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2005.00780.x). Index levels represent one positive synthetic annual total return index observed on calendar cash-flow dates. Every vintage uses the same level for the same calendar year. Residual NAV is a marked value, not a guaranteed realization. Index scaling cancels.

IRR uses a custom bisection in log gross return, with log-magnitude discounted cash flows to avoid overflow and spurious endpoint roots. Times are annual by default and may be supplied explicitly. Zero cash flows are ignored for sign-counting. Only conventional outflows followed by inflows are accepted; multiple sign changes are rejected because a single IRR would be ambiguous. The numerical search is bounded in log gross return and rejects roots that cannot be bracketed or represented above negative one. TVPI is `(distributions + residual_NAV) / paid_in`; DPI is `distributions / paid_in`; RVPI is `residual_NAV / paid_in`.

## Whole-portfolio allocation

The optimizer minimizes `risk_aversion / 2 * w' covariance w - mean' w`. Constraints are full investment, long-only weights, an illiquid share cap and stressed liquidity coverage:

```text
sum(w) = 1
w >= 0
illiquid_mask' w <= maximum_illiquid_share
liquid_haircuts' w >= fixed_liquidity_need + stressed_outflows' w
```

Haircuts here are available liquid value per dollar invested, not the percentage deduction. Illiquid assets provide no liquid value. Fixed liquidity needs and each sleeve's stressed annual outflow coefficient are stylized fractions of portfolio NAV. They are separate internal planning assumptions, not fitted to the treasury example's cash ladder or regulatory stress rates.

For a small universe, the solver enumerates active inequality sets, solves each equality-constrained KKT system using NumPy, and checks primal feasibility, nonnegative inequality multipliers and stationarity. A feasible KKT solution certifies the convex optimum. A symmetric positive definite covariance is required. The implementation rejects singular or indefinite matrices and more than eight assets rather than silently regularizing or attempting expensive large-universe enumeration.

The efficient frontier starts at constrained minimum variance and increases required expected return to the maximum feasible return. Each row minimizes variance with a return floor; the upper branch has nondecreasing return and volatility. Allocation comparison holds the policy and risk aversion fixed and changes the estimated means and covariance together. Both allocations are also evaluated on the same unsmoothed covariance to expose the naive risk understatement. It is an illustrative single-period optimization, not a transaction-cost-aware or dynamically executable investment strategy.

## Validation scope

The suite checks analytic root and allocation benchmarks, NAV and unfunded conservation, program additivity, terminal liquidation, J-curve behavior, target hold in a feasible pacing case, cash and facility reconciliation, repayment horizons, policy threshold transitions, exact synthetic unsmoothing recovery, index-matched PME, known IRRs and monotone efficient frontiers. Edge cases reject invalid inputs and ambiguous IRRs. Examples write all outputs to `examples/output`, and the committed result tables are reproducible from the runner.

Source verification covers paper structures and explicitly identified sections. All fund parameters, policy bands, facility terms, haircuts, stress multipliers and allocation expectations remain stylized. No live market data, regulatory compliance determination, legal interpretation, tax model, fee waterfall or production funding guarantee is included.

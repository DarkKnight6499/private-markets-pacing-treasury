"""Run every offline example and write synthetic CSV and Markdown artifacts."""

from pathlib import Path
import sys
import json
import math
import unittest
from dataclasses import asdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pm_pacing.ta_model import FundParameters, Vintage, project_fund, project_program, STYLIZED_PARAMETERS
from pm_pacing.pacing import solve_pacing, sensitivity_table
from pm_pacing.treasury import TreasuryPolicy, cash_ladder, quarterly_flows, stress_flows
from pm_pacing.compliance import Exposure, Holding, check_trade, overall_status
from pm_pacing.cma import unsmooth, smooth, estimate_ar1, autocorrelation, capital_market_assumptions, irr, ks_pme, multiples
from pm_pacing.unsmoothing import unsmooth_ma1, estimate_ma1
from pm_pacing.allocation import AllocationConstraints, optimize, efficient_frontier
from pm_pacing.data import SEED, ASSET_NAMES, synthetic_glm_returns, synthetic_index, synthetic_vintages, synthetic_exposures, stylized_rules

OUTPUT = ROOT / "examples" / "output"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a plain Markdown table without an optional formatting dependency."""
    def cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if math.isfinite(value):
                return f"{0.0 if abs(value) < 0.5e-6 else value:.6f}"
            return str(value)
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("-" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines) + "\n"


def write_table(name: str, frame: pd.DataFrame) -> None:
    """Write deterministic data and a readable synthetic table."""
    frame.to_csv(OUTPUT / f"{name}.csv", index=False, float_format="%.12g")
    (OUTPUT / f"{name}.md").write_text(
        f"# {name.replace('_', ' ').title()}\n\nSynthetic data and stylized assumptions.\n\n"
        + markdown_table(frame), encoding="utf-8")


def run() -> dict[str, object]:
    """Exercise all modules and return measured summary metrics."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vintages = synthetic_vintages()
    detail, program = project_program(vintages)
    write_table("fund_cash_flows", detail)
    write_table("program_cash_flows", program)
    parameter_rows = [{"strategy": name, "life": p.life, "growth": p.growth,
                       "call_rates": str(p.call_rates), "bow": p.bow, "yield_rate": p.yield_rate,
                       "source_status": "stylized", "fee_carry_assumption": "none" if name == "co_invest" else "embedded in assumed net growth"}
                      for name, p in STYLIZED_PARAMETERS.items()]
    write_table("fund_parameters", pd.DataFrame(parameter_rows))
    calendar_index = pd.Series(synthetic_index(len(program)), index=program.year)
    write_table("public_index", calendar_index.rename("index").rename_axis("year").reset_index())
    fund_performance = []
    for fund in vintages:
        flows = detail[detail.fund == fund.name]
        index = calendar_index.loc[flows.year].to_numpy()
        write_table(f"index_{fund.name}", pd.DataFrame({"year": flows.year, "index": index}))
        fund_performance.append({"fund": fund.name, "irr": irr(flows.net_cash_flow),
                                 "ks_pme": ks_pme(flows.calls, flows.distributions, index),
                                 **multiples(flows.calls, flows.distributions, float(flows.nav.iloc[-1]))})
    performance = pd.DataFrame(fund_performance)
    write_table("fund_performance", performance)
    standard = solve_pacing(1000, .25, 25, STYLIZED_PARAMETERS["buyout"], 400)
    feasible_parameters = FundParameters(6, .1, (.5,), 1)
    feasible = solve_pacing(1000, .25, 25, feasible_parameters, 1000)
    write_table("pacing_standard_buyout", standard)
    write_table("pacing_feasible_ramp", feasible)
    write_table("pacing_assumptions", pd.DataFrame([
        {"scenario": "standard_buyout", "initial_public_nav": 1000, "target": .25,
         "years": 25, "ramp_years": 4, "annual_cap": 400, "public_growth": .05,
         "life": 12, "fund_growth": .12, "call_rates": "(0.25, 0.35, 0.50)", "bow": 2.5},
        {"scenario": "feasible_ramp", "initial_public_nav": 1000, "target": .25,
         "years": 25, "ramp_years": 4, "annual_cap": 1000, "public_growth": .05,
         "life": 6, "fund_growth": .1, "call_rates": "(0.5,)", "bow": 1}]))
    write_table("pacing_sensitivity", sensitivity_table(1000, .25, 25, STYLIZED_PARAMETERS["buyout"], 400))
    quarters = quarterly_flows(program)
    write_table("quarterly_flows", quarters)
    policy = TreasuryPolicy(line_limit=30, repayment_quarters=2)
    write_table("treasury_policy", pd.DataFrame([{**asdict(policy), "initial_cash": 15,
                                                "source_status": "stylized"}]))
    scenarios = {"base": (quarters.calls.to_numpy(), quarters.distributions.to_numpy()),
                 "distributions_halved": stress_flows(quarters.calls, quarters.distributions, acceleration=0),
                 "calls_accelerated": stress_flows(quarters.calls, quarters.distributions, distribution_multiplier=1),
                 "combined_stress": stress_flows(quarters.calls, quarters.distributions)}
    treasury_rows = []
    ladders = {}
    for name, (calls, distributions) in scenarios.items():
        ladder = cash_ladder(calls, distributions, policy, initial_cash=15, periods=quarters.period.tolist())
        ladders[name] = ladder
        write_table(f"treasury_{name}", ladder)
        finite_coverage = ladder.loc[np.isfinite(ladder.liquidity_coverage_12m), "liquidity_coverage_12m"]
        treasury_rows.append({"scenario": name, "lp_funding": ladder.lp_funding.sum(),
                              "interest": ladder.interest.sum(), "cash_income": ladder.cash_income.sum(),
                              "peak_line": ladder.closing_line.max(), "final_line": ladder.closing_line.iloc[-1],
                              "minimum_coverage": finite_coverage.min(),
                              "peak_funding_shortfall": ladder.funding_shortfall.max()})
    treasury = pd.DataFrame(treasury_rows)
    write_table("treasury_summary", treasury)
    exposures = synthetic_exposures()
    write_table("compliance_portfolio", pd.DataFrame([
        {"fund": e.fund, "manager": e.manager, "strategy": e.strategy,
         "vintage": e.vintage, "geography": e.geography, "nav": e.nav,
         "unfunded": e.unfunded, "economic_exposure": e.economic_exposure}
        for e in exposures]))
    write_table("issuer_lookthrough", pd.DataFrame([
        {"fund": e.fund, "issuer": h.issuer, "weight": h.weight,
         "sector": h.sector, "jurisdiction": h.jurisdiction,
         "economic_exposure": h.weight * e.economic_exposure}
        for e in exposures for h in e.holdings]))
    rules = stylized_rules()
    (OUTPUT / "policy_rules.json").write_text(json.dumps(rules, indent=2) + "\n", encoding="utf-8")
    candidates = {
        "pass": Exposure("new_fund", "new_manager", "buyout", 2026, "Europe", 0, 10,
                         (Holding("new_issuer", 1, "technology", "US"),)),
        "warn": Exposure("fund_0", "manager_0", "buyout", 2026, "North America", 0, 25,
                         (Holding("new_issuer", 1, "technology", "US"),)),
        "fail": Exposure("large_co_invest", "manager_0", "co_invest", 2026, "North America", 160, 0,
                         (Holding("shared_issuer", 1, "thermal_coal", "Excluded_Jurisdiction"),)),
    }
    compliance_rows = []
    for name, trade in candidates.items():
        checks = check_trade(exposures, trade, 1000, rules)
        write_table(f"compliance_{name}", checks)
        compliance_rows.append({"candidate": name, "overall": overall_status(checks),
                                "pass": int((checks.status == "pass").sum()),
                                "warn": int((checks.status == "warn").sum()),
                                "fail": int((checks.status == "fail").sum())})
    write_table("compliance_summary", pd.DataFrame(compliance_rows))
    truth, reported, coefficients = synthetic_glm_returns()
    estimated = reported.iloc[1:].copy()
    exact = reported.copy()
    diagnostics = []
    for name, coefficient in coefficients.items():
        alpha = estimate_ma1(reported[name])
        estimated[name] = unsmooth_ma1(reported[name], alpha, previous_true=0)[1:]
        exact[name] = unsmooth_ma1(reported[name], coefficient, previous_true=0)
        diagnostics.append({"asset": name, "true_lag_weight": coefficient, "estimated_lag_weight": alpha,
                            "reported_ar1_slope": estimate_ar1(reported[name]),
                            "reported_volatility": reported[name].iloc[1:].std()*2,
                            "unsmoothed_volatility": estimated[name].std()*2,
                            "reported_autocorrelation": autocorrelation(reported[name]),
                            "unsmoothed_autocorrelation": autocorrelation(estimated[name]),
                            "known_weight_max_error": np.abs(exact[name] - truth[name]).max()})
    diagnostics_frame = pd.DataFrame(diagnostics)
    write_table("unsmoothing_diagnostics", diagnostics_frame)
    write_table("returns_true", truth.rename_axis("quarter_index").reset_index())
    write_table("returns_reported", reported.rename_axis("quarter_index").reset_index())
    write_table("returns_unsmoothed", estimated.rename_axis("quarter_index").reset_index())
    reported_mean, reported_cov = capital_market_assumptions(reported.iloc[1:])
    unsmoothed_mean, unsmoothed_cov = capital_market_assumptions(estimated)
    write_table("cma_comparison", pd.DataFrame({"asset": ASSET_NAMES,
                "reported_mean": reported_mean.to_numpy(), "unsmoothed_mean": unsmoothed_mean.to_numpy(),
                "reported_volatility": np.sqrt(np.diag(reported_cov)),
                "unsmoothed_volatility": np.sqrt(np.diag(unsmoothed_cov))}))
    write_table("covariance_reported", reported_cov.rename_axis("asset").reset_index())
    write_table("covariance_unsmoothed", unsmoothed_cov.rename_axis("asset").reset_index())
    limits = AllocationConstraints((False, False, True, True, True), .45,
                                  (.6, .9, 0, 0, 0), (0, 0, .5, .4, .45), .15)
    write_table("allocation_policy", pd.DataFrame([
        {"asset": name, "illiquid": illiquid, "available_liquid_fraction": haircut,
         "stressed_outflow_12m_per_nav": stress,
         "max_illiquid_share": limits.max_illiquid,
         "fixed_liquidity_need": limits.fixed_liquidity_need, "source_status": "stylized"}
        for name, illiquid, haircut, stress in zip(ASSET_NAMES, limits.illiquid,
                                                  limits.liquid_haircuts, limits.stressed_outflows)]))
    allocation_rows = []
    allocation_results = {}
    for name, mean, covariance in [("reported", reported_mean, reported_cov),
                                   ("unsmoothed", unsmoothed_mean, unsmoothed_cov)]:
        result = optimize(mean, covariance, limits, risk_aversion=4)
        allocation_results[name] = result
        allocation_rows.append({"cma": name, "expected_return": result.expected_return,
                                "modeled_volatility": result.volatility,
                                "volatility_on_unsmoothed_covariance": math.sqrt(float(result.weights @ unsmoothed_cov @ result.weights)),
                                "illiquid_share": result.weights[2:].sum(),
                                "liquidity_surplus": result.liquidity_surplus,
                                "stationarity_residual": result.stationarity_residual,
                                **dict(zip(ASSET_NAMES, result.weights.tolist()))})
        write_table(f"frontier_{name}", efficient_frontier(mean, covariance, limits, asset_names=ASSET_NAMES))
    allocations = pd.DataFrame(allocation_rows)
    write_table("allocation_comparison", allocations)
    buyout = performance[performance.fund == "buyout_A"].iloc[0]
    base_ladder = ladders["base"]
    nav_error = np.abs(detail.nav - detail.opening_nav - detail.growth_pnl - detail.calls + detail.distributions).max()
    cash_error = np.abs(base_ladder.closing_cash - base_ladder.opening_cash - base_ladder.cash_income
                        - base_ladder.distributions - base_ladder.draw - base_ladder.lp_funding
                        + base_ladder.calls + base_ladder.repayment + base_ladder.interest).max()
    summary = {"seed": SEED, "synthetic_funds": len(vintages),
               "buyout_irr": float(buyout.irr), "buyout_tvpi": float(buyout.tvpi), "buyout_pme": float(buyout.ks_pme),
               "program_calls": float(program.calls.sum()), "program_distributions": float(program.distributions.sum()),
               "maximum_nav_identity_error": float(nav_error), "maximum_treasury_cash_error": float(cash_error),
               "feasible_target": .25, "feasible_final_allocation": float(feasible.allocation.iloc[-1]),
               "feasible_max_target_error": float(feasible.target_error.abs().max()),
               "feasible_final_commitment": float(feasible.commitment.iloc[-1]),
               "standard_buyout_max_overshoot": float(standard.target_error.max()),
               "base_treasury_interest": float(treasury.loc[treasury.scenario == "base", "interest"].iloc[0]),
               "base_lp_funding": float(treasury.loc[treasury.scenario == "base", "lp_funding"].iloc[0]),
               "stress_lp_funding": float(treasury.loc[treasury.scenario == "combined_stress", "lp_funding"].iloc[0]),
               "peak_line": float(base_ladder.closing_line.max()), "line_limit": policy.line_limit,
               "pe_reported_volatility": float(diagnostics_frame.iloc[0].reported_volatility),
               "pe_unsmoothed_volatility": float(diagnostics_frame.iloc[0].unsmoothed_volatility),
               "pe_reported_autocorrelation": float(diagnostics_frame.iloc[0].reported_autocorrelation),
               "pe_unsmoothed_autocorrelation": float(diagnostics_frame.iloc[0].unsmoothed_autocorrelation),
               "known_weight_recovery_error": float(diagnostics_frame.known_weight_max_error.max()),
               "allocation_reported_private_equity": float(allocation_results["reported"].weights[2]),
               "allocation_unsmoothed_private_equity": float(allocation_results["unsmoothed"].weights[2]),
               "unsmoothed_allocation_return": allocation_results["unsmoothed"].expected_return,
               "unsmoothed_allocation_volatility": allocation_results["unsmoothed"].volatility,
               "unsmoothed_allocation_liquidity_surplus": allocation_results["unsmoothed"].liquidity_surplus,
               "analytical_irr": irr([-100, 0, 121]),
               "index_matched_pme": ks_pme([100, 0, 0], [0, 0, 121], [100, 110, 121]),
               "test_count": unittest.TestLoader().discover(str(ROOT / "tests")).countTestCases()}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_table("summary", pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()]))
    report = "# Synthetic results\n\nAll values below are computed by this example. Monetary units are synthetic millions.\n\n"
    for title, frame in [("Fund performance", performance), ("Treasury", treasury),
                         ("Unsmoothing", diagnostics_frame), ("Allocation", allocations)]:
        report += f"## {title}\n\n" + markdown_table(frame) + "\n"
    report += "The subscription line shifts LP funding dates and adds interest. LP top-ups are explicit funding requirements. "
    report += "The standard buyout ramp has unavoidable overshoot from scheduled future calls. "
    report += "The faster distribution example demonstrates a feasible target ramp. "
    report += "Appraisal filtering uses an invertible MA(1) specialization of Getmansky-Lo-Makarov, verified in Section 4, equations (21) to (23). Lag weights are stylized; estimation uses a method-of-moments equation derived for iid true returns.\n"
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

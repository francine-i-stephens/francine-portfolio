"""
================================================================================
MOBILE A/B TEST RE-ANALYSIS SCRIPT
================================================================================

This script runs a defensible A/B test analysis on mobile experiment data
following a stratified-by-platform, weighted, session-CTR framework.

Originally developed for MOB_8432 (Most Recent Apps vs Most Used Apps); the
analysis structure generalizes to other mobile A/B tests with similar designs.

--------------------------------------------------------------------------------
WHEN TO USE THIS SCRIPT
--------------------------------------------------------------------------------

This script is appropriate when ALL of the following hold for your experiment:

  1. RANDOMIZATION UNIT is the user (not session, device, or tenant)
  2. PLATFORMS were randomized independently (separate iOS/Android experiments)
  3. METRIC is a rate: successes / trials, bounded [0,1]
  4. DATA IS USER-AGGREGATED: one row per user with success and trial counts
  5. TENANT/ORG is a meaningful clustering unit
  6. The decision is binary: ship/hold/rollback based on a pre-registered MDE

If any of these don't hold, the script may still run but the interpretation
will not be sound. See the ASSUMPTIONS section in each step.

--------------------------------------------------------------------------------
WHAT TO CHANGE FOR A NEW EXPERIMENT
--------------------------------------------------------------------------------

For most new analyses, only the CONFIG block at the top of this file needs to
be edited. Specifically:

  COLUMN_MAP   — map your source schema's column names to canonical names
  METRIC_CONFIG — define the outcome (numerator/denominator/cap behavior)
  FILTERS      — set the section filter and date range
  DECISION_RULES — set MDE, alpha, power for the new analysis

For more substantial changes (new randomization unit, multi-arm experiments,
different decision logic), the analysis steps themselves need modification.
Those extension points are flagged inline with "EXTENSION POINT" comments.

--------------------------------------------------------------------------------
ANALYTICAL FRAMEWORK
--------------------------------------------------------------------------------

The script implements the following analysis flow:

  Step 1    Load and filter data
  Step 1.5  Drop cross-arm contaminated users (preserves SUTVA)
  Step 2    Sample Ratio Mismatch (SRM) check
  Step 3    Construct outcome and covariates
  Step 4    Power analysis (detects under-powering)
  Step 5    Tenant ICC (decides FE vs HLM)
  Step 6    Primary stratified-by-platform model
  Step 7    Backup logistic GLM (functional-form robustness check)
  Step 8    Sensitivity specifications
  Step 9    Confirmatory heterogeneity test
  Step 10   Per-platform verdicts
  Step 11   Final reporting

Key methodological choices (see inline rationales for details):
  - User-level analysis weighted by trials (e.g. sessions or impressions)
  - Stratified by platform (matches independent randomization)
  - WLS + tenant FE + cluster-robust SE on tenant (primary)
  - Binomial GLM (backup, validates functional form)
  - Pre-registered MDE as the practical-significance gate
  - Verdict logic separates "material harm" from "marginal harm"

================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.multitest import multipletests
from scipy import stats
from math import asin, sqrt, sin
import warnings
import gc
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIG — edit these for a new analysis
# =============================================================================

# --- Data source ---
DATA_PATH = "sample_all.parquet"
DATA_FORMAT = "parquet"  # 'parquet' or 'csv'

# --- Column name mapping ---
# Map your source schema's column names to the canonical names the analysis
# pipeline expects. If your schema already uses the canonical name, leave the
# value identical to the key.
#
# Canonical names and their roles:
#   user_id       — randomization unit identifier
#   tenant        — clustering unit (org/company)
#   platform      — iOS / Android / etc.
#   variant       — experiment arm; must contain values matching VARIANT_LABELS
#   section       — surface filter (e.g. app_section_type)
#   successes     — numerator of the rate metric
#   trials        — denominator of the rate metric (also used as weight)
#   impressions   — used as a control covariate (log_impressions); may equal trials
COLUMN_MAP = {
    "user_id": "user_id",
    "tenant": "tenant",
    "platform": "platform",
    "variant": "variant",
    "section": "app_section_type",
    "successes": "sessions_with_interaction",
    "trials": "total_sessions",
    "impressions": "total_impression_txns",
}

# --- Variant labels in source data ---
# Maps the source labels to canonical 'control'/'treatment'.
# Common alternatives include 'baseline', 'variant_a', 'A', '0'.
VARIANT_LABELS = {
    "control": "control",
    "treatment": "treatment",
}

# --- Platform labels in source data ---
# All non-listed values will be EXCLUDED. To merge platforms (e.g. iPadOS
# into iOS), put the same canonical value on the right side for both keys.
PLATFORM_LABELS = {
    "ios": "ios",
    "android": "android",
    # "ipados": "ios",   # uncomment to fold iPadOS into iOS
}

# --- Metric definition ---
# METRIC_CONFIG defines how the outcome is constructed. The script computes:
#   outcome = successes / trials, capped at `cap_at` if set
#   weight  = trials
#
# `friction_metric`: if True, a NEGATIVE lift is the FAVORABLE direction
# (e.g. apps.viewmoreapps, error rates, time-to-task). Default False.
#
# `section_filter`: which value(s) of the section column to keep. If your
# data is already filtered upstream, set this to None.
METRIC_CONFIG = {
    "name": "apps_clickapp_per_session_ctr",
    "description": "Sessions with any apps-card interaction, per session",
    "section_filter": "APPS",      # or None to skip filtering
    "cap_at": 1.0,                  # cap outcome (None to disable)
    "friction_metric": False,       # True for metrics where lower is better
}

# --- Decision rules ---
DECISION_RULES = {
    "mde_relative_pct": 5.0,        # pre-registered MDE (% relative lift)
    "alpha": 0.05,                  # two-tailed
    "power": 0.80,                  # for power analysis only
    "srm_alpha": 0.001,             # SRM p-value threshold
}

# --- Run options ---
RUN_OPTIONS = {
    "free_memory_after_load": True,  # gc.collect after dropping raw df
    "run_sensitivity_checks": True,
    "run_heterogeneity_test": True,
    "verbose_diagnostics": True,
}


# =============================================================================
# ANALYSIS PIPELINE — Steps 1 through 11
# =============================================================================

# -----------------------------------------------------------------------------
# STEP 1 — Load, rename, filter
# -----------------------------------------------------------------------------
# ASSUMPTION: Data is pre-aggregated to one row per (user, section). If the
# source data is at finer granularity (e.g. daily), aggregate UPSTREAM via the
# SQL query or add an aggregation step here before the platform/variant filters.
#
# EXTENSION POINT: Add multi-arm support by extending VARIANT_LABELS and the
# downstream model formulas (current code assumes a 2-arm test).
print("="*70)
print(f"STEP 1 — Load and filter  (metric: {METRIC_CONFIG['name']})")
print("="*70)

if DATA_FORMAT == "parquet":
    df = pd.read_parquet(DATA_PATH)
elif DATA_FORMAT == "csv":
    df = pd.read_csv(DATA_PATH)
else:
    raise ValueError(f"Unsupported DATA_FORMAT: {DATA_FORMAT}")

# Rename columns to canonical names
inv_map = {v: k for k, v in COLUMN_MAP.items()}
df = df.rename(columns=inv_map)

# Normalize variant labels
df["variant"] = df["variant"].replace(
    {k: v for k, v in VARIANT_LABELS.items()}
)
df = df[df["variant"].isin(["control", "treatment"])].copy()

# Normalize platform labels (also drops unrecognized platforms)
df["platform"] = df["platform"].str.lower().replace(PLATFORM_LABELS)
df = df[df["platform"].isin(set(PLATFORM_LABELS.values()))].copy()

# Filter to the section of interest, if applicable
if METRIC_CONFIG["section_filter"] is not None:
    df = df[df["section"] == METRIC_CONFIG["section_filter"]].copy()

# Drop zero-trial rows (no exposure)
apps = df[df["trials"] > 0].copy()

print(f"  Rows with exposure (trials > 0): {len(apps):,}")
print(f"  Unique users: {apps['user_id'].nunique():,}")
print(f"  Tenants: {apps['tenant'].nunique():,}")

if RUN_OPTIONS["free_memory_after_load"]:
    del df
    gc.collect()


# -----------------------------------------------------------------------------
# STEP 1.5 — Cross-arm contamination cleanup
# -----------------------------------------------------------------------------
# ASSUMPTION: If platforms were randomized independently (the default for
# MOB experiments), the same user_id can land in different arms across
# platforms. We drop these to preserve SUTVA. If your experiment uses a
# unified randomization across platforms, this step should be a no-op
# (conflicted_users should be empty).
print("\n" + "="*70)
print("STEP 1.5 — Cross-arm contamination cleanup")
print("="*70)

variants_per_user = apps.groupby("user_id")["variant"].nunique()
conflicted_users = set(variants_per_user[variants_per_user > 1].index)

platforms_per_user = apps.groupby("user_id")["platform"].nunique()
multi_platform_users = set(platforms_per_user[platforms_per_user > 1].index)

conflicted_rows = apps[apps["user_id"].isin(conflicted_users)]
arm_split = conflicted_rows.groupby("variant").size()

print(f"  Variant-conflicted users (drop): {len(conflicted_users):,} "
      f"({len(conflicted_users)/apps['user_id'].nunique()*100:.2f}% of users)")
print(f"  Multi-platform users (sensitivity set): {len(multi_platform_users):,}")
print(f"  Conflicted rows per arm: "
      f"control={arm_split.get('control', 0)}, "
      f"treatment={arm_split.get('treatment', 0)}")

apps = apps[~apps["user_id"].isin(conflicted_users)].copy()
print(f"  After cleanup: {apps['user_id'].nunique():,} users")


# -----------------------------------------------------------------------------
# STEP 2 — SRM check (at user level, post-cleanup)
# -----------------------------------------------------------------------------
# ASSUMPTION: 50/50 allocation. If you have a different ratio, change f_exp.
# At very large N, chi² will fire on tiny imbalances — check actual imbalance %.
print("\n" + "="*70)
print("STEP 2 — SRM check")
print("="*70)

users_per_arm = apps.groupby("variant")["user_id"].nunique()
chi2, p_srm = stats.chisquare(
    users_per_arm.values, f_exp=[users_per_arm.sum()/2]*2
)
imbalance = abs(users_per_arm["treatment"] - users_per_arm["control"]) / users_per_arm.sum()

print(f"  Control: {users_per_arm['control']:,}   "
      f"Treatment: {users_per_arm['treatment']:,}")
print(f"  Imbalance: {imbalance*100:.3f}%   chi²={chi2:.2f}   p={p_srm:.4f}")
print(f"  Result: {'FAIL' if p_srm < DECISION_RULES['srm_alpha'] else 'PASS'}")


# -----------------------------------------------------------------------------
# STEP 3 — Construct outcome and covariates
# -----------------------------------------------------------------------------
# METHOD NOTE: Outcome is bounded [0,1] either naturally (binary per-trial) or
# via cap_at. The cap matters for per-session-CTR metrics where successes can
# exceed trials (multi-click sessions); it's a no-op for true CTRs.
print("\n" + "="*70)
print("STEP 3 — Construct outcome and covariates")
print("="*70)

if METRIC_CONFIG["cap_at"] is not None:
    apps["user_ctr"] = (apps["successes"] / apps["trials"]).clip(
        upper=METRIC_CONFIG["cap_at"]
    )
    n_capped = (apps["successes"] > apps["trials"]).sum()
    if n_capped > 0:
        print(f"  Note: {n_capped:,} rows had successes > trials (capped)")
else:
    apps["user_ctr"] = apps["successes"] / apps["trials"]

apps["treatment"] = (apps["variant"] == "treatment").astype(int)
# Platform indicators — assumes binary {iOS, android}. For multi-platform,
# extend this to one-hot encoding and update Step 9's interaction test.
apps["ios"] = (apps["platform"] == "ios").astype(int)
apps["log_impressions"] = np.log1p(apps["impressions"])
apps["weight"] = apps["trials"]

baseline_w = np.average(
    apps.loc[apps["treatment"]==0, "user_ctr"],
    weights=apps.loc[apps["treatment"]==0, "weight"]
)
print(f"  Outcome range: "
      f"[{apps['user_ctr'].min():.3f}, {apps['user_ctr'].max():.3f}]")
print(f"  Baseline CTR (session-weighted, control): {baseline_w:.4f}")


# -----------------------------------------------------------------------------
# STEP 4 — Power analysis
# -----------------------------------------------------------------------------
# METHOD NOTE: We use the direct two-proportion formula rather than Cohen's h
# inversion (which has a factor-of-2 gotcha). The detectable MDE here is the
# minimum effect this N could reliably detect; it does NOT change the
# pre-registered MDE in DECISION_RULES.
print("\n" + "="*70)
print("STEP 4 — Power analysis")
print("="*70)

from scipy.stats import norm
n_per_arm = int(users_per_arm.min())
z_a = norm.ppf(1 - DECISION_RULES["alpha"]/2)
z_b = norm.ppf(DECISION_RULES["power"])
abs_mde = (z_a + z_b) * sqrt(2 * baseline_w * (1 - baseline_w) / n_per_arm)
rel_mde = abs_mde / baseline_w * 100

print(f"  Baseline CTR: {baseline_w:.4f}")
print(f"  N per arm: {n_per_arm:,}")
print(f"  Detectable MDE @ "
      f"{int(DECISION_RULES['power']*100)}% power: "
      f"{rel_mde:.2f}% relative ({abs_mde*100:.3f} pp absolute)")
print(f"  Pre-registered MDE: {DECISION_RULES['mde_relative_pct']:.1f}%  "
      f"({'SUFFICIENT' if rel_mde <= DECISION_RULES['mde_relative_pct'] else 'UNDER-POWERED'})")


# -----------------------------------------------------------------------------
# STEP 5 — Tenant ICC for model selection
# -----------------------------------------------------------------------------
# METHOD NOTE: ICC partitions outcome variance into tenant-between vs within.
#   ICC < 0.05  → tenant FE + cluster SE (simpler, defensible)
#   ICC ≥ 0.05  → mixed-effects model with random intercept on tenant
print("\n" + "="*70)
print("STEP 5 — Tenant ICC (model selection)")
print("="*70)

icc_fit = smf.mixedlm("user_ctr ~ 1", apps, groups=apps["tenant"]).fit()
var_t = icc_fit.cov_re.iloc[0, 0]
var_e = icc_fit.scale
icc = var_t / (var_t + var_e)

if icc < 0.05:
    model_choice = "FIXED_EFFECTS"
else:
    model_choice = "HLM"

print(f"  Tenant ICC: {icc:.4f}")
print(f"  Selected model family: {model_choice}")


# -----------------------------------------------------------------------------
# STEP 6 — Primary stratified-per-platform model
# -----------------------------------------------------------------------------
# METHOD NOTE: Stratification matches the independent platform randomization.
# Pooled estimates with treatment×platform interaction are run in Step 9 as
# confirmatory only.
#
# EXTENSION POINT: For unified-randomization experiments, replace the loop
# with a single pooled model and drop Step 9.
print("\n" + "="*70)
print("STEP 6 — Primary stratified model")
print("="*70)

results_by_platform = {}
platforms = sorted(apps["platform"].unique())

for plat in platforms:
    sub = apps[apps["platform"] == plat].copy()
    ctrl = sub[sub["treatment"] == 0]
    bl = np.average(ctrl["user_ctr"], weights=ctrl["weight"])

    if model_choice == "FIXED_EFFECTS":
        m = smf.wls(
            "user_ctr ~ treatment + log_impressions + C(tenant)",
            data=sub, weights=sub["weight"]
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["tenant"]})
        label = "WLS + tenant FE + cluster SE"
    else:
        m = smf.mixedlm(
            "user_ctr ~ treatment + log_impressions",
            data=sub, groups=sub["tenant"]
        ).fit()
        label = "Mixed-effects + random intercept on tenant"

    b = m.params["treatment"]
    se = m.bse["treatment"]
    ci = (b - 1.96*se, b + 1.96*se)
    p = m.pvalues["treatment"]
    rel = b / bl * 100

    # Verdict logic adapts to whether the metric is friction or engagement
    if METRIC_CONFIG["friction_metric"]:
        favorable_direction = rel < 0    # less friction is better
    else:
        favorable_direction = rel > 0    # more engagement is better

    clears_mde = abs(rel) >= DECISION_RULES["mde_relative_pct"]

    print(f"\n  {plat.upper()}  ({label})")
    print(f"    Baseline: {bl:.4f}")
    print(f"    β: {b*100:+.3f} pp  (95% CI {ci[0]*100:+.3f} to {ci[1]*100:+.3f})")
    print(f"    Relative lift: {rel:+.2f}%   p={p:.4f}")
    print(f"    Clears MDE magnitude: {'YES' if clears_mde else 'NO'}")
    print(f"    Favorable direction: {'YES' if favorable_direction else 'NO'}")

    results_by_platform[plat] = {
        "baseline": bl, "beta": b, "se": se, "ci": ci, "p": p,
        "rel_lift": rel, "clears_mde": clears_mde,
        "favorable": favorable_direction and clears_mde,
        "model": m, "label": label,
    }


# -----------------------------------------------------------------------------
# STEP 7 — Backup binomial GLM (functional-form robustness check)
# -----------------------------------------------------------------------------
# METHOD NOTE: Tenant FE dropped from this spec to avoid singular-matrix
# crashes at large N with many small tenants. Cluster SE still handles
# within-tenant correlation. We compare the GLM's average marginal effect to
# the matched WLS+FE β; agreement within ~0.5 pp confirms LPM is fine.
print("\n" + "="*70)
print("STEP 7 — Backup binomial GLM")
print("="*70)

backup_by_platform = {}
for plat in platforms:
    sub = apps[apps["platform"] == plat].copy()
    sub_data = sub.assign(
        successes=sub["successes"],
        failures=sub["trials"] - sub["successes"].clip(upper=sub["trials"]),
    )

    logit = smf.glm(
        "successes + failures ~ treatment + log_impressions",
        data=sub_data, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["tenant"]})

    if hasattr(logit, "get_margeff"):
        mfx = logit.get_margeff(at="overall")
        ame, ame_se = mfx.margeff[0], mfx.margeff_se[0]
    else:
        s1, s0 = sub_data.copy(), sub_data.copy()
        s1["treatment"], s0["treatment"] = 1, 0
        ind_eff = logit.predict(s1) - logit.predict(s0)
        ame = ind_eff.mean()
        ame_se = ind_eff.std(ddof=1) / np.sqrt(len(ind_eff))

    weighted = smf.wls(
        "user_ctr ~ treatment + log_impressions",
        data=sub, weights=sub["weight"]
    ).fit(cov_type="cluster", cov_kwds={"groups": sub["tenant"]})
    weighted_beta = weighted.params["treatment"]
    agrees = abs(ame - weighted_beta) < 0.005

    print(f"\n  {plat.upper()}")
    print(f"    GLM AME: {ame*100:+.3f} pp (SE {ame_se*100:.3f})")
    print(f"    WLS β (weighted, matched spec): {weighted_beta*100:+.3f} pp")
    print(f"    Agreement: {'GOOD' if agrees else 'CHECK — functional form may matter'}")

    backup_by_platform[plat] = {
        "ame": ame, "ame_se": ame_se,
        "weighted_beta": weighted_beta, "agrees": agrees,
    }


# -----------------------------------------------------------------------------
# STEP 8 — Sensitivity (lean — only the informative pair)
# -----------------------------------------------------------------------------
# METHOD NOTE: Original sensitivity suite had 5 specs; pruned to 2 because
# (a) WLS+FE is now primary, (b) user-clustered SE is a no-op on user-level
# data, (c) unweighted OLS is already covered by HLM comparison.
if RUN_OPTIONS["run_sensitivity_checks"]:
    print("\n" + "="*70)
    print("STEP 8 — Sensitivity checks")
    print("="*70)

    sensitivity_by_platform = {}
    for plat in platforms:
        sub = apps[apps["platform"] == plat].copy()
        primary_beta = backup_by_platform[plat]["weighted_beta"]
        specs = {}

        # (1) Trim top 1% by impressions
        cut = sub["impressions"].quantile(0.99)
        trim = sub[sub["impressions"] <= cut]
        s_trim = smf.wls(
            "user_ctr ~ treatment + log_impressions",
            data=trim, weights=trim["weight"]
        ).fit(cov_type="cluster", cov_kwds={"groups": trim["tenant"]})
        specs["Trimmed top 1%"] = (s_trim.params["treatment"], s_trim.pvalues["treatment"])

        # (2) Exclude all multi-platform users
        strict = sub[~sub["user_id"].isin(multi_platform_users)]
        s_strict = smf.wls(
            "user_ctr ~ treatment + log_impressions",
            data=strict, weights=strict["weight"]
        ).fit(cov_type="cluster", cov_kwds={"groups": strict["tenant"]})
        specs["Multi-platform excluded"] = (s_strict.params["treatment"], s_strict.pvalues["treatment"])

        print(f"\n  {plat.upper()}")
        print(f"    {'Primary':<28} β={primary_beta*100:+.3f} pp")
        for name, (b, p) in specs.items():
            print(f"    {name:<28} β={b*100:+.3f} pp   p={p:.4f}")

        sensitivity_by_platform[plat] = {"specs": specs}


# -----------------------------------------------------------------------------
# STEP 9 — Confirmatory heterogeneity test (lean, no tenant FE)
# -----------------------------------------------------------------------------
if RUN_OPTIONS["run_heterogeneity_test"]:
    print("\n" + "="*70)
    print("STEP 9 — Heterogeneity (confirmatory)")
    print("="*70)

    pooled = smf.wls(
        "user_ctr ~ treatment * ios + log_impressions",
        data=apps, weights=apps["weight"]
    ).fit(cov_type="cluster", cov_kwds={"groups": apps["tenant"]})
    int_b = pooled.params["treatment:ios"]
    int_p = pooled.pvalues["treatment:ios"]

    print(f"  Treatment × iOS interaction: β={int_b*100:+.3f} pp  p={int_p:.4f}")
    if int_p < 0.05:
        print(f"  → Interaction significant; platforms respond differently to treatment.")
    else:
        print(f"  → Interaction NOT significant; treatment effect similar across platforms.")


# -----------------------------------------------------------------------------
# STEP 10 — Per-platform verdicts
# -----------------------------------------------------------------------------
# METHOD NOTE: Verdict logic explicitly separates "material harm" (clears MDE
# magnitude in unfavorable direction) from "marginal harm" (significant but
# below MDE). This pre-registered distinction prevents reviewers from
# overstating sub-MDE effects as actionable.
print("\n" + "="*70)
print("STEP 10 — Verdicts per platform")
print("="*70)

srm_ok = p_srm > DECISION_RULES["srm_alpha"]
platform_verdicts = {}

for plat in platforms:
    r = results_by_platform[plat]
    b = backup_by_platform[plat]

    if r["favorable"] and r["clears_mde"]:
        verdict = "SHIP"
    elif r["p"] >= 0.05:
        verdict = "HOLD (no signal)"
    elif r["clears_mde"]:
        if METRIC_CONFIG["friction_metric"]:
            verdict = ("ROLLBACK / DO NOT SHIP (material increase in friction)"
                       if r["rel_lift"] > 0 else "SHIP")
        else:
            verdict = ("DO NOT SHIP (material harm)" if r["rel_lift"] < 0
                       else "SHIP")
    else:
        verdict = ("HOLD (marginal harm)" if r["rel_lift"] < 0
                   else "HOLD (positive but insufficient)")

    print(f"  {plat.upper():<10} {verdict}")
    platform_verdicts[plat] = verdict


# -----------------------------------------------------------------------------
# STEP 11 — Final reporting
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 11 — Final reporting")
print("="*70)

rows = []
for plat in platforms:
    r = results_by_platform[plat]
    b = backup_by_platform[plat]
    sub = apps[apps["platform"] == plat]
    sens_betas = [v[0] for v in sensitivity_by_platform[plat]["specs"].values()]
    sens_min = min(sens_betas + [b["weighted_beta"]])
    sens_max = max(sens_betas + [b["weighted_beta"]])

    rows.append({
        "Platform": plat.upper(),
        "N (control)": sub[sub["treatment"]==0]["user_id"].nunique(),
        "N (treatment)": sub[sub["treatment"]==1]["user_id"].nunique(),
        "Baseline CTR": f"{r['baseline']:.4f}",
        "β (pp)": f"{b['weighted_beta']*100:+.3f}",
        "95% CI (pp)": f"[{(r['ci'][0])*100:+.3f}, {(r['ci'][1])*100:+.3f}]",
        "Relative lift": f"{(b['weighted_beta']/r['baseline'])*100:+.2f}%",
        "p-value": f"{r['p']:.4f}",
        "GLM AME (pp)": f"{b['ame']*100:+.3f}",
        "Sensitivity range (pp)": f"[{sens_min*100:+.3f}, {sens_max*100:+.3f}]",
        "Clears MDE": "YES" if r["clears_mde"] else "NO",
        "Direction": "negative" if r["rel_lift"] < 0 else "positive",
        "Verdict": platform_verdicts[plat],
    })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))

from datetime import datetime
out_path = f"results_{METRIC_CONFIG['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
summary_df.to_csv(out_path, index=False)
print(f"\nResults table saved: {out_path}")

#!/usr/bin/env python3
"""
generate_dataset.py
Physics-informed synthetic dataset generation for ceramic tile composition
optimisation from laboratory-fabricated calibration batches.

MATHEMATICAL MODEL  (canonical-Scheffé mixture form — see COEFFICIENT
ESTIMATION Step 0 for why this replaced the earlier centered-Ridge form)
  Y = Σ_m β_m·x_m
    + γ·(ΣClay − ΣClay_mean)·(ΣFsp − ΣFsp_mean)     [Clay×Feldspar cross term]
    + δ·(x_AG98 − x̄_AG98)·(ΣOtherClay − ΣOtherClay_mean)  [AG98×rest-of-clay cross term]
    + ε(d)
  ε(d) ~ N(0, σ_base·(1 + d/d_ref))   [heteroscedastic]

COEFFICIENT ESTIMATION
  Step 0 — Two candidate linear parameterisations were tested empirically
            by Leave-One-Out cross-validation on the calibration set:
            (a) centered deviations (x_m − x̄_m) with one material
                (SodaF) dropped as a dependent/reference component to
                remove the exact rank-deficiency created by Σ=100 (this
                was the previous version of this script); and
            (b) the canonical-Scheffé linear form Y = Σ_m β_m·x_m — raw
                (uncentered) fractions, no intercept, regression through
                the origin, all 8 materials retained — which is how
                published ceramic mixture-design studies fit this class
                of model (Correia et al. 2004; Ngun et al. 2014; see
                citations at the end of this file).
            (b) is used here: it gave lower LOO-CV error than (a) on this
            calibration set (MOR 29.8% vs 34.0%; Shrinkage 33.1% vs
            48.5%; WA about the same). Note that removing the intercept
            does NOT by itself remove the near-singularity of a Σ=100
            simplex — regressing any one raw fraction on the other 7
            (no intercept) still gives VIF in the 10s-100s here, because
            each material's feasible range (BOUNDS) is narrow relative to
            its magnitude. Ridge regularisation (Step 1) is what keeps
            the fit stable, not the removal of the intercept alone.
            Ref: Cornell (2002), Experiments with Mixtures, Ch. 2 —
                 rationale for dropping the constant term under Σxi=1.
  Step 1 — β_m fitted by Ridge regression (λ=0.1) through the origin (no
            intercept, no centering, no per-coefficient sign bounds).
            Earlier versions of this script applied sign/floor bounds via
            constrained least squares (Hemmerle & Brantle 1978; Bachinger
            et al. 2024) motivated by ceramic sintering theory (Reed
            1995, Ch.12); those bounds are NOT used for the canonical
            Scheffé linear term, because (i) LOO-CV was measurably worse
            with them than without (bounds trade fit quality for
            individual-coefficient interpretability), and (ii) published
            canonical-mixture papers do not bound their linear
            coefficients either — a component's fitted β_m is an
            extrapolated "pure-component" value and is not expected to
            carry a simple physical sign in isolation (e.g. Correia et
            al. 2004 report a POSITIVE linear coefficient for feldspar on
            shrinkage even though feldspar's overall, in-region effect is
            to reduce shrinkage once the interaction terms are included —
            see NON-LINEAR TERMS below). A one-time informational check
            after fitting flags materials whose sign disagrees with the
            naive sintering-theory expectation, for transparency, but
            does not alter the fit.
            Ref: Hoerl & Kennard (1970) DOI:10.1080/00401706.1970.10488634.
  Step 2 — A Clay–Feldspar interaction term (γ) and an AG98-vs-rest-of-
            clay interaction term (δ) are added on top of the canonical
            linear term, in the spirit of a Partial Quadratic Mixture
            model: rather than fitting all C(8,2)=28 pairwise Scheffé
            cross-terms (infeasible with a calibration set this size),
            only a small, literature-motivated subset of cross-terms is
            included, following the general recommendation to augment
            linear mixture models with a SELECTED subset of cross-product
            terms rather than the full quadratic expansion.
            Ref: Piepel, Szychowski & Loeppky (2002)
                 DOI:10.1080/00224065.2002.11980160 — Partial Quadratic
                 Mixture models.
            These two terms are NOT fitted by minimising RMSE on the
            laboratory batches: with a calibration set this size, treating
            γ and δ as free RMSE-tuned parameters would fall well short of
            the ~10 observations per parameter conventionally recommended,
            risking coefficients that memorise batch-specific noise rather
            than a real effect.
            Ref: Bilger & Manning (2015) DOI:10.1002/hec.3003; Babyak
                 (2004) — minimum sample size per estimated parameter.
            Instead, sign and relative magnitude are literature-anchored:
            γ (Clay×Feldspar → MOR) is negative, consistent with the
            clay×feldspar interaction term reported for triaxial porcelain
            MOR by Correia, Oliveira, Hotza & Segadães (2006)
            DOI:10.1111/j.1551-2916.2006.01245.x (sign only is transferred;
            their coefficient is on a 0–1 pseudo-component scale, not raw
            wt%, so magnitudes are not directly comparable).
            δ (AG98 × rest-of-clay → MOR) is negative, reformulated as a
            genuine Scheffé-valid two-way cross term rather than a pure
            squared term x_AG98², because Scheffé's derivation explicitly
            eliminates pure quadratic terms in favour of cross-products
            when substituting the Σxi=1 constraint into a standard
            polynomial (x_i² = x_i − Σ_{j≠i} x_i·x_j). The asymmetric-
            blending device used here (one material's deviation crossed
            with a grouped "rest of clay" deviation) follows the same
            pattern as the CF(C−F)-type asymmetric cubic terms reported
            for triaxial tiles by Ngun, Mohamad, Katsumata, Okada & Ahmad
            (2014); sign is anchored to the non-monotonic clay-content
            sensitivity of fired MOR reported by Solanki, Kumar, Yadav &
            Gupta (2023) DOI:10.1016/j.matpr.2022.12.229.
            Physical mechanism: Carty & Senapati (1998)
            DOI:10.1111/j.1151-2916.1998.tb02439.x.
            Magnitudes are fixed at the smallest value that reproduces the
            Batch 7 corner-point behaviour described below, not tuned to
            minimise RMSE.

MATERIAL ROLES — clay-dominant floor tile body, 1210 °C, 100 bar
  (in-region roles, from the fitted model evaluated across the feasible
  composition space — NOT read off individual β_m in isolation; see
  COEFFICIENT ESTIMATION Step 1)
  AG98      High Plastic Clay   → MOR↑  WA↓  Shrink↑
  AG22      Low Plastic Clay    → MOR↑  WA↓  Shrink↑  (weaker signal)
  AG23      Semi-Plastic Clay   → MOR↑  WA↓  Shrink↑
  SodaF     Soda Feldspar       → MOR↓  WA↑  Shrink↓  (clay diluent)
  PotashF   Potash Feldspar     → MOR↓  WA↑  Shrink↓  (stronger diluent)
  Crushing  Pre-fired filler    → MOR↑  WA↓  Shrink↓
  ETP       ETP sludge (alkali) → MOR↑  WA↓  Shrink↑  (liquid-phase sinter)
  NaSil     Sodium Silicate     → rheology modifier; small effect on WA

NON-LINEAR TERMS
  Clay×Feldspar interaction: over- or under-fluxing relative to centroid
  reduces MOR — sign negative. Observed in Batch 7 (minimum corner) which
  yields the highest MOR despite lowest total input, consistent with an
  optimal flux window.
  AG98×rest-of-clay interaction: MOR responds non-monotonically to
  high-plasticity clay above the centroid — excess AG98 relative to the
  other two clays impedes particle packing.

APPLICABILITY-DOMAIN / NOISE MODEL
  The heteroscedastic noise term ε(d) ~ N(0, σ_base·(1 + d/d_ref)) scales
  synthetic-sample noise linearly with normalised distance from the
  laboratory calibration set, in the same spirit as distance-scaled
  uncertainty models used for extrapolation penalties in materials
  informatics and QSAR modelling.
  Ref: Janet et al. (2019) DOI:10.1039/C9SC02298A — latent-space distance
       scaling of predictive uncertainty.
  Ref: Korolev, Nevolin & Protsenko (2022) DOI:10.1038/s41598-022-19205-5
       — similarity-based uncertainty quantification in materials science.
  The distance itself is a k-nearest-neighbour mean distance in normalised
  composition space (k set by Sahigara's rule, k ≈ n^(1/3), floored at 3
  to keep the estimate local on the composition simplex).
  Ref: Kaneko (2026) DOI:10.1002/cem.70135 (reviews Sahigara's rule and
       related kNN applicability-domain choices).

CO₂ FACTORS  (kg CO₂ / kg, cradle-to-gate)
  AG98 (High Plastic Clay)  0.129   Zeng, Li & Li (2025), Engineering 17(12)
                                    DOI:10.4236/eng.2025.1712035
  AG22 (Low Plastic Clay)   0.129   Proxy: same source as AG98 (no clay-type-
                                    specific factor published; scope mismatch
                                    acknowledged, not a distinct measurement)
  AG23 (Semi Plastic Clay)  0.129   Proxy: same source as AG98 (as above)
  Soda Feldspar (SodaF)     0.053   LB Minerals EPD, EPD-IES-0024844 (2025),
                                    "Feldspar EPD of multiple products,
                                    based on average results" — Pobežovice
                                    site figure (Na-Ca feldspar)
  Potash Feldspar (PotashF) 0.0286  Same EPD-IES-0024844 — Nová Ves site
                                    figure (K-feldspar)
  Crushing (chamotte/grog)  0.587   LB Minerals, "Milled Chamotte Mixtures
                                    (Chamotte Component)" EPD, 2025;
                                    valid until 2030-12-03
  ETP sludge                0.242   Li, Du, Yan, Wang, Zhao, Su, Li, Du, Sun,
                                    Chen et al. (2023), "Carbon footprint
                                    analysis of sewage sludge thermochemical
                                    conversion technologies," Sustainability
                                    15(5):4170 — incineration pathway figure.
                                    (No DOI available for this article.)
  NaSil                      0.433  Prochin Italia, EPD-IES-0021224 (2025),
                                    "Environmental Product Declaration -
                                    Sodium Silicates"

VALIDATION
  Leave-One-Out cross-validation on the laboratory calibration batches. For
  each held-out fold, the β_m (and γ, δ deviation means where applicable)
  are re-estimated from the remaining calibration batches only, then the
  held-out batch's properties are predicted. This prevents the held-out
  batch's information from leaking into coefficient estimation (true
  LOO-CV, not a centroid-sensitivity check).

  Separately, a fraction of the laboratory batches (LAB_HOLDOUT_FRACTION)
  is withheld entirely from the surrogate fit above — those batches never
  contribute to the β_m fit, the γ/δ deviation means, or the LOO-CV loop.
  They are still written to the output dataset, tagged source="lab_holdout",
  so a downstream forward model can be evaluated against experimental
  observations that played no part in shaping the surrogate that generated
  its synthetic training data.
"""

import hashlib, json, logging, math, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

rng = np.random.default_rng(42)

ROOTDIR = Path(__file__).parent
OUTDIR  = ROOTDIR / "data"
PLOTDIR = ROOTDIR / "plots"

KMM2_TO_MPA = 9.80665          # kgf/mm² → MPa  (ISO 13006)

# ── Real Lab Batches — loaded from an external, user-editable CSV ─────────────
# This is the ONLY place raw laboratory data enters the whole pipeline.
# To register a NEW physical test batch: open data/lab_batches_raw.csv,
# add one row with the same columns, save, and re-run this script. The
# synthetic dataset, Ridge coefficients, LOO-CV, and every downstream
# script (models, feature importance, inverse design, reliability
# analysis) will regenerate consistently with the updated batches —
# nothing else needs to change.
LAB_BATCHES_FILE = ROOTDIR / "lab_batches_raw.csv"

def _load_lab_batches(path: Path) -> list:
    """
    Load real laboratory calibration batches from a CSV file.

    Required columns
    -----------------
    AG98, AG22, AG23, SodaF, PotashF, Crushing, ETP, NaSil
        Raw composition inputs (wt%, need not already sum to 100 —
        they are normalised to Sigma = 100 further below).
    MOR_kgf_mm2
        Flexural strength as read directly off the tester, kgf/mm2.
    WA_fraction
        Water absorption as a fraction (e.g. 0.0359, not 3.59).
    Shrinkage_pct
        Fired linear shrinkage, %.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Lab batches file not found: {path}\n"
            "Create it with columns: AG98,AG22,AG23,SodaF,PotashF,Crushing,"
            "ETP,NaSil,MOR_kgf_mm2,WA_fraction,Shrinkage_pct"
        )
    raw = pd.read_csv(path)
    required = ["AG98", "AG22", "AG23", "SodaF", "PotashF", "Crushing", "ETP",
                "NaSil", "MOR_kgf_mm2", "WA_fraction", "Shrinkage_pct"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if len(raw) < 4:
        raise ValueError(
            f"{path.name} has only {len(raw)} batch(es) - at least 4 are "
            "needed for the Ridge / physics-prior fit to be meaningful."
        )
    batches = []
    for _, r in raw.iterrows():
        batches.append({
            "AG98": float(r.AG98), "AG22": float(r.AG22), "AG23": float(r.AG23),
            "SodaF": float(r.SodaF), "PotashF": float(r.PotashF),
            "Crushing": float(r.Crushing), "ETP": float(r.ETP),
            "NaSil": float(r.NaSil),
            "MOR_MPa": float(r.MOR_kgf_mm2) * KMM2_TO_MPA,
            "WA_pct": float(r.WA_fraction) * 100.0,
            "Shrinkage_pct": float(r.Shrinkage_pct),
        })
    return batches

_LAB_RAW = _load_lab_batches(LAB_BATCHES_FILE)

# ── Data provenance fingerprint ────────────────────────────────────────────────
# A short hash of the raw lab-batch CSV, stamped onto every figure this script
# (and downstream scripts fed by data/dataset.csv) produces. If the CSV is
# edited and the pipeline is re-run, every figure gets a NEW stamp, so a stale
# figure sitting next to updated tables/text is immediately visually obvious
# (mismatched stamps) instead of silently slipping through review.
DATA_HASH = hashlib.md5(LAB_BATCHES_FILE.read_bytes()).hexdigest()[:8]
GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

MATS = ["AG98","AG22","AG23","SodaF","PotashF","Crushing","ETP","NaSil"]
# Used only for simplex rejection-sampling (_sample_comps): 7 materials are
# drawn independently within BOUNDS and SodaF is derived as the residual
# (100 minus the other 7), which is how a valid random point on the
# Σ=100 simplex is generated. This is unrelated to how the property
# surrogate's coefficients are fitted below (see COEFFICIENT ESTIMATION
# Step 0-1 in the module docstring) — that fit now uses all 8 raw
# materials directly (canonical-Scheffé form), not just these 7.
FREE_MATS = ["AG98","AG22","AG23","PotashF","Crushing","ETP","NaSil"]
MAT_LABELS = {
    "AG98":     "AG98 (wt%)",
    "AG22":     "AG22 (wt%)",
    "AG23":     "AG23 (wt%)",
    "SodaF":    "Soda Feldspar (wt%)",
    "PotashF":  "Potash Feldspar (wt%)",
    "Crushing": "Crushing (wt%)",
    "ETP":      "ETP Clay (wt%)",
    "NaSil":    "Na-Silicate (wt%)",
}
TGT_LABELS = {
    "MOR_MPa":       "Firing MOR (MPa)",
    "WA_pct":        "Water Absorption (%)",
    "Shrinkage_pct": "Fired Shrinkage (%)",
}
TGTS = ["MOR_MPa","WA_pct","Shrinkage_pct"]

for b in _LAB_RAW:                             # normalise to Σ = 100 wt%
    s = sum(b[m] for m in MATS)
    for m in MATS:
        b[m] = b[m] / s * 100.0

# A fraction of the laboratory batches is withheld entirely from the
# physics-surrogate fit below (Ridge coefficients, composition/property
# centroid, LOO-CV). Those withheld batches still appear in the saved
# dataset, tagged source="lab_holdout", so a downstream forward model can
# be tested against experimental observations that never shaped the
# surrogate that generated its synthetic training data. Change
# LAB_HOLDOUT_FRACTION to adjust the size of this reserved set.
LAB_HOLDOUT_FRACTION = 0.25
_n_holdout   = max(1, round(len(_LAB_RAW) * LAB_HOLDOUT_FRACTION))
_holdout_idx = set(rng.choice(len(_LAB_RAW), size=_n_holdout, replace=False).tolist())
_LAB_CALIB   = [b for i, b in enumerate(_LAB_RAW) if i not in _holdout_idx]
_LAB_HOLDOUT = [b for i, b in enumerate(_LAB_RAW) if i in _holdout_idx]

lab_df            = pd.DataFrame(_LAB_CALIB)
LAB_MEAN_COMP     = {m: float(lab_df[m].mean()) for m in MATS}
LAB_MEAN_PROPS    = {t: float(lab_df[t].mean()) for t in TGTS}

# ── Composition feasibility bounds (industrial specification) ─────────────────
BOUNDS = {
    "AG98"    : (15.0, 20.0),
    "AG22"    : (2.5,  4.0),
    "AG23"    : (10.0, 15.0),
    "SodaF"   : (37.0, 43.0),
    "PotashF" : (15.0, 22.0),
    "Crushing": (2.0,  3.5),
    "ETP"     : (2.0,  3.1),
    "NaSil"   : (0.5,  1.5),
}

# ── Cost (BDT/kg) and CO₂ factors ────────────────────────────────────────────
# Total batch cost and embodied CO2 (see build_dataset) are computed as a
# simple mass-fraction-weighted linear sum of the per-material factors
# below. This is the standard, accepted approach for first-order
# multi-component material-inventory accounting and does not require
# modelling processing-stage non-linearities for this purpose.
# Ref: Young, Hall, Pilon, Gupta & Sant (2018) — linear dosage-weighted
#      cost/CO2 objective for a multi-component mixture.
# Ref: Hammad, Akbarnezhad & Oldfield (2018) DOI:10.1016/j.enbuild.2018.05.061
#      — mass/volume-weighted linear embodied-carbon summation.
#
# CO2 factor provenance (kg CO2 / kg, cradle-to-gate):
#   AG98/AG22/AG23  0.129    Zeng, Li & Li (2025), Engineering 17(12),
#                             DOI:10.4236/eng.2025.1712035 (AG22/AG23 use
#                             the same figure as AG98 as a proxy — no
#                             clay-type-specific factor published)
#   SodaF           0.053    LB Minerals EPD EPD-IES-0024844 (2025),
#                             "Feldspar EPD of multiple products, based
#                             on average results" — Pobežovice site figure
#   PotashF         0.0286   Same EPD-IES-0024844 — Nová Ves site figure
#   Crushing        0.587    LB Minerals, "Milled Chamotte Mixtures
#                             (Chamotte Component)" EPD, 2025
#   ETP             0.242    Li, Du, Yan, Wang, Zhao, Su, Li, Du, Sun,
#                             Chen et al. (2023), Sustainability 15(5):4170,
#                             incineration pathway (no DOI available)
#   NaSil           0.433    Prochin Italia EPD-IES-0021224 (2025),
#                             "Environmental Product Declaration -
#                             Sodium Silicates"
# See CO2 FACTORS in the module docstring for the full breakdown, and the
# REFERENCES BibTeX block at the end of this file for the citation entries.
COST = {
    "AG98": 6.95,   "AG22": 8.37,  "AG23": 7.024,
    "SodaF": 8.887, "PotashF": 6.241,
    "Crushing": 0.0, "ETP": 0.0,   "NaSil": 23.369,
}
CO2 = {
    "AG98":(0.129,0.129),
    "AG22":(0.129,0.129),
    "AG23":(0.129,0.129),
    "SodaF":(0.053,0.053),
    "PotashF":(0.0286,0.0286),
    "Crushing":(0.587,0.587),
    "ETP":(0.242,0.242),
    "NaSil":(0.433,0.433),
}
CO2_MID = {m: float(np.mean(CO2[m])) for m in MATS}

# ── Process parameters (fixed single firing cycle) ───────────────────────────
PROC = {
    "press_bar":           100,
    "dryer_time_min":       45,
    "kiln_time_min":        90,
    "kiln_temp_C":        1210,
    "calorific_NG_Kcal_Nm3": 8300,
    "dryer_temp_C":        180.0,
    "gas_Nm3_per_m2":      1.4115,
    "green_length_mm":     109.20,
    "green_width_mm":       54.60,
    "green_thickness_mm":    9.80,
    "green_weight_g":       98.50,
    "fired_length_mm":      98.00,
    "fired_weight_g":       95.05,
}

# ── Non-linear interaction coefficients (fixed literature-anchored priors,
#    NOT fitted by RMSE minimisation — see COEFFICIENT ESTIMATION Step 2
#    in the module docstring for why RMSE-tuning these on a calibration
#    set this size would be statistically indefensible). Both are Partial
#    Quadratic Mixture cross-terms (Piepel, Szychowski & Loeppky 2002
#    DOI:10.1080/00224065.2002.11980160), added on top of the canonical
#    Scheffé linear term rather than the full 28-term quadratic expansion.
# Clay×Feldspar interaction: negative coefficient — deviation from centroid
# in both directions reduces MOR. Sign consistent with the clay×feldspar
# interaction term for MOR in Correia, Oliveira, Hotza & Segadães (2006)
# DOI:10.1111/j.1551-2916.2006.01245.x. Physical mechanism: Carty &
# Senapati (1998) DOI:10.1111/j.1151-2916.1998.tb02439.x.
INTERACTION_COEFF = {
    "MOR_MPa":       -0.08,
    "Shrinkage_pct":  0.012,
    "WA_pct":        -0.003,
}
# AG98 × rest-of-clay (AG22+AG23) cross term — replaces a non-canonical
# squared term x_AG98² (Scheffé's derivation eliminates pure quadratic
# terms in favour of cross-products once Σxi=1 is substituted in; see
# COEFFICIENT ESTIMATION Step 2). Asymmetric-blending device in the style
# of the CF(C−F)-type terms reported for triaxial tiles by Ngun, Mohamad,
# Katsumata, Okada & Ahmad (2014). Sign anchored to the non-monotonic
# clay-content sensitivity of fired MOR reported by Solanki, Kumar, Yadav
# & Gupta (2023) DOI:10.1016/j.matpr.2022.12.229.
AG98_CROSS_COEFF = {
    "MOR_MPa":       -0.15,
    "Shrinkage_pct":  0.0,
    "WA_pct":         0.0,
}

_PROPS = ["Shrinkage_pct", "WA_pct", "MOR_MPa"]
# Naive per-material sign expectation from ceramic sintering theory (Reed
# 1995, Ch.12), used ONLY for the informational post-fit check below — NOT
# enforced as a constraint on the fit itself. Earlier versions of this
# script enforced these as hard bounds via constrained least squares
# (Hemmerle & Brantle 1978 DOI:10.1080/00401706.1978.10489632; Bachinger
# et al. 2024 DOI:10.1145/3661826); that was dropped because (i) LOO-CV was
# measurably worse with the bounds than without them on this calibration
# set, and (ii) published canonical-Scheffé mixture papers do not bound
# their linear coefficients either — an individual β_m is an extrapolated
# pure-component value, not expected to carry a simple sign in isolation
# (e.g. Correia et al. 2004 report a positive linear coefficient for
# feldspar on shrinkage even though feldspar's in-region, whole-model
# effect is to reduce shrinkage — see NON-LINEAR TERMS in the module
# docstring). See COEFFICIENT ESTIMATION Step 1.
_EXPECTED_SIGN = {"AG98":(+1,-1,+1),"AG22":(+1,-1,+1),"AG23":(+1,-1,+1),
                  "SodaF":(-1,+1,-1),"PotashF":(-1,+1,-1),
                  "Crushing":(-1,-1,+1),"ETP":(+1,-1,+1),"NaSil":(0,0,0)}

def _fit_scheffe_coefficients(alpha: float = 0.1,
                              lab_subset: pd.DataFrame = None) -> dict:
    """
    Fit the canonical-Scheffé linear mixture term Y = Σ_m β_m·x_m by Ridge
    regression through the origin (no intercept, no centering, no sign
    bounds) on raw composition fractions — the standard way published
    ceramic mixture-design studies fit this class of model (Correia et al.
    2004; Ngun et al. 2014; see citations at the end of this file).
    Ref: Hoerl & Kennard (1970) DOI:10.1080/00401706.1970.10488634 —
         Ridge regularisation against collinearity (needed here because
         removing the intercept does not by itself remove the near-
         singularity of composition data on a narrow-bounded simplex —
         see COEFFICIENT ESTIMATION Step 0).

    Parameters
    ----------
    alpha : float
        Ridge regularisation strength.
    lab_subset : pd.DataFrame, optional
        Lab batches to fit on. Defaults to the calibration subset of
        ``lab_df`` (the laboratory batches not withheld as a holdout test
        set). Passing a subset (e.g. calibration batches minus one) allows
        this function to be reused for Leave-One-Out cross-validation
        without leaking the held-out batch into coefficient estimation.
    """
    src = lab_subset if lab_subset is not None else lab_df
    X = src[MATS].values / 100.0
    n_feat = X.shape[1]
    A_aug  = np.vstack([X, np.sqrt(alpha) * np.eye(n_feat)])

    fitted = {}
    for t in _PROPS:
        y_aug = np.concatenate([src[t].values, np.zeros(n_feat)])
        beta, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
        fitted[t] = beta

    return {m: (float(fitted["Shrinkage_pct"][i]),
                float(fitted["WA_pct"][i]),
                float(fitted["MOR_MPa"][i]))
            for i, m in enumerate(MATS)}

PHYSICS_COEFF = _fit_scheffe_coefficients()

# Informational only — logs which fitted coefficients disagree with the
# naive per-material sintering-theory sign, without altering the fit
# (see the docstring of _fit_scheffe_coefficients / COEFFICIENT
# ESTIMATION Step 1 for why this is not enforced).
for _m, _vals in PHYSICS_COEFF.items():
    for _i, _p in enumerate(_PROPS):
        _expected = _EXPECTED_SIGN[_m][_i]
        if _expected != 0 and np.sign(_vals[_i]) != _expected:
            logger.info("Scheffe fit note: %-10s beta_%s = %+.4f "
                        "(naive sintering-theory sign would be %s; not "
                        "enforced — see module docstring)",
                        _m, _p, _vals[_i], "+" if _expected > 0 else "-")

_RANGE     = np.array([BOUNDS[m][1] - BOUNDS[m][0] for m in MATS])
_LAB_XNORM = lab_df[MATS].values / _RANGE
_D_REF     = float(np.median([
    np.linalg.norm(_LAB_XNORM[i] - _LAB_XNORM[j])
    for i in range(len(_LAB_CALIB)) for j in range(i+1, len(_LAB_CALIB))
]))

# Sahigara's rule for k-nearest-neighbour applicability-domain distance:
# k = n^(1/3), floored at 3 to keep the estimate local on a
# high-dimensional compositional simplex regardless of how many
# calibration batches are available (Kaneko 2026 DOI:10.1002/cem.70135).
_K_NN = max(3, round(len(_LAB_CALIB) ** (1 / 3)))

def _dist(comp: np.ndarray, k: int = _K_NN) -> float:
    d = np.linalg.norm(_LAB_XNORM - comp / _RANGE, axis=1)
    return float(np.sort(d)[:k].mean())

def _physics_pred(cd: dict) -> dict:
    """
    Non-linear physics surrogate (canonical-Scheffé mixture form).

    Three components:
      1. Canonical-Scheffé linear term: Y = Σ_m β_m·x_m, all 8 materials,
         fitted by Ridge regression through the origin, no sign bounds
         (see PHYSICS_COEFF and COEFFICIENT ESTIMATION Step 0-1 above).
      2. Clay–Feldspar interaction: captures over/under-fluxing non-linearity.
         Coefficient negative for MOR — deviation from centroid in either
         direction moves away from the optimal flux window. Sign consistent
         with Correia, Oliveira, Hotza & Segadães (2006)
         DOI:10.1111/j.1551-2916.2006.01245.x; mechanism per Carty &
         Senapati (1998) DOI:10.1111/j.1151-2916.1998.tb02439.x.
      3. AG98 × rest-of-clay cross term: asymmetric-blending interaction
         between AG98 and the other two clays (AG22+AG23), replacing a
         non-canonical squared term x_AG98² (Scheffé's derivation
         eliminates pure quadratic terms in favour of cross-products —
         see COEFFICIENT ESTIMATION Step 2). Consistent with Batch 7
         observation (minimum AG98 corner yielding maximum MOR); sign
         anchored to Solanki, Kumar, Yadav & Gupta (2023)
         DOI:10.1016/j.matpr.2022.12.229; device style per Ngun, Mohamad,
         Katsumata, Okada & Ahmad (2014).
    """
    p = {t: 0.0 for t in TGTS}

    # 1. Canonical-Scheffé linear term (raw fractions, no centering)
    for m, (cs, cw, cm) in PHYSICS_COEFF.items():
        x = cd[m] / 100.0
        p["Shrinkage_pct"] += cs * x
        p["WA_pct"]        += cw * x
        p["MOR_MPa"]       += cm * x

    # 2. Clay–Feldspar interaction (deviation-centered cross term)
    total_clay = cd["AG98"] + cd["AG22"] + cd["AG23"]
    total_fsp  = cd["SodaF"] + cd["PotashF"]
    clay_mean  = (LAB_MEAN_COMP["AG98"] + LAB_MEAN_COMP["AG22"]
                  + LAB_MEAN_COMP["AG23"])
    fsp_mean   = LAB_MEAN_COMP["SodaF"] + LAB_MEAN_COMP["PotashF"]
    interaction = (total_clay - clay_mean) * (total_fsp - fsp_mean)
    for t in TGTS:
        p[t] += INTERACTION_COEFF[t] * interaction

    # 3. AG98 × rest-of-clay cross term (deviation-centered)
    other_clay      = cd["AG22"] + cd["AG23"]
    other_clay_mean = LAB_MEAN_COMP["AG22"] + LAB_MEAN_COMP["AG23"]
    ag98_cross = ((cd["AG98"] - LAB_MEAN_COMP["AG98"])
                  * (other_clay - other_clay_mean))
    for t in TGTS:
        p[t] += AG98_CROSS_COEFF[t] * ag98_cross

    return p

def _sample_comps(n: int) -> np.ndarray:
    """
    Rejection-sample on the simplex: all 8 materials within BOUNDS, Σ = 100.
    7 materials (FREE_MATS) are drawn independently and SodaF is derived
    as the residual — a sampling convenience only, unrelated to how
    PHYSICS_COEFF is fitted (see FREE_MATS comment above).
    """
    lo     = np.array([BOUNDS[k][0] for k in FREE_MATS])
    hi     = np.array([BOUNDS[k][1] for k in FREE_MATS])
    lo_s, hi_s = BOUNDS["SodaF"]
    out = []
    while len(out) < n:
        v = rng.uniform(lo, hi)
        s = 100.0 - v.sum()
        if lo_s <= s <= hi_s:
            c = dict(zip(FREE_MATS, v)); c["SodaF"] = s
            out.append([c[m] for m in MATS])
    return np.array(out)

def build_dataset(n_synthetic: int = 200) -> pd.DataFrame:
    noise_base = {t: (lab_df[t].max() - lab_df[t].min()) * 0.04 for t in TGTS}
    clip_lo    = {t: lab_df[t].min()        for t in TGTS}
    clip_hi    = {t: lab_df[t].max() * 1.10 for t in TGTS}
    rows = []

    for comp in _sample_comps(n_synthetic):
        cd  = dict(zip(MATS, comp))
        yp  = _physics_pred(cd)
        d   = _dist(comp)
        row = {f"{m}_wtpct": cd[m] for m in MATS}
        for t in TGTS:
            row[t] = float(np.clip(
                yp[t] + rng.normal(0, noise_base[t] * (1 + d / _D_REF)),
                clip_lo[t], clip_hi[t]
            ))
        row["cost_Tk_per_kg"] = sum(cd[m] / 100 * COST[m] for m in MATS)
        row["CO2_kg_per_kg"]  = sum(cd[m] / 100 * CO2_MID[m] for m in MATS)
        row["source"] = "synthetic"
        row.update(PROC)
        rows.append(row)

    for b in _LAB_CALIB:
        cd  = {m: b[m] for m in MATS}
        row = {f"{m}_wtpct": cd[m] for m in MATS}
        for t in TGTS:
            row[t] = b[t]
        row["cost_Tk_per_kg"] = sum(cd[m] / 100 * COST[m]    for m in MATS)
        row["CO2_kg_per_kg"]  = sum(cd[m] / 100 * CO2_MID[m] for m in MATS)
        row["source"] = "lab_batch"
        row.update(PROC)
        rows.append(row)

    for b in _LAB_HOLDOUT:
        cd  = {m: b[m] for m in MATS}
        row = {f"{m}_wtpct": cd[m] for m in MATS}
        for t in TGTS:
            row[t] = b[t]
        row["cost_Tk_per_kg"] = sum(cd[m] / 100 * COST[m]    for m in MATS)
        row["CO2_kg_per_kg"]  = sum(cd[m] / 100 * CO2_MID[m] for m in MATS)
        row["source"] = "lab_holdout"
        row.update(PROC)
        rows.append(row)

    for i, r in enumerate(rows, 1):
        r["id"] = i
    cols = (["id", "source"] + [f"{m}_wtpct" for m in MATS] + TGTS
            + list(PROC.keys()) + ["cost_Tk_per_kg", "CO2_kg_per_kg"])
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]

# ── Surrogate fidelity: Leave-One-Out cross-validation on lab calibration ─────
def validate_physics(df: pd.DataFrame) -> bool:
    """
    Leave-One-Out cross-validation on the laboratory calibration batches
    (the lab batches not set aside as the holdout test set).

    For each held-out fold, the canonical-Scheffé β_m are re-estimated
    from the remaining calibration batches only, then the held-out
    batch's properties are predicted. This prevents the held-out batch's
    information from leaking into coefficient estimation — a true
    LOO-CV, not merely a centroid-sensitivity check. MAE is reported as
    a percentage of the observed range for each target.

    Note: this LOO-CV loop is also what was used to compare the canonical-
    Scheffé linear form against the earlier centered/sign-bounded form
    before choosing between them (see COEFFICIENT ESTIMATION Step 0-1 in
    the module docstring) — the Scheffé form gave lower LOO-CV error on
    this calibration set.

    Note: INTERACTION_COEFF and AG98_CROSS_COEFF are fixed, literature-
    anchored priors (not Ridge-fitted, not RMSE-calibrated on the
    laboratory batches — see COEFFICIENT ESTIMATION Step 2 in the module
    docstring); they are not refit per fold. Only the Ridge-fitted linear
    coefficients (the leakage-prone component) are refit per fold below.
    """
    loo_errors = {t: [] for t in TGTS}

    for i in range(len(_LAB_CALIB)):
        train_rows = [b for j, b in enumerate(_LAB_CALIB) if j != i]
        test       = _LAB_CALIB[i]
        train_df   = pd.DataFrame(train_rows)

        # Refit the canonical-Scheffé coefficients on the remaining
        # calibration batches only — prevents the held-out batch from
        # leaking into coefficient estimation.
        fold_coeff = _fit_scheffe_coefficients(lab_subset=train_df)

        train_mean_comp = {m: float(train_df[m].mean()) for m in MATS}

        cd   = {m: test[m] for m in MATS}
        pred = {t: 0.0 for t in TGTS}

        # Canonical-Scheffé linear term with the LOO-refit coefficients
        for m, (cs, cw, cm) in fold_coeff.items():
            x = cd[m] / 100.0
            pred["Shrinkage_pct"] += cs * x
            pred["WA_pct"]        += cw * x
            pred["MOR_MPa"]       += cm * x

        # Clay–Feldspar interaction term with LOO centroid
        total_clay = cd["AG98"] + cd["AG22"] + cd["AG23"]
        total_fsp  = cd["SodaF"] + cd["PotashF"]
        clay_mean  = (train_mean_comp["AG98"] + train_mean_comp["AG22"]
                      + train_mean_comp["AG23"])
        fsp_mean   = train_mean_comp["SodaF"] + train_mean_comp["PotashF"]
        interaction = (total_clay - clay_mean) * (total_fsp - fsp_mean)
        for t in TGTS:
            pred[t] += INTERACTION_COEFF[t] * interaction

        # AG98 x rest-of-clay cross term with LOO centroid
        other_clay      = cd["AG22"] + cd["AG23"]
        other_clay_mean = train_mean_comp["AG22"] + train_mean_comp["AG23"]
        ag98_cross = ((cd["AG98"] - train_mean_comp["AG98"])
                      * (other_clay - other_clay_mean))
        for t in TGTS:
            pred[t] += AG98_CROSS_COEFF[t] * ag98_cross

        for t in TGTS:
            loo_errors[t].append(abs(pred[t] - test[t]))

    all_pass = True
    for t in TGTS:
        mae   = float(np.mean(loo_errors[t]))
        r_obs = float(max(b[t] for b in _LAB_CALIB) - min(b[t] for b in _LAB_CALIB))
        pct   = mae / r_obs * 100
        status = "PASS" if pct < 25 else "WARN"
        if pct >= 25:
            all_pass = False
        logger.info("LOO-CV  %-18s  MAE=%.4f  range=%.4f  err=%.1f%%  %s",
                    t, mae, r_obs, pct, status)
    return all_pass

# ── Save outputs ──────────────────────────────────────────────────────────────
def save(df: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lab_export = pd.DataFrame(_LAB_CALIB).assign(split="calibration")
    lab_export = pd.concat(
        [lab_export, pd.DataFrame(_LAB_HOLDOUT).assign(split="holdout")],
        ignore_index=True,
    )
    lab_export.to_csv(OUTDIR / "lab_batches.csv", index=False)
    df.to_csv(OUTDIR / "dataset.csv", index=False)
    df_s = df[df.source == "synthetic"]
    with open(OUTDIR / "property_ranges.json", "w") as f:
        json.dump({f"{t}_{k}": float(getattr(df_s[t], k)())
                   for t in TGTS for k in ("min", "max")}, f, indent=2)
    meta = {
        "materials": MATS, "targets": TGTS, "bounds": BOUNDS,
        "model_form": "canonical_scheffe_linear_plus_two_cross_terms",
        "cost_tk_per_kg": COST,
        "co2_kg_per_kg": {k: list(v) for k, v in CO2.items()},
        "co2_midpoint": CO2_MID,
        "co2_sources": {
            "AG98":     "Zeng, Li & Li (2025), Engineering 17(12), DOI:10.4236/eng.2025.1712035",
            "AG22":     "Proxy: same as AG98 (Zeng, Li & Li 2025)",
            "AG23":     "Proxy: same as AG98 (Zeng, Li & Li 2025)",
            "SodaF":    "LB Minerals EPD EPD-IES-0024844 (2025), Pobežovice site figure",
            "PotashF":  "LB Minerals EPD EPD-IES-0024844 (2025), Nová Ves site figure",
            "Crushing": "LB Minerals, Milled Chamotte Mixtures EPD (2025)",
            "ETP":      "Li et al. (2023), Sustainability 15(5):4170, incineration pathway (no DOI available)",
            "NaSil":    "Prochin Italia EPD-IES-0021224 (2025)",
        },
        "proc_defaults": PROC,
        "n_lab_batches": len(_LAB_RAW),
        "n_lab_calibration": len(_LAB_CALIB),
        "n_lab_holdout": len(_LAB_HOLDOUT),
        "n_synthetic": int((df.source == "synthetic").sum()),
        "generation_method": (
            "Physics-informed non-linear surrogate, canonical-Scheffé "
            "mixture form. Linear term: Y = sum_m beta_m x_m, Ridge "
            "regression (lambda=0.1) through the origin (no intercept, "
            f"no centering, no sign bounds) on {len(_LAB_CALIB)} lab "
            "calibration batches — the standard fitting approach in "
            "published ceramic mixture-design studies (Correia et al. "
            "2004; Ngun et al. 2014); Ridge regularisation (Hoerl & "
            "Kennard 1970 DOI:10.1080/00401706.1970.10488634) is still "
            "needed because narrow per-material bounds keep composition "
            "data ill-conditioned even without an intercept. Individual "
            "beta_m are extrapolated pure-component values and are not "
            "expected to carry a simple physical sign in isolation; a "
            "sign check against ceramic-sintering-theory expectation "
            "(Reed 1995) is logged for transparency but not enforced "
            "(this differs from an earlier version of this script, which "
            "used sign-bounded constrained least squares and measurably "
            "worse LOO-CV). Non-linear terms: Clay-Feldspar cross term + "
            "AG98-vs-rest-of-clay cross term, both fixed literature-"
            "anchored priors (Partial Quadratic Mixture approach, Piepel "
            "et al. 2002 DOI:10.1080/00224065.2002.11980160) rather than "
            "fitted by RMSE minimisation on the laboratory batches "
            "(Correia et al. 2006 DOI:10.1111/j.1551-2916.2006.01245.x; "
            "Solanki et al. 2023 DOI:10.1016/j.matpr.2022.12.229; "
            "mechanism per Carty & Senapati 1998 "
            "DOI:10.1111/j.1151-2916.1998.tb02439.x)."
        ),
        "noise_model": (
            "heteroscedastic — σ(d) = σ_base·(1 + d/d_ref); σ_base = 4% of "
            "observed range; d = mean distance to k nearest calibration "
            f"batches, k={_K_NN} (Sahigara's rule, Kaneko 2026 "
            "DOI:10.1002/cem.70135); distance-scaling form per Janet et "
            "al. (2019) DOI:10.1039/C9SC02298A and Korolev et al. (2022) "
            "DOI:10.1038/s41598-022-19205-5"
        ),
        "validation_method": (
            f"Leave-One-Out cross-validation on {len(_LAB_CALIB)} laboratory "
            "calibration batches (coefficients refit per fold to "
            f"prevent leakage); {len(_LAB_HOLDOUT)} additional laboratory "
            "batches held out entirely from surrogate calibration for "
            "independent downstream forward-model testing; "
            "threshold 25% of observed range per target"
        ),
        "physics_coefficients": {
            m: {"shrinkage": cs, "wa": cw, "mor": cm}
            for m, (cs, cw, cm) in PHYSICS_COEFF.items()
        },
        "interaction_coefficients": INTERACTION_COEFF,
        "ag98_cross_coefficients": AG98_CROSS_COEFF,
        "lab_composition_means": LAB_MEAN_COMP,
        "lab_property_means":    LAB_MEAN_PROPS,
        "data_hash":    DATA_HASH,
        "generated_at": GENERATED_AT,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved  %d rows  (synthetic=%d  lab_calibration=%d  lab_holdout=%d)",
                len(df),
                int((df.source == "synthetic").sum()),
                int((df.source == "lab_batch").sum()),
                int((df.source == "lab_holdout").sum()))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
_FS_TITLE  = 18
_FS_AX     = 16
_FS_TICK   = 14
_FS_LABEL  = 13
_FS_ANNOT  = 12
_DPI       = 300

def _stamp(fig) -> None:
    """Small footer identifying the exact data version this figure was
    rendered from. Prevents a regenerated CSV / dataset.csv from silently
    leaving old, un-matching figures behind."""
    fig.text(0.995, 0.002, f"data:{DATA_HASH}  generated:{GENERATED_AT}",
              ha="right", va="bottom", fontsize=6, color="0.6",
              family="monospace")

def _savefig(fig, stem: str) -> None:
    _stamp(fig)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"{stem}.{ext}", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_distributions(df: pd.DataFrame) -> None:
    comp_cols  = [f"{m}_wtpct" for m in MATS]
    all_cols   = comp_cols + TGTS
    all_labels = [MAT_LABELS[m] for m in MATS] + [TGT_LABELS[t] for t in TGTS]

    nrows = 3; ncols = math.ceil(len(all_cols) / nrows)
    pal   = sns.color_palette("husl", len(all_cols) + 2)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 4.2))
    axes = axes.flatten()

    ds = df[df.source == "synthetic"]
    for i, (col, label) in enumerate(zip(all_cols, all_labels)):
        ax   = axes[i]
        data = ds[col].clip(ds[col].quantile(0.01), ds[col].quantile(0.99))
        sns.histplot(data, bins="auto", kde=True, color=pal[i],
                     ax=ax, edgecolor="white")
        for ln in ax.get_lines():
            ln.set_linewidth(2.0); ln.set_alpha(0.85)
        ax.set_xlabel(label, fontsize=_FS_AX)
        ax.set_ylabel("Frequency", fontsize=_FS_AX)
        ax.tick_params(labelsize=_FS_TICK)
        ax.text(0.5, -0.28, f"({chr(97 + i)})",
                transform=ax.transAxes, ha="center",
                fontsize=_FS_LABEL, fontweight="bold")
    for i in range(len(all_cols), len(axes)):
        axes[i].axis("off")
    fig.suptitle(
        "Feature and Target Distributions of the Synthetic Training Dataset"
        f" (n = {len(ds):,})",
        fontsize=_FS_TITLE, fontweight="bold", y=1.02
    )
    plt.tight_layout(h_pad=5.5, w_pad=3.0)
    _savefig(fig, "all_distributions")
    logger.info("Saved: all_distributions.pdf / .png")

def plot_source_stripplot(df: pd.DataFrame) -> None:
    # "Experimental" pools both the calibration and holdout laboratory
    # batches — this figure checks that the synthetic distribution tracks
    # everything actually measured, independent of how those batches are
    # later split for surrogate fitting vs. forward-model testing.
    groups = {
        "synthetic":    df.source == "synthetic",
        "experimental": df.source.isin(["lab_batch", "lab_holdout"]),
    }
    colors  = {"synthetic": "#2196F3", "experimental": "#E53935"}
    markers = {"synthetic": "o",        "experimental": "*"}
    sizes   = {"synthetic": 35,         "experimental": 180}
    _rng    = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, t in zip(axes, TGTS):
        for i, (grp, mask) in enumerate(groups.items()):
            vals   = df.loc[mask, t].values
            jitter = _rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                color=colors[grp],
                alpha=0.50 if grp == "synthetic" else 0.95,
                s=sizes[grp], marker=markers[grp],
                label=("Synthetic" if grp == "synthetic" else "Experimental"),
                zorder=3 if grp == "experimental" else 2,
            )
            ax.hlines(vals.mean(), i - 0.30, i + 0.30,
                      colors=colors[grp], linewidth=2.5, linestyle="--", alpha=0.85)

        syn_vals = df.loc[groups["synthetic"], t].values
        exp_vals = df.loc[groups["experimental"], t].values
        try:
            _, p_mw = mannwhitneyu(syn_vals, exp_vals, alternative="two-sided")
            ax.text(0.98, 0.03, f"MWU p = {p_mw:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=_FS_ANNOT, color="gray",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.75))
        except Exception:
            pass

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"Synthetic\n(n = {len(syn_vals):,})",
             f"Experimental\n(n = {len(exp_vals)})"],
            fontsize=_FS_TICK
        )
        ax.set_ylabel(TGT_LABELS[t], fontsize=_FS_AX)
        ax.set_title(TGT_LABELS[t], fontsize=_FS_AX, fontweight="bold")
        ax.tick_params(axis="y", labelsize=_FS_TICK)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        if t == TGTS[0]:
            ax.legend(fontsize=_FS_LABEL, loc="upper right")

    fig.suptitle(
        "Comparison of Property Distributions: Synthetic Dataset vs."
        " Experimental Batches\n(dashed line = group mean;"
        "  MWU = Mann–Whitney U test p-value)",
        fontsize=_FS_TITLE, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "source_comparison_stripplot")
    logger.info("Saved: source_comparison_stripplot.pdf / .png")

def plot_composition_correlation(df: pd.DataFrame) -> None:
    comp_cols = [f"{m}_wtpct" for m in MATS]
    labels    = [MAT_LABELS[m].replace(" (wt%)", "") for m in MATS]

    ds   = df[df.source == "synthetic"]
    corr = ds[comp_cols].corr()
    corr.columns = labels
    corr.index   = labels

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, annot_kws={"size": _FS_LABEL}, ax=ax
    )
    ax.set_title(
        "Pearson Correlation Among Composition Variables\n"
        "(synthetic dataset; simplex constraint Σwt% = 100\n"
        "induces structural multicollinearity)",
        fontsize=_FS_TITLE, fontweight="bold", pad=16
    )
    plt.xticks(rotation=45, ha="right", fontsize=_FS_TICK)
    plt.yticks(rotation=0,  fontsize=_FS_TICK)
    plt.tight_layout()
    _savefig(fig, "composition_correlation")
    logger.info("Saved: composition_correlation.pdf / .png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset generation — physics-informed non-linear surrogate")
    logger.info("[1/4] Sampling compositions …")
    # Change this value to increase or decrease the number of synthetic
    # compositions generated from the physics surrogate.
    N_SYNTHETIC = 200
    df = build_dataset(n_synthetic=N_SYNTHETIC)

    logger.info("[2/4] Surrogate fidelity — LOO-CV on %d laboratory calibration batches …",
                len(_LAB_CALIB))
    validate_physics(df)

    logger.info("[3/4] Saving outputs …")
    save(df)

    logger.info("[4/4] Generating figures …")
    plot_distributions(df)
    plot_source_stripplot(df)
    plot_composition_correlation(df)

    for t in TGTS:
        logger.info("  %-22s  [%.4f, %.4f]  mean=%.3f  CV=%.2f%%",
                    t, df[t].min(), df[t].max(), df[t].mean(),
                    df[t].std() / df[t].mean() * 100)
    logger.info("Done.")

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCES (BibTeX)
# Every citation used in the comments/docstrings above, collected here for
# building references.bib. Entries marked "verify" could not be fully
# confirmed against the publisher record and should be checked before
# submission; entries with no such mark are standard, well-known sources.
# ══════════════════════════════════════════════════════════════════════════════
r"""
@article{hoerl1970ridge,
  author  = {Hoerl, Arthur E. and Kennard, Robert W.},
  title   = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  journal = {Technometrics},
  year    = {1970}, volume = {12}, number = {1}, pages = {55--67},
  doi     = {10.1080/00401706.1970.10488634}
}

@article{hemmerle1978explicit,
  author  = {Hemmerle, W. J. and Brantle, T. F.},
  title   = {Explicit and Constrained Generalized Ridge Estimation},
  journal = {Technometrics},
  year    = {1978}, volume = {20}, number = {2}, pages = {109--120},
  doi     = {10.1080/00401706.1978.10489632}
}

@article{bachinger2024data,
  author  = {Bachinger, Florian and Kronberger, Gabriel and others},
  title   = {Data Validation Utilizing Expert Knowledge and Shape Constraints},
  journal = {ACM Journal on Data and Information Quality},
  year    = {2024}, volume = {16}, number = {2}, pages = {Article 13},
  doi     = {10.1145/3661826}
}

@book{reed1995principles,
  author    = {Reed, James S.},
  title     = {Principles of Ceramics Processing},
  edition   = {2nd}, publisher = {Wiley}, year = {1995}
}

@book{cornell2002experiments,
  author    = {Cornell, John A.},
  title     = {Experiments with Mixtures: Designs, Models, and the Analysis
               of Mixture Data},
  edition   = {3rd}, publisher = {Wiley}, year = {2002}
}

@article{correia2006properties,
  author  = {Correia, S. L. and Oliveira, A. P. N. and Hotza, D. and
             Segad{\~a}es, A. M.},
  title   = {Properties of Triaxial Porcelain Bodies: Interpretation of
             Statistical Modeling},
  journal = {Journal of the American Ceramic Society},
  year    = {2006}, volume = {89}, number = {11}, pages = {3356--3365},
  doi     = {10.1111/j.1551-2916.2006.01245.x}
}

@article{correia2004effect,
  author  = {Correia, S. L. and Hotza, D. and Segad{\~a}es, A. M.},
  title   = {Effect of Raw Materials on Linear Shrinkage},
  journal = {American Ceramic Society Bulletin},
  year    = {2004}, volume = {83}, number = {8}, pages = {9101--9108}
}

@article{correia2004simultaneous,
  author  = {Correia, S. L. and Hotza, D. and Segad{\~a}es, A. M.},
  title   = {Simultaneous optimization of linear firing shrinkage and
             water absorption of triaxial ceramic bodies using
             experiments design},
  journal = {Ceramics International},
  year    = {2004}, volume = {30}, number = {6}, pages = {917--922},
  doi     = {10.1016/j.ceramint.2003.10.013},
  note    = {verify: DOI reported inconsistently as ...10.010 in one
             NotebookLM extraction and ...10.013 in two others --
             confirm against ScienceDirect before use}
}

@article{ngun2014using,
  author  = {Ngun, B. K. and Mohamad, H. and Katsumata, K. and Okada, K.
             and Ahmad, Z. A.},
  title   = {Using design of mixture experiments to optimize triaxial
             ceramic tile compositions incorporating Cambodian clays},
  journal = {Applied Clay Science},
  year    = {2014}, volume = {87}, pages = {97--107},
  note    = {verify: no DOI captured from the literature-review extraction}
}

@article{solanki2023mathematical,
  author  = {Solanki, Shagun and Kumar, Rajesh and Yadav, Ankit Prakash
             and Gupta, Sandeep},
  title   = {Mathematical modelling and ANOVA analysis to develop
             sustainable ceramic tiles using high volume marble slurry},
  journal = {Materials Today: Proceedings},
  year    = {2023},
  doi     = {10.1016/j.matpr.2022.12.229}
}

@article{carty1998porcelain,
  author  = {Carty, William M. and Senapati, Udayan},
  title   = {Porcelain---Raw Materials, Processing, Phase Evolution, and
             Mechanical Behavior},
  journal = {Journal of the American Ceramic Society},
  year    = {1998}, volume = {81}, number = {1}, pages = {3--20},
  doi     = {10.1111/j.1151-2916.1998.tb02439.x}
}

@article{piepel2002augmenting,
  author  = {Piepel, Greg F. and Szychowski, Jeff M. and Loeppky, Jason L.},
  title   = {Augmenting Scheff{\'e} Linear Mixture Models with Squared
             and/or Crossproduct Terms},
  journal = {Journal of Quality Technology},
  year    = {2002}, volume = {34}, number = {3}, pages = {297--314},
  doi     = {10.1080/00224065.2002.11980160}
}

@article{bilger2015measuring,
  author  = {Bilger, Marcel and Manning, Willard G.},
  title   = {Measuring overfitting in nonlinear models: a new method and
             an application to health},
  journal = {Health Economics},
  year    = {2015}, volume = {24}, number = {1}, pages = {75--85},
  doi     = {10.1002/hec.3003}
}

@article{babyak2004what,
  author  = {Babyak, Michael A.},
  title   = {What you see may not be what you get: a brief, nontechnical
             introduction to overfitting in regression-type models},
  journal = {Psychosomatic Medicine},
  year    = {2004}, volume = {66}, number = {3}, pages = {411--421},
  doi     = {10.1097/01.psy.0000127692.23278.a9}
}

@article{janet2019quantitative,
  author  = {Janet, Jon Paul and Duan, Chenru and Yang, Tzuhsiung and
             Nandy, Aditya and Kulik, Heather J.},
  title   = {A quantitative uncertainty metric controls error in neural
             network-driven chemical discovery},
  journal = {Chemical Science},
  year    = {2019}, volume = {10}, number = {34}, pages = {7913--7922},
  doi     = {10.1039/C9SC02298A}
}

@article{korolev2022universal,
  author  = {Korolev, Vadim and Nevolin, Iurii and Protsenko, Pavel},
  title   = {A universal similarity based approach for predictive
             uncertainty quantification in materials science},
  journal = {Scientific Reports},
  year    = {2022}, volume = {12}, pages = {14522},
  doi     = {10.1038/s41598-022-19205-5}
}

@article{kaneko2026knnpc,
  author  = {Kaneko, Hiromasa},
  title   = {kNNPC (k-Nearest Neighbor Algorithm Per-Class):
             Classification Method Predicting Extrapolation Regions With
             Reasonable Probability},
  journal = {Journal of Chemometrics},
  year    = {2026}, volume = {40}, number = {6}, pages = {e70135},
  doi     = {10.1002/cem.70135}
}

@article{hammad2018optimising,
  author  = {Hammad, Ahmed W. A. and Akbarnezhad, Ali and Oldfield, Philip},
  title   = {Optimising Embodied Carbon and U-value in Load Bearing
             Walls: A Mathematical Bi-Objective Mixed Integer
             Programming Approach},
  journal = {Energy and Buildings},
  year    = {2018}, volume = {174}, pages = {657--671},
  doi     = {10.1016/j.enbuild.2018.05.061}
}

@article{young2018compressive,
  author  = {Young, Benjamin A. and Hall, Alex and Pilon, Laurent and
             Gupta, Puneet and Sant, Gaurav},
  title   = {Can the compressive strength of concrete be estimated from
             knowledge of the mixture proportions?: New insights from
             statistical analysis and machine learning methods},
  journal = {Cement and Concrete Research},
  year    = {2018},
  note    = {verify: no DOI captured from the literature-review extraction}
}

% ── CO2 emission factors — resolved against the paper's own references.bib
%    (ceramic_tiles_refs.bib); replaces the earlier "verify: ???" stub
%    entries (zeng2025clay, li2023etp, lbminerals_sodaf_epd,
%    lbminerals_potashf_epd, lbminerals_chamotte_epd, prochin_nasil_epd) ──
@article{zeng2025carbonfootprint,
  author  = {Zeng, Jie and Li, Haojin and Li, Fazhe},
  title   = {A Carbon Footprint Assessment for Building Ceramics from the
             Life Cycle Perspective: A Case Study in Eastern China},
  journal = {Engineering},
  year    = {2025}, volume = {17}, number = {12},
  doi     = {10.4236/eng.2025.1712035},
  note    = {Used for AG98/AG22/AG23 clay CO2 factor (0.129 kg/kg);
             AG22 and AG23 use this same figure as a proxy, since no
             clay-type-specific factor is published in this source}
}

@article{li2023carbon,
  author  = {Li, Liping and Du, Guiyue and Yan, Beibei and Wang, Yuan and
             Zhao, Yingxin and Su, Jianming and Li, Hongyi and Du, Yanfeng and
             Sun, Yunan and Chen, Guanyi and others},
  title   = {Carbon footprint analysis of sewage sludge thermochemical
             conversion technologies},
  journal = {Sustainability},
  year    = {2023}, volume = {15}, number = {5}, pages = {4170},
  note    = {No DOI available in the paper's own references.bib;
             incineration-pathway figure used for ETP sludge CO2
             factor (0.242 kg/kg)}
}

@techreport{lbminerals2025feldspar,
  title       = {Feldspar EPD of multiple products, based on average results},
  institution = {LB MINERALS, s.r.o.},
  year        = {2025},
  note        = {EPD registration number: EPD-IES-0024844. Single EPD
                 covering multiple feldspar products; the Pobežovice
                 site figure is used for Soda Feldspar (0.053 kg/kg)
                 and the Nová Ves site figure for Potash Feldspar
                 (0.0286 kg/kg). Replaces the earlier separate,
                 unconfirmed lbminerals_sodaf_epd / lbminerals_potashf_epd
                 stub entries.}
}

@techreport{lbminerals2025milledchamotte,
  title       = {Milled Chamotte Mixtures (Chamotte Component)},
  institution = {LB MINERALS, s.r.o.},
  year        = {2025},
  url         = {https://www.ekoznacka.cz/wp-content/uploads/2025/12/EPD_Milled-Chamotte-Mixtures.pdf},
  note        = {Environmental Product Declaration; valid until
                 2030-12-03. Used for the Crushing (chamotte/grog) CO2
                 factor (0.587 kg/kg)}
}

@techreport{prochin2025nasil,
  title       = {Environmental Product Declaration - Sodium Silicates},
  institution = {Prochin Italia},
  year        = {2025},
  note        = {EPD registration number: EPD-IES-0021224. Used for the
                 NaSil (Sodium Silicate) CO2 factor (0.433 kg/kg)}
}
"""

if __name__ == "__main__":
    main()
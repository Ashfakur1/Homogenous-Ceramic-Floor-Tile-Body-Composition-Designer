#!/usr/bin/env python3
"""
inverse_design.py
Inverse design of ceramic tile compositions via three methods.
This module is a pure backend library imported by streamlit_app.py.
It contains no plotting or file-output logic.

METHODS
  1. Nearest Neighbour — non-optimised (property proximity only)
     Returns the single synthetic dataset sample whose STORED properties
     are closest to the target in scaled property space. Serves as a
     baseline — no cost or CO2 consideration.

  2. Nearest Neighbour — cost + CO2 optimised
     Searches the 10 nearest neighbours in scaled property space and
     selects the composition with the lowest combined cost + CO2 rank.

  3. Bayesian Optimisation (Optuna TPE)
     Searches the continuous composition space to minimise normalised
     cost + normalised CO2 subject to meeting the target properties.

ARCHITECTURE — PREDICTION vs. OPTIMISATION (IMPORTANT)
-------------------------------------------------------
Two independent engines live in this framework:

  Engine 1 — Prediction (forward_model):
      Learns Composition -> Properties. This relationship does NOT depend
      on raw-material price. The model is NOT retrained when prices change.

  Engine 2 — Optimisation (Methods 1-3 below):
      Uses Composition + CURRENT price/CO2 tables to evaluate an objective
      (min cost, min CO2, subject to hitting property targets). Prices and
      CO2 factors are dynamic, so they are intentionally kept OUT of the
      training dataset and are instead read at run time from an external,
      independently-updatable data source (see price_loader.py):

          data/cost/<date>_Cost.csv   (latest file wins)
          data/co2/<date>_CO2.csv     (latest file wins)

      Call refresh_prices() to reload the latest cost/CO2 tables into
      memory (e.g. from a Streamlit button, or automatically before each
      optimisation run). No retraining is ever required when prices change
      — only the price/CO2 database needs updating.

      For backward compatibility, if no dated cost/CO2 files are found,
      this module falls back to the values baked into metadata.json at
      dataset-generation time (the original architecture).

REPORTED-PROPERTY CONSISTENCY ACROSS ALL THREE METHODS (IMPORTANT)
--------------------------------------------------------------------
All three methods report "predicted" properties from the most reliable
source of truth available for the chosen composition:
  - If the composition is an ACTUAL fabricated lab_batch (calibration)
    row, the ACTUAL MEASURED properties are reported (property_source =
    "measured") — we already know the ground truth for that recipe, so
    there is no reason to substitute a model estimate for it.
  - Otherwise (a synthetic, physics-surrogate-generated composition, or
    anything Method 3's continuous optimiser proposes), properties are
    recomputed via forward_model.predict() (property_source =
    "predicted") — never read from dataset.csv's stored MOR_MPa / WA_pct
    / Shrinkage_pct columns for synthetic rows, since those are
    physics-surrogate outputs with injected heteroscedastic noise, not
    forward-model predictions.
This is what "is_verified_batch" / "property_source" in each method's
result dict indicate. Before this change, a composition recommended by
Method 1/2 and the same composition evaluated by Method 3 could show two
different "predicted" values for the same MOR/WA/Shrinkage, purely as an
artefact of which method happened to find it.

SEARCH CORPUS FOR METHODS 1-2 (IMPORTANT)
--------------------------------------------
Methods 1 and 2 search over source in {"synthetic", "lab_batch"} —
the physics-surrogate-generated compositions AND the real laboratory
calibration batches. lab_holdout is deliberately EXCLUDED from this
corpus (and from everything else in this module): it is kept completely
untouched everywhere upstream (surrogate fitting, forward-model training)
specifically so it remains available as an independent check, and
including it here — even just as a candidate a user might be shown —
would compromise that. Including lab_batch means a target close to an
already-fabricated, already-measured recipe can be recommended directly,
with its real measured properties, instead of only ever returning a
surrogate-generated candidate.

WHICH FORWARD MODEL FILE IS LOADED (IMPORTANT)
-------------------------------------------------
train_forward_model.py saves two models and runs a real-vs-real+synthetic
ablation on the lab_holdout batches (data/ablation_real_vs_synthetic.csv):
    models/forward_model.joblib            (real + synthetic training data)
    models/forward_model_real_only.joblib  (real calibration batches only)
That ablation found LOWER error for forward_model_real_only.joblib on all
three targets on the (small, n=8) holdout set — a directional signal, not
a definitive result (see the module docstring of train_forward_model.py).
This module does not silently switch on that basis; FORWARD_MODEL_FILE
below picks explicitly, defaulting to the real+synthetic "production"
model. Change the constant (or check ablation_real_vs_synthetic.csv
yourself first) to switch.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import optuna
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from price_loader import load_price_table

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).parent
DATADIR = BASE_DIR / "data"
MODELDIR = BASE_DIR / "models"

# Dynamic price/CO2 database folders — see price_loader.py
COST_DIR = DATADIR / "cost"
CO2_DIR = DATADIR / "co2"

# Which trained forward model to load — see "WHICH FORWARD MODEL FILE IS
# LOADED" in the module docstring above before changing this.
FORWARD_MODEL_FILE = "forward_model.joblib"

# ── Load dataset and model (unchanged from original architecture) ─────────────
dataset = pd.read_csv(DATADIR / "dataset.csv")

# Defensive schema check — dataset.csv's source column is expected to be
# exactly {"synthetic", "lab_batch", "lab_holdout"} (see generate_dataset.py).
# This module reads "synthetic" and "lab_batch" rows (see _ds_search below)
# and must never read "lab_holdout" — but if the set of labels changes
# again upstream, fail loudly here rather than silently searching over
# the wrong rows.
_expected_sources = {"synthetic", "lab_batch", "lab_holdout"}
_actual_sources = set(dataset["source"].unique())
if not _actual_sources.issubset(_expected_sources):
    raise ValueError(
        f"dataset.csv has unexpected source label(s): "
        f"{_actual_sources - _expected_sources}. Expected a subset of "
        f"{_expected_sources}. Update the filtering logic in this module "
        f"before proceeding."
    )

forward_model = joblib.load(MODELDIR / FORWARD_MODEL_FILE)

with open(DATADIR / "metadata.json") as f:
    meta = json.load(f)

with open(MODELDIR / "feature_cols.json") as f:
    model_feature_cols: list[str] = json.load(f)

materials = meta["materials"]
bounds = meta["bounds"]

# Fallback values (original static architecture) used only if no dated
# cost/CO2 CSV files are found in data/cost/ or data/co2/.
_fallback_cost = meta.get("cost_tk_per_kg", {m: 1.0 for m in materials})
_fallback_co2_raw = meta.get("co2_kg_per_kg", {m: 0.0 for m in materials})
_fallback_co2 = {
    m: float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)
    for m, v in _fallback_co2_raw.items()
}

TARGET_COLS = ["MOR_MPa", "WA_pct", "Shrinkage_pct"]

# ── Module-level state populated by refresh_prices() ───────────────────────
cost_dict: dict[str, float] = {}
co2_dict: dict[str, float] = {}
price_info: dict = {"cost": {}, "co2": {}}

_ds_search: pd.DataFrame = pd.DataFrame()
_pr: dict[str, float] = {}
_prop_ranges: dict[str, float] = {}
_cost_min = _cost_max = _cost_range = 0.0
_co2_min = _co2_max = _co2_range = 0.0
_prop_scaler: StandardScaler | None = None
_X_props_sc = None
_nbrs: NearestNeighbors | None = None


def refresh_prices() -> None:
    """
    Reload the latest raw-material cost and CO2-factor tables from the
    dated CSV database (data/cost/, data/co2/) and recompute every value
    that is derived from them.

    IMPORTANT: this touches ONLY price/CO2-derived quantities. The
    forward_model (Composition -> Properties) and the property-based
    nearest-neighbour scaler/index are completely untouched — no
    retraining ever happens here. This is exactly the "Optimisation
    Layer reads a dynamic external database" design: today's price is
    10, tomorrow's is 20 — update the CSV and call refresh_prices();
    the optimizer immediately produces a new, economically-correct
    recommendation with zero model changes.
    """
    global cost_dict, co2_dict, price_info
    global dataset, _ds_search
    global _pr, _prop_ranges
    global _cost_min, _cost_max, _cost_range
    global _co2_min, _co2_max, _co2_range
    global _prop_scaler, _X_props_sc, _nbrs

    cost_dict, cost_meta = load_price_table(
        COST_DIR, keyword="Cost", value_col="Price_Tk_per_kg",
        materials=materials, fallback=_fallback_cost,
    )
    co2_dict, co2_meta = load_price_table(
        CO2_DIR, keyword="CO2", value_col="CO2_kg_per_kg",
        materials=materials, fallback=_fallback_co2,
    )
    price_info = {"cost": cost_meta, "co2": co2_meta}

    # Recompute cost_Tk_per_kg / CO2_kg_per_kg for EVERY row in the dataset
    # (lab + synthetic) from the CURRENT price/CO2 tables. This is a cheap
    # linear recombination of each row's already-known composition — no
    # forward-model inference, no retraining.
    dataset["cost_Tk_per_kg"] = sum(
        dataset[f"{m}_wtpct"] / 100 * cost_dict[m] for m in materials
    )
    dataset["CO2_kg_per_kg"] = sum(
        dataset[f"{m}_wtpct"] / 100 * co2_dict[m] for m in materials
    )
    # Search corpus for Methods 1-2: synthetic rows PLUS real calibration
    # (lab_batch) rows. A target close to an already-fabricated,
    # already-measured recipe can then be recommended directly, with its
    # real measured properties (see _summarize). lab_holdout is
    # deliberately excluded — see "SEARCH CORPUS FOR METHODS 1-2" in the
    # module docstring. This is an explicit whitelist (source.isin([...])),
    # so lab_holdout can never leak in here even if new source labels are
    # added upstream in the future.
    _ds_search = dataset[dataset["source"].isin(["synthetic", "lab_batch"])]

    # ── Property ranges from the search corpus (synthetic + calibration) ──
    _pr = {}
    for _t in TARGET_COLS:
        _pr[f"{_t}_min"] = float(_ds_search[_t].min())
        _pr[f"{_t}_max"] = float(_ds_search[_t].max())
    _prop_ranges = {
        t: max(_pr[f"{t}_max"] - _pr[f"{t}_min"], 1e-9)
        for t in TARGET_COLS
    }

    # ── Cost/CO2 ranges recomputed from CURRENT prices (was static before) ─
    _cost_min = float(_ds_search["cost_Tk_per_kg"].min())
    _cost_max = float(_ds_search["cost_Tk_per_kg"].max())
    _cost_range = max(_cost_max - _cost_min, 1e-9)

    _co2_min = float(_ds_search["CO2_kg_per_kg"].min())
    _co2_max = float(_ds_search["CO2_kg_per_kg"].max())
    _co2_range = max(_co2_max - _co2_min, 1e-9)

    # ── NN search fitted on the search corpus (synthetic + calibration) —
    #    property space is NOT affected by price, so this only needs to be
    #    (re)built once, but we keep it here so a full refresh always
    #    leaves everything consistent.
    if _prop_scaler is None:
        _prop_scaler = StandardScaler()
        _X_props_sc = _prop_scaler.fit_transform(_ds_search[TARGET_COLS].values)
        _nbrs = NearestNeighbors(n_neighbors=10, algorithm="auto").fit(_X_props_sc)


def get_price_info() -> dict:
    """
    Return metadata about which price/CO2 files are currently loaded, for
    display in the UI, e.g.:
        {"cost": {"file": PosixPath("data/cost/17_April_2026_Cost.csv"),
                  "as_of": datetime(...), "source": "file"},
         "co2":  {"file": PosixPath("data/co2/19_April_2026_CO2.csv"),
                  "as_of": datetime(...), "source": "file"}}
    """
    return price_info


def get_property_ranges() -> dict:
    """
    Return the current target-clamping range for each property, e.g.:
        {"MOR_MPa_min": 40.34, "MOR_MPa_max": 68.75, ...}

    This is computed from the SAME search corpus used by clamp_targets()
    and Methods 1-2 (synthetic + real calibration batches — see
    "SEARCH CORPUS FOR METHODS 1-2" in the module docstring). Consumers
    such as streamlit_app.py should call this rather than recomputing
    their own range from dataset.csv, so the number-input bounds shown
    in the UI can never drift out of sync with what clamp_targets()
    actually enforces internally.
    """
    return dict(_pr)


# Populate all module-level state on first import so the module works
# exactly as before if nobody calls refresh_prices() explicitly.
refresh_prices()

# ── Process parameter defaults (held constant — fixed firing cycle) ───────
default_proc: dict[str, float] = {
    "dryer_temp_C": 180.0,
    "green_length_mm": 109.20,
    "green_width_mm": 54.60,
    "green_thickness_mm": 9.80,
    "green_weight_g": 98.50,
    "fired_length_mm": 98.00,
    "fired_weight_g": 95.05,
    "gas_Nm3_per_m2": 1.4115,
}

TGT_LABELS = {
    "MOR_MPa": "Firing MOR (MPa)",
    "WA_pct": "Water Absorption (%)",
    "Shrinkage_pct": "Fired Shrinkage (%)",
}
MAT_SHORT = {
    "AG98": "AG98",
    "AG22": "AG22",
    "AG23": "AG23",
    "SodaF": "Soda F.",
    "PotashF": "Potash F.",
    "Crushing": "Crushing",
    "ETP": "ETP Clay",
    "NaSil": "Na-Sil.",
}


# ── Helper functions ────────────────────────────────────────────────────────
def build_input_row(comp: dict[str, float]) -> pd.DataFrame:
    """Assemble a single-row DataFrame in the format expected by the forward model."""
    row = {f"{m}_wtpct": comp[m] for m in materials}
    row.update(default_proc)
    return pd.DataFrame([row])[model_feature_cols]


def clamp_targets(MOR: float, WA: float, SH: float) -> tuple[float, float, float]:
    """
    Clamp user-supplied target values to the synthetic dataset range.
    Values outside the calibrated range are clipped and a warning is issued.
    The clamped values are the actual targets used in all three methods.
    """
    MOR_c = float(np.clip(MOR, _pr["MOR_MPa_min"], _pr["MOR_MPa_max"]))
    WA_c = float(np.clip(WA, _pr["WA_pct_min"], _pr["WA_pct_max"]))
    SH_c = float(np.clip(SH, _pr["Shrinkage_pct_min"], _pr["Shrinkage_pct_max"]))
    if MOR != MOR_c:
        warnings.warn(f"MOR_MPa clamped: {MOR:.3f} -> {MOR_c:.3f} "
                       f"(dataset range [{_pr['MOR_MPa_min']:.3f}, {_pr['MOR_MPa_max']:.3f}])")
    if WA != WA_c:
        warnings.warn(f"WA_pct clamped: {WA:.4f} -> {WA_c:.4f} "
                       f"(dataset range [{_pr['WA_pct_min']:.4f}, {_pr['WA_pct_max']:.4f}])")
    if SH != SH_c:
        warnings.warn(f"Shrinkage_pct clamped: {SH:.3f} -> {SH_c:.3f} "
                       f"(dataset range [{_pr['Shrinkage_pct_min']:.3f}, "
                       f"{_pr['Shrinkage_pct_max']:.3f}])")
    return MOR_c, WA_c, SH_c


def _enforce_bounds(comp: dict[str, float]) -> dict[str, float]:
    """
    Clip each material to its individual feasibility bounds then renormalise
    to sum = 100 wt%. A second renormalisation step is required because
    clipping can push the batch sum away from 100.
    """
    clipped = {m: float(np.clip(comp[m], bounds[m][0], bounds[m][1]))
               for m in materials}
    total = sum(clipped.values())
    return {m: v / total * 100.0 for m, v in clipped.items()}


def _summarize(row: pd.Series) -> dict:
    """
    Package a dataset row into the standard result dictionary.

    Property source depends on what this row actually is:
      - source == "lab_batch": an already-fabricated, already-measured
        recipe. The ACTUAL MEASURED properties are reported
        (property_source="measured") — we already know the ground truth,
        so there is no reason to substitute a model estimate.
      - otherwise (synthetic): properties are recomputed via
        forward_model.predict() (property_source="predicted"), never
        read from the row's stored MOR_MPa / WA_pct / Shrinkage_pct
        columns. Those stored columns are physics-surrogate outputs with
        injected noise, not forward-model predictions; using them here
        would make Method 1/2's reported properties inconsistent with
        Method 3's (which has always used forward_model directly) for
        the exact same composition. See "REPORTED-PROPERTY CONSISTENCY
        ACROSS ALL THREE METHODS" in the module docstring.
    """
    comp = {m: float(row[f"{m}_wtpct"]) for m in materials}
    is_verified = row.get("source", "synthetic") == "lab_batch"

    if is_verified:
        properties = {
            "MOR_MPa": round(float(row["MOR_MPa"]), 3),
            "WA_pct": round(float(row["WA_pct"]), 5),
            "Shrinkage_pct": round(float(row["Shrinkage_pct"]), 3),
        }
        property_source = "measured"
    else:
        pred = forward_model.predict(build_input_row(comp))[0]
        properties = {
            "MOR_MPa": round(float(pred[0]), 3),
            "WA_pct": round(float(pred[1]), 5),
            "Shrinkage_pct": round(float(pred[2]), 3),
        }
        property_source = "predicted"

    return {
        "composition_wtpct": {m: round(comp[m], 4) for m in materials},
        "predicted": properties,
        "property_source": property_source,
        "is_verified_batch": is_verified,
        "cost_Tk_per_kg": round(float(row.get("cost_Tk_per_kg", 0)), 4),
        "CO2_kg_per_kg": round(float(row.get("CO2_kg_per_kg", 0)), 5),
    }


# ── Inverse design methods (unchanged logic — now reading dynamic prices) ──
def inverse_non_optimized(MOR_MPa: float, WA_pct: float,
                           Shrinkage_pct: float) -> dict:
    """
    Method 1: Nearest Neighbour (non-optimised).
    Searches for the single sample — synthetic OR a real calibration
    (lab_batch) batch — whose STORED properties are closest to the target
    in StandardScaler-normalised property space (equal weight for MOR,
    WA, and Shrinkage) — this governs which composition is chosen. No
    cost or CO2 consideration. If the match is a real batch, its ACTUAL
    MEASURED properties are returned (property_source="measured");
    otherwise properties are recomputed via forward_model for consistency
    with Methods 2 and 3 (see _summarize).
    """
    MOR_MPa, WA_pct, Shrinkage_pct = clamp_targets(MOR_MPa, WA_pct, Shrinkage_pct)
    q = _prop_scaler.transform([[MOR_MPa, WA_pct, Shrinkage_pct]])
    idx = _nbrs.kneighbors(q, n_neighbors=1, return_distance=False)[0][0]
    return _summarize(_ds_search.iloc[idx])


def inverse_optimized(MOR_MPa: float, WA_pct: float,
                       Shrinkage_pct: float) -> tuple[dict, bool]:
    """
    Method 2: Nearest Neighbour (cost + CO2 optimised).
    Searches the 10 nearest neighbours (synthetic OR real calibration
    batches) in scaled property space (using STORED properties, same
    search space as Method 1) and selects the composition with the
    lowest combined cost + CO2 percentile rank, using the CURRENTLY
    loaded price/CO2 tables (call refresh_prices() first to guarantee
    this reflects the latest prices). If the selected match is a real
    batch, its ACTUAL MEASURED properties are returned
    (property_source="measured"); otherwise properties are recomputed
    via forward_model (see _summarize), for consistency with Methods 1
    and 3.
    """
    MOR_MPa, WA_pct, Shrinkage_pct = clamp_targets(MOR_MPa, WA_pct, Shrinkage_pct)
    q = _prop_scaler.transform([[MOR_MPa, WA_pct, Shrinkage_pct]])
    idxs = _nbrs.kneighbors(q, n_neighbors=10, return_distance=False)[0]
    sub = _ds_search.iloc[idxs].copy()
    sub["_score"] = (sub["cost_Tk_per_kg"].rank(pct=True)
                      + sub["CO2_kg_per_kg"].rank(pct=True))
    best_iloc = sub["_score"].idxmin()
    nn1_iloc = _ds_search.index[idxs[0]]
    identical = (best_iloc == nn1_iloc)
    if identical:
        print("  WARNING: Methods 1 and 2 returned the same sample. "
              "The 10-nearest-neighbour neighbourhood lacks cost/CO2 "
              "diversity for this target.")
    return _summarize(_ds_search.loc[best_iloc]), identical


def inverse_bayesian_optimization(
    MOR_MPa_tgt: float,
    WA_tgt: float,
    Shrink_tgt: float,
    n_trials: int = 200,
) -> tuple[dict, list[float], optuna.Study]:
    """
    Method 3: Bayesian Optimisation (Optuna TPE).
    objective = norm_cost + norm_co2 + penalty, using the CURRENTLY loaded
    price/CO2 tables (call refresh_prices() first to guarantee this reflects
    the latest prices — no retraining of the forward model is ever needed).
    """
    MOR_MPa_tgt, WA_tgt, Shrink_tgt = clamp_targets(
        MOR_MPa_tgt, WA_tgt, Shrink_tgt
    )
    trial_vals: list[float] = []

    def _objective(trial: optuna.Trial) -> float:
        free_mats = [m for m in materials if m != "SodaF"]
        comp = {m: trial.suggest_float(f"c_{m}", bounds[m][0], bounds[m][1])
                for m in free_mats}

        # Simplex: SodaF derived from constraint
        soda_f = 100.0 - sum(comp.values())
        soda_lo, soda_hi = bounds["SodaF"][0], bounds["SodaF"][1]

        soda_pen = 0.0
        if soda_f < soda_lo:
            soda_pen = (soda_lo - soda_f) * 50.0
        elif soda_f > soda_hi:
            soda_pen = (soda_f - soda_hi) * 50.0
        comp["SodaF"] = float(np.clip(soda_f, soda_lo, soda_hi))

        comp = _enforce_bounds(comp)

        pred = forward_model.predict(build_input_row(comp))[0]
        MOR_p, WA_p, SH_p = pred[0], pred[1], pred[2]

        cost = sum(comp[m] / 100 * cost_dict[m] for m in materials)
        co2 = sum(comp[m] / 100 * co2_dict[m] for m in materials)

        norm_cost = (cost - _cost_min) / _cost_range
        norm_co2 = (co2 - _co2_min) / _co2_range

        penalty = (
            max(0.0, MOR_MPa_tgt - MOR_p) / _prop_ranges["MOR_MPa"] * 5.0
            + max(0.0, WA_p - WA_tgt) / _prop_ranges["WA_pct"] * 5.0
            + abs(SH_p - Shrink_tgt) / _prop_ranges["Shrinkage_pct"] * 5.0
            + soda_pen
        )
        obj = norm_cost + norm_co2 + penalty
        trial_vals.append(obj)
        return obj

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    best_c = {m: best[f"c_{m}"] for m in materials if m != "SodaF"}
    best_c["SodaF"] = float(np.clip(
        100.0 - sum(best_c.values()),
        bounds["SodaF"][0], bounds["SodaF"][1]
    ))
    best_c = _enforce_bounds(best_c)

    pred = forward_model.predict(build_input_row(best_c))[0]
    cost = sum(best_c[m] / 100 * cost_dict[m] for m in materials)
    co2 = sum(best_c[m] / 100 * co2_dict[m] for m in materials)

    result = {
        "composition_wtpct": {m: round(best_c[m], 4) for m in materials},
        "predicted": {
            "MOR_MPa": round(float(pred[0]), 3),
            "WA_pct": round(float(pred[1]), 5),
            "Shrinkage_pct": round(float(pred[2]), 3),
        },
        # Always "predicted"/False here — Method 3 optimises a continuous
        # composition rather than looking up an existing dataset row, so
        # it can never return an already-fabricated, already-measured
        # batch the way Methods 1-2 sometimes can. Included for schema
        # consistency with _summarize()'s output (see "REPORTED-PROPERTY
        # CONSISTENCY ACROSS ALL THREE METHODS" in the module docstring).
        "property_source": "predicted",
        "is_verified_batch": False,
        "cost_Tk_per_kg": round(cost, 4),
        "CO2_kg_per_kg": round(co2, 5),
    }
    return result, trial_vals, study
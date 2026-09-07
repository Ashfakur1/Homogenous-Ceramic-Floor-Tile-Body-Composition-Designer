#!/usr/bin/env python3
"""
train_forward_model.py
Trains a multi-output ML forward model mapping ceramic tile composition
variables to mechanical and dimensional properties.

KEY DECISIONS & JUSTIFICATIONS
  [A] Three-way data split, lab_holdout is the ONLY independent test set.
      dataset.csv carries three source labels (see generate_dataset.py):
        "synthetic"    — physics-surrogate-generated compositions.
        "lab_batch"    — real laboratory batches used to calibrate the
                          physics surrogate (Ridge/Scheffe coefficients).
        "lab_holdout"  — real laboratory batches withheld entirely from
                          surrogate calibration; never touched by anything
                          upstream of this script.
      "lab_batch" rows are NOT used for evaluation, because they directly
      shaped the physics surrogate that generated the synthetic training
      data — evaluating on them would be circular. They ARE used for
      training (see [G]), since more real signal is otherwise wasted.
      "lab_holdout" rows are used ONLY for evaluation, never for training,
      anywhere in this script.
      An earlier version of this script only recognised two source labels
      ("synthetic" and "lab_batch") and treated everything else — which,
      after generate_dataset.py added the lab_holdout split, silently
      included the lab_holdout rows — as "synthetic" for train/test
      splitting purposes. That let a majority of the untouched holdout
      batches leak into the training set at random. See [H] for the
      guard added against this recurring.

  [B] Six candidate models evaluated via 5-fold cross-validation on the
      synthetic training set only (never lab_batch or lab_holdout, so
      model-architecture selection cannot be influenced by the small
      number of real batches). The best-performing model (highest mean
      R²) is selected. That architecture (fresh, uninitialised instances)
      is then used consistently for both arms of the ablation in [G] and
      for the production model in [G], so all comparisons isolate the
      effect of the TRAINING DATA, not the model architecture.
      Native multi-output RandomForest preferred over MultiOutputRegressor
      where physically motivated — MOR, WA, and fired shrinkage are
      correlated via sintering density; assuming output independence
      underestimates target covariance. However, final model selection is
      CV-driven. If a wrapped model achieves superior cross-validated R²,
      it is retained on empirical grounds and the output-independence
      assumption is noted as a limitation.

  [C] PDP computed manually on ORIGINAL-SCALE X using the full pipeline.
      Passing scaled data to PartialDependenceDisplay yields standardised
      x-axes (z-score units) that are uninterpretable by practitioners.
      Manual computation calls pipeline.predict(X_modified) directly so
      x-values are naturally in wt%. Marginalised over the production
      model's actual training pool (synthetic train split + lab_batch
      calibration rows — see [G]), not the synthetic split alone, since
      that is what the deployed model was actually trained on.

  [D] Single rectangular heatmap (composition features × targets) only.
      Process variables confirmed to have near-zero Pearson correlation
      with all three targets (|r| < 0.07) and are excluded from the
      heatmap.

  [E] Feature importance bar chart retains composition variables only,
      as process variables are held constant across all batches and carry
      no predictive signal.

  [F] Composition variables only are used as model features. Process
      variables (press_bar, kiln_temp_C, etc.) are held constant at the
      values recorded during fabrication of the calibration batches and
      confirmed to show near-zero correlation with all three targets
      (|r| < 0.07). Including them would add noise without predictive
      benefit and would prevent deployment in settings where process
      parameters differ slightly from the calibration conditions.

  [G] Ablation: does synthetic augmentation actually help? Two models
      (same architecture, chosen in [B]) are trained on:
        Arm "real_only"          — lab_batch calibration rows alone.
        Arm "real_plus_synthetic" — lab_batch calibration rows + the
                                     synthetic training split.
      Both are evaluated exclusively on lab_holdout (real batches neither
      arm has ever seen). This directly answers whether the physics-
      surrogate-generated synthetic data improves forward-model accuracy
      on genuinely new experimental compositions, or whether it dilutes
      the signal from the (now much larger) real calibration set. The
      "real_plus_synthetic" arm is saved as the production model, since
      using more information is preferable when the ablation does not
      show it to be worse — but the per-target ablation numbers are
      printed and plotted so this can be checked, not assumed.

  [H] Defensive schema check. The three source masks (synthetic,
      lab_batch, lab_holdout) are asserted to partition the dataset
      exactly (no row unmatched, no row double-matched). If
      generate_dataset.py's source labelling scheme changes again in the
      future, this raises immediately instead of silently mis-splitting
      data the way the two-label version of this script did (see [A]).

NOTE — Experimental-batch (lab_holdout) performance metrics
  R² is undefined or misleading for a small number of held-out points
  drawn from a distribution shift relative to the synthetic training set.
  RMSE is reported instead for the lab_holdout subset, as it carries the
  same physical units as the target and is robust to small sample sizes.
"""

import json, math, warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk", font_scale=1.1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOTDIR  = Path(__file__).parent
DATADIR  = ROOTDIR / "data"
MODELDIR = ROOTDIR / "models"
PLOTDIR  = ROOTDIR / "plots"
for p in [MODELDIR, PLOTDIR]:
    p.mkdir(exist_ok=True, parents=True)

# ── Unified font scale ────────────────────────────────────────────────────────
_FS_TITLE = 18
_FS_AX    = 16
_FS_TICK  = 14
_FS_LABEL = 13
_FS_ANNOT = 12
_DPI      = 300

# ── Load data ─────────────────────────────────────────────────────────────────
df   = pd.read_csv(DATADIR / "dataset.csv")
with open(DATADIR / "metadata.json") as f:
    meta = json.load(f)

# Data-provenance stamp — carried over from generate_dataset.py so every
# figure in this script is tied to the SAME raw-data version as the figures
# produced upstream. If dataset.csv is regenerated from an updated CSV but
# this script is run against a stale copy (or vice versa), the stamps will
# visibly disagree instead of the mismatch hiding in the numbers alone.
DATA_HASH    = meta.get("data_hash", "unknown")
GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def _stamp(fig) -> None:
    fig.text(0.995, 0.002, f"data:{DATA_HASH}  generated:{GENERATED_AT}",
              ha="right", va="bottom", fontsize=6, color="0.6",
              family="monospace")

def savefig(fig, stem: str) -> None:
    _stamp(fig)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"{stem}.{ext}", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

materials   = meta["materials"]
TARGET_COLS = ["MOR_MPa", "WA_pct", "Shrinkage_pct"]

TGT_LABELS = {
    "MOR_MPa":       "Firing MOR (MPa)",
    "WA_pct":        "Water Absorption (%)",
    "Shrinkage_pct": "Fired Shrinkage (%)",
}
MAT_SHORT = {
    "AG98_wtpct":     "AG98",
    "AG22_wtpct":     "AG22",
    "AG23_wtpct":     "AG23",
    "SodaF_wtpct":    "Soda Feldspar",
    "PotashF_wtpct":  "Potash Feldspar",
    "Crushing_wtpct": "Crushing",
    "ETP_wtpct":      "ETP Clay",
    "NaSil_wtpct":    "Na-Silicate",
}
units = {"MOR_MPa": "MPa", "WA_pct": "%", "Shrinkage_pct": "%"}

# ── Feature selection: composition variables only (see Note [F]) ──────────────
feature_cols = [f"{m}_wtpct" for m in materials
                if f"{m}_wtpct" in df.columns]
comp_cols    = feature_cols

print(f"Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols]
y = df[TARGET_COLS]

# ── Three-way source split (see [A], [H]) ──────────────────────────────────────
synth_mask   = df["source"] == "synthetic"
calib_mask   = df["source"] == "lab_batch"
holdout_mask = df["source"] == "lab_holdout"

# Defensive schema check — see [H]. Catches exactly the class of bug that
# a two-label version of this script had against a three-label dataset.
_n_matched = int(synth_mask.sum() + calib_mask.sum() + holdout_mask.sum())
_unmatched = df.loc[~(synth_mask | calib_mask | holdout_mask), "source"].unique()
if _n_matched != len(df) or len(_unmatched) > 0:
    raise ValueError(
        f"source column has values not recognised by this script: "
        f"{list(_unmatched)}. Expected exactly 'synthetic', 'lab_batch', "
        f"'lab_holdout'. Update the three masks above before proceeding — "
        f"do NOT fall back to a '~synth_mask' catch-all, since that is "
        f"what caused lab_holdout rows to leak into training previously."
    )

synth_idx   = df[synth_mask].index
calib_idx   = df[calib_mask].index
holdout_idx = df[holdout_mask].index

# Synthetic-only train/test split — used for (i) 5-fold CV model-architecture
# selection in [B], and (ii) a synthetic-only parity sanity check alongside
# the real lab_holdout evaluation. calib_idx and holdout_idx never appear
# in this split.
synth_train_idx, synth_test_idx = train_test_split(
    synth_idx, test_size=0.20, random_state=7
)

# Production training pool: synthetic training rows + ALL real calibration
# rows (see [G], "real_plus_synthetic" arm / final production model).
train_idx_aug = synth_train_idx.tolist() + calib_idx.tolist()

# Evaluation pool for parity plots etc.: synthetic test rows (diagnostic)
# + lab_holdout (the only genuinely independent real-data test). calib_idx
# is deliberately excluded — it was used to train the production model.
test_idx = synth_test_idx.tolist() + holdout_idx.tolist()

X_train_cv = X.loc[synth_train_idx]
y_train_cv = y.loc[synth_train_idx].values

X_train_aug, y_train_aug = X.loc[train_idx_aug], y.loc[train_idx_aug].values
X_test,      y_test      = X.loc[test_idx],      y.loc[test_idx].values

print(f"Synthetic: {len(synth_idx)}  (train {len(synth_train_idx)} / "
      f"test {len(synth_test_idx)})  |  "
      f"Calibration (lab_batch): {len(calib_idx)}  |  "
      f"Holdout (lab_holdout, independent test): {len(holdout_idx)}")
print(f"Production training pool: {len(train_idx_aug)} rows "
      f"({len(synth_train_idx)} synthetic + {len(calib_idx)} calibration)")

preproc = ColumnTransformer(
    [("num", StandardScaler(), feature_cols)], remainder="drop"
)

# ── Input–output correlation heatmap ─────────────────────────────────────────
# Shows the Pearson correlation between each composition variable and each
# target property. Provides the scientific basis for understanding which
# raw materials drive which properties — essential context for interpreting
# the forward model and the inverse design outputs.
comp_only_X = X[comp_cols]
corr_io = pd.concat([comp_only_X, y], axis=1).corr().loc[comp_cols, TARGET_COLS]

corr_io_display = corr_io.copy()
corr_io_display.index   = [MAT_SHORT.get(c, c) for c in corr_io_display.index]
corr_io_display.columns = [TGT_LABELS.get(c, c) for c in corr_io_display.columns]
corr_io.to_csv(DATADIR / "input_output_correlation.csv")

fig, ax = plt.subplots(figsize=(11, 10))
sns.heatmap(corr_io_display,
            annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 16}, ax=ax)
ax.set_title(
    "Pearson Correlation Coefficients Between\n"
    "Composition Variables and Target Properties",
    pad=20, fontsize=20, fontweight="bold"
)
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(rotation=0,  fontsize=16)
ax.collections[0].colorbar.ax.tick_params(labelsize=16)
plt.tight_layout()
savefig(fig, "input_output_correlation_heatmap")
print("Saved: input_output_correlation_heatmap.pdf / .png")

# ── Candidate models (see [B]) ────────────────────────────────────────────────
candidate_models: dict = {
    "RandomForest_native": RandomForestRegressor(
        n_estimators=500, random_state=7, n_jobs=-1
    ),
    "RandomForest_wrapped": MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=7, n_jobs=-1)
    ),
    "MLP": MultiOutputRegressor(
        MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=3000,
                     random_state=42, early_stopping=True,
                     n_iter_no_change=30)
    ),
}
for _lib, _cls, _name, _kw in [
    ("xgboost",  "XGBRegressor",      "XGB",
     {"n_estimators":500,"random_state":7,"n_jobs":-1,"verbosity":0}),
    ("lightgbm", "LGBMRegressor",     "LGBM",
     {"n_estimators":500,"random_state":7,"n_jobs":-1,
      "verbose":-1,"min_gain_to_split":0.0}),
    ("catboost", "CatBoostRegressor", "CatBoost",
     {"iterations":500,"verbose":0,"random_seed":7}),
]:
    try:
        candidate_models[_name] = MultiOutputRegressor(
            getattr(__import__(_lib), _cls)(**_kw)
        )
    except Exception:
        pass

# ── 5-fold CV on synthetic training rows only (see [B]) ──────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=7)
model_scores: dict[str, float] = {}
print("\nCross-validation (5-fold, synthetic training rows only, R²):")
for name, model in candidate_models.items():
    try:
        pipe   = Pipeline([("preproc", preproc), ("reg", clone(model))])
        scores = cross_val_score(pipe, X_train_cv, y_train_cv,
                                 cv=cv, scoring="r2", n_jobs=-1)
        model_scores[name] = float(np.mean(scores))
        print(f"  {name:<28s}  mean={model_scores[name]:.4f}  "
              f"std={np.std(scores):.4f}")
    except Exception as e:
        print(f"  {name:<28s}  skipped: {e}")

best_name = max(model_scores, key=model_scores.get)
print(f"\nBest model: {best_name}  (R² = {model_scores[best_name]:.4f})")

# ── Ablation: real-only vs real+synthetic, evaluated on lab_holdout (see [G]) ─
print(f"\nAblation — same architecture ({best_name}), different training data, "
      f"evaluated on {len(holdout_idx)} lab_holdout batches:")

model_real_only = Pipeline([
    ("preproc", preproc),
    ("reg",     clone(candidate_models[best_name])),
])
model_real_only.fit(X.loc[calib_idx], y.loc[calib_idx].values)

model_real_plus_synth = Pipeline([
    ("preproc", preproc),
    ("reg",     clone(candidate_models[best_name])),
])
model_real_plus_synth.fit(X_train_aug, y_train_aug)

X_holdout = X.loc[holdout_idx]
y_holdout = y.loc[holdout_idx].values

ablation_rows = []
for arm_name, arm_model in [("real_only", model_real_only),
                            ("real_plus_synthetic", model_real_plus_synth)]:
    y_pred_arm = arm_model.predict(X_holdout)
    for i, t in enumerate(TARGET_COLS):
        mae_a  = float(np.mean(np.abs(y_holdout[:, i] - y_pred_arm[:, i])))
        rmse_a = float(np.sqrt(np.mean((y_holdout[:, i] - y_pred_arm[:, i]) ** 2)))
        ablation_rows.append({"arm": arm_name, "target": t,
                              "mae": mae_a, "rmse": rmse_a})
        print(f"  {arm_name:<20s}  {t:<15s}  "
              f"MAE={mae_a:.4f} {units[t]}  RMSE={rmse_a:.4f} {units[t]}")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(DATADIR / "ablation_real_vs_synthetic.csv", index=False)

print(f"\nAblation verdict (n={len(holdout_idx)} holdout batches — small-n, "
      f"treat as a directional signal, not a definitive result):")
for t in TARGET_COLS:
    rmse_real  = ablation_df[(ablation_df.arm == "real_only") &
                             (ablation_df.target == t)]["rmse"].iloc[0]
    rmse_synth = ablation_df[(ablation_df.arm == "real_plus_synthetic") &
                             (ablation_df.target == t)]["rmse"].iloc[0]
    better = "real_plus_synthetic" if rmse_synth < rmse_real else "real_only"
    delta  = abs(rmse_synth - rmse_real)
    print(f"  {t:<18s}  lower RMSE: {better:<20s}  "
          f"(real_only={rmse_real:.4f} vs real+synth={rmse_synth:.4f}, "
          f"Δ={delta:.4f} {units[t]})")

# Ablation comparison plot — RMSE per target, real_only vs real+synthetic,
# both scored on the same lab_holdout batches.
fig, ax = plt.subplots(figsize=(9, 6))
arm_labels  = {"real_only": "Real batches only",
              "real_plus_synthetic": "Real + synthetic"}
arm_colors  = {"real_only": "#90A4AE", "real_plus_synthetic": "#1565C0"}
width = 0.35
xpos  = np.arange(len(TARGET_COLS))
for j, arm in enumerate(["real_only", "real_plus_synthetic"]):
    vals = [ablation_df[(ablation_df.arm == arm) &
                        (ablation_df.target == t)]["rmse"].iloc[0]
            for t in TARGET_COLS]
    ax.bar(xpos + (j - 0.5) * width, vals, width,
          label=arm_labels[arm], color=arm_colors[arm], edgecolor="white")
ax.set_xticks(xpos)
ax.set_xticklabels([TGT_LABELS[t] for t in TARGET_COLS], fontsize=_FS_TICK)
ax.set_ylabel("RMSE on lab_holdout (native units)", fontsize=_FS_AX)
ax.set_title(
    f"Does Synthetic Augmentation Help? ({best_name}, "
    f"n={len(holdout_idx)} independent real batches)",
    fontsize=_FS_TITLE, fontweight="bold", pad=14
)
ax.legend(fontsize=_FS_LABEL)
ax.tick_params(labelsize=_FS_TICK)
plt.tight_layout()
savefig(fig, "ablation_real_vs_synthetic")
print("Saved: ablation_real_vs_synthetic.pdf / .png")

# ── Final (production) model: real + synthetic (see [G]) ─────────────────────
final_model = model_real_plus_synth
y_pred = final_model.predict(X_test)

mae = np.mean(np.abs(y_test - y_pred), axis=0)
r2  = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(3)]

print("\nTest-set performance (synthetic-test + lab_holdout combined; "
      "calibration rows excluded — see [A]):")
for t, m, r in zip(TARGET_COLS, mae, r2):
    print(f"  {t:<25s}  MAE={m:.4f}  R²={r:.4f}")

# ── holdout_mask_test: identify lab_holdout rows within the combined test set ──
holdout_idx_set   = set(holdout_idx)
holdout_mask_test = np.array([idx in holdout_idx_set for idx in test_idx])

if holdout_mask_test.sum() > 0:
    y_tl = y_test[holdout_mask_test]
    y_pl = y_pred[holdout_mask_test]
    print(f"\nlab_holdout-only metrics (n={int(holdout_mask_test.sum())}, "
          "fully independent, R² not reported — see Note in module "
          "docstring):")
    for i, t in enumerate(TARGET_COLS):
        mae_e  = float(np.mean(np.abs(y_tl[:, i] - y_pl[:, i])))
        rmse_e = float(np.sqrt(np.mean((y_tl[:, i] - y_pl[:, i]) ** 2)))
        print(f"  {t:<25s}  MAE={mae_e:.4f} {units[t]}"
              f"  RMSE={rmse_e:.4f} {units[t]}")

# ── Save CSV artefacts ────────────────────────────────────────────────────────
pd.DataFrame({
    **{f"y_true_{t}": y_test[:, i] for i, t in enumerate(TARGET_COLS)},
    **{f"y_pred_{t}": y_pred[:, i] for i, t in enumerate(TARGET_COLS)},
}).to_csv(DATADIR / "parity_data.csv", index=False)

# ── Feature importance ────────────────────────────────────────────────────────
# Mean feature importance averaged across all three target properties.
# Composition variables only — process variables excluded (see Note [E]).
# Computed from the production model (real + synthetic training pool).
try:
    reg = final_model.named_steps["reg"]
    imp = (reg.feature_importances_
           if hasattr(reg, "feature_importances_")
           else np.mean([e.feature_importances_ for e in reg.estimators_], axis=0))

    # Normalise to sum = 1 so x-axis is always in [0, 1]
    imp = imp / imp.sum()
    fi = (pd.DataFrame({"feature": feature_cols, "importance": imp})
          .sort_values("importance", ascending=False))
    fi["label"] = fi["feature"].map(lambda c: MAT_SHORT.get(c, c))
    fi.to_csv(DATADIR / "feature_importances.csv", index=False)

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#E53935" if "_wtpct" in f else "#90A4AE" for f in fi["feature"]]
    sns.barplot(x="importance", y="label", data=fi,
                palette=palette, ax=ax, edgecolor="white")
    ax.legend(handles=[
        Patch(color="#E53935", label="Composition variable"),
    ], fontsize=_FS_LABEL, loc="lower right")
    ax.set_title("Mean Feature Importance Across All Target Properties",
                 pad=14, fontsize=_FS_TITLE, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=_FS_AX)
    ax.set_ylabel("", fontsize=_FS_AX)
    ax.tick_params(labelsize=_FS_TICK)
    plt.tight_layout()
    savefig(fig, "feature_importances")
    print("Saved: feature_importances.pdf / .png")
except Exception as e:
    print(f"Feature importance: {e}")

# ── Parity plots ──────────────────────────────────────────────────────────────
# R² reported for the full test set (synthetic-test + lab_holdout combined).
# For the lab_holdout subset specifically, RMSE in original units is
# reported instead of R², which is unreliable at small sample sizes with a
# distribution shift relative to the synthetic training set.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, t in enumerate(TARGET_COLS):
    ax   = axes[i]
    smsk = ~holdout_mask_test

    r2_combined = r2_score(y_test[:, i], y_pred[:, i])

    if holdout_mask_test.sum() > 0:
        rmse_exp = float(np.sqrt(np.mean(
            (y_test[holdout_mask_test, i] - y_pred[holdout_mask_test, i]) ** 2
        )))
        exp_label = (f"RMSE (holdout, n={int(holdout_mask_test.sum())}) "
                    f"= {rmse_exp:.3f} {units[t]}")
    else:
        exp_label = ""

    ax.scatter(y_test[smsk, i], y_pred[smsk, i],
               alpha=0.55, s=40, color="teal", label="Synthetic (test)")
    if holdout_mask_test.sum() > 0:
        ax.scatter(y_test[holdout_mask_test, i], y_pred[holdout_mask_test, i],
                   alpha=0.9, s=130, color="crimson", marker="*",
                   label="lab_holdout (independent)", zorder=5)
    lo = min(y_test[:, i].min(), y_pred[:, i].min())
    hi = max(y_test[:, i].max(), y_pred[:, i].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_xlabel("Measured", fontsize=_FS_AX)
    ax.set_ylabel("Predicted", fontsize=_FS_AX)
    ax.tick_params(labelsize=_FS_TICK)
    ax.set_title(
        f"{TGT_LABELS[t]}\n"
        f"R² (all) = {r2_combined:.4f}  |  {exp_label}",
        fontsize=_FS_AX
    )
    ax.legend(fontsize=_FS_LABEL)

plt.suptitle("Predicted vs. Measured Values: Forward Model Parity Plots",
             fontsize=_FS_TITLE, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
savefig(fig, "parity_plots")
print("Saved: parity_plots.pdf / .png")

# ── PDP Section ───────────────────────────────────────────────────────────────
# Partial dependence plots show how each target property responds to
# variation in a single composition variable, marginalised over the
# PRODUCTION model's actual training pool (synthetic train split + real
# calibration rows — see [C], [G]). X-values are in original wt% units
# (not standardised), making the plots directly interpretable by
# practitioners. See Note [C] for the rationale for manual PDP computation.
print(f"\nComputing PDPs on original-scale composition variables "
      f"(marginalised over production training pool, "
      f"n = {len(X_train_aug)}) …")

def compute_pdp(pipeline, X_orig: pd.DataFrame,
                feature: str, n_grid: int = 60):
    grid = np.linspace(X_orig[feature].min(), X_orig[feature].max(), n_grid)
    means = []
    for val in grid:
        Xt = X_orig.copy()
        Xt[feature] = val
        means.append(pipeline.predict(Xt).mean(axis=0))
    return grid, np.array(means)

_FS_PDP_SUPTITLE = 26
_FS_PDP_TITLE    = 24
_FS_PDP_LABEL    = 20
_FS_PDP_TICK     = 18

colors = ["#1565C0", "#D32F2F", "#388E3C"]

for t_idx, tname in enumerate(TARGET_COLS):
    n_comp = len(comp_cols)
    ncols  = 4
    nrows  = math.ceil(n_comp / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7.2, nrows * 6.4),
                             dpi=250)
    axes = axes.flatten()

    for ax_idx, feat in enumerate(comp_cols):
        ax = axes[ax_idx]
        grid, means = compute_pdp(final_model, X_train_aug, feat)

        ax.plot(grid, means[:, t_idx], color=colors[t_idx], linewidth=3.5)

        short = MAT_SHORT.get(feat, feat.replace("_wtpct", ""))
        ax.set_xlabel(f"{short} (wt%)", fontsize=_FS_PDP_LABEL, labelpad=10)
        ax.set_ylabel(TGT_LABELS[tname],  fontsize=_FS_PDP_LABEL, labelpad=10)
        ax.set_title(short, fontsize=_FS_PDP_TITLE, fontweight="bold", pad=15)
        ax.tick_params(labelsize=_FS_PDP_TICK, width=1.5, length=8)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for ax in axes[n_comp:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Partial Dependence of {TGT_LABELS[tname]}\n"
        "on Composition Variables (wt%)\n"
        f"(marginalised over production training pool, n = {len(X_train_aug)})",
        fontsize=_FS_PDP_SUPTITLE, fontweight="bold", y=0.96
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    savefig(fig, f"PDP_{tname}")
    print(f"  Saved: PDP_{tname}.pdf / .png")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(final_model, MODELDIR / "forward_model.joblib")
joblib.dump(model_real_only, MODELDIR / "forward_model_real_only.joblib")
with open(MODELDIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\nAll figures → plots/")
print("Production model → models/forward_model.joblib "
      "(real + synthetic; see ablation_real_vs_synthetic.csv/.png)")
print("Reference model  → models/forward_model_real_only.joblib "
      "(real batches only, for comparison)")
print("Done.")
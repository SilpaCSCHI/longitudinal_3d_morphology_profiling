#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_final_BOW_9_vs_22_K8.py

FINAL Bag-of-Words comparison using:
    A) correlation-pruned 9-feature panel
    B) full 22-feature panel

Fixed primary vocabulary size:
    K = 8

For EACH feature set the script performs:
------------------------------------------------
1. Load all four object-level BF feature tables
2. Robust-standardize each dataset x timepoint relative to its matched control
3. Balanced MiniBatchKMeans training
4. Convert each dataset/dose/timepoint to an 8-state morphology histogram
5. Calculate Jensen-Shannon (JS) distance to matched same-time control
6. Baseline-correct morphology divergence:

       delta_JS(d,t) = JS(d,t) - JS(d,0)

7. At 96 h correlate baseline-corrected JS with:

       CTG effect = 100 - CTG viability

   across 20 treated condition-level samples

8. Calculate within-dataset correlations
9. Leave-one-treated-condition-out sensitivity (20 folds)
10. Leave-one-dose-position-out sensitivity (5 folds):
       exclude the lowest treated dose from every dataset,
       then second-lowest, ..., highest treated dose
11. Save longitudinal delta-JS trajectories
12. Save endpoint tables and summary plots

IMPORTANT
---------
- K is fixed at 8.
- CTG is NOT used to fit the clusters.
- Dose conditions are condition-level observations, not biological replicates.
- "Leave-one-dose-position-out" is used instead of excluding an absolute numeric
  concentration, because the four datasets use different compounds/units/ranges.

Primary comparison:
    Does the 9-feature pruned representation preserve the morphology-CTG
    association observed with the full 22-feature representation?
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import MiniBatchKMeans

# ============================================================================
# USER CONFIG
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MORPH_DATA_DIR", REPO_ROOT / "data" / "example"))
OUT_ROOT = Path(os.environ.get("MORPH_OUT_DIR", REPO_ROOT / "outputs" / "feature_sensitivity"))
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CTG_XLSX = Path(os.environ.get("MORPH_CTG_FILE", DATA_DIR / "CTG_4celllines.xlsx"))

FILES = {
    "BR4": DATA_DIR / "cell_features_BR4.csv",
    "A-431": DATA_DIR / "cell_features_SCC4.csv",
    "SMU030": DATA_DIR / "cell_features_SMU030.csv",
    "WMD033": DATA_DIR / "cell_features_WMD033.csv",
}

K = 8
BASELINE_H = 0
ENDPOINT_H = 96

MAX_CELLS_PER_CONDITION_FOR_KMEANS = 2500

N_PERM = int(os.environ.get("MORPH_N_PERM", "10000"))
SEED = 42

# ---------------------------------------------------------------------------
# Final correlation-pruned 9-feature panel
# ---------------------------------------------------------------------------
FEATURES_9 = [
    "volume",
    "extent",
    "elongation",
    "mean_intensity",
    "std_intensity",
    "intensity_cv",
    "ring_contrast",
    "mean_edge",
    "log_s3p0_mean",
]

# ---------------------------------------------------------------------------
# Full 22 quantitative BF-derived features
# ---------------------------------------------------------------------------
FEATURES_22 = [
    "volume",
    "extent",
    "elongation",
    "mean_intensity",
    "std_intensity",
    "intensity_cv",
    "intensity_p10",
    "intensity_p90",
    "intensity_range",
    "mean_edge",
    "edge_std",
    "ring_mean_intensity",
    "ring_contrast",
    "ring_contrast_z",
    "boundary_vox_frac",
    "boundary_edge_mean",
    "log_s1p0_mean",
    "log_s1p0_std",
    "log_s2p0_mean",
    "log_s2p0_std",
    "log_s3p0_mean",
    "log_s3p0_std",
]

FEATURE_SETS = {
    "pruned_9_features": FEATURES_9,
    "full_22_features": FEATURES_22,
}

# Dose labels in object-level feature CSVs
DOSE_MAP = {
    "BR4": {
        "Small_Untreated": 0.0,
        "Untreated": 0.0,
        "0.5uM": 0.5,
        "0.5µM": 0.5,
        "1uM": 1.0,
        "2uM": 2.0,
        "4uM": 4.0,
        "Small_8µM": 8.0,
        "8uM": 8.0,
        "8µM": 8.0,
    },
    "A-431": {
        "Untreated": 0.0,
        "1uM": 1.0,
        "2uM": 2.0,
        "5uM": 5.0,
        "10uM": 10.0,
        "20uM": 20.0,
    },
    "SMU030": {
        "DMSO_Control": 0.0,
        "Dose1": 1.565,
        "Dose2": 3.125,
        "Dose3": 6.25,
        "Dose4": 12.5,
        "Dose5": 25.0,
    },
    "WMD033": {
        "DMSO_Control": 0.0,
        "Dose1": 1.565,
        "Dose2": 3.125,
        "Dose3": 6.25,
        "Dose4": 12.5,
        "Dose5": 25.0,
    },
}


# ============================================================================
# CTG
# ============================================================================

def load_ctg_workbook(path: Path) -> pd.DataFrame:
    specs = {
        "BR4": dict(
            sheet="BR4",
            rep_rows=slice(20, 24),
            doses=[0, 0.5, 1, 2, 4, 8],
        ),
        "A-431": dict(
            sheet="SCC-4",
            rep_rows=slice(20, 24),
            doses=[0, 1, 2, 5, 10, 20],
        ),
        "SMU030": dict(
            sheet="SMU",
            rep_rows=slice(15, 18),
            doses=[0, 1.565, 3.125, 6.25, 12.5, 25],
        ),
        "WMD033": dict(
            sheet="WMD",
            rep_rows=slice(15, 18),
            doses=[0, 1.565, 3.125, 6.25, 12.5, 25],
        ),
    }

    rows = []

    for dataset, sp in specs.items():
        raw = pd.read_excel(
            path,
            sheet_name=sp["sheet"],
            header=None
        )

        vals = (
            raw.iloc[sp["rep_rows"], 1:7]
            .apply(pd.to_numeric, errors="coerce")
        )

        for rep_i, (_, row) in enumerate(
            vals.iterrows(),
            start=1
        ):
            for j, dose in enumerate(sp["doses"]):
                v = row.iloc[j]

                if pd.notna(v):
                    rows.append({
                        "dataset": dataset,
                        "dose_value": float(dose),
                        "ctg_replicate": int(rep_i),
                        "ctg_viability_pct": float(v),
                    })

    rep = pd.DataFrame(rows)

    mean = (
        rep.groupby(
            ["dataset", "dose_value"],
            as_index=False
        )
        .agg(
            ctg_viability_pct=("ctg_viability_pct", "mean"),
            ctg_sd=("ctg_viability_pct", "std"),
            ctg_n_technical=("ctg_viability_pct", "size"),
        )
    )

    mean["ctg_effect_pct"] = (
        100.0 - mean["ctg_viability_pct"]
    )

    return mean


# ============================================================================
# FEATURE TABLE LOADING
# ============================================================================

def load_one_dataset(dataset, path, features):
    needed = set(features + ["dose", "time_h"])

    df = pd.read_csv(
        path,
        usecols=lambda c: c in needed,
        low_memory=False,
    )

    missing = [
        f for f in features
        if f not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset}: missing features: {missing}"
        )

    df["dose"] = (
        df["dose"]
        .astype(str)
        .str.strip()
    )

    df["time_h"] = pd.to_numeric(
        df["time_h"],
        errors="coerce"
    )

    df = df[
        df["time_h"].notna()
        & df["dose"].ne("")
        & df["dose"].ne("nan")
    ].copy()

    mapper = DOSE_MAP[dataset]

    unknown = sorted(
        set(df["dose"]) - set(mapper)
    )

    if unknown:
        raise ValueError(
            f"{dataset}: unmapped dose labels: {unknown}"
        )

    df["dataset"] = dataset
    df["dose_value"] = (
        df["dose"]
        .map(mapper)
        .astype(float)
    )

    for f in features:
        df[f] = pd.to_numeric(
            df[f],
            errors="coerce"
        )

    df = df.dropna(
        subset=features
    )

    return df


# ============================================================================
# CONTROL-RELATIVE ROBUST STANDARDIZATION
# ============================================================================

def robust_scale_to_matched_control(df, features):
    frames = []

    for (dataset, time_h), sub in df.groupby(
        ["dataset", "time_h"],
        sort=False
    ):
        ctrl = sub[
            np.isclose(
                sub["dose_value"],
                0.0
            )
        ]

        if ctrl.empty:
            raise ValueError(
                f"No control for {dataset} at {time_h} h"
            )

        med = ctrl[features].median(axis=0)
        q25 = ctrl[features].quantile(0.25)
        q75 = ctrl[features].quantile(0.75)
        iqr = q75 - q25

        iqr = iqr.where(
            iqr.abs() > 1e-12,
            1.0
        )

        z = (
            sub[features] - med
        ) / iqr

        z = z.clip(
            -10.0,
            10.0
        )

        rec = sub[
            ["dataset", "dose_value", "time_h"]
        ].copy()

        for f in features:
            rec[f] = z[f].to_numpy(float)

        frames.append(rec)

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================================
# BALANCED K-MEANS TRAINING
# ============================================================================

def balanced_training_sample(zdf):
    rng = np.random.default_rng(SEED)
    chunks = []

    for _, sub in zdf.groupby(
        ["dataset", "dose_value", "time_h"],
        sort=False
    ):
        n = min(
            len(sub),
            MAX_CELLS_PER_CONDITION_FOR_KMEANS
        )

        if len(sub) > n:
            idx = rng.choice(
                len(sub),
                size=n,
                replace=False
            )
            chunks.append(
                sub.iloc[idx]
            )
        else:
            chunks.append(sub)

    return pd.concat(
        chunks,
        ignore_index=True
    )


# ============================================================================
# HISTOGRAMS / JS
# ============================================================================

def build_histograms(zdf, model, features):
    labels = model.predict(
        zdf[features].to_numpy(float)
    )

    work = zdf[
        ["dataset", "dose_value", "time_h"]
    ].copy()

    work["state"] = labels.astype(int)

    rows = []

    for (dataset, dose, time_h), sub in work.groupby(
        ["dataset", "dose_value", "time_h"],
        sort=True
    ):
        counts = (
            sub["state"]
            .value_counts()
            .reindex(
                range(K),
                fill_value=0
            )
            .to_numpy(float)
        )

        p = counts / counts.sum()

        rec = {
            "dataset": dataset,
            "dose_value": float(dose),
            "time_h": float(time_h),
            "n_objects": int(counts.sum()),
        }

        for k, v in enumerate(p):
            rec[f"p{k}"] = float(v)

        rows.append(rec)

    return pd.DataFrame(rows)


def calculate_js_to_control(hist):
    pcols = [
        f"p{k}"
        for k in range(K)
    ]

    rows = []

    for (dataset, time_h), sub in hist.groupby(
        ["dataset", "time_h"],
        sort=False
    ):
        ctrl = sub[
            np.isclose(
                sub["dose_value"],
                0.0
            )
        ]

        if len(ctrl) != 1:
            raise ValueError(
                f"{dataset} {time_h}h: expected 1 control histogram"
            )

        q = (
            ctrl.iloc[0][pcols]
            .to_numpy(float)
        )

        for _, r in sub.iterrows():
            p = (
                r[pcols]
                .to_numpy(float)
            )

            js = float(
                jensenshannon(
                    p,
                    q,
                    base=2.0
                )
            )

            rows.append({
                "dataset": dataset,
                "dose_value": float(r["dose_value"]),
                "time_h": float(time_h),
                "JS_distance_from_control": js,
            })

    return pd.DataFrame(rows)


def add_baseline_correction(div):
    baseline = (
        div[
            np.isclose(
                div["time_h"],
                BASELINE_H
            )
        ][
            [
                "dataset",
                "dose_value",
                "JS_distance_from_control",
            ]
        ]
        .rename(
            columns={
                "JS_distance_from_control":
                "JS_distance_at_0h"
            }
        )
    )

    out = div.merge(
        baseline,
        on=["dataset", "dose_value"],
        how="left"
    )

    if out["JS_distance_at_0h"].isna().any():
        raise ValueError(
            "Missing 0 h baseline JS value."
        )

    out["delta_JS_from_baseline"] = (
        out["JS_distance_from_control"]
        - out["JS_distance_at_0h"]
    )

    return out


# ============================================================================
# STATISTICS
# ============================================================================

def within_dataset_permutation_p(
    x,
    y,
    groups,
    n_perm=N_PERM,
    seed=SEED
):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)

    ok = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[ok]
    y = y[ok]
    groups = groups[ok]

    if len(x) < 4:
        return np.nan

    obs = spearmanr(
        x,
        y
    ).statistic

    if not np.isfinite(obs):
        return np.nan

    rng = np.random.default_rng(seed)

    extreme = 0

    for _ in range(n_perm):
        yp = y.copy()

        for g in np.unique(groups):
            idx = np.flatnonzero(
                groups == g
            )

            yp[idx] = rng.permutation(
                yp[idx]
            )

        s = spearmanr(
            x,
            yp
        ).statistic

        if (
            np.isfinite(s)
            and abs(s) >= abs(obs)
        ):
            extreme += 1

    return (
        extreme + 1
    ) / (
        n_perm + 1
    )


def calculate_pooled_stats(endpoint):
    treated = endpoint[
        endpoint["dose_value"] > 0
    ].copy()

    x = treated[
        "delta_JS_from_baseline"
    ].to_numpy(float)

    y = treated[
        "ctg_effect_pct"
    ].to_numpy(float)

    groups = treated[
        "dataset"
    ].to_numpy(str)

    sr = spearmanr(
        x,
        y
    )

    pr = pearsonr(
        x,
        y
    )

    pp = within_dataset_permutation_p(
        x,
        y,
        groups
    )

    return {
        "n_condition_samples": len(treated),
        "spearman_rho": float(sr.statistic),
        "spearman_naive_p": float(sr.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_naive_p": float(pr.pvalue),
        "within_dataset_permutation_p": float(pp),
    }


def calculate_within_dataset(endpoint):
    treated = endpoint[
        endpoint["dose_value"] > 0
    ].copy()

    rows = []

    for dataset, sub in treated.groupby(
        "dataset"
    ):
        x = sub[
            "delta_JS_from_baseline"
        ].to_numpy(float)

        y = sub[
            "ctg_effect_pct"
        ].to_numpy(float)

        sr = spearmanr(
            x,
            y
        )

        pr = pearsonr(
            x,
            y
        )

        rows.append({
            "dataset": dataset,
            "n_treated_doses": len(sub),
            "spearman_rho": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
        })

    return pd.DataFrame(rows)


# ============================================================================
# LEAVE-ONE-TREATED-CONDITION-OUT
# ============================================================================

def leave_one_condition_out(endpoint):
    treated = (
        endpoint[
            endpoint["dose_value"] > 0
        ]
        .copy()
        .reset_index(drop=True)
    )

    rows = []

    for i, held in treated.iterrows():
        train = treated.drop(
            index=i
        )

        sr = spearmanr(
            train["delta_JS_from_baseline"],
            train["ctg_effect_pct"]
        )

        pr = pearsonr(
            train["delta_JS_from_baseline"],
            train["ctg_effect_pct"]
        )

        rows.append({
            "held_out_dataset": held["dataset"],
            "held_out_dose_value": held["dose_value"],
            "n_retained": len(train),
            "spearman_rho": float(sr.statistic),
            "pearson_r": float(pr.statistic),
        })

    return pd.DataFrame(rows)


# ============================================================================
# LEAVE-ONE-DOSE-POSITION-OUT
# ============================================================================

def assign_dose_rank(endpoint):
    """
    Rank treated doses within each dataset:
       1 = lowest treated concentration
       5 = highest treated concentration
    """
    out = endpoint.copy()

    out["treated_dose_rank"] = np.nan

    for dataset, sub in out.groupby("dataset"):
        treated = (
            sub[sub["dose_value"] > 0]
            .sort_values("dose_value")
        )

        for rank, idx in enumerate(
            treated.index,
            start=1
        ):
            out.loc[
                idx,
                "treated_dose_rank"
            ] = rank

    return out


def leave_one_dose_position_out(endpoint):
    e = assign_dose_rank(
        endpoint
    )

    treated = e[
        e["dose_value"] > 0
    ].copy()

    rows = []

    for held_rank in range(1, 6):
        retained = treated[
            treated["treated_dose_rank"]
            != held_rank
        ].copy()

        held = treated[
            treated["treated_dose_rank"]
            == held_rank
        ].copy()

        sr = spearmanr(
            retained["delta_JS_from_baseline"],
            retained["ctg_effect_pct"]
        )

        pr = pearsonr(
            retained["delta_JS_from_baseline"],
            retained["ctg_effect_pct"]
        )

        pp = within_dataset_permutation_p(
            retained["delta_JS_from_baseline"],
            retained["ctg_effect_pct"],
            retained["dataset"]
        )

        rows.append({
            "held_out_treated_dose_rank": held_rank,
            "description": (
                "lowest treated dose"
                if held_rank == 1
                else "highest treated dose"
                if held_rank == 5
                else f"treated dose rank {held_rank}"
            ),
            "n_removed": len(held),
            "n_retained": len(retained),
            "spearman_rho": float(sr.statistic),
            "pearson_r": float(pr.statistic),
            "within_dataset_permutation_p": float(pp),
        })

    return pd.DataFrame(rows)


# ============================================================================
# PLOTS
# ============================================================================

def save_endpoint_scatter(endpoint, out_dir, title):
    treated = endpoint[
        endpoint["dose_value"] > 0
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for dataset, sub in treated.groupby(
        "dataset"
    ):
        ax.scatter(
            sub["delta_JS_from_baseline"],
            sub["ctg_effect_pct"],
            s=60,
            label=dataset
        )

    ax.set_xlabel(
        "Baseline-corrected JS divergence"
    )

    ax.set_ylabel(
        "CTG treatment effect (%)"
    )

    ax.set_title(title)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir /
        "deltaJS_vs_CTG_effect_96h.png",
        dpi=300
    )

    plt.close(fig)


def save_longitudinal_plots(corrected, out_dir, tag):
    for dataset, sub in corrected.groupby(
        "dataset"
    ):
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for dose, dsub in sub.groupby(
            "dose_value"
        ):
            if np.isclose(
                dose,
                0.0
            ):
                continue

            dsub = dsub.sort_values(
                "time_h"
            )

            ax.plot(
                dsub["time_h"],
                dsub["delta_JS_from_baseline"],
                marker="o",
                label=str(dose)
            )

        ax.axhline(
            0,
            linewidth=1
        )

        ax.set_xlabel(
            "Time (h)"
        )

        ax.set_ylabel(
            "Baseline-corrected JS divergence"
        )

        ax.set_title(
            f"{dataset}: {tag}"
        )

        ax.legend(
            title="Dose"
        )

        fig.tight_layout()

        fig.savefig(
            out_dir /
            f"{dataset}_deltaJS_longitudinal.png",
            dpi=300
        )

        plt.close(fig)


def save_leave_one_dose_plot(loo_dose, out_dir):
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        loo_dose[
            "held_out_treated_dose_rank"
        ],
        loo_dose[
            "spearman_rho"
        ],
        marker="o"
    )

    ax.set_xlabel(
        "Held-out treated dose rank"
    )

    ax.set_ylabel(
        "Spearman rho after exclusion"
    )

    ax.set_xticks(
        [1, 2, 3, 4, 5]
    )

    ax.set_title(
        "Leave-one-dose-position-out sensitivity"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir /
        "leave_one_dose_position_out_sensitivity.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================================
# ONE FEATURE-SET RUN
# ============================================================================

def run_feature_set(
    tag,
    features,
    ctg
):
    out_dir = (
        OUT_ROOT /
        tag
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n============================================================"
    )
    print(
        f"RUNNING: {tag}"
    )
    print(
        f"Number of features: {len(features)}"
    )
    print(
        "============================================================"
    )

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------
    cells = []

    for dataset, path in FILES.items():
        d = load_one_dataset(
            dataset,
            path,
            features
        )

        print(
            f"{dataset}: {len(d):,} objects"
        )

        cells.append(d)

    cells = pd.concat(
        cells,
        ignore_index=True
    )

    print(
        f"Total usable objects: {len(cells):,}"
    )

    # ------------------------------------------------------------------------
    # Scale
    # ------------------------------------------------------------------------
    zdf = robust_scale_to_matched_control(
        cells,
        features
    )

    # ------------------------------------------------------------------------
    # Balanced training
    # ------------------------------------------------------------------------
    train = balanced_training_sample(
        zdf
    )

    print(
        f"Balanced K-means training: {len(train):,}"
    )

    # ------------------------------------------------------------------------
    # K-means K=8
    # ------------------------------------------------------------------------
    model = MiniBatchKMeans(
        n_clusters=K,
        random_state=SEED,
        batch_size=4096,
        n_init=20,
    )

    model.fit(
        train[
            features
        ].to_numpy(float)
    )

    # ------------------------------------------------------------------------
    # Histograms and JS
    # ------------------------------------------------------------------------
    hist = build_histograms(
        zdf,
        model,
        features
    )

    div = calculate_js_to_control(
        hist
    )

    corrected = add_baseline_correction(
        div
    )

    # ------------------------------------------------------------------------
    # Endpoint merge
    # ------------------------------------------------------------------------
    endpoint = corrected[
        np.isclose(
            corrected["time_h"],
            ENDPOINT_H
        )
    ].merge(
        ctg,
        on=[
            "dataset",
            "dose_value"
        ],
        how="inner"
    )

    # ------------------------------------------------------------------------
    # Main stats
    # ------------------------------------------------------------------------
    pooled = calculate_pooled_stats(
        endpoint
    )

    within = calculate_within_dataset(
        endpoint
    )

    loo_condition = leave_one_condition_out(
        endpoint
    )

    loo_dose = leave_one_dose_position_out(
        endpoint
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------
    hist.to_csv(
        out_dir /
        "phenotype_histograms_all_timepoints.csv",
        index=False
    )

    corrected.to_csv(
        out_dir /
        "morphology_divergence_longitudinal_baseline_corrected.csv",
        index=False
    )

    endpoint.to_csv(
        out_dir /
        "morphology_divergence_vs_CTG_96h.csv",
        index=False
    )

    pd.DataFrame(
        [pooled]
    ).to_csv(
        out_dir /
        "pooled_96h_statistics.csv",
        index=False
    )

    within.to_csv(
        out_dir /
        "within_dataset_correlations.csv",
        index=False
    )

    loo_condition.to_csv(
        out_dir /
        "leave_one_condition_out_sensitivity.csv",
        index=False
    )

    loo_dose.to_csv(
        out_dir /
        "leave_one_dose_position_out_sensitivity.csv",
        index=False
    )

    pd.DataFrame({
        "feature": features
    }).to_csv(
        out_dir /
        "features_used.csv",
        index=False
    )

    # ------------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------------
    save_endpoint_scatter(
        endpoint,
        out_dir,
        f"{tag}: baseline-corrected JS vs CTG effect"
    )

    save_longitudinal_plots(
        corrected,
        out_dir,
        tag
    )

    save_leave_one_dose_plot(
        loo_dose,
        out_dir
    )

    # ------------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------------
    print(
        "\nPOOLED 96 h:"
    )

    print(
        f"Spearman rho = {pooled['spearman_rho']:.6f}"
    )

    print(
        f"Pearson r = {pooled['pearson_r']:.6f}"
    )

    print(
        f"Permutation p = {pooled['within_dataset_permutation_p']:.6f}"
    )

    print(
        "\nWITHIN-DATASET:"
    )

    print(
        within.to_string(
            index=False
        )
    )

    print(
        "\nLEAVE-ONE-DOSE-POSITION-OUT:"
    )

    print(
        loo_dose.to_string(
            index=False
        )
    )

    print(
        "\nLEAVE-ONE-CONDITION-OUT RANGE:"
    )

    print(
        f"Spearman min = "
        f"{loo_condition['spearman_rho'].min():.6f}"
    )

    print(
        f"Spearman max = "
        f"{loo_condition['spearman_rho'].max():.6f}"
    )

    return {
        "feature_set": tag,
        "n_features": len(features),
        **pooled,
        "within_rho_min":
            float(within["spearman_rho"].min()),
        "within_rho_max":
            float(within["spearman_rho"].max()),
        "leave_one_condition_rho_min":
            float(loo_condition["spearman_rho"].min()),
        "leave_one_condition_rho_max":
            float(loo_condition["spearman_rho"].max()),
        "leave_one_dose_rank_rho_min":
            float(loo_dose["spearman_rho"].min()),
        "leave_one_dose_rank_rho_max":
            float(loo_dose["spearman_rho"].max()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    ctg = load_ctg_workbook(
        CTG_XLSX
    )

    summaries = []

    for tag, features in FEATURE_SETS.items():

        summary = run_feature_set(
            tag,
            features,
            ctg
        )

        summaries.append(
            summary
        )

    comparison = pd.DataFrame(
        summaries
    )

    comparison.to_csv(
        OUT_ROOT /
        "FINAL_9_vs_22_comparison.csv",
        index=False
    )

    # ------------------------------------------------------------------------
    # Comparison bar plot
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        comparison["feature_set"],
        comparison["spearman_rho"]
    )

    ax.set_ylabel(
        "Spearman rho vs CTG effect"
    )

    ax.set_title(
        "K=8 baseline-corrected JS: 9 vs 22 features"
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    fig.tight_layout()

    fig.savefig(
        OUT_ROOT /
        "FINAL_9_vs_22_spearman_comparison.png",
        dpi=300
    )

    plt.close(fig)

    print(
        "\n============================================================"
    )
    print(
        "FINAL 9 vs 22 FEATURE COMPARISON"
    )
    print(
        "============================================================"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\nAll outputs saved to:\n"
        f"{OUT_ROOT}"
    )


if __name__ == "__main__":
    main()

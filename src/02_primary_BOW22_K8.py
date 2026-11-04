#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10_BOW22_K8_no_clipping_no_sampling.py

Sensitivity analysis for the 22-feature Bag-of-Words pipeline.

Changes from the previous primary 22-feature pipeline:
-------------------------------------------------------
1. NO clipping of robust-scaled feature values.
2. NO condition-capped / balanced sampling for K-means training.
3. ALL usable segmented objects are used to fit the K=8 MiniBatchKMeans model.

Everything else is kept the same:
---------------------------------
- 22 BF-derived object features
- matched-control robust scaling:
      z = (x - median_control) / IQR_control
- K = 8
- condition histograms
- Jensen-Shannon distance to same-time control
- baseline correction:
      delta_JS(d,t) = JS(d,t) - JS(d,0)
- pooled 96 h association with CTG treatment effect
- within-dataset correlations
- leave-one-condition-out sensitivity
- leave-one-dose-position-out sensitivity
- leave-one-dataset-out sensitivity

Purpose
-------
Test whether the strong morphology–CTG association depends on:
    a) clipping robust z-scores to [-10, 10], or
    b) capping the number of training objects per condition.

This script removes BOTH safeguards simultaneously.
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
OUT_DIR = Path(os.environ.get("MORPH_OUT_DIR", REPO_ROOT / "outputs" / "primary"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
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

N_PERM = int(os.environ.get("MORPH_N_PERM", "10000"))
SEED = 42

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

def load_one_dataset(dataset, path):
    needed = set(FEATURES_22 + ["dose", "time_h"])

    df = pd.read_csv(
        path,
        usecols=lambda c: c in needed,
        low_memory=False,
    )

    missing = [
        f for f in FEATURES_22
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

    for f in FEATURES_22:
        df[f] = pd.to_numeric(
            df[f],
            errors="coerce"
        )

    df = df.dropna(
        subset=FEATURES_22
    )

    return df


# ============================================================================
# ROBUST STANDARDIZATION — NO CLIPPING
# ============================================================================

def robust_scale_to_matched_control_no_clipping(df):
    """
    Scale every feature within each dataset x timepoint relative to its matched
    untreated control:

        z = (x - median_control) / IQR_control

    IMPORTANT:
    ---------
    Unlike the previous pipeline, values are NOT clipped to [-10, 10].
    """

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

        med = ctrl[FEATURES_22].median(axis=0)
        q25 = ctrl[FEATURES_22].quantile(0.25)
        q75 = ctrl[FEATURES_22].quantile(0.75)
        iqr = q75 - q25

        # Avoid division by zero.
        # This does NOT clip the transformed values.
        iqr = iqr.where(
            iqr.abs() > 1e-12,
            1.0
        )

        z = (
            sub[FEATURES_22] - med
        ) / iqr

        rec = sub[
            ["dataset", "dose_value", "time_h"]
        ].copy()

        for f in FEATURES_22:
            rec[f] = z[f].to_numpy(float)

        frames.append(rec)

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================================
# HISTOGRAMS / JS
# ============================================================================

def build_histograms(zdf, model):
    labels = model.predict(
        zdf[FEATURES_22].to_numpy(float)
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


def pooled_stats(endpoint):
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

    sr = spearmanr(x, y)
    pr = pearsonr(x, y)

    pp = within_dataset_permutation_p(
        x, y, groups
    )

    return {
        "n_condition_samples": len(treated),
        "spearman_rho": float(sr.statistic),
        "spearman_naive_p": float(sr.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_naive_p": float(pr.pvalue),
        "within_dataset_permutation_p": float(pp),
    }


def within_dataset_stats(endpoint):
    treated = endpoint[
        endpoint["dose_value"] > 0
    ].copy()

    rows = []

    for dataset, sub in treated.groupby(
        "dataset"
    ):
        sr = spearmanr(
            sub["delta_JS_from_baseline"],
            sub["ctg_effect_pct"]
        )

        pr = pearsonr(
            sub["delta_JS_from_baseline"],
            sub["ctg_effect_pct"]
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
# LEAVE-ONE-CONDITION-OUT
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
        retained = treated.drop(
            index=i
        )

        sr = spearmanr(
            retained["delta_JS_from_baseline"],
            retained["ctg_effect_pct"]
        )

        pr = pearsonr(
            retained["delta_JS_from_baseline"],
            retained["ctg_effect_pct"]
        )

        rows.append({
            "held_out_dataset": held["dataset"],
            "held_out_dose_value": held["dose_value"],
            "n_retained": len(retained),
            "spearman_rho": float(sr.statistic),
            "pearson_r": float(pr.statistic),
        })

    return pd.DataFrame(rows)


# ============================================================================
# LEAVE-ONE-DOSE-POSITION-OUT
# ============================================================================

def assign_dose_rank(endpoint):
    out = endpoint.copy()
    out["treated_dose_rank"] = np.nan

    for dataset, sub in out.groupby(
        "dataset"
    ):
        treated = (
            sub[
                sub["dose_value"] > 0
            ]
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
            "n_removed": int(
                (treated["treated_dose_rank"] == held_rank).sum()
            ),
            "n_retained": len(retained),
            "spearman_rho": float(sr.statistic),
            "pearson_r": float(pr.statistic),
            "within_dataset_permutation_p": float(pp),
        })

    return pd.DataFrame(rows)


# ============================================================================
# LEAVE-ONE-DATASET-OUT
# ============================================================================

def leave_one_dataset_out(endpoint):
    treated = endpoint[
        endpoint["dose_value"] > 0
    ].copy()

    rows = []

    for excluded in sorted(
        treated["dataset"].unique()
    ):
        retained = treated[
            treated["dataset"] != excluded
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
            "excluded_dataset": excluded,
            "n_retained": len(retained),
            "spearman_rho": float(sr.statistic),
            "pearson_r": float(pr.statistic),
            "within_dataset_permutation_p": float(pp),
        })

    return pd.DataFrame(rows)


# ============================================================================
# PLOTS
# ============================================================================

def save_endpoint_scatter(endpoint):
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

    ax.set_title(
        "22-feature BoW, K=8\nNo clipping + all objects for K-means"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT_DIR /
        "deltaJS_vs_CTG_effect_96h.png",
        dpi=300
    )

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "\n============================================================"
    )
    print(
        "22-FEATURE BOW | K=8 | NO CLIPPING | NO SAMPLING"
    )
    print(
        "============================================================"
    )

    # ------------------------------------------------------------------------
    # Load objects
    # ------------------------------------------------------------------------
    cells = []

    for dataset, path in FILES.items():
        d = load_one_dataset(
            dataset,
            path
        )

        print(
            f"{dataset}: {len(d):,} usable objects"
        )

        cells.append(d)

    cells = pd.concat(
        cells,
        ignore_index=True
    )

    print(
        f"\nTotal usable objects: {len(cells):,}"
    )

    # ------------------------------------------------------------------------
    # Scale WITHOUT clipping
    # ------------------------------------------------------------------------
    zdf = robust_scale_to_matched_control_no_clipping(
        cells
    )

    # Report transformed-value range for transparency
    feature_matrix = zdf[
        FEATURES_22
    ].to_numpy(float)

    print(
        "\nRobust-scaled feature values:"
    )

    print(
        f"Minimum = {np.nanmin(feature_matrix):.6f}"
    )

    print(
        f"Maximum = {np.nanmax(feature_matrix):.6f}"
    )

    print(
        f"99.9th percentile absolute value = "
        f"{np.nanpercentile(np.abs(feature_matrix), 99.9):.6f}"
    )

    # ------------------------------------------------------------------------
    # Fit K-means on ALL objects
    # ------------------------------------------------------------------------
    print(
        f"\nObjects used to fit K-means: {len(zdf):,}"
    )

    model = MiniBatchKMeans(
        n_clusters=K,
        random_state=SEED,
        batch_size=4096,
        n_init=20,
    )

    model.fit(
        feature_matrix
    )

    # ------------------------------------------------------------------------
    # Build morphology state histograms
    # ------------------------------------------------------------------------
    hist = build_histograms(
        zdf,
        model
    )

    div = calculate_js_to_control(
        hist
    )

    corrected = add_baseline_correction(
        div
    )

    # ------------------------------------------------------------------------
    # Endpoint CTG merge
    # ------------------------------------------------------------------------
    ctg = load_ctg_workbook(
        CTG_XLSX
    )

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
    # Stats
    # ------------------------------------------------------------------------
    pooled = pooled_stats(
        endpoint
    )

    within = within_dataset_stats(
        endpoint
    )

    loo_condition = leave_one_condition_out(
        endpoint
    )

    loo_dose = leave_one_dose_position_out(
        endpoint
    )

    loo_dataset = leave_one_dataset_out(
        endpoint
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------
    hist.to_csv(
        OUT_DIR /
        "phenotype_histograms_all_timepoints.csv",
        index=False
    )

    corrected.to_csv(
        OUT_DIR /
        "morphology_divergence_longitudinal_baseline_corrected.csv",
        index=False
    )

    endpoint.to_csv(
        OUT_DIR /
        "morphology_divergence_vs_CTG_96h.csv",
        index=False
    )

    pd.DataFrame(
        [pooled]
    ).to_csv(
        OUT_DIR /
        "pooled_96h_statistics.csv",
        index=False
    )

    within.to_csv(
        OUT_DIR /
        "within_dataset_correlations.csv",
        index=False
    )

    loo_condition.to_csv(
        OUT_DIR /
        "leave_one_condition_out_sensitivity.csv",
        index=False
    )

    loo_dose.to_csv(
        OUT_DIR /
        "leave_one_dose_position_out_sensitivity.csv",
        index=False
    )

    loo_dataset.to_csv(
        OUT_DIR /
        "leave_one_dataset_out_sensitivity.csv",
        index=False
    )

    pd.DataFrame({
        "feature": FEATURES_22
    }).to_csv(
        OUT_DIR /
        "features_used.csv",
        index=False
    )

    save_endpoint_scatter(
        endpoint
    )

    # ------------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------------
    print(
        "\n============================================================"
    )
    print(
        "POOLED 96 h"
    )
    print(
        "============================================================"
    )

    print(
        f"Spearman rho = {pooled['spearman_rho']:.6f}"
    )

    print(
        f"Pearson r = {pooled['pearson_r']:.6f}"
    )

    print(
        f"Permutation p = "
        f"{pooled['within_dataset_permutation_p']:.6f}"
    )

    print(
        "\n============================================================"
    )
    print(
        "WITHIN-DATASET"
    )
    print(
        "============================================================"
    )

    print(
        within.to_string(
            index=False
        )
    )

    print(
        "\n============================================================"
    )
    print(
        "LEAVE-ONE-DATASET-OUT"
    )
    print(
        "============================================================"
    )

    print(
        loo_dataset.to_string(
            index=False
        )
    )

    print(
        "\n============================================================"
    )
    print(
        "LEAVE-ONE-DOSE-POSITION-OUT"
    )
    print(
        "============================================================"
    )

    print(
        loo_dose.to_string(
            index=False
        )
    )

    print(
        "\n============================================================"
    )
    print(
        "LEAVE-ONE-CONDITION-OUT RANGE"
    )
    print(
        "============================================================"
    )

    print(
        f"Spearman min = "
        f"{loo_condition['spearman_rho'].min():.6f}"
    )

    print(
        f"Spearman max = "
        f"{loo_condition['spearman_rho'].max():.6f}"
    )

    print(
        f"\nAll outputs saved to:\n{OUT_DIR}"
    )


if __name__ == "__main__":
    main()

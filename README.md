# longitudinal_3d_morphology_profiling
Code for longitudinal population morphology profiling of drug response from label-free 3D brightfield imaging.
# Longitudinal population morphology profiling of drug response in 3D cancer models

Code accompanying:

Sindhu SC, Audish A, Rana P, Li X, Rizos H, Liu S.
"Label-free longitudinal population morphology profiling of drug response
in 3D cancer models."

## Overview

This repository implements a longitudinal population morphology-profiling
framework for label-free 3D brightfield imaging.

The analysis consists of:

1. Extraction of 22 object-level BF-derived morphology features.
2. Matched-control robust scaling using the control median and IQR.
3. Construction of a shared K=8 morphological Bag-of-Words vocabulary.
4. Representation of each condition as a morphology-state distribution.
5. Jensen-Shannon distance to the same-time matched control.
6. Baseline correction relative to 0 h.
7. Association of 96 h morphology divergence with CTG 3D treatment effect.
8. Robustness and sensitivity analyses.

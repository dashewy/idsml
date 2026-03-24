# STAT 426 Project Scripts

This repo is a collection of STAT 426 homework/final R scripts. The scripts are mostly **self-contained** and use built-in datasets (Iris, USPS digits / MNIST variants) plus one local CSV (`housing.csv`).

## Files

### `stat_426_hw_6.r`
- **Topic**: Supervised classification practice on:
  - `iris` (3-class)
  - `dslabs::mnist_127` (digits 1/2/7 with 2 engineered features)
- **Models used (via `caret`)**: Naive Bayes (`naive_bayes`), LDA (`lda`), QDA (`qda`), multinomial logistic regression / elastic net (`glmnet`).
- **Outputs**:
  - Basic EDA summaries/means
  - Scatter plot on Iris (petal length/width)
  - Confusion matrices printed to console
  - Knits itself to PDF at the bottom via `rmarkdown::render()`
- **Notes**:
  - The script installs packages at the top. On GitHub / CI you typically *don’t* want install calls in scripts; locally it’s convenient.
  - The final `render()` call uses an **absolute path**; update it to a relative path if you run it on another machine.

### `stat_426_hw_11.r`
- **Topic**: Classification + PCA on:
  - `iris`
  - `Rdimtools::USPSdigits` (16×16 grayscale digit images flattened to 256 features)
- **What it does**:
  - Computes overall pixel mean and per-digit pixel means (to identify an “average” digit).
  - Trains and evaluates multiple classifiers on Iris: Naive Bayes, LDA, QDA, SVM (linear / radial / polynomial) using `caret`.
  - Builds a binary digit task (2 vs 8) from USPS digits and evaluates similar models (some with preprocessing like PCA/centering/scaling).
  - PCA visualization for Iris (first 2 PCs) and PCA exploration for USPS digits (how many PCs for 95% variance; scatter of selected digits on first 2 PCs).
  - Knits itself to PDF at the bottom via `rmarkdown::render()`
- **Outputs**:
  - Confusion matrices and printed model summaries
  - `ggplot2` PCA plots
  - When knitting, figure files may be written under `figure/`
- **Notes**:
  - Like HW6, this script includes `install.packages()` calls and an absolute-path `render()` call.

### `stat_426_final.r`
- **Topic**: Multi-class digit classification strategies + unsupervised clustering + regression.
- **Key components**:
  - **One-vs-Rest (OvR)** and **One-vs-One (OvO)** training/prediction helpers for multi-class classification:
    - `train_one_vs_rest()` / `predict_one_vs_rest()`
    - `train_one_vs_one()` / `predict_one_vs_one()`
  - Uses `Rdimtools::USPSdigits` and focuses on digits **1, 3, 7, 8**.
  - Runs SVMs (`e1071::svm`) with polynomial and radial kernels.
  - Runs QDA-like models using `caret::train(method="rda")` (regularized DA) to avoid singularity/runtime issues on high-dimensional features.
  - Applies PCA to reduce dimensionality and reruns OvO/OvR experiments.
  - **K-means clustering**: `center_plotter()` visualizes cluster centers as 16×16 grayscale “digit” images and prints majority-label proportions by cluster.
  - **Hierarchical clustering** comparisons (complete / Ward / average linkage) and majority-label summaries via `maj_lab()`.
  - **Housing regression**: loads local `housing.csv`, cleans currency fields, builds train/test split, standardizes predictors, and compares:
    - best subset / feature selection (`leapSeq`)
    - ridge (`glmnet`, `alpha=0`)
    - lasso (`glmnet`, `alpha=1`)
    - plus residual plots and RMSE on the held-out set.
- **Inputs**:
  - Built-in: `Rdimtools::USPSdigits`
  - Local file: `housing.csv` (expected in repo root)
- **Notes**:
  - Some clustering sections call `dev.off()` to clear plots; if you’re not plotting to a device, that can error—run in an interactive session where plotting devices exist.
  - The housing CSV is read using an **absolute path**; change to `read.csv("housing.csv")` for portability.

### `stat_426_helper_func.r`
Small helper functions used for earlier coursework / derivations:
- **`sigma_lda()`**: builds a 2D covariance-like scatter matrix from two feature vectors and a mean vector.
- **`delta_lda()`**: computes the LDA discriminant score \( \delta_k(x) \) given \(\mu\), \(\Sigma\), class prior \(\pi\), and point \(x\).
- **`gradient_descent()`**: generic 2-parameter gradient descent/ascent for an `expression()` using symbolic derivatives (`D()`), returning \((\theta_0, \theta_1, f(\theta))\).

## How to run

From R (recommended: RStudio), set your working directory to the repo root and source the script you want:

```r
setwd("path/to/stat_426_proj")
source("stat_426_hw_6.r")
```

To knit the homework scripts to PDF, you can run their `rmarkdown::render()` lines, but you’ll likely want to replace the hard-coded absolute paths with relative paths, e.g.:

```r
rmarkdown::render("stat_426_hw_11.r", output_format = "pdf_document")
```

## Dependencies (high level)

These scripts primarily use:
- `caret`, `e1071`, `kernlab`, `glmnet`
- `ggplot2`
- `Rdimtools` (for `USPSdigits`)
- `dslabs` (for `mnist_127`)
- `rmarkdown`, `knitr`, `tinytex` (for PDF knitting)


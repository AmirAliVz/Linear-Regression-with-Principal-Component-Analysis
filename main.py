"""
main.py

Predicts housing prices using Principal Component Regression (PCR):
standardizes 14 continuous features, reduces dimensionality with PCA,
selects significant principal components via forward stepwise OLS regression,
and evaluates the model on held-out test data.

Author: Amir Vaziri
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import boxcox, skew
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.eval_measures import mse

# ── Module-level constants ─────────────────────────────────────────────────────
FIGURES_DIR     = "Figures"
DATA_PATH       = "data/Housing_Information.csv"
RANDOM_SEED     = 42
KAISER_THRESHOLD = 1.0    # retain PCs with eigenvalue above this threshold
THRESHOLD_IN    = 0.05    # p-value threshold for forward stepwise inclusion
THRESHOLD_OUT   = 0.05    # p-value threshold for backward stepwise removal

# Categorical and non-continuous columns excluded before PCA
EXCLUDE_COLS = ["ID", "PreviousSalePrice", "IsLuxury", "Floors",
                "Garage", "HouseColor", "Fireplace", "Price"]


# ── Visualisation helpers ──────────────────────────────────────────────────────

def plot_histogram_boxplot(
    data, filename, kde=True, display=True,
    figsize=(10, 6), fontsize=12, fontcolor="black",
):
    """
    Plot a combined box plot and histogram for a given data Series.

    Parameters
    ----------
    data : pd.Series
        Input column of data.
    filename : str
        Column label used as the plot title and saved filename.
    kde : bool, optional
        Overlay a kernel density estimate on the histogram. Default True.
    display : bool, optional
        Render and save the figure (True) or return IQR bounds only (False).
        Default True.
    figsize : tuple, optional
        Figure dimensions in inches. Default (10, 6).
    fontsize : int, optional
        Font size for box-plot annotations. Default 12.
    fontcolor : str, optional
        Font colour for box-plot annotations. Default 'black'.

    Returns
    -------
    lower_whisker : float
        Lower IQR bound (Q1 - 1.5 * IQR, floored at the data minimum).
    upper_whisker : float
        Upper IQR bound (Q3 + 1.5 * IQR, capped at the data maximum).
    """
    q1            = np.percentile(data, 25)
    median        = np.median(data)
    q3            = np.percentile(data, 75)
    iqr           = q3 - q1
    lower_whisker = max(data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)

    if not display:
        return lower_whisker, upper_whisker

    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        sharex=True,
    )

    cmap = plt.get_cmap("inferno")

    # ── Box plot ──────────────────────────────────────────────────────────────
    sns.boxplot(x=data, ax=ax_box, color="skyblue")

    box_stats = {
        "Min":    lower_whisker,
        "Q1":     q1,
        "Median": median,
        "Q3":     q3,
        "Max":    upper_whisker,
    }
    for label, val in box_stats.items():
        ax_box.text(
            val, 0.02, f"{label}\n{val:.2f}",
            ha="center", va="bottom",
            fontsize=fontsize, color=fontcolor, rotation=45, weight="bold",
        )
    ax_box.set(xlabel="")

    # ── Histogram ─────────────────────────────────────────────────────────────
    hist_plot = sns.histplot(
        data, bins="auto", kde=kde,
        color="steelblue", edgecolor="black", ax=ax_hist,
    )

    # Summary statistics annotation overlaid on the histogram
    stats_text = "\n".join([
        f"Mean: {data.mean():.2f}",
        f"Mode: {data.mode().values[0]:.2f}",
        f"Std:  {data.std():.2f}",
        f"Skew: {data.skew():.2f}",
    ])
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)
    ax_hist.text(
        0.85, 0.6, stats_text,
        transform=ax_hist.transAxes,
        verticalalignment="top", horizontalalignment="right",
        bbox=bbox_props, fontsize=fontsize + 2, color=fontcolor, weight="bold",
    )

    patches = hist_plot.patches
    counts  = [patch.get_height() for patch in patches]

    # Apply inferno colour gradient scaled to bar height
    for patch, count in zip(patches, counts):
        patch.set_facecolor(cmap(0.3 + 0.7 * count / max(counts)))

    # Annotate each bar with its count
    for patch, count in zip(patches, counts):
        if count > 0:
            ax_hist.text(
                patch.get_x() + patch.get_width() / 2,
                count,
                f"{int(count)}",
                ha="center", va="bottom", fontsize=fontsize,
            )

    ax_hist.set_title(f"Distribution of {filename}", fontweight="bold")
    ax_hist.set_xlabel(filename, fontweight="bold")
    ax_hist.set_ylabel("Count", fontweight="bold")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{filename}.jpg"), dpi=300)
    plt.show()

    return lower_whisker, upper_whisker


def scatter_pearson(x, y, c1, c2, verbose=True):
    """
    Create a scatter plot and compute the Pearson correlation between two
    continuous variables, applying Box-Cox transformation to any variable
    whose absolute skewness is ≥ 1 before computing the coefficient.

    Parameters
    ----------
    x : array-like or pd.Series
        Values for the x-axis.
    y : array-like or pd.Series
        Values for the y-axis.
    c1 : str
        Label for x (used in the plot title and filename).
    c2 : str
        Label for y.
    verbose : bool, optional
        Print the correlation coefficient and display the figure. Default True.

    Returns
    -------
    pearson_corr : float
        Pearson correlation coefficient between (possibly transformed) x and y.
    """
    c1 = c1.capitalize()
    c2 = c2.capitalize()

    # Apply Box-Cox to each variable independently if its skewness exceeds the threshold
    if abs(skew(x)) >= 1:
        x, _lambda_x = boxcox(x + 1)
    if abs(skew(y)) >= 1:
        y, _lambda_y = boxcox(y + 1)

    correlation_matrix = np.corrcoef(x, y)
    pearson_corr       = correlation_matrix[0, 1]

    if verbose:
        print(f"Pearson correlation  {c1} vs {c2}: {pearson_corr:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.5})

    # Display the correlation coefficient as an annotated text box
    bbox_props = dict(boxstyle="round4,pad=1.2", fc="seashell", ec="black", alpha=0.5)
    plt.text(
        0.9, 0.8, f"r: {pearson_corr:.2f}",
        transform=ax.transAxes,
        verticalalignment="top", horizontalalignment="right",
        bbox=bbox_props, fontsize=16, color="black", weight="bold",
    )

    plt.title(f"{c1} vs {c2}", fontsize=16, fontweight="bold")
    plt.xlabel(c1, weight="bold")
    plt.ylabel(c2, weight="bold")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, f"{c1}vs{c2}.jpg"), dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()

    return pearson_corr


# ── Feature selection ──────────────────────────────────────────────────────────

def forward_stepwise_selection(X, y, threshold_in=THRESHOLD_IN, verbose=True):
    """
    Forward stepwise feature selection for OLS linear regression.

    Iteratively adds the predictor with the lowest p-value below threshold_in,
    stopping when no remaining variable meets the inclusion criterion.

    Parameters
    ----------
    X : pd.DataFrame
        Candidate predictor variables.
    y : pd.Series
        Target variable.
    threshold_in : float, optional
        p-value threshold for adding a variable. Default 0.05.
    verbose : bool, optional
        Print each addition step. Default True.

    Returns
    -------
    list of str
        Names of selected features in the order they were added.
    """
    included = []
    while True:
        changed  = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]

        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print(f"Add  {best_feature:30}  p-value: {best_pval:.6f}")

        if not changed:
            break

    return included


def backward_stepwise_elimination(X, y, threshold_out=THRESHOLD_OUT, verbose=True):
    """
    Backward stepwise elimination for OLS linear regression.

    Starts with the full model and iteratively removes the predictor with the
    highest p-value above threshold_out until all remaining predictors are
    statistically significant.

    Parameters
    ----------
    X : pd.DataFrame
        Full set of candidate predictors.
    y : pd.Series
        Target variable.
    threshold_out : float, optional
        p-value threshold for removing a variable. Default 0.05.
    verbose : bool, optional
        Print each removal step. Default True.

    Returns
    -------
    list of str
        Names of retained features.
    """
    features = list(X.columns)
    while True:
        model    = sm.OLS(y, sm.add_constant(X[features])).fit()
        pvalues  = model.pvalues.iloc[1:]
        max_pval = pvalues.max()
        if max_pval > threshold_out:
            worst = pvalues.idxmax()
            features.remove(worst)
            if verbose:
                print(f"Remove  {worst:30}  p-value: {max_pval:.6f}")
        else:
            break

    return features


def rfe_statsmodels(X, y, n_features_to_select=5, verbose=True):
    """
    Recursive Feature Elimination using statsmodels OLS.

    Iteratively removes the least significant predictor (highest p-value) until
    the specified number of features remains.

    Parameters
    ----------
    X : pd.DataFrame
        Candidate predictor variables.
    y : pd.Series
        Target variable.
    n_features_to_select : int, optional
        Number of features to retain. Default 5.
    verbose : bool, optional
        Print each removal step. Default True.

    Returns
    -------
    list of str
        Names of retained features.
    """
    features = list(X.columns)
    while len(features) > n_features_to_select:
        model         = sm.OLS(y, sm.add_constant(X[features])).fit()
        pvalues       = model.pvalues.iloc[1:]
        worst_feature = pvalues.idxmax()
        if verbose:
            print(f"Remove  {worst_feature}  p-value: {pvalues[worst_feature]:.4f}")
        features.remove(worst_feature)

    return features


# ── Diagnostics ────────────────────────────────────────────────────────────────

def plot_residuals_analysis(model):
    """
    Plot residuals vs. fitted values and the autocorrelation function of residuals.

    Residuals vs. fitted values checks the linearity and homoscedasticity
    assumptions. The ACF plot checks the independence-of-errors assumption.

    Parameters
    ----------
    model : statsmodels RegressionResultsWrapper
        A fitted OLS model.
    """
    fitted_vals = model.fittedvalues
    residuals   = model.resid

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs. fitted values — should show no systematic pattern or fan shape
    ax[0].scatter(fitted_vals, residuals, alpha=0.5)
    ax[0].axhline(0, color="red", linestyle="--")
    ax[0].tick_params(axis="x", rotation=30)
    ax[0].set_xlabel("Fitted values")
    ax[0].set_ylabel("Residuals")
    ax[0].set_title("Residuals vs Fitted Values", weight="bold")

    # Autocorrelation of residuals — values within confidence bands indicate independence
    sm.graphics.tsa.plot_acf(residuals, lags=40, alpha=0.05, zero=False, ax=ax[1])
    ax[1].set_title("Autocorrelation of Residuals", weight="bold")
    ax[1].set_xlabel("Lag")
    ax[1].set_ylabel("Autocorrelation")

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "residuals_analysis.jpg"), dpi=300)
    plt.show()


# ── PCA visualisations ─────────────────────────────────────────────────────────

def plot_scree(eigenvalues, verbose=True):
    """
    Plot the PCA scree plot with a Kaiser criterion reference line at eigenvalue = 1.

    Components with eigenvalues above the reference line explain more variance
    than a single original standardized variable and are candidates for retention.

    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues from the fitted PCA object (pca.explained_variance_).
    verbose : bool, optional
        Display the figure interactively. Default True.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker="o", linestyle="-")
    plt.axhline(y=KAISER_THRESHOLD, color="r", linestyle="--",
                label=f"Kaiser Criterion (Eigenvalue = {KAISER_THRESHOLD})")
    plt.title("Scree Plot — Eigenvalues per Principal Component")
    plt.xlabel("Principal Component Number")
    plt.ylabel("Eigenvalue")
    plt.xticks(np.arange(1, len(eigenvalues) + 1))
    plt.legend()
    plt.grid(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "ScreePlot(EigenValuesPCA).jpg"), dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()


def plot_variance_explained(pca_var, verbose=True):
    """
    Plot the proportion of variance explained by each retained principal component.

    Parameters
    ----------
    pca_var : array-like
        Explained variance ratios for the retained components.
    verbose : bool, optional
        Display the figure interactively. Default True.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(pca_var) + 1), pca_var, marker="o", linestyle="-")
    plt.title("Variance Explained per Retained Principal Component")
    plt.xlabel("Principal Component Number")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(np.arange(1, len(pca_var) + 1))
    plt.grid(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, "VarianceperPrincipalComponent.jpg"), dpi=300)
    if verbose:
        plt.show()
    else:
        plt.close()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(verbose=True):
    """
    Orchestrate the full Principal Component Regression pipeline.

    Steps
    -----
    1. Load and prepare the dataset; extract continuous features.
    2. Standardize continuous features (required for PCA).
    3. Fit PCA; retain components with eigenvalues > 1 (Kaiser criterion).
    4. Visualize scree plot and variance explained.
    5. Train / test split (80 / 20) on the PC score matrix.
    6. Forward stepwise OLS to identify significant PCs.
    7. Evaluate on training and test sets (MSE).
    8. Residual diagnostic plots.

    Parameters
    ----------
    verbose : bool, optional
        Print progress and display figures. Default True.
    """
    plt.rcParams["font.family"] = "Georgia"
    plt.rcParams["font.size"]   = 14

    # ── Load and prepare data ─────────────────────────────────────────────────
    df       = pd.read_csv(DATA_PATH)
    df_clean = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])

    # Retain only continuous predictors for PCA
    df_continuous = df_clean.copy()

    # ── Part D: Standardise continuous variables ───────────────────────────────
    # Standardization is mandatory before PCA: without it, variables measured on
    # larger scales dominate the principal components regardless of information content
    scaler         = StandardScaler()
    X_std          = scaler.fit_transform(df_continuous)
    df_standardized = pd.DataFrame(X_std, columns=df_continuous.columns)

    # Univariate distribution plots for each standardized predictor
    for col in df_standardized.columns:
        plot_histogram_boxplot(df_standardized[col], col, display=verbose)

    # Distribution of the dependent variable (not standardized)
    plot_histogram_boxplot(df["Price"], "Price", display=verbose)

    # ── Part E: PCA ───────────────────────────────────────────────────────────
    pca       = PCA()
    pc_matrix = pca.fit_transform(df_standardized)

    eigenvalues = pca.explained_variance_
    plot_scree(eigenvalues, verbose=verbose)

    # Retain components whose eigenvalue exceeds the Kaiser threshold of 1.0
    # — each retained component explains more variance than one original variable
    retained_indices = [i for i, val in enumerate(eigenvalues) if val > KAISER_THRESHOLD]
    selected_pcs     = pc_matrix[:, retained_indices]

    pc_df = pd.DataFrame(
        selected_pcs,
        columns=[f"PC{i + 1}" for i in range(selected_pcs.shape[1])],
    )

    # Attach the target variable to the PC score DataFrame
    pc_df["Price"] = df["Price"].reset_index(drop=True)

    # Variance explained by each retained component
    pca_var = pca.explained_variance_ratio_[retained_indices]
    plot_variance_explained(pca_var, verbose=verbose)

    # ── Part F: Model training and evaluation ─────────────────────────────────

    # 80 / 20 split on the PC scores; the scaler is NOT re-applied to the test set
    # because PC scores are already in the transformed space
    train_df, test_df = train_test_split(pc_df, test_size=0.2, random_state=RANDOM_SEED)
    train_df.to_csv("data/train_dataset.csv", index=False)
    test_df.to_csv("data/test_dataset.csv",  index=False)

    if verbose:
        print("Train and test datasets saved.")

    X_train = train_df.drop(columns=["Price"])
    y_train = train_df["Price"]

    # Forward stepwise selection identifies the subset of PCs with significant OLS p-values
    selected_features = forward_stepwise_selection(X_train, y_train, verbose=verbose)
    # Alternatives (all three converge on the same feature set):
    # selected_features = backward_stepwise_elimination(X_train, y_train, verbose=verbose)
    # selected_features = rfe_statsmodels(X_train, y_train, n_features_to_select=5, verbose=verbose)

    X_train_final = sm.add_constant(X_train[selected_features])
    model         = sm.OLS(y_train, X_train_final).fit()

    if verbose:
        print(model.summary())

    # MSE on the training set
    y_train_pred   = model.predict(X_train_final)
    mse_train      = mse(y_train, y_train_pred)
    print(f"MSE — training set: {mse_train:,.0f}")

    # MSE on the held-out test set
    # The test set already contains PC scores — no re-scaling or re-encoding needed
    X_test       = test_df.drop(columns=["Price"])
    y_test       = test_df["Price"]
    X_test_final = sm.add_constant(X_test[selected_features])
    y_test_pred  = model.predict(X_test_final)
    mse_test     = mse(y_test, y_test_pred)
    print(f"MSE — test set:     {mse_test:,.0f}")

    # Residual diagnostic plots
    plot_residuals_analysis(model)

    # Scatter plots of Price vs each PC score (excluding self-correlation)
    for col in [c for c in pc_df.columns if c != "Price"]:
        scatter_pearson(pc_df["Price"], pc_df[col], "Price", col, verbose=verbose)


if __name__ == "__main__":
    main(verbose=True)

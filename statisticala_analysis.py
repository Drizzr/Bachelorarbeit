# =============================================================================
# Library Imports
# =============================================================================
import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t, ttest_ind
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve

# =============================================================================
# Global Configuration and Styling
# =============================================================================

# =============================================================================
# Global Configuration and Styling
# =============================================================================

def setup_matplotlib_for_latex():
    """Configures Matplotlib to use LaTeX for rendering text in plots."""
    try:
        # --- Updated settings for better visibility ---
        plt.rcParams.update({
            # Font settings
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,                # Base font size
            "axes.labelsize": 15,           # X and Y axis labels
            "axes.titlesize": 16,           # Plot title
            "xtick.labelsize": 14,          # X-axis tick labels
            "ytick.labelsize": 14,          # Y-axis tick labels
            "legend.fontsize": 14,          # Legend font size

            # Line and marker settings
            "lines.linewidth": 2.0,         # Thicker plot lines
            "lines.markersize": 8,          # Larger markers
            "axes.linewidth": 1.5,          # Thicker axis lines for emphasis

            # LaTeX specific
            "text.latex.preamble": r"\usepackage{gensymb}"
        })
        print("Matplotlib configured to use LaTeX for plotting (with enhanced visibility settings).")
    except Exception as e:
        print(f"Could not configure LaTeX for plots, falling back to default. Error: {e}")
        plt.rcParams.update(plt.rcParamsDefault)

# Apply plotting styles
sns.set(style="whitegrid")
setup_matplotlib_for_latex()

# --- Directories ---
RESULTS_BASE_DIR = "Results"
OVERALL_PLOTS_DIR = os.path.join(RESULTS_BASE_DIR, "Overall_Generated_Plots")
OVERALL_TABLES_DIR = os.path.join(RESULTS_BASE_DIR, "Overall_Generated_Tables")
os.makedirs(OVERALL_PLOTS_DIR, exist_ok=True)
os.makedirs(OVERALL_TABLES_DIR, exist_ok=True)

# --- Analysis Parameters ---
N_MC_ITERATIONS = 500
CONFIDENCE_LEVEL = 0.95

# --- Feature Definitions ---
segment_cols_map = {
    "T": ["t_Area", "t_T-Dist", "t_Compact", "t_Angle"],
    "QRS": ["qrs_Area", "qrs_T-Dist", "qrs_Compact", "qrs_Angle"],
    "ST": ["st_Area", "st_T-Dist", "st_Compact", "st_Angle"]
}
features_to_analyze_short = ["Area", "T-Dist", "Compact", "Angle"]

# --- Feature-Specific Analysis Configuration ---
feature_analysis_config = {
    "default": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT", "show_outliers_in_plot": True},
    "t_Area": {"hypothesis": "not_equal", "remove_outliers": True, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "t_Compact": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "t_T-Dist": {"hypothesis": "not_equal", "remove_outliers": True, "units": r"pT", "show_outliers_in_plot": True},
    "qrs_Area": {"hypothesis": "not_equal", "remove_outliers": True, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "qrs_Compact": {"hypothesis": "not_equal", "remove_outliers": True, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "qrs_T-Dist": {"hypothesis": "not_equal", "remove_outliers": True, "units": r"pT", "show_outliers_in_plot": True},
    "st_Area": {"hypothesis": "not_equal", "remove_outliers": True, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "st_Compact": {"hypothesis": "not_equal", "remove_outliers": True, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "st_T-Dist": {"hypothesis": "not_equal", "remove_outliers": True, "units": r"pT", "show_outliers_in_plot": True},
    "t_Angle": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"$\degree$", "show_outliers_in_plot": True},
    "qrs_Angle": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"$\degree$", "show_outliers_in_plot": True},
    "st_Angle": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"$\degree$", "show_outliers_in_plot": True},
}

# =============================================================================
# Demographics Data Loading and Preparation
# =============================================================================
with open("Data/setup.json") as f:
    patient_meta_data = json.load(f)

demographic_records = []
for pid, p_data in patient_meta_data.items():
    ACM_status_patient = None
    if p_data.get("runs"):
        first_run_id = next(iter(p_data["runs"]))
        ACM_status_patient = p_data["runs"][first_run_id].get("ACM")

    # Expanded height and age processing
    height_val = p_data.get("height")
    if height_val not in [None, ""]:
        height = float(height_val)
    else:
        height = None
        
    age_val = p_data.get("age")
    if age_val not in [None, ""]:
        age = int(age_val)
    else:
        age = None

    record = {
        "patient": pid, "gender": p_data.get("gender", "unknown") or "unknown",
        "height": height, "age": age, "ACM": ACM_status_patient
    }
    demographic_records.append(record)

df_demographics = pd.DataFrame(demographic_records)
df_demographics.dropna(subset=['ACM'], inplace=True)
gender_palette_global = {"male": "#555555", "female": "#AAAAAA"}

# =============================================================================
# Demographics Analysis and Plotting Functions
# =============================================================================
def plot_gender_distribution(data: pd.DataFrame, title: str, save_path: str = None):
    """Generates and saves a bar plot of gender distribution."""
    counts = data["gender"].value_counts().reindex(["male", "female"]).fillna(0)
    plt.figure(figsize=(7, 5))
    palette = [gender_palette_global.get(g, "#cccccc") for g in counts.index]
    ax = sns.barplot(x=counts.index, y=counts.values, palette=palette, edgecolor='black')

    plt.title(title)
    plt.ylabel(r"Number of Patients")
    plt.xlabel(r"Gender")

    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values, default=0) * 0.03, str(int(v)),
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.ylim(0, max(counts.values, default=1) * 1.20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_hist_and_stats(data: pd.DataFrame, column: str, title: str, color: str, save_path: str = None) -> Dict:
    """Calculates stats and plots a histogram for a given column."""
    valid_data = data[column].dropna()
    stats = {"mean": np.nan, "std": np.nan, "count": len(valid_data), "median": np.nan}

    if not valid_data.empty:
        stats.update({"mean": valid_data.mean(), "std": valid_data.std(), "median": valid_data.median()})
        print(f"{title}: Mean = {stats['mean']:.2f}, Median = {stats['median']:.2f}, "
              f"Std = {stats['std']:.2f}, N = {stats['count']}")
        plt.figure(figsize=(8, 5))
        sns.histplot(valid_data, kde=True, color=color, bins=10, edgecolor='black', linewidth=1)
        plt.axvline(stats['mean'], color='black', linestyle='--', label=fr'Mean = {stats["mean"]:.2f}')
        if not np.isnan(stats['std']):
            plt.axvline(stats['mean'] + stats['std'], color='dimgray', linestyle=':', label=fr'+1 SD = {stats["mean"] + stats["std"]:.2f}')
            plt.axvline(stats['mean'] - stats['std'], color='dimgray', linestyle=':', label=fr'-1 SD = {stats["mean"] - stats["std"]:.2f}')
        plt.title(title)
        plt.xlabel(column.capitalize())
        plt.ylabel(r"Frequency")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
    else:
        print(f"{title}: No valid data.")
    return stats

# =============================================================================
# Core Statistical & Plotting Helper Functions
# =============================================================================

def remove_outliers_iqr(data: np.ndarray) -> np.ndarray:
    """Removes outliers from a 1D array using the IQR method."""
    if data.size == 0:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def _plot_t_distribution(t_stat: float, df: int, p_value_area_cond, p_value_area_label: str, title: str, save_path: str):
    """Helper to plot the t-distribution with a shaded p-value area."""
    plt.figure(figsize=(10, 6))
    x_t = np.linspace(-4, 4, 1000)
    y_t = t.pdf(x_t, df)
    plt.plot(x_t, y_t, label=r"t-distribution", color='black')
    plt.fill_between(x_t, 0, y_t, where=p_value_area_cond(x_t), color='lightgray', alpha=0.8, label=p_value_area_label)
    plt.axvline(x=t_stat, color='black', linestyle='--', label=fr't-statistic: {t_stat:.2f}')
    plt.xlabel(r"t-value")
    plt.ylabel(r"Probability Density")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def _plot_feature_boxplot(data_raw: Tuple[np.ndarray, np.ndarray], data_cleaned: Tuple[np.ndarray, np.ndarray],
                        labels: Tuple[str, str], title: str, y_label: str, mc_medians: Tuple[List[float], List[float]],
                        threshold: float, show_outliers: bool, save_path: str):
    """Helper to plot the feature boxplot with MC medians and data points."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x_labels = [fr"{labels[i]} (N={len(d)})" for i, d in enumerate(data_cleaned)]

    sns.boxplot(data=list(data_raw), ax=ax, palette=["#DDDDDD", "#777777"], boxprops=dict(edgecolor='k'),
                medianprops=dict(color='k'), whiskerprops=dict(color='dimgray'), capprops=dict(color='k'),
                showfliers=show_outliers)

    for i, group_data in enumerate(data_cleaned):
        if len(group_data) > 0:
            sns.stripplot(y=group_data, x=[i] * len(group_data), ax=ax, color='black', size=4, alpha=0.5, jitter=0.1)

    ax.scatter([], [], color='black', alpha=0.5, s=30, label=r'Individual Data Points')

    if mc_medians and all(mc_medians):
        low_p, high_p = (1 - CONFIDENCE_LEVEL) / 2 * 100, (1 + CONFIDENCE_LEVEL) / 2 * 100
        for i, mc_data in enumerate(mc_medians):
            mean_med = np.mean(mc_data)
            low_ci, high_ci = np.percentile(mc_data, [low_p, high_p])
            y_err = [[mean_med - low_ci], [high_ci - mean_med]]
            
            if i == 0:
                label = fr'MC Median ({CONFIDENCE_LEVEL*100:.0f}\% CI)'
            else:
                label = None
                
            ax.errorbar(x=i, y=mean_med, yerr=y_err, fmt='X', color='black', ecolor='black',
                        elinewidth=2, capsize=8, capthick=2, markersize=6, label=label)

    if threshold is not None and not np.isnan(threshold):
        ax.axhline(y=threshold, color='black', linestyle='-.', label=fr'Optimal Threshold: {threshold:.2f}')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    handles, legend_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _plot_mc_distribution(data: List[float], nominal_val: float, name: str, title: str, save_path: str):
    """Helper to plot the distribution of a metric from Monte Carlo simulations."""
    if not data: return
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=True, bins=30, color='darkgray')
    plt.title(title)
    plt.xlabel(name)
    if not np.isnan(nominal_val):
        plt.axvline(nominal_val, color='black', linestyle='--', label=fr'Nominal ({nominal_val:.3f})')
    mean_val = np.mean(data)
    plt.axvline(mean_val, color='dimgray', linestyle=':', label=fr'Mean ({mean_val:.3f})')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def _plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path: str):
    """Helper to plot an ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    opt_idx = np.argmax(tpr - fpr) if len(tpr) > 0 else 0
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='black', lw=2, label=fr'ROC (AUC={roc_auc:.2f})')
    if len(fpr) > opt_idx:
        plt.plot(fpr[opt_idx], tpr[opt_idx], 'ko', markersize=8, label=r'Optimal Threshold')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel(r'False Positive Rate')
    plt.ylabel(r'True Positive Rate')
    plt.title(r'ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, display_labels: List[str], save_path: str):
    """Helper to plot a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_sum_rows = cm.sum(axis=1)[:, np.newaxis]
    cm_perc = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype('float'), cm_sum_rows, out=cm_perc, where=cm_sum_rows != 0)
    cm_perc *= 100
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_perc, annot=True, fmt=".2f", cmap="Greys", cbar=False,
                linecolor='black', linewidths=0.5,
                xticklabels=display_labels, yticklabels=display_labels)
    plt.title(r"Confusion Matrix (\%)")
    plt.xlabel(r"Predicted Label")
    plt.ylabel(r"True Label")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# =============================================================================
# Core Statistical Analysis Functions
# =============================================================================

def _perform_single_t_test_run(data1: np.ndarray, data2: np.ndarray, hypothesis: str) -> Tuple[float, float, bool]:
    """Helper for a single Welch's t-test run in an MC loop."""
    if len(data1) < 2 or len(data2) < 2:
        return np.nan, np.nan, False
    t_stat, p_val_two_tailed = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
    if np.isnan(t_stat) or np.isnan(p_val_two_tailed):
        return np.nan, np.nan, False

    if hypothesis == "data1_greater":
        p_value = p_val_two_tailed / 2 if t_stat > 0 else 1 - (p_val_two_tailed / 2)
    elif hypothesis == "data2_greater":
        p_value = p_val_two_tailed / 2 if t_stat < 0 else 1 - (p_val_two_tailed / 2)
    else:
        p_value = p_val_two_tailed
    return t_stat, p_value, p_value < 0.05

def perform_t_test(data1_nom, data2_nom, data1_unc, data2_unc, name, hypothesis, threshold, labels,
                   remove_outliers, show_outliers_in_plot, save_plots_prefix, y_label):
    """Performs t-test on nominal data and optional MC simulation."""
    data1_nom_arr, data2_nom_arr = np.asarray(data1_nom), np.asarray(data2_nom)
    if remove_outliers:
        data1_clean = remove_outliers_iqr(data1_nom_arr)
        data2_clean = remove_outliers_iqr(data2_nom_arr)
    else:
        data1_clean = data1_nom_arr
        data2_clean = data2_nom_arr

    print(f"T-Test for {name} (Nominal): N1={len(data1_clean)}, N2={len(data2_clean)}")
    nom_t, nom_p, nom_sig = np.nan, np.nan, False
    if len(data1_clean) >= 2 and len(data2_clean) >= 2:
        nom_t, nom_p, nom_sig = _perform_single_t_test_run(data1_clean, data2_clean, hypothesis)
        if not np.isnan(nom_p):
            print(f"Hypothesis: {hypothesis}, Nominal T-stat: {nom_t:.4f}, P-val: {nom_p:.4f}, Significant: {nom_sig}")
            df_freedom = len(data1_clean) + len(data2_clean) - 2
            if df_freedom > 0 and save_plots_prefix:
                if hypothesis == "data1_greater":
                    title_hyp = fr"{labels[0]} $>$ {labels[1]}"
                    cond = lambda x: x > nom_t
                    area_label = fr'p-value (t > {nom_t:.2f})'
                elif hypothesis == "data2_greater":
                    title_hyp = fr"{labels[0]} $<$ {labels[1]}"
                    cond = lambda x: x < nom_t
                    area_label = fr'p-value (t < {nom_t:.2f})'
                else: # 'not_equal'
                    title_hyp = fr"{labels[0]} $\neq$ {labels[1]}"
                    cond = lambda x: (x > abs(nom_t)) | (x < -abs(nom_t))
                    area_label = fr'p-value ($|t|$ > {abs(nom_t):.2f})'
                
                _plot_t_distribution(nom_t, df_freedom, cond, area_label, fr"t-Distribution (Hyp: {title_hyp})", f"{save_plots_prefix}_tdist.pdf")

    mc_t_stats, mc_p_values, mc_medians1, mc_medians2 = [], [], [], []
    has_unc = data1_unc is not None and data2_unc is not None
    if has_unc and N_MC_ITERATIONS > 0:
        print(f"Performing MC ({N_MC_ITERATIONS} iter) for t-test of {name}...")
        for _ in range(N_MC_ITERATIONS):
            d1_s = np.random.normal(data1_nom_arr, np.asarray(data1_unc))
            d2_s = np.random.normal(data2_nom_arr, np.asarray(data2_unc))
            d1_sc = remove_outliers_iqr(d1_s) if remove_outliers else d1_s
            d2_sc = remove_outliers_iqr(d2_s) if remove_outliers else d2_s
            if len(d1_sc) > 0: mc_medians1.append(np.median(d1_sc))
            if len(d2_sc) > 0: mc_medians2.append(np.median(d2_sc))
            t_s_mc, p_v_mc, _ = _perform_single_t_test_run(d1_sc, d2_sc, hypothesis)
            if not np.isnan(p_v_mc):
                mc_t_stats.append(t_s_mc)
                mc_p_values.append(p_v_mc)

    if save_plots_prefix:
        if hypothesis == "data1_greater":
            plot_labels = (f'{labels[0]} (Higher)', f'{labels[1]} (Lower)')
        elif hypothesis == "data2_greater":
            plot_labels = (f'{labels[0]} (Lower)', f'{labels[1]} (Higher)')
        else:
            plot_labels = labels
        _plot_feature_boxplot(data_raw=(data1_nom_arr, data2_nom_arr), data_cleaned=(data1_clean, data2_clean),
                              labels=plot_labels, title=fr"Box Plot for {name}", y_label=y_label,
                              mc_medians=(mc_medians1, mc_medians2), threshold=threshold,
                              show_outliers=show_outliers_in_plot, save_path=f"{save_plots_prefix}_boxplot.pdf")
        if mc_p_values:
            _plot_mc_distribution(mc_p_values, nom_p, r"P-value", fr"MC Distribution of P-values for {name}", f"{save_plots_prefix}_p_value_mc_dist.pdf")

    if mc_p_values:
        p_low, p_high = np.percentile(mc_p_values, [(1-CONFIDENCE_LEVEL)/2*100, (1+CONFIDENCE_LEVEL)/2*100])
        t_low, t_high = np.percentile(mc_t_stats, [(1-CONFIDENCE_LEVEL)/2*100, (1+CONFIDENCE_LEVEL)/2*100])
        mean_p, mean_t = np.mean(mc_p_values), np.mean(mc_t_stats)
        print(f"MC T-Test: Mean P-val: {mean_p:.4f} CI:[{p_low:.4f},{p_high:.4f}]. Mean T-stat: {mean_t:.4f} CI:[{t_low:.4f},{t_high:.4f}]")
        return (mean_t, t_low, t_high), (mean_p, p_low, p_high), (mean_p < 0.05)
    
    return (nom_t, np.nan, np.nan), (nom_p, np.nan, np.nan), nom_sig

def _determine_single_optimal_threshold_run(data1, data2, hypothesis):
    """Helper for a single ROC analysis run in an MC loop."""
    if len(data1) == 0 or len(data2) == 0:
        return (np.nan,) * 5
        
    y_scores = np.concatenate((data1, data2))
    if hypothesis == "data1_greater":
        y_true = np.concatenate((np.ones(len(data1)), np.zeros(len(data2))))
    elif hypothesis == "data2_greater":
        y_true = np.concatenate((np.zeros(len(data1)), np.ones(len(data2))))
    else:
        if np.mean(data1) > np.mean(data2):
            y_true = np.concatenate((np.ones(len(data1)), np.zeros(len(data2))))
        else:
            y_true = np.concatenate((np.zeros(len(data1)), np.ones(len(data2))))

    fpr, tpr, thresh_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    if len(thresh_roc) == 0:
        return np.nan, np.nan, np.nan, np.nan, roc_auc

    opt_idx = np.argmax(tpr - fpr)
    opt_thresh = thresh_roc[opt_idx]
    y_pred = (y_scores >= opt_thresh).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if np.sum(y_true == 1) > 0:
        sensitivity = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
    else:
        sensitivity = 0.0
    
    if np.sum(y_true == 0) > 0:
        specificity = np.sum((y_pred == 0) & (y_true == 0)) / np.sum(y_true == 0)
    else:
        specificity = 0.0
        
    return opt_thresh, sensitivity, specificity, f1, roc_auc

def determine_optimal_threshold(data1_nom, data2_nom, data1_unc, data2_unc, hypothesis, labels, remove_outliers, save_plots_prefix):
    """Determines optimal classification threshold and metrics with optional MC."""
    d1_nom, d2_nom = np.asarray(data1_nom), np.asarray(data2_nom)
    if remove_outliers:
        d1_clean = remove_outliers_iqr(d1_nom)
        d2_clean = remove_outliers_iqr(d2_nom)
    else:
        d1_clean = d1_nom
        d2_clean = d2_nom

    print(f"Optimal Threshold for {labels[0]} vs {labels[1]} (Nominal): N1={len(d1_clean)}, N2={len(d2_clean)}")
    
    nom_res = (np.nan,) * 5
    if len(d1_clean) > 0 and len(d2_clean) > 0:
        nom_res = _determine_single_optimal_threshold_run(d1_clean, d2_clean, hypothesis)
        print(f"Nominal Thresh:{nom_res[0]:.4f}, F1:{nom_res[3]:.2f}, Sens:{nom_res[1]:.2f}, Spec:{nom_res[2]:.2f}, AUC:{nom_res[4]:.2f}")
        if save_plots_prefix and not np.isnan(nom_res[0]):
            y_s_plot = np.concatenate((d1_clean, d2_clean))
            
            is_data1_greater = (hypothesis == "data1_greater") or \
                               (hypothesis == "not_equal" and np.mean(d1_clean) > np.mean(d2_clean))
            
            if is_data1_greater:
                y_t_plot = np.concatenate((np.ones(len(d1_clean)), np.zeros(len(d2_clean))))
                cm_labels = [labels[1], labels[0]]
            else:
                y_t_plot = np.concatenate((np.zeros(len(d1_clean)), np.ones(len(d2_clean))))
                cm_labels = list(labels)
                
            y_pred_plot = (y_s_plot >= nom_res[0]).astype(int)
            _plot_roc_curve(y_t_plot, y_s_plot, f"{save_plots_prefix}_roc.pdf")
            _plot_confusion_matrix(y_t_plot, y_pred_plot, cm_labels, f"{save_plots_prefix}_cm.pdf")

    mc_results = {k: [] for k in ["Threshold", "Sensitivity", "Specificity", "F1-Score", "AUC"]}
    has_unc = data1_unc is not None and data2_unc is not None
    if has_unc and N_MC_ITERATIONS > 0:
        print(f"Performing MC ({N_MC_ITERATIONS} iter) for ROC of {labels[0]} vs {labels[1]}...")
        for _ in range(N_MC_ITERATIONS):
            d1_s, d2_s = np.random.normal(d1_nom, np.asarray(data1_unc)), np.random.normal(d2_nom, np.asarray(data2_unc))
            if remove_outliers:
                d1_sc, d2_sc = remove_outliers_iqr(d1_s), remove_outliers_iqr(d2_s)
            else:
                d1_sc, d2_sc = d1_s, d2_s
            res = _determine_single_optimal_threshold_run(d1_sc, d2_sc, hypothesis)
            if not np.isnan(res[0]):
                for i, key in enumerate(mc_results.keys()):
                    mc_results[key].append(res[i])

    final_results = {}
    for name, data_list, nom_val in zip(mc_results.keys(), mc_results.values(), nom_res):
        if data_list:
            mean_val = np.mean(data_list)
            low_ci, high_ci = np.percentile(data_list, [(1 - CONFIDENCE_LEVEL) / 2 * 100, (1 + CONFIDENCE_LEVEL) / 2 * 100])
            final_results[name] = (mean_val, low_ci, high_ci)
            print(f"  MC Mean {name}:{mean_val:.4f} CI:[{low_ci:.4f},{high_ci:.4f}]")
        else:
            final_results[name] = (nom_val, np.nan, np.nan)
    
    if save_plots_prefix:
        for name in ["Threshold", "F1-Score"]:
            _plot_mc_distribution(mc_results[name], final_results[name][0], name, fr"MC Distribution of {name.replace('-', ' ')}", f"{save_plots_prefix}_{name.lower().replace('-', '_')}_mc_dist.pdf")
    return final_results

# =============================================================================
# Main Analysis Execution Logic
# =============================================================================

def run_full_feature_analysis(df_analysis: pd.DataFrame, analysis_name: str, clean_plot_name: str,
                             output_plots_dir: str, output_tables_dir: str) -> List[Dict]:
    """Runs the complete statistical analysis for a given dataframe of features."""
    os.makedirs(output_plots_dir, exist_ok=True)
    os.makedirs(output_tables_dir, exist_ok=True)
    analysis_records = []
    
    for seg_lbl in segment_cols_map.keys():
        for feat_short_n in features_to_analyze_short:
            nom_col = f"{seg_lbl.lower()}_{feat_short_n}"
            unc_col = f"{nom_col}_unc"
            if nom_col not in df_analysis.columns:
                continue

            has_unc = unc_col in df_analysis.columns
            if not has_unc:
                print(f"Warning: {unc_col} not found in {analysis_name}. No MC will be run.")
            print(f"\n-- Feature: {seg_lbl} {feat_short_n} in {clean_plot_name}, Has Uncertainty: {has_unc} --")

            nom_series = pd.to_numeric(df_analysis[nom_col], errors='coerce')
            valid_idx = nom_series.notna()
            
            if has_unc:
                unc_series = pd.to_numeric(df_analysis.loc[valid_idx, unc_col], errors='coerce').fillna(0).clip(lower=0)
            else:
                unc_series = pd.Series(0.0, index=nom_series[valid_idx].index)

            nom_v, unc_v, acm_l = nom_series[valid_idx].values, unc_series.values, df_analysis.loc[valid_idx, "ACM"].values
            pos_nom, neg_nom = nom_v[acm_l], nom_v[~acm_l]
            
            if has_unc:
                pos_unc, neg_unc = unc_v[acm_l], unc_v[~acm_l]
            else:
                pos_unc, neg_unc = None, None

            if len(pos_nom) < 1 or len(neg_nom) < 1:
                print(f"Skipping {nom_col} for {analysis_name}: Insufficient data.")
                continue

            cfg = feature_analysis_config.get(nom_col, feature_analysis_config["default"])
            plot_pref = os.path.join(output_plots_dir, f"{analysis_name}_{seg_lbl}_{feat_short_n}")

            # Note: Group 1 = Healthy, Group 2 = ACM for consistency
            roc_res = determine_optimal_threshold(neg_nom, pos_nom, data1_unc=neg_unc, data2_unc=pos_unc,
                hypothesis=cfg["hypothesis"], labels=("Healthy", "ACM"), remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_pref)

            ttest_res = perform_t_test(neg_nom, pos_nom, data1_unc=neg_unc, data2_unc=pos_unc,
                name=fr"{seg_lbl} {feat_short_n} ({clean_plot_name})", hypothesis=cfg["hypothesis"],
                threshold=roc_res["Threshold"][0], labels=("Healthy", "ACM"), remove_outliers=cfg["remove_outliers"],
                show_outliers_in_plot=cfg.get("show_outliers_in_plot", True),
                save_plots_prefix=plot_pref, y_label=f"{feat_short_n} in {cfg['units']}")
            
            clean_pos_nom = remove_outliers_iqr(pos_nom) if cfg["remove_outliers"] else pos_nom
            clean_neg_nom = remove_outliers_iqr(neg_nom) if cfg["remove_outliers"] else neg_nom
            is_acm_greater = (cfg["hypothesis"] == "data2_greater") or \
                             (cfg["hypothesis"] == "not_equal" and np.mean(clean_pos_nom) > np.mean(clean_neg_nom))
            
            if is_acm_greater:
                pos_class = "ACM"
            else:
                pos_class = "Healthy"
            
            rec = {"source": analysis_name, "segment": seg_lbl, "feature": feat_short_n, "column_name": nom_col,
                   "hypothesis_tested": cfg["hypothesis"], "outliers_removed": cfg["remove_outliers"],
                   "mc_iterations": N_MC_ITERATIONS if has_unc else 0,
                   "significant_ttest (mean_p<0.05)": ttest_res[2],
                   "roc_positive_class_for_metrics": pos_class,
                   "n_ACM_initial": len(pos_nom), "n_healthy_initial": len(neg_nom)}
            
            all_metrics = {"p_value": ttest_res[1], "t_stat": ttest_res[0], **roc_res}
            for key, (mean, ci_low, ci_high) in all_metrics.items():
                key_clean = key.lower().replace('-', '_')
                rec.update({f"{key_clean}_mean": mean, f"{key_clean}_ci_lower": ci_low, f"{key_clean}_ci_upper": ci_high})
            analysis_records.append(rec)
    
    if analysis_records:
        df_summary = pd.DataFrame(analysis_records)
        summary_path = os.path.join(output_tables_dir, f"{analysis_name}_analysis_summary_mc.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\nSummary for {analysis_name} saved to {summary_path}")
    return analysis_records

# =============================================================================
# Script Execution Starts Here
# =============================================================================

# --- 1. Demographics Analysis ---
print("\n--- GENDER DISTRIBUTION ---")
plot_gender_distribution(df_demographics, r"Gender Distribution (All Patients)", os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_all.pdf"))
plot_gender_distribution(df_demographics[df_demographics["ACM"] == True], r"Gender Distribution (ACM)", os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_ACM.pdf"))
plot_gender_distribution(df_demographics[df_demographics["ACM"] == False], r"Gender Distribution (Healthy)", os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_healthy.pdf"))

print("\n--- AGE & HEIGHT DISTRIBUTIONS ---")
demographics_summary_stats = {}
groups = {"All": df_demographics, "ACM": df_demographics[df_demographics["ACM"]], "Healthy": df_demographics[~df_demographics["ACM"]]}
for group_name, df_group in groups.items():
    for col in ["age", "height"]:
        demographics_summary_stats[f"{col}_{group_name.lower()}"] = plot_hist_and_stats(
            df_group, col, fr"{col.capitalize()} Distribution ({group_name})", "#555555",
            os.path.join(OVERALL_PLOTS_DIR, f"demographics_{col}_{group_name.lower()}.pdf"))
summary_path = os.path.join(OVERALL_TABLES_DIR, "demographics_summary.json")
with open(summary_path, "w") as f: json.dump(demographics_summary_stats, f, indent=4)
print(f"\nDemographics summary saved to {summary_path}")

# --- 2. Data Loading for Core Analysis ---
ACM_map = {(pid, str(run_id)): run_data["ACM"] for pid, p_data in patient_meta_data.items() for run_id, run_data in p_data.get("runs", {}).items()}
result_files = glob.glob(os.path.join(RESULTS_BASE_DIR, "*", "result.csv"))
if not result_files:
    print(f"Warning: No 'result.csv' files found in subdirectories of {RESULTS_BASE_DIR}")
all_sensor_data_dfs = {}
for f_path in result_files:
    sensor_name = os.path.basename(os.path.dirname(f_path))
    try:
        df_sensor = pd.read_csv(f_path, dtype={'run': str})
        df_sensor["ACM"] = df_sensor.apply(lambda row: ACM_map.get((row["patient"], row["run"])), axis=1)
        df_sensor.dropna(subset=["ACM"], inplace=True)
        df_sensor["ACM"] = df_sensor["ACM"].astype(bool)
        if not df_sensor.empty:
            all_sensor_data_dfs[sensor_name] = df_sensor
            print(f"Loaded {sensor_name}: {len(df_sensor)} records.")
    except Exception as e:
        print(f"Error loading {f_path}: {e}")

# --- 3. Individual Sensor Analysis ---
print("\n\n--- INDIVIDUAL SENSOR ANALYSIS ---")
overall_individual_analysis_records = []
for sensor_name, df_sensor in all_sensor_data_dfs.items():
    print(f"\n--- Analyzing Sensor: {sensor_name} ---")
    s_base_dir = os.path.dirname(glob.glob(os.path.join(RESULTS_BASE_DIR, sensor_name, "result.csv"))[0])
    s_plots_dir = os.path.join(s_base_dir, "Generated_Plots_MC")
    s_tables_dir = os.path.join(s_base_dir, "Generated_Tables_MC")
    clean_name = sensor_name.replace("_", " ").upper()
    sensor_records = run_full_feature_analysis(df_sensor, sensor_name, clean_name, s_plots_dir, s_tables_dir)
    overall_individual_analysis_records.extend(sensor_records)

if overall_individual_analysis_records:
    pd.DataFrame(overall_individual_analysis_records).to_csv(os.path.join(OVERALL_TABLES_DIR, "all_sensors_features_summary_mc.csv"), index=False)
    print(f"\nOverall summary for all individual features saved.")

# --- 4. Aggregated Projections Analysis ---
print("\n\n--- AGGREGATED PROJECTION ANALYSIS ---")
sensor_grid = [["F1", "OX", "OO", "OQ"], ["YQ", "NL", "OY", "OW"], ["OT", "F2", "F0", "C1"], ["EY", "YP", "OR", "EZ"]]

# ============================ FIX IS HERE ============================
# The definition for "BottomLeft" was incorrect. It was using the same
# column slice as "BottomRight", causing them to analyze identical data.
# Corrected "BottomLeft" to use slice(0, 2) for columns.
subsquare_definitions = {
    "TopLeft": (slice(0, 2), slice(0, 2)), 
    "TopRight": (slice(0, 2), slice(2, 4)),
    "BottomLeft": (slice(2, 4), slice(0, 2)), # Corrected from slice(2, 4)
    "BottomRight": (slice(2, 4), slice(2, 4))
}
# =====================================================================

subsquares = {}
for name, (row_slice, col_slice) in subsquare_definitions.items():
    subsquares[name] = [sensor_grid[r][c] for r in range(row_slice.start, row_slice.stop) for c in range(col_slice.start, col_slice.stop)]

SUBSQUARE_PLOT_NAMES = {"TopLeft": "Top-Left", "TopRight": "Top-Right", "BottomLeft": "Bottom-Left", "BottomRight": "Bottom-Right"}
projections_to_analyze = ["xy", "yz"]
agg_analysis_records = []

for subsquare_name, sensor_ids in subsquares.items():
    for projection in projections_to_analyze:
        agg_name = f"aggregated_{subsquare_name}_{projection}"
        plot_name = SUBSQUARE_PLOT_NAMES.get(subsquare_name, subsquare_name)
        clean_name = f"Aggregated {plot_name} Quadrant {projection.upper()}"
        
        dfs_to_aggregate = [all_sensor_data_dfs[f"{sid}_{projection}"] for sid in sensor_ids if f"{sid}_{projection}" in all_sensor_data_dfs]
        if not dfs_to_aggregate:
            print(f"No data for {clean_name}. Skipping.")
            continue
            
        df_agg = pd.concat(dfs_to_aggregate, ignore_index=True)
        print(f"\n--- Analyzing {clean_name} ({len(df_agg)} records) ---")

        agg_plots_dir = os.path.join(OVERALL_PLOTS_DIR, "Aggregated_Plots_MC")
        agg_tables_dir = os.path.join(OVERALL_TABLES_DIR, "Aggregated_Tables_MC")
        
        agg_records = run_full_feature_analysis(df_agg, agg_name, clean_name, agg_plots_dir, agg_tables_dir)
        agg_analysis_records.extend(agg_records)

if agg_analysis_records:
    pd.DataFrame(agg_analysis_records).to_csv(os.path.join(OVERALL_TABLES_DIR, "aggregated_projection_summary_mc.csv"), index=False)
    print(f"\nAggregated analysis summary saved.")

print("\n\n--- SCRIPT FINISHED ---")
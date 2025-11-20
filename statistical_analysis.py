# =============================================================================
# Library Imports
# =============================================================================
import glob
import json
import os
import functools
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t, ttest_ind
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve

# --- New Import for Uncertainty Propagation ---
try:
    from uncertainties import ufloat
    from uncertainties.umath import *
except ImportError:
    raise ImportError("Please install the 'uncertainties' library: pip install uncertainties")

# =============================================================================
# Global Configuration and Styling
# =============================================================================

def setup_matplotlib_for_latex():
    """Configures Matplotlib to use LaTeX for rendering text in plots."""
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "axes.linewidth": 1.5,
            "text.latex.preamble": r"\usepackage{gensymb}"
        })
        print("Matplotlib configured to use LaTeX for plotting.")
    except Exception as e:
        print(f"Could not configure LaTeX, falling back to default. Error: {e}")
        plt.rcParams.update(plt.rcParamsDefault)

sns.set(style="whitegrid")
setup_matplotlib_for_latex()

# --- Directories ---
RESULTS_BASE_DIR = "Results"
OVERALL_PLOTS_DIR = os.path.join(RESULTS_BASE_DIR, "Overall_Generated_Plots")
OVERALL_TABLES_DIR = os.path.join(RESULTS_BASE_DIR, "Overall_Generated_Tables")
os.makedirs(OVERALL_PLOTS_DIR, exist_ok=True)
os.makedirs(OVERALL_TABLES_DIR, exist_ok=True)

# --- Analysis Parameters ---
N_MC_ITERATIONS = 5000
UNCERTAINTY_LEVEL = 0.95

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
    "t_Area": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "t_Compact": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "t_T-Dist": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT", "show_outliers_in_plot": True},
    "qrs_Area": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "qrs_Compact": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "qrs_T-Dist": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT", "show_outliers_in_plot": True},
    "st_Area": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "st_Compact": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT$^2$", "show_outliers_in_plot": True},
    "st_T-Dist": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"pT", "show_outliers_in_plot": True},
    "t_Angle": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"$\degree$", "show_outliers_in_plot": True},
    "qrs_Angle": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"$\degree$", "show_outliers_in_plot": True},
    "st_Angle": {"hypothesis": "not_equal", "remove_outliers": False, "units": r"$\degree$", "show_outliers_in_plot": True},
}

# =============================================================================
# Helper: Aggregation with Uncertainty Library
# =============================================================================

def aggregate_dataframes_with_uncertainty(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges a list of dataframes on Patient/Run/ACM, then calculates the mean
    across sensors for every feature, propagating uncertainty using the 
    'uncertainties' library.
    """
    if not df_list:
        return pd.DataFrame()
    if len(df_list) == 1:
        return df_list[0]

    # 1. Rename columns in input DFs before merging to ensure uniqueness
    # This prevents messy pandas suffixes like _x, _y when merging >2 dfs
    renamed_dfs = []
    for i, df in enumerate(df_list):
        temp_df = df.copy()
        # Rename feature columns to include an index (e.g., t_Area -> t_Area_0)
        cols_to_rename = {}
        for col in temp_df.columns:
            if col not in ['patient', 'run', 'ACM']:
                cols_to_rename[col] = f"{col}_{i}"
        temp_df.rename(columns=cols_to_rename, inplace=True)
        renamed_dfs.append(temp_df)

    # 2. Merge all renamed dataframes into one wide dataframe
    # Outer join ensures we keep patients even if they only appear in one sensor
    df_wide = functools.reduce(
        lambda left, right: pd.merge(left, right, on=['patient', 'run', 'ACM'], how='outer'),
        renamed_dfs
    )

    # 3. Iterate through known features and calculate mean + uncertainty
    df_result = df_wide[['patient', 'run', 'ACM']].copy()

    # Helper to compute mean of ufloats for a row
    def compute_row_mean(row, value_cols, unc_cols):
        measurements = []
        for v_c, u_c in zip(value_cols, unc_cols):
            val = row.get(v_c, np.nan)
            unc = row.get(u_c, 0.0)
            
            if pd.notna(val):
                # Create ufloat. If unc is NaN, assume 0
                u_val = 0.0 if pd.isna(unc) else unc
                measurements.append(ufloat(val, u_val))
        
        if not measurements:
            return pd.Series([np.nan, np.nan])
        
        # Calculate mean (sum / count). 
        # uncertainties library handles error propagation: 1/N * sqrt(sum(sigma^2))
        mean_obj = sum(measurements) / len(measurements)
        return pd.Series([mean_obj.n, mean_obj.s])

    # Identify unique base features (e.g. t_Area, qrs_Dist)
    unique_base_features = set()
    for feats in segment_cols_map.values():
        for f in feats:
            unique_base_features.add(f)

    for base_feat in unique_base_features:
        # Identify indices present in the merge (0, 1, ...)
        relevant_indices = [str(i) for i in range(len(df_list))]
        
        val_cols = [f"{base_feat}_{i}" for i in relevant_indices if f"{base_feat}_{i}" in df_wide.columns]
        unc_cols = [f"{base_feat}_unc_{i}" for i in relevant_indices]
        
        # Filter to only columns that actually exist in the wide df
        final_val_cols = []
        final_unc_cols = []
        for vc, uc in zip(val_cols, unc_cols):
            if vc in df_wide.columns:
                final_val_cols.append(vc)
                final_unc_cols.append(uc)

        if not final_val_cols:
            continue

        # Apply calculation row-wise. Returns a DataFrame with 2 columns: [Nominal, StdDev]
        calculated = df_wide.apply(
            lambda row: compute_row_mean(row, final_val_cols, final_unc_cols), 
            axis=1
        )
        
        df_result[base_feat] = calculated[0]
        df_result[f"{base_feat}_unc"] = calculated[1]

    return df_result

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

    height_val = p_data.get("height")
    height = float(height_val) if height_val not in [None, ""] else None
    age_val = p_data.get("age")
    age = int(age_val) if age_val not in [None, ""] else None

    record = {
        "patient": pid, "gender": p_data.get("gender", "unknown") or "unknown",
        "height": height, "age": age, "ACM": ACM_status_patient
    }
    demographic_records.append(record)

df_demographics = pd.DataFrame(demographic_records)
df_demographics.dropna(subset=['ACM'], inplace=True)
gender_palette_global = {"male": "#555555", "female": "#AAAAAA"}

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_gender_distribution(data: pd.DataFrame, title: str, save_path: str = None):
    counts = data["gender"].value_counts().reindex(["male", "female"]).fillna(0)
    plt.figure(figsize=(7, 5))
    palette = [gender_palette_global.get(g, "#cccccc") for g in counts.index]
    ax = sns.barplot(x=counts.index, y=counts.values, palette=palette, edgecolor='black')
    plt.title(title)
    plt.ylabel(r"Number of Patients")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values, default=0) * 0.03, str(int(v)), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches='tight'); plt.close()
    else: plt.show()

def plot_hist_and_stats(data: pd.DataFrame, column: str, title: str, color: str, save_path: str = None) -> Dict:
    valid_data = data[column].dropna()
    stats = {"mean": np.nan, "std": np.nan, "count": len(valid_data), "median": np.nan}
    if not valid_data.empty:
        stats.update({"mean": valid_data.mean(), "std": valid_data.std(), "median": valid_data.median()})
        plt.figure(figsize=(8, 5))
        sns.histplot(valid_data, kde=True, color=color, bins=10, edgecolor='black')
        plt.axvline(stats['mean'], color='black', linestyle='--', label=fr'Mean')
        plt.title(title)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, bbox_inches='tight'); plt.close()
    return stats

# =============================================================================
# Core Statistical Functions
# =============================================================================
def remove_outliers_iqr(data: np.ndarray) -> np.ndarray:
    if data.size == 0: return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return data[(data >= q1 - 1.5 * iqr) & (data <= q3 + 1.5 * iqr)]

def _plot_feature_boxplot(data_raw, data_cleaned, labels, title, y_label, mc_medians, threshold, show_outliers, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    x_labels = [fr"{labels[i]} (N={len(d)})" for i, d in enumerate(data_cleaned)]
    sns.boxplot(data=list(data_raw), ax=ax, palette=["#DDDDDD", "#777777"], showfliers=show_outliers)
    if mc_medians and all(mc_medians):
        for i, mc_data in enumerate(mc_medians):
            mean_med = np.mean(mc_data)
            low, high = np.percentile(mc_data, [(1-UNCERTAINTY_LEVEL)/2*100, (1+UNCERTAINTY_LEVEL)/2*100])
            ax.errorbar(x=i, y=mean_med, yerr=[[mean_med-low], [high-mean_med]], fmt='X', color='black', capsize=5)
    if threshold is not None and not np.isnan(threshold):
        ax.axhline(y=threshold, color='black', linestyle='-.')
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _perform_single_t_test_run(data1, data2, hypothesis):
    if len(data1) < 2 or len(data2) < 2: return np.nan, np.nan, False
    t_stat, p_val_two = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
    if np.isnan(t_stat): return np.nan, np.nan, False
    if hypothesis == "data1_greater": p_value = p_val_two / 2 if t_stat > 0 else 1 - p_val_two / 2
    elif hypothesis == "data2_greater": p_value = p_val_two / 2 if t_stat < 0 else 1 - p_val_two / 2
    else: p_value = p_val_two
    return t_stat, p_value, p_value < 0.05

def perform_t_test(data1_nom, data2_nom, data1_unc, data2_unc, name, hypothesis, threshold, labels, remove_outliers, show_outliers_in_plot, save_plots_prefix, y_label):
    d1, d2 = np.asarray(data1_nom), np.asarray(data2_nom)
    d1_c = remove_outliers_iqr(d1) if remove_outliers else d1
    d2_c = remove_outliers_iqr(d2) if remove_outliers else d2
    
    nom_t, nom_p, nom_sig = _perform_single_t_test_run(d1_c, d2_c, hypothesis)
    
    mc_p, mc_t, mc_m1, mc_m2 = [], [], [], []
    if data1_unc is not None and N_MC_ITERATIONS > 0:
        for _ in range(N_MC_ITERATIONS):
            s1 = np.random.normal(d1, data1_unc)
            s2 = np.random.normal(d2, data2_unc)
            if remove_outliers: s1, s2 = remove_outliers_iqr(s1), remove_outliers_iqr(s2)
            if len(s1) > 0: mc_m1.append(np.median(s1))
            if len(s2) > 0: mc_m2.append(np.median(s2))
            ts, ps, _ = _perform_single_t_test_run(s1, s2, hypothesis)
            if not np.isnan(ps): mc_p.append(ps); mc_t.append(ts)
            
    if save_plots_prefix:
        _plot_feature_boxplot((d1, d2), (d1_c, d2_c), labels, fr"Box Plot for {name}", y_label, (mc_m1, mc_m2), threshold, show_outliers_in_plot, f"{save_plots_prefix}_boxplot.pdf")
    
    if mc_p:
        p_stats = (np.mean(mc_p), np.percentile(mc_p, 2.5), np.percentile(mc_p, 97.5))
        t_stats = (np.mean(mc_t), np.percentile(mc_t, 2.5), np.percentile(mc_t, 97.5))
        return t_stats, p_stats, (p_stats[0] < 0.05)
    return (nom_t, np.nan, np.nan), (nom_p, np.nan, np.nan), nom_sig

def _determine_single_roc_run(data1, data2, hypothesis):
    if len(data1) == 0 or len(data2) == 0: return (np.nan,)*5
    y_scores = np.concatenate((data1, data2))
    if hypothesis == "data1_greater": y_true = np.concatenate((np.ones(len(data1)), np.zeros(len(data2))))
    elif hypothesis == "data2_greater": y_true = np.concatenate((np.zeros(len(data1)), np.ones(len(data2))))
    else: 
        y_true = np.concatenate((np.ones(len(data1)), np.zeros(len(data2)))) if np.mean(data1) > np.mean(data2) else np.concatenate((np.zeros(len(data1)), np.ones(len(data2))))
    
    fpr, tpr, thres = roc_curve(y_true, y_scores)
    auc_val = auc(fpr, tpr)
    if len(thres) == 0: return (np.nan,)*5
    idx = np.argmax(tpr - fpr)
    opt_t = thres[idx]
    y_pred = (y_scores >= opt_t).astype(int)
    f1 = f1_score(y_true, y_pred)
    sens = np.sum((y_pred==1)&(y_true==1))/np.sum(y_true==1) if np.sum(y_true==1)>0 else 0
    spec = np.sum((y_pred==0)&(y_true==0))/np.sum(y_true==0) if np.sum(y_true==0)>0 else 0
    return opt_t, sens, spec, f1, auc_val

def determine_optimal_threshold(data1_nom, data2_nom, data1_unc, data2_unc, hypothesis, labels, remove_outliers, save_plots_prefix):
    d1, d2 = np.asarray(data1_nom), np.asarray(data2_nom)
    d1_c = remove_outliers_iqr(d1) if remove_outliers else d1
    d2_c = remove_outliers_iqr(d2) if remove_outliers else d2
    
    nom_res = _determine_single_roc_run(d1_c, d2_c, hypothesis)
    
    mc_results = {k:[] for k in ["Threshold", "Sensitivity", "Specificity", "F1-Score", "AUC"]}
    if data1_unc is not None and N_MC_ITERATIONS > 0:
        for _ in range(N_MC_ITERATIONS):
            s1 = np.random.normal(d1, data1_unc)
            s2 = np.random.normal(d2, data2_unc)
            if remove_outliers: s1, s2 = remove_outliers_iqr(s1), remove_outliers_iqr(s2)
            res = _determine_single_roc_run(s1, s2, hypothesis)
            if not np.isnan(res[0]):
                for i, k in enumerate(mc_results.keys()): mc_results[k].append(res[i])
                
    final = {}
    for i, k in enumerate(mc_results.keys()):
        if mc_results[k]:
            final[k] = (np.mean(mc_results[k]), np.percentile(mc_results[k], 2.5), np.percentile(mc_results[k], 97.5))
        else:
            final[k] = (nom_res[i], np.nan, np.nan)
    return final

def run_full_feature_analysis(df_analysis: pd.DataFrame, analysis_name: str, clean_plot_name: str, output_plots_dir: str, output_tables_dir: str):
    os.makedirs(output_plots_dir, exist_ok=True)
    os.makedirs(output_tables_dir, exist_ok=True)
    records = []
    
    for seg_lbl in segment_cols_map.keys():
        for feat_short in features_to_analyze_short:
            nom_col = f"{seg_lbl.lower()}_{feat_short}"
            unc_col = f"{nom_col}_unc"
            if nom_col not in df_analysis.columns: continue
            
            has_unc = unc_col in df_analysis.columns
            series_nom = pd.to_numeric(df_analysis[nom_col], errors='coerce')
            valid = series_nom.notna()
            
            nom_vals = series_nom[valid].values
            unc_vals = pd.to_numeric(df_analysis.loc[valid, unc_col], errors='coerce').fillna(0).values if has_unc else np.zeros_like(nom_vals)
            acm_vals = df_analysis.loc[valid, "ACM"].astype(bool).values
            
            pos_n, neg_n = nom_vals[acm_vals], nom_vals[~acm_vals]
            pos_u, neg_u = unc_vals[acm_vals], unc_vals[~acm_vals]
            
            if len(pos_n) < 2 or len(neg_n) < 2: continue
            
            cfg = feature_analysis_config.get(nom_col, feature_analysis_config["default"])
            plot_pref = os.path.join(output_plots_dir, f"{analysis_name}_{seg_lbl}_{feat_short}")
            
            roc = determine_optimal_threshold(neg_n, pos_n, neg_u, pos_u, cfg["hypothesis"], ("Healthy", "ACM"), cfg["remove_outliers"], plot_pref)
            
            ttest = perform_t_test(neg_n, pos_n, neg_u, pos_u, fr"{seg_lbl} {feat_short} ({clean_plot_name})", cfg["hypothesis"], roc["Threshold"][0], ("Healthy", "ACM"), cfg["remove_outliers"], cfg["show_outliers_in_plot"], plot_pref, f"{feat_short} {cfg['units']}")
            
            rec = {"source": analysis_name, "segment": seg_lbl, "feature": feat_short, "hypothesis": cfg["hypothesis"], "significant": ttest[2]}
            for k, v in roc.items(): rec[f"{k}_mean"], rec[f"{k}_lower"], rec[f"{k}_upper"] = v
            rec["p_val_mean"], rec["p_val_lower"], rec["p_val_upper"] = ttest[1]
            records.append(rec)
            
    if records: pd.DataFrame(records).to_csv(os.path.join(output_tables_dir, f"{analysis_name}_summary.csv"), index=False)
    return records

# =============================================================================
# Main Execution
# =============================================================================

# 1. Demographics
print("\n--- DEMOGRAPHICS ---")
plot_gender_distribution(df_demographics, "Gender Distribution (All)", os.path.join(OVERALL_PLOTS_DIR, "gender_all.pdf"))

# 2. Load Data
print("\n--- LOADING DATA ---")
ACM_map = {(pid, str(run_id)): run_data["ACM"] for pid, p_data in patient_meta_data.items() for run_id, run_data in p_data.get("runs", {}).items()}
result_files = glob.glob(os.path.join(RESULTS_BASE_DIR, "*", "result.csv"))
all_sensor_dfs = {}
for f_path in result_files:
    s_name = os.path.basename(os.path.dirname(f_path))
    df = pd.read_csv(f_path, dtype={'run': str})
    df["ACM"] = df.apply(lambda r: ACM_map.get((r["patient"], r["run"])), axis=1)
    df.dropna(subset=["ACM"], inplace=True)
    if not df.empty: all_sensor_dfs[s_name] = df

# 3. Individual Analysis
print("\n--- INDIVIDUAL SENSORS ---")
indiv_recs = []
for s_name, df in all_sensor_dfs.items():
    print(f"Analyzing {s_name}...")
    s_dir = os.path.dirname(glob.glob(os.path.join(RESULTS_BASE_DIR, s_name, "result.csv"))[0])
    recs = run_full_feature_analysis(df, s_name, s_name.replace("_", " "), os.path.join(s_dir, "Plots"), os.path.join(s_dir, "Tables"))
    indiv_recs.extend(recs)
if indiv_recs: pd.DataFrame(indiv_recs).to_csv(os.path.join(OVERALL_TABLES_DIR, "all_sensors_summary.csv"), index=False)

# 4. Aggregated Analysis (MEAN + UNCERTAINTY)
print("\n--- AGGREGATED ANALYSIS (MEAN + UNCERTAINTY) ---")
sensor_grid = [["F1", "OX", "OO", "OQ"], ["YQ", "NL", "OY", "OW"], ["OT", "F2", "F0", "C1"], ["EY", "YP", "OR", "EZ"]]

subsquares = {
    "TopLeft": (slice(0, 2), slice(0, 2)), 
    "TopRight": (slice(0, 2), slice(2, 4)),
    "BottomLeft": (slice(2, 4), slice(0, 2)), 
    "BottomRight": (slice(2, 4), slice(2, 4))
}

agg_recs = []
for sub_name, (r_slice, c_slice) in subsquares.items():
    ids = [sensor_grid[r][c] for r in range(r_slice.start, r_slice.stop) for c in range(c_slice.start, c_slice.stop)]
    for proj in ["xy", "yz"]:
        dfs_to_agg = [all_sensor_dfs[f"{sid}_{proj}"] for sid in ids if f"{sid}_{proj}" in all_sensor_dfs]
        
        if not dfs_to_agg:
            print(f"Skipping {sub_name} {proj}: No data.")
            continue

        agg_name = f"aggregated_{sub_name}_{proj}"
        print(f"Processing {agg_name} (Merging {len(dfs_to_agg)} sensors)...")
        
        df_averaged = aggregate_dataframes_with_uncertainty(dfs_to_agg)
        
        if df_averaged.empty:
            print(f"  Resulting dataframe empty for {agg_name}.")
            continue
            
        clean_title = f"{sub_name} {proj.upper()} (Averaged)"
        recs = run_full_feature_analysis(
            df_averaged, 
            agg_name, 
            clean_title, 
            os.path.join(OVERALL_PLOTS_DIR, "Aggregated_MC"), 
            os.path.join(OVERALL_TABLES_DIR, "Aggregated_MC")
        )
        agg_recs.extend(recs)

if agg_recs: 
    pd.DataFrame(agg_recs).to_csv(os.path.join(OVERALL_TABLES_DIR, "aggregated_summary_mc.csv"), index=False)
    print("Aggregated analysis complete.")

print("\n--- FINISHED ---")
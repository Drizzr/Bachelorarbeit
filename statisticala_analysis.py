
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_ind
# import uncertainties as unc # Not used
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import json
import os
import glob



# Seaborn styling
sns.set(style="whitegrid")

# Define BASE output directories
RESULTS_BASE_DIR = "Results" 
OVERALL_PLOTS_DIR = os.path.join(RESULTS_BASE_DIR, "Overall_Generated_Plots")
OVERALL_TABLES_DIR = os.path.join(RESULTS_BASE_DIR, "Overall_Generated_Tables")

os.makedirs(OVERALL_PLOTS_DIR, exist_ok=True)
os.makedirs(OVERALL_TABLES_DIR, exist_ok=True)


# %% [markdown]
# ## Study Population (Demographics)

# %%

# Load the JSON data for patient demographics and ARVC status
with open("Data/setup.json") as f:
    patient_meta_data = json.load(f)

# Flatten data for demographics DataFrame
demographic_records = []
for pid, p_data in patient_meta_data.items():
    gender = p_data.get("gender")
    height = p_data.get("height")
    age = p_data.get("age")
    arvc_status_patient = None
    if p_data.get("runs"):
        first_run_id = next(iter(p_data["runs"]))
        arvc_status_patient = p_data["runs"][first_run_id].get("ARVC")

    demographic_records.append({
        "patient": pid,
        "gender": gender if gender else "unknown",
        "height": float(height) if height not in [None, ""] else None,
        "age": int(age) if age not in [None, ""] else None,
        "ARVC": arvc_status_patient
    })

df_demographics = pd.DataFrame(demographic_records)
df_demographics.dropna(subset=['ARVC'], inplace=True)

gender_palette_global = {
    "male": "#4C72B0", "female": "#DD8452", "unknown": "#A9A9A9"
}

def plot_gender_distribution(data, title, save_path=None):
    counts = data["gender"].value_counts().reindex(["male", "female", "unknown"]).fillna(0)
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=[gender_palette_global.get(g, "#cccccc") for g in counts.index])
    plt.title(title)
    plt.ylabel("Number of Patients"); plt.xlabel("Gender")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values, default=0) * 0.03, str(int(v)), ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.ylim(0, max(counts.values, default=1) * 1.20)
    plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show()

print("\n--- GENDER DISTRIBUTION ---")
plot_gender_distribution(df_demographics, "Gender Distribution (All Patients)", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_all.png"))
plot_gender_distribution(df_demographics[df_demographics["ARVC"] == True], "Gender Distribution (ARVC)", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_arvc.png"))
plot_gender_distribution(df_demographics[df_demographics["ARVC"] == False], "Gender Distribution (Healthy)", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_healthy.png"))

def plot_hist_and_stats(data, column, title, color, save_path=None):
    valid_data = data[column].dropna()
    stats = {"mean": np.nan, "std": np.nan, "count": len(valid_data), "median": np.nan}
    if len(valid_data) > 0:
        stats["mean"] = valid_data.mean(); stats["std"] = valid_data.std(); stats["median"] = valid_data.median()
        print(f"{title}: Mean = {stats['mean']:.2f}, Median = {stats['median']:.2f}, Std = {stats['std']:.2f}, N = {stats['count']}")
        plt.figure(figsize=(8, 5))
        sns.histplot(valid_data, kde=True, color=color, bins=10)
        plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean = {stats["mean"]:.2f}')
        if not np.isnan(stats['std']):
            plt.axvline(stats['mean'] + stats['std'], color='green', linestyle='--', label=f'+1 SD = {stats["mean"]+stats["std"]:.2f}')
            plt.axvline(stats['mean'] - stats['std'], color='green', linestyle='--', label=f'-1 SD = {stats["mean"]-stats["std"]:.2f}')
        plt.title(title); plt.xlabel(column.capitalize()); plt.ylabel("Frequency")
        plt.legend(); plt.tight_layout()
        if save_path: plt.savefig(save_path); plt.close()
        else: plt.show()
    else: print(f"{title}: No valid data.")
    return stats

demographics_summary_stats = {}
print("\n--- AGE & HEIGHT (All Patients) ---")
demographics_summary_stats["age_all"] = plot_hist_and_stats(df_demographics, "age", "Age Distribution (All)", "#3E8E7E", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_age_all.png"))
demographics_summary_stats["height_all"] = plot_hist_and_stats(df_demographics, "height", "Height Distribution (All)", "#5DADEC", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_height_all.png"))
print("\n--- AGE & HEIGHT (ARVC Positive) ---")
demographics_summary_stats["age_arvc"] = plot_hist_and_stats(df_demographics[df_demographics["ARVC"] == True], "age", "Age Distribution (ARVC)", "#C44E52", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_age_arvc.png"))
demographics_summary_stats["height_arvc"] = plot_hist_and_stats(df_demographics[df_demographics["ARVC"] == True], "height", "Height Distribution (ARVC)", "#C44E52", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_height_arvc.png"))
print("\n--- AGE & HEIGHT (ARVC Negative) ---")
demographics_summary_stats["age_healthy"] = plot_hist_and_stats(df_demographics[df_demographics["ARVC"] == False], "age", "Age Distribution (Healthy)", "#4C72B0", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_age_healthy.png"))
demographics_summary_stats["height_healthy"] = plot_hist_and_stats(df_demographics[df_demographics["ARVC"] == False], "height", "Height Distribution (Healthy)", "#4C72B0", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_height_healthy.png"))

with open(os.path.join(OVERALL_TABLES_DIR, "demographics_summary.json"), "w") as f:
    json.dump(demographics_summary_stats, f, indent=4)
print(f"\nDemographics summary stats saved to {os.path.join(OVERALL_TABLES_DIR, 'demographics_summary.json')}")



def remove_outliers_iqr(data):
    """ User's original remove_outliers_iqr """
    data_arr = np.asarray(data)
    if len(data_arr) == 0: 
        return data_arr
    q1 = np.percentile(data_arr, 25)
    q3 = np.percentile(data_arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data_arr[(data_arr >= lower) & (data_arr <= upper)]

def perform_t_test(data1, data2, name="", hypothesis="data1_greater", threshold=None, labels=("Group 1", "Group 2"), remove_outliers=False, save_plots_prefix=None):
    data1_arr = np.asarray(data1); data2_arr = np.asarray(data2)
    data1_clean, data2_clean = (remove_outliers_iqr(data1_arr), remove_outliers_iqr(data2_arr)) if remove_outliers else (data1_arr, data2_arr)
    
    print(f"T-Test for {name}:") 
    print(f"Original data 1: {len(data1_arr)}, Original data 2: {len(data2_arr)}")
    print(f"Cleaned data 1: {len(data1_clean)}, Cleaned data 2: {len(data2_clean)}")

    if len(data1_clean) < 2 or len(data2_clean) < 2: print("Not enough data for t-test after cleaning."); return np.nan, np.nan, False

    t_stat, p_val_two_tailed = ttest_ind(data1_clean, data2_clean, equal_var=False, nan_policy='omit')
    if np.isnan(t_stat) or np.isnan(p_val_two_tailed): print("T-test resulted in NaN."); return t_stat, p_val_two_tailed, False

    if hypothesis == "data1_greater":
        group1_plot_label, group2_plot_label = f'{labels[0]} (Higher)', f'{labels[1]} (Lower)'
        p_value = p_val_two_tailed / 2 if t_stat > 0 else 1 - (p_val_two_tailed / 2)
        hypothesis_text_for_plot = f"{labels[0]} > {labels[1]}" 
    elif hypothesis == "data2_greater":
        group1_plot_label, group2_plot_label = f'{labels[0]} (Lower)', f'{labels[1]} (Higher)'
        p_value = p_val_two_tailed / 2 if t_stat < 0 else 1 - (p_val_two_tailed / 2)
        hypothesis_text_for_plot = f"{labels[0]} < {labels[1]}" 
    else:  
        group1_plot_label, group2_plot_label = labels[0], labels[1]
        p_value = p_val_two_tailed
        hypothesis_text_for_plot = f"{labels[0]} ≠ {labels[1]}" 
    
    significant = p_value < 0.05
    print(f"Hypothesis: {hypothesis_text_for_plot.replace(labels[0],'data1').replace(labels[1],'data2')}") 
    print(f"T-statistic: {t_stat:.4f}")
    if hypothesis == "not_equal": print(f"Two-tailed p-value: {p_value:.4f}")
    else: print(f"One-tailed p-value: {p_value:.4f}"); print(f"Two-tailed p-value: {p_val_two_tailed:.4f}")
    if significant: print(f"The result is statistically significant (p-value = {p_value:.4f})")
    else: print(f"The result is not statistically significant (p-value = {p_value:.4f})")
    
    df_freedom = len(data1_clean) + len(data2_clean) - 2
    if df_freedom > 0:
        x_t = np.linspace(-4, 4, 1000) 
        y_t = t.pdf(x_t, df_freedom)
        plt.figure(figsize=(10, 6)); plt.plot(x_t, y_t, label="t-distribution", color='blue') 
        fill_label_t_dist = f'p-value Area (t > {t_stat:.2f})' if hypothesis == "data1_greater" \
            else f'p-value Area (t < {t_stat:.2f})' if hypothesis == "data2_greater" \
            else f'p-value Area (|t| > {abs(t_stat):.2f})'
        fill_cond = (x_t > t_stat) if hypothesis == "data1_greater" else (x_t < t_stat) if hypothesis == "data2_greater" \
            else ((x_t > abs(t_stat)) | (x_t < -abs(t_stat)))
        plt.fill_between(x_t, 0, y_t, where=fill_cond, color='red', alpha=0.5, label=fill_label_t_dist) 
        plt.axvline(x=t_stat, color='green', linestyle='--', label=f't-statistic: {t_stat:.2f}') 
        plt.xlabel("t-value"); plt.ylabel("Probability Density"); plt.title(f"t-Distribution with Shaded p-value Area ({hypothesis_text_for_plot.replace(labels[0],'data1').replace(labels[1],'data2')})"); plt.legend() 
        if save_plots_prefix: plt.savefig(f"{save_plots_prefix}_tdist.png"); plt.close()
        else: plt.show()

    # Prepare data for boxplot, ensuring they are not empty after potential cleaning
    plot_data_for_boxplot = []
    if len(data1_clean) > 0:
        plot_data_for_boxplot.append(data1_clean)
    else: 
        plot_data_for_boxplot.append(np.array([])) # Pass empty array if data1_clean is empty
        
    if len(data2_clean) > 0:
        plot_data_for_boxplot.append(data2_clean)
    else:
        plot_data_for_boxplot.append(np.array([])) # Pass empty array if data2_clean is empty
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=plot_data_for_boxplot, 
                palette=["green", "red"], 
                notch=True) 
    plt.xticks([0, 1], [f"{group1_plot_label} (N={len(data1_clean)})", f"{group2_plot_label} (N={len(data2_clean)})"]); 
    plt.title(f"Boxplot of Cleaned Data for {name}"); 
    if threshold is not None: plt.axhline(y=threshold, color='orange', linestyle='--', label=f'Optimal Threshold: {threshold:.2f}'); plt.legend() 
    if save_plots_prefix: plt.savefig(f"{save_plots_prefix}_boxplot.png"); plt.close()
    else: plt.show()
    return t_stat, p_value, significant

def determine_optimal_threshold(data1, data2, hypothesis="data2_greater", labels=("Class 0", "Class 1"), remove_outliers=False, save_plots_prefix=None):
    data1_arr = np.asarray(data1); data2_arr = np.asarray(data2)
    print(f"data 1: {len(data1_arr)}, data 2: {len(data2_arr)}") 
    data1_clean, data2_clean = (remove_outliers_iqr(data1_arr), remove_outliers_iqr(data2_arr)) if remove_outliers else (data1_arr, data2_arr)
    print(f"Cleaned data 1: {len(data1_clean)}, Cleaned data 2: {len(data2_clean)}") 
    if len(data1_clean) == 0 or len(data2_clean) == 0: print("Not enough data for ROC after cleaning."); return np.nan, np.nan, np.nan, np.nan, np.nan

    y_scores = np.concatenate((data1_clean, data2_clean))
    cm_display_labels_for_plot = list(labels) 
    if hypothesis == "data1_greater":
        y_true = np.concatenate((np.ones(len(data1_clean)), np.zeros(len(data2_clean))))
        cm_display_labels_for_plot = [labels[1], labels[0]] 
    elif hypothesis == "data2_greater":
        y_true = np.concatenate((np.zeros(len(data1_clean)), np.ones(len(data2_clean))))
    else:  
        mean1 = np.mean(data1_clean) if len(data1_clean) > 0 else -np.inf
        mean2 = np.mean(data2_clean) if len(data2_clean) > 0 else -np.inf
        if mean1 > mean2:
            y_true = np.concatenate((np.ones(len(data1_clean)), np.zeros(len(data2_clean))))
            cm_display_labels_for_plot = [labels[1], labels[0]]
            print("Automatic arrangement: data1 values are higher (class 1), data2 values are lower (class 0)") 
        else:
            y_true = np.concatenate((np.zeros(len(data1_clean)), np.ones(len(data2_clean))))
            print("Automatic arrangement: data1 values are lower (class 0), data2 values are higher (class 1)") 

    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores); roc_auc = auc(fpr, tpr)
    opt_idx = np.argmax(tpr - fpr); opt_thresh = thresholds_roc[opt_idx]
    print(f"Optimal Threshold: {opt_thresh:.4f}")
    print(f"Maximum Youden Index: {(tpr - fpr)[opt_idx]:.4f}")
    
    plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})') 
    plt.plot(fpr[opt_idx], tpr[opt_idx], 'ro', markersize=8, label='Optimal Threshold'); 
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); 
    plt.title('Receiver Operating Characteristic (ROC) Curve'); plt.legend(loc='lower right') 
    if save_plots_prefix: plt.savefig(f"{save_plots_prefix}_roc.png"); plt.close()
    else: plt.show()
    
    y_pred = (y_scores >= opt_thresh).astype(int); cm = confusion_matrix(y_true, y_pred)
    cm_sum_axis1 = cm.sum(axis=1)[:, np.newaxis]; cm_perc = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype('float'), cm_sum_axis1, out=cm_perc, where=cm_sum_axis1!=0); cm_perc *= 100
    plt.figure(figsize=(6, 5)); sns.heatmap(cm_perc, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=cm_display_labels_for_plot, yticklabels=cm_display_labels_for_plot) 
    plt.title("Confusion Matrix (Percentage)"); plt.xlabel("Predicted"); plt.ylabel("True") 
    if save_plots_prefix: plt.savefig(f"{save_plots_prefix}_cm.png"); plt.close()
    else: plt.show()
        
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sens_class1 = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0.0
    spec_class0 = np.sum((y_pred == 0) & (y_true == 0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0.0
    print(f"F1 Score: {f1:.2f}")
    print(f"Sensitivity: {sens_class1:.2f}")
    print(f"Specificity: {spec_class0:.2f}")
    return opt_thresh, sens_class1, spec_class0, f1, roc_auc


t_cols = ["t_Area", "t_T-Dist", "t_Compact", "t_Angle"]
qrs_cols = ["qrs_Area", "qrs_T-Dist", "qrs_Compact", "qrs_Angle"]
st_cols = ["st_Area", "st_T-Dist", "st_Compact", "st_Angle"]
segment_cols_map = {"T": t_cols, "QRS": qrs_cols, "ST": st_cols}
features_to_analyze_short = ["Area", "T-Dist", "Compact", "Angle"] 

feature_analysis_config = {
    "default": {"hypothesis": "data2_greater", "remove_outliers": False},
    "t_Area": {"hypothesis": "data2_greater", "remove_outliers": True}, "t_Compact": {"hypothesis": "data2_greater", "remove_outliers": False}, "t_T-Dist": {"hypothesis": "data2_greater", "remove_outliers": True},
    "qrs_Area": {"hypothesis": "data2_greater", "remove_outliers": True}, "qrs_Compact": {"hypothesis": "data2_greater", "remove_outliers": True}, "qrs_T-Dist": {"hypothesis": "data2_greater", "remove_outliers": True},
    "st_Area": {"hypothesis": "data2_greater", "remove_outliers": True}, "st_Compact": {"hypothesis": "data2_greater", "remove_outliers": True}, "st_T-Dist": {"hypothesis": "data2_greater", "remove_outliers": True},
    "t_Angle": {"hypothesis": "not_equal", "remove_outliers": False}, "qrs_Angle": {"hypothesis": "not_equal", "remove_outliers": False}, "st_Angle": {"hypothesis": "not_equal", "remove_outliers": False},
}

# %% [markdown]
# ## Load and Process All `result.csv` Files

# %%
arvc_map = {(pid, run_id): run_data["ARVC"] for pid, p_data in patient_meta_data.items() for run_id, run_data in p_data.get("runs", {}).items()}
result_files_pattern = os.path.join(RESULTS_BASE_DIR, "*", "result.csv")
result_files = glob.glob(result_files_pattern)
if not result_files: print(f"Warning: No 'result.csv' files found: {result_files_pattern}")

all_sensor_data_dfs = {}
sensor_file_paths = {} 
for f_path in result_files:
    sensor_projection_name = os.path.basename(os.path.dirname(f_path))
    try:
        df_sensor = pd.read_csv(f_path)
        df_sensor["ARVC"] = df_sensor.apply(lambda row: arvc_map.get((row["patient"], str(row["run"])), None), axis=1)
        if df_sensor["ARVC"].isna().sum() > 0: print(f"Warning: {sensor_projection_name} - {df_sensor['ARVC'].isna().sum()} rows missing ARVC after map!")
        df_sensor.dropna(subset=["ARVC"], inplace=True)
        df_sensor["ARVC"] = df_sensor["ARVC"].astype(bool)
        if not df_sensor.empty:
            all_sensor_data_dfs[sensor_projection_name] = df_sensor
            sensor_file_paths[sensor_projection_name] = f_path 
            print(f"Loaded {sensor_projection_name}: {len(df_sensor)} records with ARVC info.")
        else: print(f"Warning: {sensor_projection_name} - No data after ARVC map/NA removal.")
    except Exception as e: print(f"Error loading/processing {f_path}: {e}")

# %% [markdown]
# ## Analysis for Individual Sensors/Projections

# %%
overall_individual_analysis_records = []

for sensor_name, df_sensor_iter in all_sensor_data_dfs.items():
    print(f"\n--- Analyzing Sensor/Projection: {sensor_name} ---")
    
    sensor_original_file_path = sensor_file_paths[sensor_name]
    sensor_base_dir = os.path.dirname(sensor_original_file_path) 
    sensor_plots_dir = os.path.join(sensor_base_dir, "Generated_Plots")
    sensor_tables_dir = os.path.join(sensor_base_dir, "Generated_Tables")
    os.makedirs(sensor_plots_dir, exist_ok=True)
    os.makedirs(sensor_tables_dir, exist_ok=True)
    
    current_sensor_analysis_records = []

    for segment_label, cols_list_for_segment in segment_cols_map.items():
        for feature_short_name in features_to_analyze_short:
            full_col_name = f"{segment_label.lower()}_{feature_short_name}"
            if full_col_name not in df_sensor_iter.columns: continue
            print(f"\n-- Feature: {segment_label} {feature_short_name} ({full_col_name}) --")

            feature_data_series = pd.to_numeric(df_sensor_iter[full_col_name], errors='coerce')
            valid_indices = feature_data_series.notna()
            feature_data_cleaned_vals = feature_data_series[valid_indices].values
            arvc_labels_cleaned_vals = df_sensor_iter.loc[valid_indices, "ARVC"].values

            arvc_pos_data = feature_data_cleaned_vals[arvc_labels_cleaned_vals == True]
            arvc_neg_data = feature_data_cleaned_vals[arvc_labels_cleaned_vals == False]

            if len(arvc_pos_data) < 1 or len(arvc_neg_data) < 1:
                print(f"Skipping {full_col_name} for {sensor_name}: Insufficient data for ARVC+ ({len(arvc_pos_data)}) or Healthy ({len(arvc_neg_data)}) before outlier consideration.")
                continue

            cfg = feature_analysis_config.get(full_col_name, feature_analysis_config["default"])
            plot_file_prefix_for_sensor = os.path.join(sensor_plots_dir, f"{sensor_name}_{segment_label}_{feature_short_name}")

            opt_res = determine_optimal_threshold(arvc_pos_data, arvc_neg_data, hypothesis=cfg["hypothesis"], labels=("ARVC", "Healthy"), remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_file_prefix_for_sensor)
            ttest_res = perform_t_test(arvc_pos_data, arvc_neg_data, name=f"{segment_label} {feature_short_name}", hypothesis=cfg["hypothesis"], threshold=opt_res[0], labels=("ARVC", "Healthy"), remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_file_prefix_for_sensor)
            
            if cfg["hypothesis"] == "data1_greater": pos_class_roc = "ARVC"
            elif cfg["hypothesis"] == "data2_greater": pos_class_roc = "Healthy"
            else: 
                d1_comp = remove_outliers_iqr(arvc_pos_data) if cfg["remove_outliers"] else np.asarray(arvc_pos_data)
                d2_comp = remove_outliers_iqr(arvc_neg_data) if cfg["remove_outliers"] else np.asarray(arvc_neg_data)
                mean1_comp = np.mean(d1_comp) if len(d1_comp) > 0 else -np.inf
                mean2_comp = np.mean(d2_comp) if len(d2_comp) > 0 else -np.inf
                pos_class_roc = "ARVC" if mean1_comp > mean2_comp else "Healthy"
            
            record = {
                "source": sensor_name, "segment": segment_label, "feature": feature_short_name, "column_name": full_col_name,
                "hypothesis_tested": cfg["hypothesis"], "outliers_removed": cfg["remove_outliers"],
                "p_value": ttest_res[1], "t_statistic": ttest_res[0], "significant_ttest (p<0.05)": ttest_res[2],
                "optimal_threshold": opt_res[0], "roc_auc": opt_res[4], 
                "roc_positive_class_for_metrics": pos_class_roc, 
                "sensitivity_of_roc_pos_class": opt_res[1], 
                "specificity_for_other_class": opt_res[2], 
                "f1_score_of_roc_pos_class": opt_res[3],   
                "n_arvc_initial": len(arvc_pos_data), "n_healthy_initial": len(arvc_neg_data),
                "plot_roc_path": f"{plot_file_prefix_for_sensor}_roc.png", "plot_cm_path": f"{plot_file_prefix_for_sensor}_cm.png",
                "plot_boxplot_path": f"{plot_file_prefix_for_sensor}_boxplot.png", "plot_ttest_dist_path": f"{plot_file_prefix_for_sensor}_tdist.png"
            }
            current_sensor_analysis_records.append(record)
            overall_individual_analysis_records.append(record) 

    if current_sensor_analysis_records:
        df_current_sensor_summary = pd.DataFrame(current_sensor_analysis_records)
        sensor_summary_path = os.path.join(sensor_tables_dir, f"{sensor_name}_analysis_summary.csv")
        df_current_sensor_summary.to_csv(sensor_summary_path, index=False)
        print(f"\nSensor-specific analysis summary for {sensor_name} saved to {sensor_summary_path}")

if overall_individual_analysis_records:
    df_overall_individual_summary = pd.DataFrame(overall_individual_analysis_records)
    overall_summary_path = os.path.join(OVERALL_TABLES_DIR, "all_sensors_individual_features_summary.csv")
    df_overall_individual_summary.to_csv(overall_summary_path, index=False)
    print(f"\nOverall summary of individual sensor analyses saved to {overall_summary_path}")
else:
    print("\nNo individual sensor analysis was performed or recorded.")


# %% [markdown]
# ## Aggregated Analysis (All XY, All YZ)

# %%
print("\n\n--- AGGREGATED ANALYSIS ---")
projection_dfs_to_aggregate = {"xy": [], "yz": []}
for sensor_name_iter, df_sensor_iter in all_sensor_data_dfs.items():
    if sensor_name_iter.endswith("_xy"): projection_dfs_to_aggregate["xy"].append(df_sensor_iter)
    elif sensor_name_iter.endswith("_yz"): projection_dfs_to_aggregate["yz"].append(df_sensor_iter)

aggregated_analysis_records = []
for proj_suffix, dfs_list in projection_dfs_to_aggregate.items():
    if not dfs_list: print(f"No data for aggregated {proj_suffix.upper()} projection."); continue
    df_aggregated = pd.concat(dfs_list, ignore_index=True)
    source_name_agg = f"aggregated_{proj_suffix}" 
    print(f"\n--- Analyzing {source_name_agg.upper()} Data ({len(df_aggregated)} records) ---")
    
    for segment_label, cols_list_for_segment in segment_cols_map.items():
        for feature_short_name in features_to_analyze_short:
            full_col_name = f"{segment_label.lower()}_{feature_short_name}"
            if full_col_name not in df_aggregated.columns: continue
            print(f"\n-- Feature: {segment_label} {feature_short_name} ({full_col_name}) --")

            feature_data_series = pd.to_numeric(df_aggregated[full_col_name], errors='coerce')
            valid_indices = feature_data_series.notna()
            feature_data_cleaned_vals = feature_data_series[valid_indices].values
            arvc_labels_cleaned_vals = df_aggregated.loc[valid_indices, "ARVC"].values

            arvc_pos_data = feature_data_cleaned_vals[arvc_labels_cleaned_vals == True]
            arvc_neg_data = feature_data_cleaned_vals[arvc_labels_cleaned_vals == False]

            if len(arvc_pos_data) < 1 or len(arvc_neg_data) < 1: 
                print(f"Skipping {full_col_name} for {source_name_agg}: Insufficient data for ARVC+ ({len(arvc_pos_data)}) or Healthy ({len(arvc_neg_data)}).")
                continue
            
            cfg = feature_analysis_config.get(full_col_name, feature_analysis_config["default"])
            plot_file_prefix_aggregated = os.path.join(OVERALL_PLOTS_DIR, f"{source_name_agg}_{segment_label}_{feature_short_name}")

            opt_res = determine_optimal_threshold(arvc_pos_data, arvc_neg_data, hypothesis=cfg["hypothesis"], labels=("ARVC", "Healthy"), remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_file_prefix_aggregated)
            ttest_res = perform_t_test(arvc_pos_data, arvc_neg_data, name=f"{segment_label} {feature_short_name} ({source_name_agg})", hypothesis=cfg["hypothesis"], threshold=opt_res[0], labels=("ARVC", "Healthy"), remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_file_prefix_aggregated)
            
            if cfg["hypothesis"] == "data1_greater": pos_class_roc = "ARVC"
            elif cfg["hypothesis"] == "data2_greater": pos_class_roc = "Healthy"
            else:
                d1_comp = remove_outliers_iqr(arvc_pos_data) if cfg["remove_outliers"] else np.asarray(arvc_pos_data)
                d2_comp = remove_outliers_iqr(arvc_neg_data) if cfg["remove_outliers"] else np.asarray(arvc_neg_data)
                mean1_comp = np.mean(d1_comp) if len(d1_comp) > 0 else -np.inf
                mean2_comp = np.mean(d2_comp) if len(d2_comp) > 0 else -np.inf
                pos_class_roc = "ARVC" if mean1_comp > mean2_comp else "Healthy"
            
            aggregated_analysis_records.append({
                "source": source_name_agg, "segment": segment_label, "feature": feature_short_name, "column_name": full_col_name,
                "hypothesis_tested": cfg["hypothesis"], "outliers_removed": cfg["remove_outliers"],
                "p_value": ttest_res[1], "t_statistic": ttest_res[0], "significant_ttest (p<0.05)": ttest_res[2],
                "optimal_threshold": opt_res[0], "roc_auc": opt_res[4], 
                "roc_positive_class_for_metrics": pos_class_roc,
                "sensitivity_of_roc_pos_class": opt_res[1], 
                "specificity_for_other_class": opt_res[2], 
                "f1_score_of_roc_pos_class": opt_res[3],   
                "n_arvc_initial": len(arvc_pos_data), "n_healthy_initial": len(arvc_neg_data),
                "plot_roc_path": f"{plot_file_prefix_aggregated}_roc.png", "plot_cm_path": f"{plot_file_prefix_aggregated}_cm.png",
                "plot_boxplot_path": f"{plot_file_prefix_aggregated}_boxplot.png", "plot_ttest_dist_path": f"{plot_file_prefix_aggregated}_tdist.png"
            })

if aggregated_analysis_records:
    df_aggregated_summary = pd.DataFrame(aggregated_analysis_records)
    summary_path = os.path.join(OVERALL_TABLES_DIR, "aggregated_projection_analysis_summary.csv")
    df_aggregated_summary.to_csv(summary_path, index=False)
    print(f"\nAggregated projection analysis summary saved to {summary_path}")
else:
    print("\nNo aggregated analysis was performed or recorded.")

print("\n\n--- SCRIPT FINISHED ---")
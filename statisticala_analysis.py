# =============================================================================
# Library Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_ind, norm
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import json
import os
import glob

# =============================================================================
# Global Configuration and Styling
# =============================================================================

# Apply a clean, white grid style to all Seaborn plots.
sns.set(style="whitegrid")

# Define base output directories for storing results.
RESULTS_BASE_DIR = "Results"
# Directory for plots summarizing all data (e.g., demographics, aggregated analyses).
OVERALL_PLOTS_DIR = os.path.join(RESULTS_BASE_DIR, "Overall_Generated_Plots")
# Directory for tables/CSVs summarizing all data.
OVERALL_TABLES_DIR = os.path.join(RESULTS_BASE_DIR, "Overall_Generated_Tables")

# Create the output directories if they do not already exist.
os.makedirs(OVERALL_PLOTS_DIR, exist_ok=True)
os.makedirs(OVERALL_TABLES_DIR, exist_ok=True)

# Number of Monte Carlo iterations for uncertainty analysis.
# A higher number (e.g., 1000) provides more robust results but takes longer.
N_MC_ITERATIONS = 100
# Confidence level for calculating confidence intervals from MC simulations.
CONFIDENCE_LEVEL = 0.95

# Define the feature columns associated with different ECG wave segments.
t_cols = ["t_Area", "t_T-Dist", "t_Compact", "t_Angle"]
qrs_cols = ["qrs_Area", "qrs_T-Dist", "qrs_Compact", "qrs_Angle"]
st_cols = ["st_Area", "st_T-Dist", "st_Compact", "st_Angle"]
segment_cols_map = {"T": t_cols, "QRS": qrs_cols, "ST": st_cols}

# Short names for features, used for looping and labeling.
features_to_analyze_short = ["Area", "T-Dist", "Compact", "Angle"]

# Configuration for statistical analysis of each feature.
# Defines the hypothesis for the t-test and whether to remove outliers.
# 'data2_greater': Healthy > ARVC (e.g., area is expected to be larger in healthy patients).
# 'not_equal': Two-tailed test where the direction is not pre-specified.
feature_analysis_config = {
    "default": {"hypothesis": "not_equal", "remove_outliers": False},
    "t_Area": {"hypothesis": "not_equal", "remove_outliers": True},
    "t_Compact": {"hypothesis": "not_equal", "remove_outliers": False},
    "t_T-Dist": {"hypothesis": "not_equal", "remove_outliers": True},
    "qrs_Area": {"hypothesis": "not_equal", "remove_outliers": True},
    "qrs_Compact": {"hypothesis": "not_equal", "remove_outliers": True},
    "qrs_T-Dist": {"hypothesis": "not_equal", "remove_outliers": True},
    "st_Area": {"hypothesis": "not_equal", "remove_outliers": True},
    "st_Compact": {"hypothesis": "not_equal", "remove_outliers": True},
    "st_T-Dist": {"hypothesis": "not_equal", "remove_outliers": True},
    "t_Angle": {"hypothesis": "not_equal", "remove_outliers": False},
    "qrs_Angle": {"hypothesis": "not_equal", "remove_outliers": False},
    "st_Angle": {"hypothesis": "not_equal", "remove_outliers": False},
}


# =============================================================================
# Demographics Data Loading and Preparation
# =============================================================================

# Load patient metadata from the setup JSON file.
with open("Data/setup.json") as f:
    patient_meta_data = json.load(f)

# Process the raw metadata into a structured list of records.
demographic_records = []
for pid, p_data in patient_meta_data.items():
    arvc_status_patient = None
    # Extract ARVC status from the first available run for the patient.
    if p_data.get("runs"):
        first_run_id = next(iter(p_data["runs"]))
        arvc_status_patient = p_data["runs"][first_run_id].get("ARVC")

    # Create a dictionary for the patient's demographic information.
    record = {
        "patient": pid,
        "gender": p_data.get("gender", "unknown") or "unknown", # Handle None or empty strings
        "height": float(p_data.get("height")) if p_data.get("height") not in [None, ""] else None,
        "age": int(p_data.get("age")) if p_data.get("age") not in [None, ""] else None,
        "ARVC": arvc_status_patient
    }
    demographic_records.append(record)

# Convert the list of records into a pandas DataFrame.
df_demographics = pd.DataFrame(demographic_records)

# Remove patients where ARVC status could not be determined.
df_demographics.dropna(subset=['ARVC'], inplace=True)

# Define a consistent B&W palette for gender across all plots.
gender_palette_global = {"male": "#555555", "female": "#AAAAAA", "unknown": "#E0E0E0"}


# =============================================================================
# Demographics Analysis and Plotting Functions
# =============================================================================

def plot_gender_distribution(data, title, save_path=None):
    """
    Generates and saves a bar plot showing the gender distribution.

    Args:
        data (pd.DataFrame): DataFrame containing a 'gender' column.
        title (str): The title for the plot.
        save_path (str, optional): Path to save the plot image. If None, shows the plot.
    """
    # Count occurrences of each gender, ensuring all categories are present.
    counts = data["gender"].value_counts().reindex(["male", "female", "unknown"]).fillna(0)

    # Create the figure and axes for the plot.
    plt.figure(figsize=(7, 5))
    # Generate the bar plot using the B&W palette and add black edges for clarity.
    palette = [gender_palette_global.get(g, "#cccccc") for g in counts.index]
    ax = sns.barplot(x=counts.index, y=counts.values, palette=palette, edgecolor='black')

    # Set plot titles and labels.
    plt.title(title)
    plt.ylabel("Number of Patients")
    plt.xlabel("Gender")

    # Add text labels on top of each bar to show the count.
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values, default=0) * 0.03, str(int(v)),
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Adjust y-axis limits for better visualization and apply tight layout.
    plt.ylim(0, max(counts.values, default=1) * 1.20)
    plt.tight_layout()

    # Save or display the plot.
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# --- Plotting Gender Distributions ---
print("\n--- GENDER DISTRIBUTION ---")
plot_gender_distribution(
    df_demographics,
    "Gender Distribution (All Patients)",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_all.png")
)
plot_gender_distribution(
    df_demographics[df_demographics["ARVC"] == True],
    "Gender Distribution (ARVC)",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_arvc.png")
)
plot_gender_distribution(
    df_demographics[df_demographics["ARVC"] == False],
    "Gender Distribution (Healthy)",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_healthy.png")
)


def plot_hist_and_stats(data, column, title, color, save_path=None):
    """
    Calculates summary statistics, prints them, and plots a histogram for a given column.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): The column to analyze (e.g., 'age', 'height').
        title (str): The title for the plot.
        color (str): The color for the histogram (will be converted to grayscale).
        save_path (str, optional): Path to save the plot. If None, shows the plot.

    Returns:
        dict: A dictionary of calculated statistics (mean, std, count, median).
    """
    # Remove missing values for the specified column.
    valid_data = data[column].dropna()
    # Initialize stats dictionary.
    stats = {"mean": np.nan, "std": np.nan, "count": len(valid_data), "median": np.nan}

    if not valid_data.empty:
        # Calculate statistics.
        stats["mean"] = valid_data.mean()
        stats["std"] = valid_data.std()
        stats["median"] = valid_data.median()
        print(f"{title}: Mean = {stats['mean']:.2f}, Median = {stats['median']:.2f}, "
              f"Std = {stats['std']:.2f}, N = {stats['count']}")

        # Create the plot.
        plt.figure(figsize=(8, 5))
        sns.histplot(valid_data, kde=True, color=color, bins=10) # `color` is now a shade of gray from the call.

        # Add vertical lines for mean and standard deviation using black/gray and different linestyles.
        plt.axvline(stats['mean'], color='black', linestyle='--', label=f'Mean = {stats["mean"]:.2f}')
        if not np.isnan(stats['std']):
            plt.axvline(stats['mean'] + stats['std'], color='dimgray', linestyle=':', label=f'+1 SD = {stats["mean"] + stats["std"]:.2f}')
            plt.axvline(stats['mean'] - stats['std'], color='dimgray', linestyle=':', label=f'-1 SD = {stats["mean"] - stats["std"]:.2f}')

        # Set plot titles, labels, and legend.
        plt.title(title)
        plt.xlabel(column.capitalize())
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        # Save or display the plot.
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    else:
        print(f"{title}: No valid data.")

    return stats

# --- Plotting Age and Height Distributions ---
demographics_summary_stats = {}
print("\n--- AGE & HEIGHT (All Patients) ---")
# Use shades of gray for the histogram color argument
demographics_summary_stats["age_all"] = plot_hist_and_stats(
    df_demographics, "age", "Age Distribution (All)", "darkgray",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_age_all.png")
)
demographics_summary_stats["height_all"] = plot_hist_and_stats(
    df_demographics, "height", "Height Distribution (All)", "darkgray",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_height_all.png")
)

print("\n--- AGE & HEIGHT (ARVC Positive) ---")
demographics_summary_stats["age_arvc"] = plot_hist_and_stats(
    df_demographics[df_demographics["ARVC"] == True], "age", "Age Distribution (ARVC)", "darkgray",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_age_arvc.png")
)
demographics_summary_stats["height_arvc"] = plot_hist_and_stats(
    df_demographics[df_demographics["ARVC"] == True], "height", "Height Distribution (ARVC)", "darkgray",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_height_arvc.png")
)

print("\n--- AGE & HEIGHT (ARVC Negative) ---")
demographics_summary_stats["age_healthy"] = plot_hist_and_stats(
    df_demographics[df_demographics["ARVC"] == False], "age", "Age Distribution (Healthy)", "darkgray",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_age_healthy.png")
)
demographics_summary_stats["height_healthy"] = plot_hist_and_stats(
    df_demographics[df_demographics["ARVC"] == False], "height", "Height Distribution (Healthy)", "darkgray",
    save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_height_healthy.png")
)

# Save the collected demographic stats to a JSON file.
summary_path = os.path.join(OVERALL_TABLES_DIR, "demographics_summary.json")
with open(summary_path, "w") as f:
    json.dump(demographics_summary_stats, f, indent=4)
print(f"\nDemographics summary stats saved to {summary_path}")


# =============================================================================
# Core Statistical Analysis Functions
# =============================================================================

def remove_outliers_iqr(data):
    """
    Removes outliers from a 1D array using the Interquartile Range (IQR) method.

    Args:
        data (array-like): The input data.

    Returns:
        np.ndarray: Data with outliers removed.
    """
    data_arr = np.asarray(data)
    if data_arr.size == 0:
        return data_arr

    # Calculate the first (Q1) and third (Q3) quartiles.
    q1, q3 = np.percentile(data_arr, [25, 75])
    # Calculate the Interquartile Range (IQR).
    iqr = q3 - q1
    # Define the lower and upper bounds for outlier detection.
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Return the data within the non-outlier range.
    return data_arr[(data_arr >= lower_bound) & (data_arr <= upper_bound)]


def _perform_single_t_test_run(data1_sample, data2_sample, hypothesis, labels):
    """
    Helper function to perform a single Welch's t-test for one MC iteration.

    Args:
        data1_sample (np.ndarray): Sample data for group 1.
        data2_sample (np.ndarray): Sample data for group 2.
        hypothesis (str): The hypothesis ('data1_greater', 'data2_greater', 'not_equal').
        labels (tuple): Names of the two groups.

    Returns:
        tuple: (t-statistic, p-value, is_significant).
    """
    # Ensure there is enough data in both samples to perform a test.
    if len(data1_sample) < 2 or len(data2_sample) < 2:
        return np.nan, np.nan, False

    # Perform Welch's t-test (assumes unequal variances).
    t_stat, p_val_two_tailed = ttest_ind(data1_sample, data2_sample, equal_var=False, nan_policy='omit')

    if np.isnan(t_stat) or np.isnan(p_val_two_tailed):
        return np.nan, np.nan, False

    # Calculate one-tailed p-value from the two-tailed result based on the hypothesis.
    if hypothesis == "data1_greater":
        p_value = p_val_two_tailed / 2 if t_stat > 0 else 1 - (p_val_two_tailed / 2)
    elif hypothesis == "data2_greater":
        p_value = p_val_two_tailed / 2 if t_stat < 0 else 1 - (p_val_two_tailed / 2)
    else:  # 'not_equal'
        p_value = p_val_two_tailed

    return t_stat, p_value, p_value < 0.05


def perform_t_test(data1_nominal, data2_nominal, data1_unc=None, data2_unc=None, name="",
                   hypothesis="data1_greater", threshold=None, labels=("Group 1", "Group 2"),
                   remove_outliers=False, save_plots_prefix=None):
    """
    Performs a t-test on nominal data and optionally a Monte Carlo simulation if uncertainties are provided.
    Generates plots for the t-distribution and violin plots with MC median error bars.

    Args:
        data1_nominal, data2_nominal (array-like): Nominal data for group 1 and 2.
        data1_unc, data2_unc (array-like, optional): Uncertainties for group 1 and 2.
        name (str): Name of the feature being tested, for titles.
        hypothesis (str): The hypothesis type.
        threshold (float, optional): An optimal threshold to display on the plot.
        labels (tuple): Names for the two groups.
        remove_outliers (bool): If True, remove outliers using IQR method.
        save_plots_prefix (str, optional): Prefix for saving plot files.

    Returns:
        tuple: Contains results for (t-stat, p-value, significance). Each result is a
               tuple of (mean, ci_lower, ci_upper). For nominal-only, CIs are NaN.
    """
    data1_nom_arr = np.asarray(data1_nominal)
    data2_nom_arr = np.asarray(data2_nominal)

    # Clean data for plotting and nominal analysis by optionally removing outliers.
    if remove_outliers:
        data1_plot_clean = remove_outliers_iqr(data1_nom_arr)
        data2_plot_clean = remove_outliers_iqr(data2_nom_arr)
    else:
        data1_plot_clean = data1_nom_arr
        data2_plot_clean = data2_nom_arr

    print(f"T-Test for {name} (Nominal Values): Orig D1:{len(data1_nom_arr)}, D2:{len(data2_nom_arr)}. "
          f"Clean D1:{len(data1_plot_clean)}, D2:{len(data2_plot_clean)}")

    # --- Nominal T-Test ---
    nominal_t_stat, nominal_p_value, nominal_significant = np.nan, np.nan, False
    group1_plot_label, group2_plot_label = labels[0], labels[1]
    hypothesis_text_for_plot = f"{labels[0]} ≠ {labels[1]}" # Default for 'not_equal'

    if len(data1_plot_clean) >= 2 and len(data2_plot_clean) >= 2:
        nominal_t_stat, nominal_p_val_two_tailed = ttest_ind(
            data1_plot_clean, data2_plot_clean, equal_var=False, nan_policy='omit'
        )
        if not (np.isnan(nominal_t_stat) or np.isnan(nominal_p_val_two_tailed)):
            # Adjust p-value and labels based on the one-tailed hypothesis.
            if hypothesis == "data1_greater":
                group1_plot_label = f'{labels[0]} (Higher)'
                group2_plot_label = f'{labels[1]} (Lower)'
                nominal_p_value = nominal_p_val_two_tailed / 2 if nominal_t_stat > 0 else 1 - (nominal_p_val_two_tailed / 2)
                hypothesis_text_for_plot = f"{labels[0]} > {labels[1]}"
            elif hypothesis == "data2_greater":
                group1_plot_label = f'{labels[0]} (Lower)'
                group2_plot_label = f'{labels[1]} (Higher)'
                nominal_p_value = nominal_p_val_two_tailed / 2 if nominal_t_stat < 0 else 1 - (nominal_p_val_two_tailed / 2)
                hypothesis_text_for_plot = f"{labels[0]} < {labels[1]}"
            else:  # 'not_equal'
                nominal_p_value = nominal_p_val_two_tailed

            nominal_significant = nominal_p_value < 0.05
            print(f"Hypothesis: {hypothesis_text_for_plot.replace(labels[0],'data1').replace(labels[1],'data2')}, T-statistic: {nominal_t_stat:.4f}")
            if hypothesis == "not_equal":
                print(f"Two-tailed p-value: {nominal_p_value:.4f}")
            else:
                print(f"One-tailed p-value: {nominal_p_value:.4f}")
                print(f"Two-tailed p-value: {nominal_p_val_two_tailed:.4f}")
            print(f"Result is statistically {'significant' if nominal_significant else 'not significant'} (p-value = {nominal_p_value:.4f})")

            # --- Plot t-Distribution ---
            df_freedom = len(data1_plot_clean) + len(data2_plot_clean) - 2
            if df_freedom > 0:
                plt.figure(figsize=(10, 6))
                x_t = np.linspace(-4, 4, 1000)
                y_t = t.pdf(x_t, df_freedom)
                plt.plot(x_t, y_t, label="t-distribution", color='black')

                # Define conditions for shading the p-value area.
                if hypothesis == "data1_greater":
                    fill_cond = x_t > nominal_t_stat
                    fill_label = f'p-value Area (t > {nominal_t_stat:.2f})'
                elif hypothesis == "data2_greater":
                    fill_cond = x_t < nominal_t_stat
                    fill_label = f'p-value Area (t < {nominal_t_stat:.2f})'
                else: # 'not_equal'
                    fill_cond = (x_t > abs(nominal_t_stat)) | (x_t < -abs(nominal_t_stat))
                    fill_label = f'p-value Area (|t| > {abs(nominal_t_stat):.2f})'

                plt.fill_between(x_t, 0, y_t, where=fill_cond, color='lightgray', alpha=0.8, label=fill_label)
                plt.axvline(x=nominal_t_stat, color='black', linestyle='--', label=f't-statistic: {nominal_t_stat:.2f}')
                plt.xlabel("t-value")
                plt.ylabel("Probability Density")
                plt.title(f"t-Distribution with Shaded p-value Area ({hypothesis_text_for_plot.replace(labels[0],'data1').replace(labels[1],'data2')})")
                plt.legend()
                if save_plots_prefix:
                    plt.savefig(f"{save_plots_prefix}_tdist.png")
                    plt.close()
                else:
                    plt.show()
        else:
            print("Nominal t-test resulted in NaN.")
    else:
        print("Not enough data for nominal t-test after cleaning.")

    # --- Monte Carlo Simulation ---
    mc_t_stats, mc_p_values, mc_medians_g1, mc_medians_g2 = [], [], [], []
    if data1_unc is not None and data2_unc is not None and N_MC_ITERATIONS > 0:
        print(f"\nPerforming MC ({N_MC_ITERATIONS} iter) for t-test of {name}...")
        data1_unc_arr = np.asarray(data1_unc)
        data2_unc_arr = np.asarray(data2_unc)
        for i in range(N_MC_ITERATIONS):
            if i % (N_MC_ITERATIONS // 10) == 0 and i > 0:
                print(f"MC iter {i}...")

            # Generate random samples based on nominal values and uncertainties.
            d1_s = np.random.normal(data1_nom_arr, data1_unc_arr)
            d2_s = np.random.normal(data2_nom_arr, data2_unc_arr)
            d1_sc = remove_outliers_iqr(d1_s) if remove_outliers else d1_s
            d2_sc = remove_outliers_iqr(d2_s) if remove_outliers else d2_s
            
            # Store medians for visualization
            if len(d1_sc) > 0: mc_medians_g1.append(np.median(d1_sc))
            if len(d2_sc) > 0: mc_medians_g2.append(np.median(d2_sc))

            t_s_mc, p_v_mc, _ = _perform_single_t_test_run(d1_sc, d2_sc, hypothesis, labels)

            if not np.isnan(p_v_mc):
                mc_t_stats.append(t_s_mc)
                mc_p_values.append(p_v_mc)

    
    # --- Plot Box Plot with Nominal Data Points and MC Median Error Bars ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data container for plotting
    plot_data_bp = [
        data1_plot_clean if len(data1_plot_clean) > 0 else np.array([]),
        data2_plot_clean if len(data2_plot_clean) > 0 else np.array([])
    ]

    group_labels = [group1_plot_label, group2_plot_label]
    x_labels = [f"{group_labels[i]} (N={len(plot_data_bp[i])})" for i in range(2)]

    # Box plot for nominal data distribution
    sns.boxplot(data=plot_data_bp, ax=ax,
                palette=["#DDDDDD", "#777777"],
                boxprops=dict(edgecolor='k'),
                medianprops=dict(color='k'),
                whiskerprops=dict(color='k'),
                capprops=dict(color='k'),
                showfliers=True)

    # Overlay individual data points using stripplot
    for i, group_data in enumerate(plot_data_bp):
        if len(group_data) > 0:
            sns.stripplot(y=group_data, x=[i] * len(group_data), ax=ax,
                        color='black', size=4, alpha=0.5, jitter=0.1)

    # Add dummy scatter point for legend
    scatter_handle = ax.scatter([], [], color='black', alpha=0.5, s=30, label='Individual Data Points')

    # Plot MC medians with confidence intervals
    if mc_medians_g1 and mc_medians_g2:
        low_p = (1 - CONFIDENCE_LEVEL) / 2 * 100
        high_p = (1 + CONFIDENCE_LEVEL) / 2 * 100

        def plot_mc_median(x, mc_data, label=None):
            mean_med = np.mean(mc_data)
            low_ci, high_ci = np.percentile(mc_data, [low_p, high_p])
            y_err = [[mean_med - low_ci], [high_ci - mean_med]]
            ax.errorbar(x=x, y=mean_med, yerr=y_err, fmt='kx', markersize=8, capsize=5,
                        label=label)

        plot_mc_median(0, mc_medians_g1, label=f'MC Median ({CONFIDENCE_LEVEL*100:.0f}% CI)')
        plot_mc_median(1, mc_medians_g2)

    # Add threshold line if applicable
    if threshold is not None and not np.isnan(threshold):
        ax.axhline(y=threshold, color='black', linestyle='-.', label=f'Optimal Threshold: {threshold:.2f}')

    # Label and format axes
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Value")
    ax.set_title(f"Box Plot of Nominal Data with MC Medians for {name}")

    # Clean and deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    # Save or display
    plt.tight_layout()
    if save_plots_prefix:
        plt.savefig(f"{save_plots_prefix}_boxplot.png", dpi=300)
        plt.close()
    else:
        plt.show()

    # --- Process and Return MC Results ---
    if mc_p_values:
        p_low, p_high = np.percentile(mc_p_values, [(1 - CONFIDENCE_LEVEL) / 2 * 100, (1 + CONFIDENCE_LEVEL) / 2 * 100])
        t_low, t_high = np.percentile(mc_t_stats, [(1 - CONFIDENCE_LEVEL) / 2 * 100, (1 + CONFIDENCE_LEVEL) / 2 * 100])
        m_p, med_p = np.mean(mc_p_values), np.median(mc_p_values)
        m_t, med_t = np.mean(mc_t_stats), np.median(mc_t_stats)
        print(f"\nMC T-Test: Mean P-val: {m_p:.4f} (Med:{med_p:.4f}) CI:[{p_low:.4f},{p_high:.4f}]. "
            f"Mean T-stat: {m_t:.4f} (Med:{med_t:.4f}) CI:[{t_low:.4f},{t_high:.4f}]")
        
        # Format p-values: use scientific notation if < 0.001
        def format_pval(p):
            return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"

        # --- Plot P-value Distribution ---
        if save_plots_prefix:
            plt.figure(figsize=(8, 5))
            sns.histplot(mc_p_values, kde=True, bins=30, color='darkgray')
            plt.title(f"MC Distribution of P-values for {name}")
            plt.xlabel("P-value")
            

            plt.axvline(nominal_p_value, color='black', linestyle='--',
                        label=f'Nominal P ({format_pval(nominal_p_value)})')
            plt.axvline(m_p, color='dimgray', linestyle=':',
                        label=f'Mean P ({format_pval(m_p)})')

            plt.legend()
            plt.savefig(f"{save_plots_prefix}_p_value_mc_dist.png")
            plt.close()

        return (m_t, t_low, t_high), (m_p, p_low, p_high), (m_p < 0.05)
    else: # MC was not run or failed
        # Fallback to nominal results if MC fails.
        return (nominal_t_stat, np.nan, np.nan), (nominal_p_value, np.nan, np.nan), nominal_significant
    


def _determine_single_optimal_threshold_run(data1_sample, data2_sample, hypothesis, labels):
    """
    Helper to find the optimal classification threshold for one MC iteration.

    Args:
        data1_sample (np.ndarray): Sample data for group 1.
        data2_sample (np.ndarray): Sample data for group 2.
        hypothesis (str): The hypothesis defining which group has higher values.
        labels (tuple): Names of the two groups.

    Returns:
        tuple: (optimal_threshold, sensitivity, specificity, f1_score, roc_auc).
    """
    if len(data1_sample) == 0 or len(data2_sample) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # y_scores are the feature values. y_true are the binary class labels (0 or 1).
    y_scores = np.concatenate((data1_sample, data2_sample))

    # Assign true labels. Class 1 is the group expected to have higher values.
    if hypothesis == "data1_greater":
        y_true = np.concatenate((np.ones(len(data1_sample)), np.zeros(len(data2_sample))))
    elif hypothesis == "data2_greater":
        y_true = np.concatenate((np.zeros(len(data1_sample)), np.ones(len(data2_sample))))
    else:  # 'not_equal', determine direction from sample means.
        mean1 = np.mean(data1_sample) if len(data1_sample) > 0 else -np.inf
        mean2 = np.mean(data2_sample) if len(data2_sample) > 0 else -np.inf
        if mean1 > mean2:
            y_true = np.concatenate((np.ones(len(data1_sample)), np.zeros(len(data2_sample))))
        else:
            y_true = np.concatenate((np.zeros(len(data1_sample)), np.ones(len(data2_sample))))

    # Calculate ROC curve and AUC.
    fpr, tpr, thresh_roc = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    if len(thresh_roc) == 0 or len(tpr) == 0 or len(fpr) == 0:
        return np.nan, np.nan, np.nan, np.nan, roc_auc

    # Find optimal threshold using Youden's J-statistic (maximizes tpr - fpr).
    opt_idx = np.argmax(tpr - fpr)
    opt_thresh = thresh_roc[opt_idx]

    # Calculate metrics at the optimal threshold.
    y_pred = (y_scores >= opt_thresh).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sensitivity = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0.0
    specificity = np.sum((y_pred == 0) & (y_true == 0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0.0

    return opt_thresh, sensitivity, specificity, f1, roc_auc

def determine_optimal_threshold(data1_nominal, data2_nominal, data1_unc=None, data2_unc=None,
                                hypothesis="data2_greater", labels=("Class 0", "Class 1"),
                                remove_outliers=False, save_plots_prefix=None):
    """
    Determines optimal classification threshold and metrics (AUC, F1, etc.),
    with optional Monte Carlo simulation if uncertainties are provided.
    Generates ROC curve and confusion matrix plots.

    Args:
        data1_nominal, data2_nominal (array-like): Nominal data for group 1 and 2.
        data1_unc, data2_unc (array-like, optional): Uncertainties for group 1 and 2.
        hypothesis (str): The hypothesis type.
        labels (tuple): Names for the two groups (Class 0, Class 1).
        remove_outliers (bool): If True, remove outliers using IQR method.
        save_plots_prefix (str, optional): Prefix for saving plot files.

    Returns:
        dict: A dictionary of classification metrics, each a tuple of (mean, ci_lower, ci_upper).
    """
    data1_nom_arr = np.asarray(data1_nominal)
    data2_nom_arr = np.asarray(data2_nominal)
    print(f"Optimal Threshold for {labels[0]} vs {labels[1]} (Nominal Values): d1:{len(data1_nom_arr)}, d2:{len(data2_nom_arr)}")

    # Clean data for plotting and nominal analysis.
    d1_plot_c = remove_outliers_iqr(data1_nom_arr) if remove_outliers else data1_nom_arr
    d2_plot_c = remove_outliers_iqr(data2_nom_arr) if remove_outliers else data2_nom_arr
    print(f"Cleaned d1:{len(d1_plot_c)}, d2:{len(d2_plot_c)}")

    # --- Nominal ROC Analysis ---
    nom_opt_t, nom_sens, nom_spec, nom_f1, nom_auc = (np.nan,) * 5
    cm_disp_labels_plot = list(labels) # Default for data2_greater and some not_equal

    if not (len(d1_plot_c) == 0 or len(d2_plot_c) == 0):
        nom_opt_t, nom_sens, nom_spec, nom_f1, nom_auc = _determine_single_optimal_threshold_run(d1_plot_c, d2_plot_c, hypothesis, labels)

        if not np.isnan(nom_opt_t):
            # The positive class is the one with higher values. For plotting, this is Class 1.
            # If data1 is higher, we need to swap the labels for the confusion matrix.
            if hypothesis == "data1_greater":
                cm_disp_labels_plot = [labels[1], labels[0]]
            elif hypothesis == "not_equal":
                m1 = np.mean(d1_plot_c) if len(d1_plot_c) > 0 else -np.inf
                m2 = np.mean(d2_plot_c) if len(d2_plot_c) > 0 else -np.inf
                if m1 > m2:
                    cm_disp_labels_plot = [labels[1], labels[0]]
                    print("Auto-detected direction: data1 has a higher mean (becomes class 1)")
                else:
                    print("Auto-detected direction: data2 has a higher mean (becomes class 1)")

            print(f"Optimal Thresh:{nom_opt_t:.4f}, F1:{nom_f1:.2f}, Sens:{nom_sens:.2f}, Spec:{nom_spec:.2f}, AUC:{nom_auc:.2f}")

            # Prepare data for plotting ROC and CM.
            y_s_plot = np.concatenate((d1_plot_c, d2_plot_c))
            if hypothesis == "data1_greater":
                y_t_plot = np.concatenate((np.ones(len(d1_plot_c)), np.zeros(len(d2_plot_c))))
            elif hypothesis == "data2_greater":
                y_t_plot = np.concatenate((np.zeros(len(d1_plot_c)), np.ones(len(d2_plot_c))))
            else: # not_equal
                m1 = np.mean(d1_plot_c) if len(d1_plot_c) > 0 else -np.inf
                m2 = np.mean(d2_plot_c) if len(d2_plot_c) > 0 else -np.inf
                y_t_plot = np.concatenate((np.ones(len(d1_plot_c)), np.zeros(len(d2_plot_c)))) if m1 > m2 else \
                           np.concatenate((np.zeros(len(d1_plot_c)), np.ones(len(d2_plot_c))))

            # --- Plot ROC Curve ---
            fpr_p, tpr_p, _ = roc_curve(y_t_plot, y_s_plot)
            opt_idx_p = np.argmax(tpr_p - fpr_p) if len(tpr_p) > 0 else 0

            plt.figure(figsize=(8, 6))
            plt.plot(fpr_p, tpr_p, color='black', lw=2, label=f'ROC (AUC={nom_auc:.2f})')
            if len(fpr_p) > opt_idx_p:
                plt.plot(fpr_p[opt_idx_p], tpr_p[opt_idx_p], 'ko', markersize=8, label='Optimal Threshold')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            if save_plots_prefix:
                plt.savefig(f"{save_plots_prefix}_roc.png")
                plt.close()
            else:
                plt.show()

            # --- Plot Confusion Matrix ---
            y_pred_p = (y_s_plot >= nom_opt_t).astype(int)
            cm_p = confusion_matrix(y_t_plot, y_pred_p)
            # Normalize to percentages.
            cm_sum_rows = cm_p.sum(axis=1)[:, np.newaxis]
            cm_perc_p = np.zeros_like(cm_p, dtype=float)
            np.divide(cm_p.astype('float'), cm_sum_rows, out=cm_perc_p, where=cm_sum_rows != 0)
            cm_perc_p *= 100

            plt.figure(figsize=(6, 5))
            # Use a grayscale colormap for the heatmap
            sns.heatmap(cm_perc_p, annot=True, fmt=".2f", cmap="Greys", cbar=False,
                        linecolor='black', linewidths=0.5,
                        xticklabels=cm_disp_labels_plot, yticklabels=cm_disp_labels_plot)
            plt.title("Confusion Matrix (%)")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            if save_plots_prefix:
                plt.savefig(f"{save_plots_prefix}_cm.png")
                plt.close()
            else:
                plt.show()
        else:
            print("Nominal ROC analysis failed (e.g., could not determine threshold).")
    else:
        print("Not enough data for nominal ROC analysis.")

    # --- Monte Carlo Simulation for ROC metrics ---
    if data1_unc is not None and data2_unc is not None and N_MC_ITERATIONS > 0:
        print(f"\nPerforming MC ({N_MC_ITERATIONS} iter) for ROC of {labels[0]} vs {labels[1]}...")
        mc_thresholds, mc_sens, mc_spec, mc_f1s, mc_aucs = [], [], [], [], []
        d1_u_arr, d2_u_arr = np.asarray(data1_unc), np.asarray(data2_unc)

        for i in range(N_MC_ITERATIONS):
            if i % (N_MC_ITERATIONS // 10) == 0 and i > 0:
                print(f"MC iter {i}...")
            # Generate random samples.
            d1_s = np.random.normal(data1_nom_arr, d1_u_arr)
            d2_s = np.random.normal(data2_nom_arr, d2_u_arr)
            d1_sc = remove_outliers_iqr(d1_s) if remove_outliers else d1_s
            d2_sc = remove_outliers_iqr(d2_s) if remove_outliers else d2_s
            # Run analysis on the sample.
            res = _determine_single_optimal_threshold_run(d1_sc, d2_sc, hypothesis, labels)
            if not np.isnan(res[0]):
                mc_thresholds.append(res[0])
                mc_sens.append(res[1])
                mc_spec.append(res[2])
                mc_f1s.append(res[3])
                mc_aucs.append(res[4])

        results_mc = {}
        mc_metrics_to_process = [
            ("Threshold", mc_thresholds, nom_opt_t), ("F1-Score", mc_f1s, nom_f1),
            ("Sensitivity", mc_sens, nom_sens), ("Specificity", mc_spec, nom_spec),
            ("AUC", mc_aucs, nom_auc)
        ]
        # Only plot distributions for a subset of key metrics to avoid clutter.
        mc_metrics_to_plot = [("Threshold", mc_thresholds, nom_opt_t), ("F1-Score", mc_f1s, nom_f1)]

        if mc_thresholds: # Check if any valid MC runs occurred.
            for name, data_list, _ in mc_metrics_to_process:
                mean_val = np.mean(data_list) if data_list else np.nan
                median_val = np.median(data_list) if data_list else np.nan
                low_ci, high_ci = (np.percentile(data_list, [(1-CONFIDENCE_LEVEL)/2*100, (1+CONFIDENCE_LEVEL)/2*100])
                                   if data_list else (np.nan, np.nan))
                results_mc[name] = (mean_val, low_ci, high_ci)
                print(f"  MC Mean {name}:{mean_val:.4f} (Med:{median_val:.4f}) CI:[{low_ci:.4f},{high_ci:.4f}]")

            for name, data_list, nom_val in mc_metrics_to_plot:
                 if save_plots_prefix and data_list:
                    plt.figure(figsize=(8, 5))
                    sns.histplot(data_list, kde=True, bins=30, color='darkgray')
                    plt.title(f"MC Distribution of {name}")
                    plt.xlabel(name)
                    if not np.isnan(nom_val):
                        plt.axvline(nom_val, color='black', linestyle='--', label=f'Nominal ({nom_val:.3f})')
                    mean_plot = np.mean(data_list)
                    plt.axvline(mean_plot, color='dimgray', linestyle=':', label=f'Mean ({mean_plot:.3f})')
                    plt.legend()
                    save_name = f"{save_plots_prefix}_{name.lower().replace('-', '_')}_mc_dist.png"
                    plt.savefig(save_name)
                    plt.close()
            return results_mc
        else:
            print("MC ROC analysis yielded no valid results.")

    # Fallback to nominal results if MC failed or was not run.
    fallback_results = {
        "Threshold": (nom_opt_t, np.nan, np.nan), "Sensitivity": (nom_sens, np.nan, np.nan),
        "Specificity": (nom_spec, np.nan, np.nan), "F1-Score": (nom_f1, np.nan, np.nan),
        "AUC": (nom_auc, np.nan, np.nan)
    }
    return fallback_results


# =============================================================================
# Main Analysis Execution: Individual Sensors
# =============================================================================

# Create a mapping from (patient_id, run_id) to ARVC status for easy lookup.
arvc_map = {
    (pid, run_id): run_data["ARVC"]
    for pid, p_data in patient_meta_data.items()
    for run_id, run_data in p_data.get("runs", {}).items()
}

# Find all 'result.csv' files within subdirectories of the base results directory.
result_files_pattern = os.path.join(RESULTS_BASE_DIR, "*", "result.csv")
result_files = glob.glob(result_files_pattern)
if not result_files:
    print(f"Warning: No 'result.csv' files found matching pattern: {result_files_pattern}")

# Load and prepare data from each result file.
all_sensor_data_dfs = {}
sensor_file_paths = {}
for f_path in result_files:
    # The sensor/projection name is the name of the parent directory.
    sensor_projection_name = os.path.basename(os.path.dirname(f_path))
    try:
        df_sensor = pd.read_csv(f_path)
        if 'run' in df_sensor.columns:
            df_sensor['run'] = df_sensor['run'].astype(str)

        # Map the ARVC status to each row using the patient and run IDs.
        df_sensor["ARVC"] = df_sensor.apply(
            lambda row: arvc_map.get((row["patient"], row["run"]), None), axis=1
        )

        if df_sensor["ARVC"].isna().sum() > 0:
            print(f"Warning: {sensor_projection_name} - {df_sensor['ARVC'].isna().sum()} rows missing ARVC status!")

        # Clean data by dropping rows without an ARVC label.
        df_sensor.dropna(subset=["ARVC"], inplace=True)
        df_sensor["ARVC"] = df_sensor["ARVC"].astype(bool)

        if not df_sensor.empty:
            all_sensor_data_dfs[sensor_projection_name] = df_sensor
            sensor_file_paths[sensor_projection_name] = f_path
            print(f"Loaded {sensor_projection_name}: {len(df_sensor)} records.")
        else:
            print(f"Warning: {sensor_projection_name} - No data after ARVC mapping and NA removal.")
    except Exception as e:
        print(f"Error loading {f_path}: {e}")

# --- Loop Through Each Loaded Sensor Dataset for Analysis ---
overall_individual_analysis_records = []
for sensor_name, df_sensor_iter in all_sensor_data_dfs.items():
    print(f"\n--- Analyzing Sensor: {sensor_name} ---")
    # Set up output directories for this specific sensor's results.
    s_base_dir = os.path.dirname(sensor_file_paths[sensor_name])
    s_plots_dir = os.path.join(s_base_dir, "Generated_Plots_MC")
    s_tables_dir = os.path.join(s_base_dir, "Generated_Tables_MC")
    os.makedirs(s_plots_dir, exist_ok=True)
    os.makedirs(s_tables_dir, exist_ok=True)

    current_sensor_records = []
    # Iterate through each segment (T, QRS, ST) and feature (Area, Compact, etc.).
    for seg_lbl, _ in segment_cols_map.items():
        for feat_short_n in features_to_analyze_short:
            # Construct column names for nominal value and its uncertainty.
            nom_col = f"{seg_lbl.lower()}_{feat_short_n}"
            unc_col = f"{seg_lbl.lower()}_{feat_short_n}_unc"

            if nom_col not in df_sensor_iter.columns:
                continue

            # Check if uncertainty data is available for Monte Carlo simulation.
            has_unc = unc_col in df_sensor_iter.columns
            if not has_unc:
                print(f"Warning: {unc_col} not found for {nom_col} in {sensor_name}. No MC will be run for this feature.")
            print(f"\n-- Feature: {seg_lbl} {feat_short_n} ({nom_col}), Has Uncertainty: {has_unc} --")

            # Prepare data series, ensuring correct types and handling missing values.
            nom_series = pd.to_numeric(df_sensor_iter[nom_col], errors='coerce')
            valid_idx = nom_series.notna()
            unc_series = (pd.to_numeric(df_sensor_iter.loc[valid_idx, unc_col], errors='coerce').fillna(0).clip(lower=0)
                          if has_unc else pd.Series(0.0, index=nom_series[valid_idx].index))

            # Split data into ARVC (positive) and Healthy (negative) groups.
            nom_v, unc_v, arvc_l = nom_series[valid_idx].values, unc_series.values, df_sensor_iter.loc[valid_idx, "ARVC"].values
            pos_nom, neg_nom = nom_v[arvc_l == True], nom_v[arvc_l == False]
            pos_unc, neg_unc = (unc_v[arvc_l == True], unc_v[arvc_l == False]) if has_unc else (None, None)

            if len(pos_nom) < 1 or len(neg_nom) < 1:
                print(f"Skipping {nom_col} for {sensor_name}: Insufficient nominal data in one or both groups.")
                continue

            # Get the specific analysis configuration for this feature.
            cfg = feature_analysis_config.get(nom_col, feature_analysis_config["default"])
            plot_pref = os.path.join(s_plots_dir, f"{sensor_name}_{seg_lbl}_{feat_short_n}")

            # Run ROC analysis to find the optimal threshold.
            roc_res_mc = determine_optimal_threshold(
                pos_nom, neg_nom, data1_unc=pos_unc, data2_unc=neg_unc,
                hypothesis=cfg["hypothesis"], labels=("ARVC", "Healthy"),
                remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_pref
            )
            # Use the nominal threshold from ROC for the t-test boxplot.
            nom_thresh_plot = roc_res_mc["Threshold"][0]

            # Run t-test analysis.
            ttest_res_mc = perform_t_test(
                pos_nom, neg_nom, data1_unc=pos_unc, data2_unc=neg_unc,
                name=f"{seg_lbl} {feat_short_n}", hypothesis=cfg["hypothesis"],
                threshold=nom_thresh_plot, labels=("ARVC", "Healthy"),
                remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_pref
            )
            
            # Determine which group was treated as the "positive" class for metrics like sensitivity.
            if cfg["hypothesis"] == "data1_greater":
                pos_class = "ARVC" # ARVC is data1
            elif cfg["hypothesis"] == "data2_greater":
                pos_class = "Healthy" # Healthy is data2
            else: # not_equal
                pos_mean = np.mean(remove_outliers_iqr(pos_nom) if cfg["remove_outliers"] else pos_nom)
                neg_mean = np.mean(remove_outliers_iqr(neg_nom) if cfg["remove_outliers"] else neg_nom)
                pos_class = "ARVC" if pos_mean > neg_mean else "Healthy"

            # Compile all results into a single record.
            rec = {
                "source": sensor_name, "segment": seg_lbl, "feature": feat_short_n, "column_name": nom_col,
                "hypothesis_tested": cfg["hypothesis"], "outliers_removed": cfg["remove_outliers"],
                "mc_iterations": N_MC_ITERATIONS if has_unc else 0,
                "p_value_mean": ttest_res_mc[1][0], "p_value_ci_lower": ttest_res_mc[1][1], "p_value_ci_upper": ttest_res_mc[1][2],
                "t_stat_mean": ttest_res_mc[0][0], "t_stat_ci_lower": ttest_res_mc[0][1], "t_stat_ci_upper": ttest_res_mc[0][2],
                "significant_ttest (mean_p<0.05)": ttest_res_mc[2],
                "opt_threshold_mean": roc_res_mc["Threshold"][0], "opt_threshold_ci_lower": roc_res_mc["Threshold"][1], "opt_threshold_ci_upper": roc_res_mc["Threshold"][2],
                "roc_auc_mean": roc_res_mc["AUC"][0], "roc_auc_ci_lower": roc_res_mc["AUC"][1], "roc_auc_ci_upper": roc_res_mc["AUC"][2],
                "roc_positive_class_for_metrics": pos_class,
                "sensitivity_mean": roc_res_mc["Sensitivity"][0], "sensitivity_ci_lower": roc_res_mc["Sensitivity"][1], "sensitivity_ci_upper": roc_res_mc["Sensitivity"][2],
                "specificity_mean": roc_res_mc["Specificity"][0], "specificity_ci_lower": roc_res_mc["Specificity"][1], "specificity_ci_upper": roc_res_mc["Specificity"][2],
                "f1_score_mean": roc_res_mc["F1-Score"][0], "f1_score_ci_lower": roc_res_mc["F1-Score"][1], "f1_score_ci_upper": roc_res_mc["F1-Score"][2],
                "n_arvc_initial": len(pos_nom), "n_healthy_initial": len(neg_nom),
                "plot_nominal_tdist_path": f"{plot_pref}_tdist.png", "plot_nominal_boxplot_path": f"{plot_pref}_boxplot.png",
                "plot_nominal_roc_path": f"{plot_pref}_roc.png", "plot_nominal_cm_path": f"{plot_pref}_cm.png",
                "plot_mc_pvalue_dist_path": f"{plot_pref}_p_value_mc_dist.png" if has_unc and N_MC_ITERATIONS > 0 else None,
                "plot_mc_threshold_dist_path": f"{plot_pref}_threshold_mc_dist.png" if has_unc and N_MC_ITERATIONS > 0 else None,
                "plot_mc_f1_dist_path": f"{plot_pref}_f1_score_mc_dist.png" if has_unc and N_MC_ITERATIONS > 0 else None
            }
            current_sensor_records.append(rec)
            overall_individual_analysis_records.append(rec)

    # Save the summary table for the current sensor.
    if current_sensor_records:
        df_sum = pd.DataFrame(current_sensor_records)
        sum_path = os.path.join(s_tables_dir, f"{sensor_name}_analysis_summary_mc.csv")
        df_sum.to_csv(sum_path, index=False)
        print(f"\nMC summary for {sensor_name} saved to {sum_path}")

# Save the overall summary table for all individual sensor analyses.
if overall_individual_analysis_records:
    df_overall_sum = pd.DataFrame(overall_individual_analysis_records)
    overall_sum_path = os.path.join(OVERALL_TABLES_DIR, "all_sensors_features_summary_mc.csv")
    df_overall_sum.to_csv(overall_sum_path, index=False)
    print(f"\nOverall MC summary for all individual features saved to {overall_sum_path}")
else:
    print("\nNo individual sensor MC analysis was recorded.")


# =============================================================================
# Main Analysis Execution: Aggregated Projections
# =============================================================================

print("\n\n--- AGGREGATED ANALYSIS ---")
# Group sensor data by projection type (xy, yz) for combined analysis.
proj_dfs_agg = {"xy": [], "yz": []}
for s_name_iter, df_s_iter in all_sensor_data_dfs.items():
    if s_name_iter.endswith("_xy"):
        proj_dfs_agg["xy"].append(df_s_iter)
    elif s_name_iter.endswith("_yz"):
        proj_dfs_agg["yz"].append(df_s_iter)

agg_analysis_records = []
# Loop through each aggregated projection dataset.
for proj_suf, dfs_list in proj_dfs_agg.items():
    if not dfs_list:
        print(f"No data found for aggregated {proj_suf.upper()} projection. Skipping.")
        continue

    # Concatenate all dataframes for the current projection into one.
    df_agg = pd.concat(dfs_list, ignore_index=True)
    src_name_agg = f"aggregated_{proj_suf}"
    # Create a clean, human-readable name for plot titles
    clean_proj_name = f"Aggregated {proj_suf.upper()}"
    print(f"\n--- Analyzing {clean_proj_name} Data ({len(df_agg)} records) ---")

    # Create a dedicated directory for aggregated plots.
    agg_plots_dir = os.path.join(OVERALL_PLOTS_DIR, "Aggregated_Plots_MC")
    os.makedirs(agg_plots_dir, exist_ok=True)

    # This loop is identical to the individual sensor analysis loop but runs on aggregated data.
    for seg_lbl, _ in segment_cols_map.items():
        for feat_short_n in features_to_analyze_short:
            nom_col = f"{seg_lbl.lower()}_{feat_short_n}"
            unc_col = f"{seg_lbl.lower()}_{feat_short_n}_unc"
            if nom_col not in df_agg.columns:
                continue

            has_unc_agg = unc_col in df_agg.columns
            if not has_unc_agg:
                print(f"Warning: {unc_col} not found for {nom_col} in {src_name_agg}. No MC will be run.")
            print(f"\n-- Aggregated Feature: {seg_lbl} {feat_short_n} ({nom_col}), Has Uncertainty: {has_unc_agg} --")

            # Prepare data series from the aggregated dataframe.
            nom_series = pd.to_numeric(df_agg[nom_col], errors='coerce')
            valid_idx = nom_series.notna()
            unc_series = (pd.to_numeric(df_agg.loc[valid_idx, unc_col], errors='coerce').fillna(0).clip(lower=0)
                          if has_unc_agg else pd.Series(0.0, index=nom_series[valid_idx].index))

            # Split data into ARVC and Healthy groups.
            nom_v, unc_v, arvc_l = nom_series[valid_idx].values, unc_series.values, df_agg.loc[valid_idx, "ARVC"].values
            pos_nom, neg_nom = nom_v[arvc_l == True], nom_v[arvc_l == False]
            pos_unc_agg, neg_unc_agg = (unc_v[arvc_l == True], unc_v[arvc_l == False]) if has_unc_agg else (None, None)

            if len(pos_nom) < 1 or len(neg_nom) < 1:
                print(f"Skipping {nom_col} for {src_name_agg}: Insufficient data.")
                continue

            cfg = feature_analysis_config.get(nom_col, feature_analysis_config["default"])
            plot_pref_agg = os.path.join(agg_plots_dir, f"{src_name_agg}_{seg_lbl}_{feat_short_n}")

            # Run ROC and t-test analyses on the aggregated data.
            roc_res_mc_agg = determine_optimal_threshold(
                pos_nom, neg_nom, data1_unc=pos_unc_agg, data2_unc=neg_unc_agg,
                hypothesis=cfg["hypothesis"], labels=("ARVC", "Healthy"),
                remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_pref_agg
            )
            nom_thresh_plot_agg = roc_res_mc_agg["Threshold"][0]
            
            # Use the clean name for the plot title
            plot_title_name = f"{seg_lbl} {feat_short_n} ({clean_proj_name})"
            ttest_res_mc_agg = perform_t_test(
                pos_nom, neg_nom, data1_unc=pos_unc_agg, data2_unc=neg_unc_agg,
                name=plot_title_name, hypothesis=cfg["hypothesis"],
                threshold=nom_thresh_plot_agg, labels=("ARVC", "Healthy"),
                remove_outliers=cfg["remove_outliers"], save_plots_prefix=plot_pref_agg
            )
            
            # Determine the positive class for metrics.
            if cfg["hypothesis"] == "data1_greater":
                pos_class = "ARVC"
            elif cfg["hypothesis"] == "data2_greater":
                pos_class = "Healthy"
            else: # not_equal
                pos_mean = np.mean(remove_outliers_iqr(pos_nom) if cfg["remove_outliers"] else pos_nom)
                neg_mean = np.mean(remove_outliers_iqr(neg_nom) if cfg["remove_outliers"] else neg_nom)
                pos_class = "ARVC" if pos_mean > neg_mean else "Healthy"

            # Compile results into a record for the aggregated analysis.
            agg_analysis_records.append({
                "source": src_name_agg, "segment": seg_lbl, "feature": feat_short_n, "column_name": nom_col,
                "hypothesis_tested": cfg["hypothesis"], "outliers_removed": cfg["remove_outliers"],
                "mc_iterations": N_MC_ITERATIONS if has_unc_agg else 0,
                "p_value_mean": ttest_res_mc_agg[1][0], "p_value_ci_lower": ttest_res_mc_agg[1][1], "p_value_ci_upper": ttest_res_mc_agg[1][2],
                "t_stat_mean": ttest_res_mc_agg[0][0], "t_stat_ci_lower": ttest_res_mc_agg[0][1], "t_stat_ci_upper": ttest_res_mc_agg[0][2],
                "significant_ttest (mean_p<0.05)": ttest_res_mc_agg[2],
                "opt_threshold_mean": roc_res_mc_agg["Threshold"][0], "opt_threshold_ci_lower": roc_res_mc_agg["Threshold"][1], "opt_threshold_ci_upper": roc_res_mc_agg["Threshold"][2],
                "roc_auc_mean": roc_res_mc_agg["AUC"][0], "roc_auc_ci_lower": roc_res_mc_agg["AUC"][1], "roc_auc_ci_upper": roc_res_mc_agg["AUC"][2],
                "roc_positive_class_for_metrics": pos_class,
                "sensitivity_mean": roc_res_mc_agg["Sensitivity"][0], "sensitivity_ci_lower": roc_res_mc_agg["Sensitivity"][1], "sensitivity_ci_upper": roc_res_mc_agg["Sensitivity"][2],
                "specificity_mean": roc_res_mc_agg["Specificity"][0], "specificity_ci_lower": roc_res_mc_agg["Specificity"][1], "specificity_ci_upper": roc_res_mc_agg["Specificity"][2],
                "f1_score_mean": roc_res_mc_agg["F1-Score"][0], "f1_score_ci_lower": roc_res_mc_agg["F1-Score"][1], "f1_score_ci_upper": roc_res_mc_agg["F1-Score"][2],
                "n_arvc_initial": len(pos_nom), "n_healthy_initial": len(neg_nom),
                "plot_nominal_tdist_path": f"{plot_pref_agg}_tdist.png", "plot_nominal_boxplot_path": f"{plot_pref_agg}_boxplot.png",
                "plot_nominal_roc_path": f"{plot_pref_agg}_roc.png", "plot_nominal_cm_path": f"{plot_pref_agg}_cm.png",
                "plot_mc_pvalue_dist_path": f"{plot_pref_agg}_p_value_mc_dist.png" if has_unc_agg and N_MC_ITERATIONS > 0 else None,
                "plot_mc_threshold_dist_path": f"{plot_pref_agg}_threshold_mc_dist.png" if has_unc_agg and N_MC_ITERATIONS > 0 else None,
                "plot_mc_f1_dist_path": f"{plot_pref_agg}_f1_score_mc_dist.png" if has_unc_agg and N_MC_ITERATIONS > 0 else None
            })

# Save the summary table for the aggregated analysis.
if agg_analysis_records:
    df_agg_sum = pd.DataFrame(agg_analysis_records)
    sum_path_agg = os.path.join(OVERALL_TABLES_DIR, "aggregated_projection_summary_mc.csv")
    df_agg_sum.to_csv(sum_path_agg, index=False)
    print(f"\nAggregated MC summary saved to {sum_path_agg}")
else:
    print("\nNo aggregated MC analysis was recorded.")

print("\n\n--- SCRIPT FINISHED ---")
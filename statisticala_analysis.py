# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_ind, norm
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

N_MC_ITERATIONS = 100 # Reduced for faster testing, increase for production (e.g., 1000)
CONFIDENCE_LEVEL = 0.95


with open("Data/setup.json") as f:
    patient_meta_data = json.load(f)
demographic_records = []
for pid, p_data in patient_meta_data.items():
    arvc_status_patient = None
    if p_data.get("runs"):
        first_run_id = next(iter(p_data["runs"]))
        arvc_status_patient = p_data["runs"][first_run_id].get("ARVC")
    demographic_records.append({
        "patient": pid, "gender": p_data.get("gender", "unknown") or "unknown",
        "height": float(p_data.get("height")) if p_data.get("height") not in [None, ""] else None,
        "age": int(p_data.get("age")) if p_data.get("age") not in [None, ""] else None,
        "ARVC": arvc_status_patient})
df_demographics = pd.DataFrame(demographic_records)
df_demographics.dropna(subset=['ARVC'], inplace=True)
gender_palette_global = {"male": "#4C72B0", "female": "#DD8452", "unknown": "#A9A9A9"}
def plot_gender_distribution(data, title, save_path=None):
    counts = data["gender"].value_counts().reindex(["male", "female", "unknown"]).fillna(0)
    plt.figure(figsize=(7, 5)); ax = sns.barplot(x=counts.index, y=counts.values, palette=[gender_palette_global.get(g, "#cccccc") for g in counts.index])
    plt.title(title); plt.ylabel("Number of Patients"); plt.xlabel("Gender")
    for i, v in enumerate(counts.values): ax.text(i, v + max(counts.values, default=0) * 0.03, str(int(v)), ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.ylim(0, max(counts.values, default=1) * 1.20); plt.tight_layout()
    if save_path: plt.savefig(save_path); plt.close()
    else: plt.show()
print("\n--- GENDER DISTRIBUTION ---")
plot_gender_distribution(df_demographics, "Gender Distribution (All Patients)", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_all.png"))
plot_gender_distribution(df_demographics[df_demographics["ARVC"] == True], "Gender Distribution (ARVC)", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_arvc.png"))
plot_gender_distribution(df_demographics[df_demographics["ARVC"] == False], "Gender Distribution (Healthy)", save_path=os.path.join(OVERALL_PLOTS_DIR, "demographics_gender_healthy.png"))
def plot_hist_and_stats(data, column, title, color, save_path=None):
    valid_data = data[column].dropna(); stats = {"mean": np.nan, "std": np.nan, "count": len(valid_data), "median": np.nan}
    if len(valid_data) > 0:
        stats["mean"] = valid_data.mean(); stats["std"] = valid_data.std(); stats["median"] = valid_data.median()
        print(f"{title}: Mean = {stats['mean']:.2f}, Median = {stats['median']:.2f}, Std = {stats['std']:.2f}, N = {stats['count']}")
        plt.figure(figsize=(8, 5)); sns.histplot(valid_data, kde=True, color=color, bins=10)
        plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean = {stats["mean"]:.2f}')
        if not np.isnan(stats['std']):
            plt.axvline(stats['mean'] + stats['std'], color='green', linestyle='--', label=f'+1 SD = {stats["mean"]+stats["std"]:.2f}')
            plt.axvline(stats['mean'] - stats['std'], color='green', linestyle='--', label=f'-1 SD = {stats["mean"]-stats["std"]:.2f}')
        plt.title(title); plt.xlabel(column.capitalize()); plt.ylabel("Frequency"); plt.legend(); plt.tight_layout()
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
with open(os.path.join(OVERALL_TABLES_DIR, "demographics_summary.json"), "w") as f: json.dump(demographics_summary_stats, f, indent=4)
print(f"\nDemographics summary stats saved to {os.path.join(OVERALL_TABLES_DIR, 'demographics_summary.json')}")

# %% [markdown]
# ## Statistical Analysis Functions (Modified for Monte Carlo)
# %%
def remove_outliers_iqr(data):
    data_arr = np.asarray(data); 
    if len(data_arr) == 0: return data_arr
    q1, q3 = np.percentile(data_arr, [25, 75]); iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data_arr[(data_arr >= lower) & (data_arr <= upper)]

def _perform_single_t_test_run(data1_sample, data2_sample, hypothesis, labels):
    if len(data1_sample) < 2 or len(data2_sample) < 2: return np.nan, np.nan, False
    t_stat, p_val_two_tailed = ttest_ind(data1_sample, data2_sample, equal_var=False, nan_policy='omit')
    if np.isnan(t_stat) or np.isnan(p_val_two_tailed): return np.nan, np.nan, False
    p_value = (p_val_two_tailed / 2 if t_stat > 0 else 1-(p_val_two_tailed/2)) if hypothesis == "data1_greater" else \
              (p_val_two_tailed / 2 if t_stat < 0 else 1-(p_val_two_tailed/2)) if hypothesis == "data2_greater" else p_val_two_tailed
    return t_stat, p_value, p_value < 0.05

def perform_t_test(data1_nominal, data2_nominal, data1_unc=None, data2_unc=None, name="", hypothesis="data1_greater", threshold=None, labels=("Group 1", "Group 2"), remove_outliers=False, save_plots_prefix=None):
    data1_nom_arr, data2_nom_arr = np.asarray(data1_nominal), np.asarray(data2_nominal)
    data1_plot_clean, data2_plot_clean = (remove_outliers_iqr(data1_nom_arr), remove_outliers_iqr(data2_nom_arr)) if remove_outliers else (data1_nom_arr, data2_nom_arr)
    print(f"T-Test for {name} (Nominal Values): Orig D1:{len(data1_nom_arr)}, D2:{len(data2_nom_arr)}. Clean D1:{len(data1_plot_clean)}, D2:{len(data2_plot_clean)}")
    nominal_t_stat, nominal_p_value, nominal_significant = np.nan, np.nan, False
    group1_plot_label, group2_plot_label, hypothesis_text_for_plot = labels[0], labels[1], f"{labels[0]} ≠ {labels[1]}" # Defaults

    if len(data1_plot_clean) >= 2 and len(data2_plot_clean) >= 2:
        nominal_t_stat, nominal_p_val_two_tailed = ttest_ind(data1_plot_clean, data2_plot_clean, equal_var=False, nan_policy='omit')
        if not (np.isnan(nominal_t_stat) or np.isnan(nominal_p_val_two_tailed)):
            if hypothesis == "data1_greater":
                group1_plot_label, group2_plot_label, nominal_p_value, hypothesis_text_for_plot = f'{labels[0]} (Higher)', f'{labels[1]} (Lower)', (nominal_p_val_two_tailed / 2 if nominal_t_stat > 0 else 1 - (nominal_p_val_two_tailed/2)), f"{labels[0]} > {labels[1]}"
            elif hypothesis == "data2_greater":
                group1_plot_label, group2_plot_label, nominal_p_value, hypothesis_text_for_plot = f'{labels[0]} (Lower)', f'{labels[1]} (Higher)', (nominal_p_val_two_tailed / 2 if nominal_t_stat < 0 else 1 - (nominal_p_val_two_tailed/2)), f"{labels[0]} < {labels[1]}"
            else: nominal_p_value = nominal_p_val_two_tailed
            nominal_significant = nominal_p_value < 0.05
            print(f"Hypothesis: {hypothesis_text_for_plot.replace(labels[0],'data1').replace(labels[1],'data2')}, T-statistic: {nominal_t_stat:.4f}")
            if hypothesis=="not_equal": print(f"Two-tailed p-value: {nominal_p_value:.4f}")
            else: print(f"One-tailed p-value: {nominal_p_value:.4f}"); print(f"Two-tailed p-value: {nominal_p_val_two_tailed:.4f}")
            print(f"Result is statistically {'significant' if nominal_significant else 'not significant'} (p-value = {nominal_p_value:.4f})")
            df_freedom = len(data1_plot_clean) + len(data2_plot_clean) - 2
            if df_freedom > 0:
                x_t = np.linspace(-4, 4, 1000); y_t = t.pdf(x_t, df_freedom)
                plt.figure(figsize=(10,6)); plt.plot(x_t, y_t, label="t-distribution", color='blue')
                fill_label = f'p-value Area (t > {nominal_t_stat:.2f})' if hypothesis=="data1_greater" else f'p-value Area (t < {nominal_t_stat:.2f})' if hypothesis=="data2_greater" else f'p-value Area (|t| > {abs(nominal_t_stat):.2f})'
                fill_cond = (x_t > nominal_t_stat) if hypothesis=="data1_greater" else (x_t < nominal_t_stat) if hypothesis=="data2_greater" else ((x_t > abs(nominal_t_stat))|(x_t < -abs(nominal_t_stat)))
                plt.fill_between(x_t,0,y_t,where=fill_cond,color='red',alpha=0.5,label=fill_label)
                plt.axvline(x=nominal_t_stat,color='green',linestyle='--',label=f't-statistic: {nominal_t_stat:.2f}')
                plt.xlabel("t-value");plt.ylabel("Probability Density");plt.title(f"t-Distribution with Shaded p-value Area ({hypothesis_text_for_plot.replace(labels[0],'data1').replace(labels[1],'data2')})");plt.legend()
                if save_plots_prefix: plt.savefig(f"{save_plots_prefix}_tdist.png"); plt.close()
                else: plt.show()
        else: print("Nominal t-test resulted in NaN.")
    else: print("Not enough data for nominal t-test after cleaning.")
    
    plot_data_bp = [data1_plot_clean if len(data1_plot_clean)>0 else np.array([]), data2_plot_clean if len(data2_plot_clean)>0 else np.array([])]
    plt.figure(figsize=(8,6)); sns.boxplot(data=plot_data_bp,palette=["green","red"],notch=True)
    plt.xticks([0,1],[f"{group1_plot_label} (N={len(data1_plot_clean)})",f"{group2_plot_label} (N={len(data2_plot_clean)})"])
    plt.title(f"Boxplot of Cleaned Data for {name}")
    if threshold is not None and not np.isnan(threshold): plt.axhline(y=threshold,color='orange',linestyle='--',label=f'Optimal Threshold: {threshold:.2f}'); plt.legend()
    if save_plots_prefix: plt.savefig(f"{save_plots_prefix}_boxplot.png"); plt.close()
    else: plt.show()

    if data1_unc is not None and data2_unc is not None and N_MC_ITERATIONS > 0:
        print(f"\nPerforming MC ({N_MC_ITERATIONS} iter) for t-test of {name}...")
        mc_t_stats, mc_p_values = [], []
        data1_unc_arr, data2_unc_arr = np.asarray(data1_unc), np.asarray(data2_unc)
        for i in range(N_MC_ITERATIONS):
            if i%(N_MC_ITERATIONS//10)==0 and i>0: print(f"MC iter {i}...")
            d1_s, d2_s = np.random.normal(data1_nom_arr,data1_unc_arr), np.random.normal(data2_nom_arr,data2_unc_arr)
            d1_sc, d2_sc = (remove_outliers_iqr(d1_s),remove_outliers_iqr(d2_s)) if remove_outliers else (d1_s,d2_s)
            t_s_mc, p_v_mc, _ = _perform_single_t_test_run(d1_sc,d2_sc,hypothesis,labels)
            if not np.isnan(p_v_mc): mc_t_stats.append(t_s_mc); mc_p_values.append(p_v_mc)
        if mc_p_values:
            m_p, med_p, p_low, p_high = np.mean(mc_p_values), np.median(mc_p_values), np.percentile(mc_p_values,(1-CONFIDENCE_LEVEL)/2*100), np.percentile(mc_p_values,(1+CONFIDENCE_LEVEL)/2*100)
            m_t, med_t, t_low, t_high = np.mean(mc_t_stats), np.median(mc_t_stats), np.percentile(mc_t_stats,(1-CONFIDENCE_LEVEL)/2*100), np.percentile(mc_t_stats,(1+CONFIDENCE_LEVEL)/2*100)
            print(f"\nMC T-Test: Mean P-val: {m_p:.4f} (Med:{med_p:.4f}) CI:[{p_low:.4f},{p_high:.4f}]. Mean T-stat: {m_t:.4f} (Med:{med_t:.4f}) CI:[{t_low:.4f},{t_high:.4f}]")
            if save_plots_prefix: # Only plot P-value distribution
                plt.figure(figsize=(8,5));sns.histplot(mc_p_values,kde=True,bins=30);plt.title(f"MC Distribution of P-values for {name}");plt.xlabel("P-value")
                plt.axvline(nominal_p_value,color='red',linestyle='--',label=f'Nominal P ({nominal_p_value:.3f})');plt.axvline(m_p,color='blue',linestyle=':',label=f'Mean P ({m_p:.3f})');plt.legend()
                plt.savefig(f"{save_plots_prefix}_p_value_mc_dist.png");plt.close()
            return (m_t,t_low,t_high), (m_p,p_low,p_high), (m_p < 0.05)
        else: print("MC t-test no valid p-values."); return (nominal_t_stat,np.nan,np.nan),(nominal_p_value,np.nan,np.nan),nominal_significant
    return (nominal_t_stat,np.nan,np.nan),(nominal_p_value,np.nan,np.nan),nominal_significant

def _determine_single_optimal_threshold_run(data1_sample, data2_sample, hypothesis, labels):
    if len(data1_sample)==0 or len(data2_sample)==0: return np.nan,np.nan,np.nan,np.nan,np.nan
    y_scores = np.concatenate((data1_sample,data2_sample))
    y_true = (np.concatenate((np.ones(len(data1_sample)),np.zeros(len(data2_sample))))) if hypothesis=="data1_greater" else \
             (np.concatenate((np.zeros(len(data1_sample)),np.ones(len(data2_sample))))) if hypothesis=="data2_greater" else \
             (np.concatenate((np.ones(len(data1_sample)),np.zeros(len(data2_sample))))) if (np.mean(data1_sample) if len(data1_sample)>0 else -np.inf) > (np.mean(data2_sample) if len(data2_sample)>0 else -np.inf) else \
             (np.concatenate((np.zeros(len(data1_sample)),np.ones(len(data2_sample)))))
    fpr,tpr,thresh_roc=roc_curve(y_true,y_scores); roc_auc=auc(fpr,tpr)
    if len(thresh_roc)==0 or len(tpr)==0 or len(fpr)==0: return np.nan,np.nan,np.nan,np.nan,roc_auc
    opt_idx=np.argmax(tpr-fpr); opt_thresh=thresh_roc[opt_idx]
    y_pred=(y_scores>=opt_thresh).astype(int); f1=f1_score(y_true,y_pred,zero_division=0)
    sens1=np.sum((y_pred==1)&(y_true==1))/np.sum(y_true==1) if np.sum(y_true==1)>0 else 0.0
    spec0=np.sum((y_pred==0)&(y_true==0))/np.sum(y_true==0) if np.sum(y_true==0)>0 else 0.0
    return opt_thresh,sens1,spec0,f1,roc_auc

def determine_optimal_threshold(data1_nominal, data2_nominal, data1_unc=None, data2_unc=None, hypothesis="data2_greater", labels=("Class 0","Class 1"), remove_outliers=False, save_plots_prefix=None):
    data1_nom_arr,data2_nom_arr=np.asarray(data1_nominal),np.asarray(data2_nominal)
    print(f"Optimal Threshold for {labels[0]} vs {labels[1]} (Nominal Values): d1:{len(data1_nom_arr)}, d2:{len(data2_nom_arr)}")
    d1_plot_c,d2_plot_c=(remove_outliers_iqr(data1_nom_arr),remove_outliers_iqr(data2_nom_arr)) if remove_outliers else (data1_nom_arr,data2_nom_arr)
    print(f"Cleaned d1:{len(d1_plot_c)}, d2:{len(d2_plot_c)}")
    nom_opt_t,nom_sens,nom_spec,nom_f1,nom_auc = np.nan,np.nan,np.nan,np.nan,np.nan
    cm_disp_labels_plot = list(labels) # Default

    if not (len(d1_plot_c)==0 or len(d2_plot_c)==0):
        nom_opt_t,nom_sens,nom_spec,nom_f1,nom_auc = _determine_single_optimal_threshold_run(d1_plot_c,d2_plot_c,hypothesis,labels)
        if not np.isnan(nom_opt_t):
            if hypothesis=="data1_greater": cm_disp_labels_plot=[labels[1],labels[0]]
            elif hypothesis=="not_equal":
                m1,m2=(np.mean(d1_plot_c) if len(d1_plot_c)>0 else -np.inf),(np.mean(d2_plot_c) if len(d2_plot_c)>0 else -np.inf)
                if m1>m2: cm_disp_labels_plot=[labels[1],labels[0]]; print("Auto: d1 higher (class 1)")
                else: print("Auto: d1 lower (class 0)")
            print(f"Optimal Thresh:{nom_opt_t:.4f}, F1:{nom_f1:.2f}, Sens:{nom_sens:.2f}, Spec:{nom_spec:.2f}, AUC:{nom_auc:.2f}")
            
            y_s_plot=np.concatenate((d1_plot_c,d2_plot_c))
            y_t_plot=(np.concatenate((np.ones(len(d1_plot_c)),np.zeros(len(d2_plot_c))))) if hypothesis=="data1_greater" else \
                      (np.concatenate((np.zeros(len(d1_plot_c)),np.ones(len(d2_plot_c))))) if hypothesis=="data2_greater" else \
                      (np.concatenate((np.ones(len(d1_plot_c)),np.zeros(len(d2_plot_c))))) if (np.mean(d1_plot_c) if len(d1_plot_c)>0 else -np.inf) > (np.mean(d2_plot_c) if len(d2_plot_c)>0 else -np.inf) else \
                      (np.concatenate((np.zeros(len(d1_plot_c)),np.ones(len(d2_plot_c)))))
            fpr_p,tpr_p,thr_p=roc_curve(y_t_plot,y_s_plot); opt_idx_p=np.argmax(tpr_p-fpr_p) if len(tpr_p)>0 else 0
            plt.figure(figsize=(8,6));plt.plot(fpr_p,tpr_p,color='blue',lw=2,label=f'ROC (AUC={nom_auc:.2f})')
            if len(fpr_p)>opt_idx_p: plt.plot(fpr_p[opt_idx_p],tpr_p[opt_idx_p],'ro',markersize=8,label='Optimal Threshold')
            plt.plot([0,1],[0,1],color='gray',linestyle='--');plt.xlabel('FPR');plt.ylabel('TPR');plt.title('ROC Curve');plt.legend(loc='lower right')
            if save_plots_prefix: plt.savefig(f"{save_plots_prefix}_roc.png");plt.close()
            else: plt.show()
            y_pred_p=(y_s_plot>=nom_opt_t).astype(int); cm_p=confusion_matrix(y_t_plot,y_pred_p)
            cm_sum1=cm_p.sum(axis=1)[:,np.newaxis]; cm_perc_p=np.zeros_like(cm_p,dtype=float)
            np.divide(cm_p.astype('float'),cm_sum1,out=cm_perc_p,where=cm_sum1!=0); cm_perc_p*=100
            plt.figure(figsize=(6,5));sns.heatmap(cm_perc_p,annot=True,fmt=".2f",cmap="Blues",cbar=False,xticklabels=cm_disp_labels_plot,yticklabels=cm_disp_labels_plot)
            plt.title("Confusion Matrix (%)");plt.xlabel("Predicted");plt.ylabel("True")
            if save_plots_prefix: plt.savefig(f"{save_plots_prefix}_cm.png");plt.close()
            else: plt.show()
        else: print("Nominal ROC failed.")
    else: print("Not enough data for nominal ROC.")

    if data1_unc is not None and data2_unc is not None and N_MC_ITERATIONS > 0:
        print(f"\nPerforming MC ({N_MC_ITERATIONS} iter) for ROC of {labels[0]} vs {labels[1]}...")
        mc_ts,mc_s,mc_sp,mc_f1s,mc_aucs = [],[],[],[],[]
        d1_u_arr,d2_u_arr = np.asarray(data1_unc),np.asarray(data2_unc)
        for i in range(N_MC_ITERATIONS):
            if i%(N_MC_ITERATIONS//10)==0 and i>0: print(f"MC iter {i}...")
            d1_s,d2_s = np.random.normal(data1_nom_arr,d1_u_arr),np.random.normal(data2_nom_arr,d2_u_arr)
            d1_sc,d2_sc = (remove_outliers_iqr(d1_s),remove_outliers_iqr(d2_s)) if remove_outliers else (d1_s,d2_s)
            res = _determine_single_optimal_threshold_run(d1_sc,d2_sc,hypothesis,labels)
            if not np.isnan(res[0]): mc_ts.append(res[0]);mc_s.append(res[1]);mc_sp.append(res[2]);mc_f1s.append(res[3]);mc_aucs.append(res[4])
        
        results_mc = {}
        mc_metrics_to_process = [("Threshold",mc_ts,nom_opt_t),("F1-Score",mc_f1s,nom_f1),("Sensitivity",mc_s,nom_sens),("Specificity",mc_sp,nom_spec),("AUC",mc_aucs,nom_auc)]
        mc_metrics_to_plot = [("Threshold",mc_ts,nom_opt_t),("F1-Score",mc_f1s,nom_f1)] # Only Threshold and F1

        if mc_ts: # Check if any valid MC runs
            for name, data_l, nom_v in mc_metrics_to_process:
                m,med,low,high = (np.mean(data_l) if data_l else np.nan, np.median(data_l) if data_l else np.nan, 
                                  np.percentile(data_l,(1-CONFIDENCE_LEVEL)/2*100) if data_l else np.nan, 
                                  np.percentile(data_l,(1+CONFIDENCE_LEVEL)/2*100) if data_l else np.nan)
                results_mc[name] = (m,low,high)
                print(f"  MC Mean {name}:{m:.4f} (Med:{med:.4f}) CI:[{low:.4f},{high:.4f}]")
            
            for name, data_l, nom_v in mc_metrics_to_plot: # Plot only selected
                 if save_plots_prefix and data_l:
                    plt.figure(figsize=(8,5));sns.histplot(data_l,kde=True,bins=30);plt.title(f"MC Dist of {name}");plt.xlabel(name)
                    if not np.isnan(nom_v): plt.axvline(nom_v,color='red',linestyle='--',label=f'Nominal ({nom_v:.3f})')
                    m_plot = np.mean(data_l)
                    plt.axvline(m_plot,color='blue',linestyle=':',label=f'Mean ({m_plot:.3f})');plt.legend()
                    plt.savefig(f"{save_plots_prefix}_{name.lower().replace('-','_')}_mc_dist.png");plt.close()
            return results_mc
        else: print("MC ROC no valid results.")
    # Fallback if MC failed or not run
    return {"Threshold":(nom_opt_t,np.nan,np.nan),"Sensitivity":(nom_sens,np.nan,np.nan),"Specificity":(nom_spec,np.nan,np.nan),"F1-Score":(nom_f1,np.nan,np.nan),"AUC":(nom_auc,np.nan,np.nan)}


# %% [markdown]
# ## Feature Analysis Configuration 
# %%
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

arvc_map = {(pid, run_id): run_data["ARVC"] for pid, p_data in patient_meta_data.items() for run_id, run_data in p_data.get("runs", {}).items()}
result_files_pattern = os.path.join(RESULTS_BASE_DIR, "*", "result.csv")
result_files = glob.glob(result_files_pattern)
if not result_files: print(f"Warning: No 'result.csv' files found: {result_files_pattern}")
all_sensor_data_dfs, sensor_file_paths = {}, {}
for f_path in result_files:
    sensor_projection_name = os.path.basename(os.path.dirname(f_path))
    try:
        df_sensor = pd.read_csv(f_path)
        if 'run' in df_sensor.columns: df_sensor['run'] = df_sensor['run'].astype(str)
        df_sensor["ARVC"] = df_sensor.apply(lambda row: arvc_map.get((row["patient"], row["run"]), None), axis=1)
        if df_sensor["ARVC"].isna().sum() > 0: print(f"Warning: {sensor_projection_name} - {df_sensor['ARVC'].isna().sum()} rows missing ARVC!")
        df_sensor.dropna(subset=["ARVC"], inplace=True); df_sensor["ARVC"] = df_sensor["ARVC"].astype(bool)
        if not df_sensor.empty: all_sensor_data_dfs[sensor_projection_name], sensor_file_paths[sensor_projection_name] = df_sensor, f_path; print(f"Loaded {sensor_projection_name}: {len(df_sensor)} recs.")
        else: print(f"Warning: {sensor_projection_name} - No data after ARVC map/NA removal.")
    except Exception as e: print(f"Error loading {f_path}: {e}")

overall_individual_analysis_records = []
for sensor_name, df_sensor_iter in all_sensor_data_dfs.items():
    print(f"\n--- Analyzing Sensor: {sensor_name} ---")
    s_base_dir = os.path.dirname(sensor_file_paths[sensor_name]) 
    s_plots_dir = os.path.join(s_base_dir, "Generated_Plots_MC"); os.makedirs(s_plots_dir, exist_ok=True)
    s_tables_dir = os.path.join(s_base_dir, "Generated_Tables_MC"); os.makedirs(s_tables_dir, exist_ok=True)
    current_sensor_records = []
    for seg_lbl, _ in segment_cols_map.items():
        for feat_short_n in features_to_analyze_short:
            nom_col, unc_col = f"{seg_lbl.lower()}_{feat_short_n}", f"{seg_lbl.lower()}_{feat_short_n}_unc"
            if nom_col not in df_sensor_iter.columns: continue
            has_unc = unc_col in df_sensor_iter.columns
            if not has_unc: print(f"Warning: {unc_col} not found for {nom_col} in {sensor_name}. No MC for this feature.")
            print(f"\n-- Feature: {seg_lbl} {feat_short_n} ({nom_col}), Has Unc: {has_unc} --")
            
            nom_series = pd.to_numeric(df_sensor_iter[nom_col],errors='coerce'); valid_idx = nom_series.notna()
            unc_series = pd.to_numeric(df_sensor_iter.loc[valid_idx,unc_col],errors='coerce').fillna(0).clip(lower=0) if has_unc else pd.Series(0.0,index=nom_series[valid_idx].index)
            nom_v, unc_v, arvc_l = nom_series[valid_idx].values, unc_series.values, df_sensor_iter.loc[valid_idx,"ARVC"].values
            pos_nom, neg_nom = nom_v[arvc_l==True], nom_v[arvc_l==False]
            pos_unc, neg_unc = (unc_v[arvc_l==True], unc_v[arvc_l==False]) if has_unc else (None, None)

            if len(pos_nom)<1 or len(neg_nom)<1: print(f"Skipping {nom_col} for {sensor_name}: Insufficient nominal data."); continue
            cfg = feature_analysis_config.get(nom_col,feature_analysis_config["default"])
            plot_pref = os.path.join(s_plots_dir,f"{sensor_name}_{seg_lbl}_{feat_short_n}")
            
            roc_res_mc = determine_optimal_threshold(pos_nom,neg_nom,data1_unc=pos_unc,data2_unc=neg_unc,hypothesis=cfg["hypothesis"],labels=("ARVC","Healthy"),remove_outliers=cfg["remove_outliers"],save_plots_prefix=plot_pref)
            nom_thresh_plot = roc_res_mc["Threshold"][0]
            ttest_res_mc = perform_t_test(pos_nom,neg_nom,data1_unc=pos_unc,data2_unc=neg_unc,name=f"{seg_lbl} {feat_short_n}",hypothesis=cfg["hypothesis"],threshold=nom_thresh_plot,labels=("ARVC","Healthy"),remove_outliers=cfg["remove_outliers"],save_plots_prefix=plot_pref)
            
            pos_class = "ARVC" if cfg["hypothesis"]=="data1_greater" else "Healthy" if cfg["hypothesis"]=="data2_greater" else ("ARVC" if np.mean(remove_outliers_iqr(pos_nom) if cfg["remove_outliers"] else pos_nom) > np.mean(remove_outliers_iqr(neg_nom) if cfg["remove_outliers"] else neg_nom) else "Healthy")
            rec = {"source":sensor_name,"segment":seg_lbl,"feature":feat_short_n,"column_name":nom_col,"hypothesis_tested":cfg["hypothesis"],"outliers_removed":cfg["remove_outliers"],"mc_iterations":N_MC_ITERATIONS if has_unc else 0,
                   "p_value_mean":ttest_res_mc[1][0],"p_value_ci_lower":ttest_res_mc[1][1],"p_value_ci_upper":ttest_res_mc[1][2],
                   "t_stat_mean":ttest_res_mc[0][0],"t_stat_ci_lower":ttest_res_mc[0][1],"t_stat_ci_upper":ttest_res_mc[0][2],"significant_ttest (mean_p<0.05)":ttest_res_mc[2],
                   "opt_threshold_mean":roc_res_mc["Threshold"][0],"opt_threshold_ci_lower":roc_res_mc["Threshold"][1],"opt_threshold_ci_upper":roc_res_mc["Threshold"][2],
                   "roc_auc_mean":roc_res_mc["AUC"][0],"roc_auc_ci_lower":roc_res_mc["AUC"][1],"roc_auc_ci_upper":roc_res_mc["AUC"][2],
                   "roc_positive_class_for_metrics":pos_class, "sensitivity_mean":roc_res_mc["Sensitivity"][0],"sensitivity_ci_lower":roc_res_mc["Sensitivity"][1],"sensitivity_ci_upper":roc_res_mc["Sensitivity"][2],
                   "specificity_mean":roc_res_mc["Specificity"][0],"specificity_ci_lower":roc_res_mc["Specificity"][1],"specificity_ci_upper":roc_res_mc["Specificity"][2],
                   "f1_score_mean":roc_res_mc["F1-Score"][0],"f1_score_ci_lower":roc_res_mc["F1-Score"][1],"f1_score_ci_upper":roc_res_mc["F1-Score"][2],
                   "n_arvc_initial":len(pos_nom),"n_healthy_initial":len(neg_nom),
                   "plot_nominal_tdist_path":f"{plot_pref}_tdist.png","plot_nominal_boxplot_path":f"{plot_pref}_boxplot.png","plot_nominal_roc_path":f"{plot_pref}_roc.png","plot_nominal_cm_path":f"{plot_pref}_cm.png",
                   "plot_mc_pvalue_dist_path":f"{plot_pref}_p_value_mc_dist.png" if has_unc and N_MC_ITERATIONS>0 else None,
                   "plot_mc_threshold_dist_path":f"{plot_pref}_threshold_mc_dist.png" if has_unc and N_MC_ITERATIONS>0 else None,
                   "plot_mc_f1_dist_path":f"{plot_pref}_f1_score_mc_dist.png" if has_unc and N_MC_ITERATIONS>0 else None} # Corrected key
            current_sensor_records.append(rec); overall_individual_analysis_records.append(rec)
    if current_sensor_records:
        df_sum=pd.DataFrame(current_sensor_records); sum_path=os.path.join(s_tables_dir,f"{sensor_name}_analysis_summary_mc.csv")
        df_sum.to_csv(sum_path,index=False); print(f"\nMC summary for {sensor_name} saved to {sum_path}")
if overall_individual_analysis_records:
    df_overall_sum=pd.DataFrame(overall_individual_analysis_records); overall_sum_path=os.path.join(OVERALL_TABLES_DIR,"all_sensors_features_summary_mc.csv")
    df_overall_sum.to_csv(overall_sum_path,index=False); print(f"\nOverall MC summary saved to {overall_sum_path}")
else: print("\nNo individual sensor MC analysis recorded.")


print("\n\n--- AGGREGATED ANALYSIS ---")
proj_dfs_agg = {"xy":[],"yz":[]}
for s_name_iter, df_s_iter in all_sensor_data_dfs.items():
    if s_name_iter.endswith("_xy"): proj_dfs_agg["xy"].append(df_s_iter)
    elif s_name_iter.endswith("_yz"): proj_dfs_agg["yz"].append(df_s_iter)
agg_analysis_records = []
for proj_suf, dfs_l in proj_dfs_agg.items():
    if not dfs_l: print(f"No data for aggregated {proj_suf.upper()} proj."); continue
    df_agg = pd.concat(dfs_l,ignore_index=True); src_name_agg=f"aggregated_{proj_suf}"
    print(f"\n--- Analyzing {src_name_agg.upper()} Data ({len(df_agg)} recs) ---")
    agg_plots_dir = os.path.join(OVERALL_PLOTS_DIR,"Aggregated_Plots_MC"); os.makedirs(agg_plots_dir,exist_ok=True)
    for seg_lbl, _ in segment_cols_map.items():
        for feat_short_n in features_to_analyze_short:
            nom_col,unc_col = f"{seg_lbl.lower()}_{feat_short_n}",f"{seg_lbl.lower()}_{feat_short_n}_unc"
            if nom_col not in df_agg.columns: continue
            has_unc_agg = unc_col in df_agg.columns
            if not has_unc_agg: print(f"Warning: {unc_col} not found for {nom_col} in {src_name_agg}. No MC.")
            print(f"\n-- Agg Feature: {seg_lbl} {feat_short_n} ({nom_col}), Has Unc: {has_unc_agg} --")
            nom_series = pd.to_numeric(df_agg[nom_col],errors='coerce'); valid_idx = nom_series.notna()
            unc_series = pd.to_numeric(df_agg.loc[valid_idx,unc_col],errors='coerce').fillna(0).clip(lower=0) if has_unc_agg else pd.Series(0.0,index=nom_series[valid_idx].index)
            nom_v,unc_v,arvc_l = nom_series[valid_idx].values,unc_series.values,df_agg.loc[valid_idx,"ARVC"].values
            pos_nom,neg_nom = nom_v[arvc_l==True],nom_v[arvc_l==False]
            pos_unc_agg,neg_unc_agg = (unc_v[arvc_l==True],unc_v[arvc_l==False]) if has_unc_agg else (None,None)
            if len(pos_nom)<1 or len(neg_nom)<1: print(f"Skipping {nom_col} for {src_name_agg}: Insufficient data."); continue
            cfg = feature_analysis_config.get(nom_col,feature_analysis_config["default"])
            plot_pref_agg = os.path.join(agg_plots_dir,f"{src_name_agg}_{seg_lbl}_{feat_short_n}")
            
            roc_res_mc_agg = determine_optimal_threshold(pos_nom,neg_nom,data1_unc=pos_unc_agg,data2_unc=neg_unc_agg,hypothesis=cfg["hypothesis"],labels=("ARVC","Healthy"),remove_outliers=cfg["remove_outliers"],save_plots_prefix=plot_pref_agg)
            nom_thresh_plot_agg = roc_res_mc_agg["Threshold"][0]
            ttest_res_mc_agg = perform_t_test(pos_nom,neg_nom,data1_unc=pos_unc_agg,data2_unc=neg_unc_agg,name=f"{seg_lbl} {feat_short_n} ({src_name_agg})",hypothesis=cfg["hypothesis"],threshold=nom_thresh_plot_agg,labels=("ARVC","Healthy"),remove_outliers=cfg["remove_outliers"],save_plots_prefix=plot_pref_agg)
            
            pos_class = "ARVC" if cfg["hypothesis"]=="data1_greater" else "Healthy" if cfg["hypothesis"]=="data2_greater" else ("ARVC" if np.mean(remove_outliers_iqr(pos_nom) if cfg["remove_outliers"] else pos_nom) > np.mean(remove_outliers_iqr(neg_nom) if cfg["remove_outliers"] else neg_nom) else "Healthy")
            agg_analysis_records.append({"source":src_name_agg,"segment":seg_lbl,"feature":feat_short_n,"column_name":nom_col,"hypothesis_tested":cfg["hypothesis"],"outliers_removed":cfg["remove_outliers"],"mc_iterations":N_MC_ITERATIONS if has_unc_agg else 0,
                "p_value_mean":ttest_res_mc_agg[1][0],"p_value_ci_lower":ttest_res_mc_agg[1][1],"p_value_ci_upper":ttest_res_mc_agg[1][2],
                "t_stat_mean":ttest_res_mc_agg[0][0],"t_stat_ci_lower":ttest_res_mc_agg[0][1],"t_stat_ci_upper":ttest_res_mc_agg[0][2],"significant_ttest (mean_p<0.05)":ttest_res_mc_agg[2],
                "opt_threshold_mean":roc_res_mc_agg["Threshold"][0],"opt_threshold_ci_lower":roc_res_mc_agg["Threshold"][1],"opt_threshold_ci_upper":roc_res_mc_agg["Threshold"][2],
                "roc_auc_mean":roc_res_mc_agg["AUC"][0],"roc_auc_ci_lower":roc_res_mc_agg["AUC"][1],"roc_auc_ci_upper":roc_res_mc_agg["AUC"][2],
                "roc_positive_class_for_metrics":pos_class, "sensitivity_mean":roc_res_mc_agg["Sensitivity"][0],"sensitivity_ci_lower":roc_res_mc_agg["Sensitivity"][1],"sensitivity_ci_upper":roc_res_mc_agg["Sensitivity"][2],
                "specificity_mean":roc_res_mc_agg["Specificity"][0],"specificity_ci_lower":roc_res_mc_agg["Specificity"][1],"specificity_ci_upper":roc_res_mc_agg["Specificity"][2],
                "f1_score_mean":roc_res_mc_agg["F1-Score"][0],"f1_score_ci_lower":roc_res_mc_agg["F1-Score"][1],"f1_score_ci_upper":roc_res_mc_agg["F1-Score"][2],
                "n_arvc_initial":len(pos_nom),"n_healthy_initial":len(neg_nom),
                "plot_nominal_tdist_path":f"{plot_pref_agg}_tdist.png","plot_nominal_boxplot_path":f"{plot_pref_agg}_boxplot.png","plot_nominal_roc_path":f"{plot_pref_agg}_roc.png","plot_nominal_cm_path":f"{plot_pref_agg}_cm.png",
                "plot_mc_pvalue_dist_path":f"{plot_pref_agg}_p_value_mc_dist.png" if has_unc_agg and N_MC_ITERATIONS>0 else None,
                "plot_mc_threshold_dist_path":f"{plot_pref_agg}_threshold_mc_dist.png" if has_unc_agg and N_MC_ITERATIONS>0 else None,
                "plot_mc_f1_dist_path":f"{plot_pref_agg}_f1_score_mc_dist.png" if has_unc_agg and N_MC_ITERATIONS>0 else None}) # Corrected key
if agg_analysis_records:
    df_agg_sum=pd.DataFrame(agg_analysis_records); sum_path_agg=os.path.join(OVERALL_TABLES_DIR,"aggregated_projection_summary_mc.csv")
    df_agg_sum.to_csv(sum_path_agg,index=False); print(f"\nAggregated MC summary saved to {sum_path_agg}")
else: print("\nNo aggregated MC analysis recorded.")
print("\n\n--- SCRIPT FINISHED ---")
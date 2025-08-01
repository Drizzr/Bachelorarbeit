import importlib
import json
import os
import analyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyzer import Analyzer

# =============================================================================
# Configuration
# =============================================================================

# Configure Matplotlib to use LaTeX for rendering text in plots
plt.rcParams['text.usetex'] = True

CONFIG = {
    "setup_file": "Data/setup.json",
    "data_base_dir": "Data",
    "results_base_dir": "Results",
    "model_checkpoint_dir": "MCG_segmentation/trained_models/UNet_1D_15M/checkpoints/acm_study_checkpoint",
    "model_class_name": "UNet1D",  # Store as string for dynamic import if needed
    "overall_plots_dir": "Results/Overall_Generated_Plots/Patient_Heart_Vectors_4_x_4",
    "grayscale_map": {
        "xy-Projection": '#2F2F2F',  # Dark gray
        "xz-Projection": '#5F5F5F',  # Medium gray
        "yz-Projection": "#C9C1C1"   # Light gray
    }
}

# Ensure the main output directory exists
os.makedirs(CONFIG["overall_plots_dir"], exist_ok=True)

# =============================================================================
# User Interaction
# =============================================================================

def get_user_inputs() -> tuple[str, str | None]:
    """
    Prompts the user to enter patient and run IDs and validates the input.
    """
    print("🫀 ECG Vector Metrics Export Tool 🧪\n")

    patient = ""
    while not patient.startswith("P"):
        patient = input("Enter patient ID (e.g., P004): ").strip()
        if not patient.startswith("P"):
            print("❌ Invalid patient ID. It should start with 'P' (e.g., P004).")

    run = input("Enter run ID (e.g., S00) [optional, defaults to first available]: ").strip()
    if not run:
        run = None

    print(f"\n✅ Patient: {patient} | Run: {run or 'Default'}\n")
    return patient, run

# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_patient_data(patient_id: str, run_id: str | None) -> tuple | None:
    """
    Loads patient data using setup.json and returns a configured Analyzer instance.

    Returns:
        A tuple containing (Analyzer instance, interval start, interval end,
        ICA filter settings, run ID) or None if loading fails.
    """
    with open(CONFIG["setup_file"], "r") as f:
        setup_data = json.load(f)

    if patient_id not in setup_data:
        print(f"❌ Error: Patient '{patient_id}' not found in {CONFIG['setup_file']}.")
        return None

    patient_info = setup_data[patient_id]
    data_dir = os.path.join(CONFIG["data_base_dir"], patient_id)

    # Determine the run to process
    if not run_id:
        available_runs = patient_info.get("runs", {})
        if not available_runs:
            print(f"❌ Error: No runs found for patient '{patient_id}'.")
            return None
        run_id = next(iter(available_runs))  # Default to the first run
        print(f"ℹ️ No run specified. Defaulting to first available run: {run_id}")

    run_info = patient_info["runs"].get(run_id)
    if not run_info:
        print(f"❌ Error: Run '{run_id}' not found for patient '{patient_id}'.")
        return None

    # Find patient data files
    main_file, add_file = None, None
    for file in os.listdir(data_dir):
        is_patient_file = file.endswith(".tdms") and file.startswith(patient_id)
        is_correct_run = run_id in file if run_id else True
        if is_patient_file and is_correct_run:
            if "addCh" in file:
                add_file = os.path.join(data_dir, file)
            else:
                main_file = os.path.join(data_dir, file)
    
    if not main_file:
        print(f"❌ Error: Could not find main .tdms file for {patient_id}, run {run_id}.")
        return None

    # Extract configuration
    sensors_to_exclude = run_info.get("sensors_to_exclude", {})
    interval_start = run_info.get("interval_start")
    interval_end = run_info.get("interval_end")
    ica_filter = run_info.get("ICA_filter")
    log_file_path = os.path.join(data_dir, "QZFM_log_file.txt")

    print(f"--- Loading Configuration for {patient_id} | {run_id} ---")
    print(f"  - Interval: {interval_start} - {interval_end}")
    print(f"  - ICA Filter Thresholds: {ica_filter}")
    print(f"  - Excluding Sensors: {sensors_to_exclude}")
    
    # Dynamically get model class from the analyzer module
    # Assuming 'analyzer' module has UNet1D defined.
    model_class = getattr(analyzer, CONFIG["model_class_name"])
    
    analysis_instance = Analyzer(
        filename=main_file,
        add_filename=add_file,
        log_file_path=log_file_path,
        model_checkpoint_dir=CONFIG["model_checkpoint_dir"],
        model_class=model_class,
        sensor_channels_to_exclude=sensors_to_exclude
    )
    
    return analysis_instance, interval_start, interval_end, ica_filter, run_id


def apply_ica_and_prepare_data(analysis, x, y, z, ica_settings, key, interval):
    """Slices data to interval, applies ICA filtering, and inverts field directions."""
    print("\n--- Applying ICA Filtering ---")
    
    # Slice data to the specified interval
    x_interval = x[:, :, interval[0]:interval[1]]
    y_interval = y[:, :, interval[0]:interval[1]]
    z_interval = z[:, :, interval[0]:interval[1]]

    # Apply ICA filter to each component
    x_filtered, _, _, _ = analysis.ICA_filter(x_interval, heart_beat_score_threshold=ica_settings[0], plot_result=True, max_iter=20000)
    y_filtered, _, _, _ = analysis.ICA_filter(y_interval, heart_beat_score_threshold=ica_settings[1], plot_result=True, max_iter=20000)
    z_filtered, _, _, _ = analysis.ICA_filter(z_interval, heart_beat_score_threshold=ica_settings[2], plot_result=True, max_iter=20000)
    
    # Reconstruct the single run signal from filtered components
    single_run_filtered = analysis.invert_field_directions(x_filtered, y_filtered, z_filtered, key, 48)
    
    return single_run_filtered, x_filtered, y_filtered, z_filtered


def detect_peaks_and_segment(analysis, single_run_data):
    """Detects R-peaks, allows manual segmentation editing, and returns segments."""
    print("\n--- Detecting Peaks and Segmenting Signal ---")
    
    # Detect QRS peaks from the cleanest channel
    peak_positions, ch, labels, _, _ = analysis.detect_qrs_complex_peaks_cleanest_channel(
        single_run_data, print_heart_rate=True, confidence_threshold=0.5, 
        confidence_weight=0.9, plausibility_weight=0.1
    )

    if peak_positions is None or len(peak_positions) == 0:
        print("❌ Error: No R-peaks detected. Cannot proceed.")
        return None, None, None, None, None
    
    # Plot detected peaks
    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(single_run_data[ch, :], label='Signal', linewidth=1.2)
    plt.plot(peak_positions, single_run_data[ch, peak_positions], "ro", markersize=6, label='R Peaks')
    plt.title(fr"QRS Detection - Cleanest Channel {ch + 1}")
    plt.xlabel(r"Time Step (Sample Index)")
    plt.ylabel(r"Amplitude (pT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Average beats
    avg_channels, _ = analysis.avg_window(
        single_run_data, peak_positions, window_left=0.3, window_right=0.4, 
        sigma=1, heart_beat_score_threshold=0.50
    )
    
    # Find cleanest channel in averaged data for segmentation
    best_channel, labels, _, _ = analysis.find_cleanest_channel(
        avg_channels, confidence_weight=0.7, plausibility_weight=0.3
    )
    
    # Allow manual editing of segments
    edited_labels = analysis.plot_segments_with_editing(avg_channels[best_channel], labels[best_channel])

    # Extract segment boundaries from edited labels
    mask_t = (edited_labels == 3) & (np.arange(len(edited_labels)) >= 110) & (np.arange(len(edited_labels)) <= 175)
    t_indices = np.where(mask_t)[0]
    t_start, t_end = t_indices[0], t_indices[-1]

    mask_qrs = (edited_labels == 2) & (np.arange(len(edited_labels)) >= 50) & (np.arange(len(edited_labels)) <= (len(edited_labels) - 50))
    qrs_indices = np.where(mask_qrs)[0]
    qrs_start, qrs_end = qrs_indices[0], qrs_indices[-1]
    
    return avg_channels, t_start, t_end, qrs_start, qrs_end

# =============================================================================
# Main Processing Loop
# =============================================================================

def process_and_save_sensor_projections(analysis, avg_channels, key, patient_id, run_id, segments):
    """Processes each sensor, calculates metrics, and saves all results and plots."""
    print("\n--- Processing Sensor Projections and Saving Results ---")
    
    t_start, t_end, qrs_start, qrs_end = segments
    x_data, y_data, z_data = analysis.get_field_directions(avg_channels, key)

    figs = {
        "t": plt.subplots(4, 4, figsize=(16, 16)),
        "qrs": plt.subplots(4, 4, figsize=(16, 16)),
        "st": plt.subplots(4, 4, figsize=(16, 16))
    }
    
    for row_idx, row_sensors in enumerate(analysis.quspin_position_list):
        for col_idx, quspin_id in enumerate(row_sensors):
            # Determine which projections are possible for this sensor (xy, xz, or yz)
            sensor_data_map = {'x': x_data[row_idx, col_idx, :], 'y': y_data[row_idx, col_idx, :], 'z': z_data[row_idx, col_idx, :]}
            
            # Check for non-zero data
            valid_components = {k: v for k, v in sensor_data_map.items() if not np.all(v == 0)}
            
            if len(valid_components) < 2:
                continue # Not enough data for a 2D projection

            # Determine projection type
            if 'x' in valid_components and 'y' in valid_components:
                proj_name = "xy-Projection"
                sensor_data = np.array([valid_components['x'], valid_components['y']])
            elif 'x' in valid_components and 'z' in valid_components:
                proj_name = "xz-Projection"
                sensor_data = np.array([valid_components['x'], valid_components['z']])
            elif 'y' in valid_components and 'z' in valid_components:
                proj_name = "yz-Projection"
                sensor_data = np.array([valid_components['y'], valid_components['z']])
                # Set background color for yz projection plots in the grid view
                for fig_ax in figs.values():
                    fig_ax[1][row_idx, col_idx].set_facecolor("#DCDCDC")
            else:
                continue
                
            print(f"  - Processing Sensor: {quspin_id} ({proj_name})")

            # --- Calculate Metrics for each segment ---
            segments_info = {
                "t": (t_start, t_end, "T"),
                "qrs": (qrs_start, qrs_end, "QRS"),
                "st": (qrs_end + 1, t_start, "ST")
            }
            
            # --- MODIFICATION START: Define paths for saving individual plots ---
            sensor_base_path = os.path.join(CONFIG["results_base_dir"], f"{quspin_id}_{proj_name[:2]}")
            patient_run_path = os.path.join(sensor_base_path, "Patients", f"{patient_id}_{run_id}")
            # --- MODIFICATION END ---

            metrics_all_segments = {}
            for seg_key, (start, end, seg_name) in segments_info.items():
                
                # --- MODIFICATION START: Create directories and define save path ---
                segment_plot_dir = os.path.join(patient_run_path, f"{seg_name}_heart_vector")
                os.makedirs(segment_plot_dir, exist_ok=True)
                individual_plot_save_path = os.path.join(segment_plot_dir, f"{quspin_id}_{proj_name[:2]}.pdf")
                # --- MODIFICATION END ---
                
                _, metrics = analysis.visualize_heart_vector(
                    original_data=sensor_data, segment_start_global=start, segment_end_global=end,
                    proj_name=proj_name, title_suffix=fr"{seg_name} Segment - {quspin_id}",
                    show=False, 
                    save_path=individual_plot_save_path, # Pass the save path
                    uncertainty_ms=40
                )
                metrics_all_segments[seg_key] = metrics

            # --- Plot Grid Views ---
            plot_lims = {"t": 20, "qrs": 45, "st": 3}
            for seg_key, (start, end, _) in segments_info.items():
                ax = figs[seg_key][1][row_idx, col_idx]
                analysis._trajectory_plot(
                    ax, component1=sensor_data[0, start:end + 1], component2=sensor_data[1, start:end + 1],
                    proj_name=proj_name, plot_color=CONFIG["grayscale_map"][proj_name]
                )
                lim = plot_lims[seg_key]
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)

            # --- Save Results to CSV ---
            output_file = os.path.join(sensor_base_path, "result.csv")
            os.makedirs(sensor_base_path, exist_ok=True)
            
            # Create a row dictionary for the new data
            new_row = {"patient": patient_id, "run": run_id}
            for seg_key, metrics in metrics_all_segments.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        new_row[f"{seg_key}_{k}"] = float(v.n)
                        new_row[f"{seg_key}_{k}_unc"] = float(v.s)
            
            # Read existing data, append new row, and save
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
            df.to_csv(output_file, index=False)

    # --- Finalize and Save Grid Plots ---
    for seg_key, (fig, _) in figs.items():
        title = fr"{seg_key.upper()} Segment Heart Vector Projections - {patient_id}_{run_id}"
        fig.suptitle(title, fontsize=24, y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(CONFIG["overall_plots_dir"], f"{seg_key.upper()}_heart_vectors_{patient_id}_{run_id}.pdf")
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to run the entire patient data processing pipeline."""
    # 1. Get user input
    patient_id, run_id = get_user_inputs()
    
    # 2. Load data and initialize Analyzer
    load_result = load_patient_data(patient_id, run_id)
    if load_result is None:
        return
    analysis, interval_start, interval_end, ica_filter, run_id = load_result

    # 3. Prepare raw data
    # Find the primary data key (e.g., 'Brustlage')
    primary_key = None
    for k in analysis.key_list:
        if k in ["Brustlage", "Brust", "Bauchlage", "Bauch", "Rene_Brust", "Lena_Brustlage"]:
            primary_key = k
            break
    if not primary_key:
        print("❌ Error: Could not find a primary data key like 'Brustlage'.")
        return
        
    (x_raw, y_raw, z_raw), _, _ = analysis.prepare_data(primary_key, apply_default_filter=True, plot_alignment=False)
    
    # 4. Apply ICA filtering and get single-run signal
    single_run_filtered, x_filt, y_filt, z_filt = apply_ica_and_prepare_data(
        analysis, x_raw, y_raw, z_raw, ica_filter, primary_key, (interval_start, interval_end)
    )

    # 5. Detect peaks and define segments
    segment_results = detect_peaks_and_segment(analysis, single_run_filtered)
    if segment_results[0] is None: # Check if peak detection failed
        return
    avg_channels, t_start, t_end, qrs_start, qrs_end = segment_results
    
    # 6. Process all sensors, save metrics and plots
    process_and_save_sensor_projections(
        analysis, avg_channels, primary_key, patient_id, run_id, 
        (t_start, t_end, qrs_start, qrs_end)
    )
    
    print("\n✅ --- Pipeline finished successfully! --- ✅")

if __name__ == "__main__":
    # Reload the analyzer module to pick up any changes during development
    importlib.reload(analyzer)
    main()
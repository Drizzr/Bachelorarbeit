import importlib
import analyzer
importlib.reload(analyzer)
from analyzer import Analyzer
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import json
import numpy as np

def get_user_inputs():
    print("🫀 ECG Vector Metrics Export Tool 🧪\n")
    
    patient = input("Enter patient ID (e.g., P004): ").strip()
    while not patient.startswith("P"):
        print("❌ Invalid patient ID. It should start with 'P' (e.g., P004).")
        patient = input("Enter patient ID (e.g., P004): ").strip()
    
    run = input("Enter run ID (e.g., S00) if left empty this will defualt to the first run: ").strip()
    if not run:
        run = None
    
    print(f"\n✅ Patient: {patient} | Run: {run}\n")
    return patient, run

# Usage
Patient, run = get_user_inputs()


######
# Load patient data
######

def load_patient_data(patient: str, run: str = None):
    """
    Load patient data using Data/setup.json and return the configured Analyzer instance
    along with interval and ICA filter data.
    """
    setup_path = "Data/setup.json"

    with open(setup_path, "r") as f:
        setup = json.load(f)

    if patient not in setup:
        print(f"No data found for patient {patient} in setup.json.")
        return

    patient_data = setup[patient]
    dir = f"Data/{patient}/"

    if not run:
        available_runs = patient_data.get("runs", {})
        print(f"Available runs for patient {patient}: {list(available_runs.keys())}")
        if not available_runs:
            print(f"No runs found for patient {patient}. Skipping...")
            return
        run = next(iter(available_runs))  # default to first available run

    run_data = patient_data["runs"].get(run)
    if not run_data:
        print(f"Run {run} not found for patient {patient}. Skipping...")
        return

    sensor_channels_to_exclude = run_data.get("sensors_to_exclude", {})
    intervall_start = run_data.get("interval_start")
    intervall_end = run_data.get("interval_end")
    ica_filter = run_data.get("ICA_filter")

    print(f"Intervall: {intervall_start} - {intervall_end}")
    print(f"ICA Filter: {ica_filter}")

    add_filename, file_name = None, None

    for file in os.listdir(dir):
        if file.endswith(".tdms") and file.startswith(patient):
            print(file)
            if run and run not in file:
                continue
            if "addCh" in file:
                add_filename = os.path.join(dir, file)
            else:
                file_name = os.path.join(dir, file)

    log_file_path = os.path.join(dir, "QZFM_log_file.txt")

    return Analyzer(
        filename=file_name,
        add_filename=add_filename,
        log_file_path=log_file_path,
        #model_checkpoint_dir="MCG_segmentation/trained_models/MCGSegmentator_s",
        sensor_channels_to_exclude=sensor_channels_to_exclude
    ), intervall_start, intervall_end, ica_filter

analysis, intervall_start, intervall_end, ica_filter = load_patient_data(Patient, run)

for k in analysis.key_list:
    if k in ["Brustlage", "Brust", "Bauchlage", "Bauch", "Rene_Brust", "Lena_Brustlage"]:
        key = k
        break
    
(x_data, y_data, z_data), time, single_run = analysis.prepare_data(key, apply_default_filter=True, plot_alignment=False)


########
# Apply ICA filtering
########

x_data_intervall = x_data[:, :, intervall_start:intervall_end]
y_data_intervall = y_data[:, :, intervall_start:intervall_end]
z_data_intervall = z_data[:, :, intervall_start:intervall_end]
time_intervall = time[intervall_start:intervall_end]
single_run_intervall = single_run[:, intervall_start:intervall_end]


x_data_filtered, _, _, _ = analysis.ICA_filter(x_data_intervall, heart_beat_score_threshold=ica_filter[0], plot_result=True)
y_data_filtered, ica_components, _, _ = analysis.ICA_filter(y_data_intervall, heart_beat_score_threshold=ica_filter[1], plot_result=True)
z_data_filtered, _, _, _ = analysis.ICA_filter(z_data_intervall, heart_beat_score_threshold=ica_filter[2], plot_result=True)
single_run_filtered = analysis.invert_field_directions(x_data_filtered, y_data_filtered, z_data_filtered, key, 48)

#single_run_filtered = single_run_intervall.copy()


########
# Visualize the filtered data and apply window averaging
########


#analysis.butterfly_plot(single_run_filtered, time_intervall, 48, f"Original {key}")

# use cleanest channel for peak detection
peak_positions, ch, labels, _, _ = analysis.detect_qrs_complex_peaks_cleanest_channel(single_run_filtered, print_heart_rate=True, confidence_threshold=0.7, confidence_weight=0.9, plausibility_weight=0.1)
if peak_positions is not None and len(peak_positions) > 0:
    plt.figure(figsize=(12, 4))
    plt.plot(single_run_filtered[ch, :], label='Signal', linewidth=1.2)
    #plt.plot(resampled_data[ch, :], label='Signal', linewidth=1.2)
    plt.plot(peak_positions, single_run_filtered[ch, peak_positions], "ro", markersize=6, label='R Peaks')
    plt.title(f"QRS Detection - Cleanest Channel {ch + 1}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No R peaks detected or `peak_positions` is empty.")
#analysis.plot_segmented_signal(single_run_filtered[ch, :], labels[ch, :])


# window averaging
avg_channels, time_window = analysis.avg_window(single_run_filtered, peak_positions, window_left=0.3, window_right=0.5)
analysis.butterfly_plot(avg_channels, time_window, 48, f"Original {key}")


avg_channels = np.array(avg_channels)
# --- Load averaged field data ---
x_data_window, y_data_window, z_data_window = analysis.get_field_directions(avg_channels, key)

#analysis.plot_sensor_matrix(x_data_window, time_window, name="X-Field")
analysis.plot_sensor_matrix(y_data_window, time_window, name="Y-Field")
#analysis.plot_sensor_matrix(z_data_window, time_window, name="Z-Field")

# Use a sample vector for projection
f1_data = np.array([x_data_window[0, 1, :], y_data_window[0, 1, :]])
print(f"f1_data shape: {f1_data.shape}")

# --- Find cleanest channel ---
best_channel, labels, confidence, _ = analysis.find_cleanest_channel(
    avg_channels, confidence_weight=0.7, plausibility_weight=0.3
)


# option of manual segmentation of the cleanest channel
edited_labels = analysis.plot_segments_with_editing(avg_channels[best_channel], labels[best_channel])

# --- Extract T-wave segment ---
mask_t = edited_labels == 3
mask_t[:110] = False  # Ignore early segment
mask_t[175:] = False  # Ignore late segment

t_indices = np.where(mask_t)[0]
t_start, t_end = t_indices[0], t_indices[-1] 



# --- Extract QRS-wave segment ---
mask_qrs = edited_labels == 2
mask_qrs[:50] = False
mask_qrs[-50:] = False
t_start_qrs = np.where(mask_qrs)[0][0] 
t_end_qrs = np.where(mask_qrs)[0][-1] 


# Set default run if not provided
run_id = run if run else "S01"

for row_idx, row in enumerate(analysis.quspin_position_list):
    for col_idx, quspin_id in enumerate(row):
        sensor_data = []
        suffixes = []
        for suffix, target in zip(['_x', '_y', '_z'], [x_data_window, y_data_window, z_data_window]):
            channel_name = quspin_id + suffix
            channel_index = analysis.quspin_channel_dict.get(channel_name)
            if channel_index is None or (analysis.sensor_channels_to_exclude.get(key) and channel_name in analysis.sensor_channels_to_exclude.get(key, [])) or \
            (analysis.sensor_channels_to_exclude.get(key) and f"*{suffix}" in analysis.sensor_channels_to_exclude.get(key, [])):
                continue

            channel_index = abs(int(channel_index))
            if not np.all(target[row_idx, col_idx, :] == 0):
                sensor_data.append(target[row_idx, col_idx, :])
                suffixes.append(suffix)
            else:
                print(f"Skipping {channel_name}{suffix} as it contains only zeros. (To change this decrease the threshold in the window averaging step.)")
        
        if len(sensor_data) >= 2:
            if len(sensor_data) == 3:
                sensor_data = sensor_data[:2]
                suffixes = suffixes[:2]
            sensor_data = np.array(sensor_data)

            if "_x" in suffixes and "_y" in suffixes:
                name = "xy-Projection"
            elif "_x" in suffixes and "_z" in suffixes:
                name = "xz-Projection"
            elif "_y" in suffixes and "_z" in suffixes:
                name = "yz-Projection"


            print(f"processing Sensor: {quspin_id}")

            # Create the new directory structure
            # Results/sensor_name/Patients/patient_id_run/**_heart_vector
            sensor_base_path = f"Results/{quspin_id}_{name[:2]}"
            patient_run_path = f"{sensor_base_path}/Patients/{Patient}_{run_id}"
            
            # Create directories for heart vector plots
            t_path = f"{patient_run_path}/T_heart_vector"
            qrs_path = f"{patient_run_path}/QRS_heart_vector"
            st_path = f"{patient_run_path}/ST_heart_vector"
            
            for path in [t_path, qrs_path, st_path]:
                if not os.path.exists(path):
                    os.makedirs(path)

            # --- Plot and calculate uncertainty metrics using updated method ---
            t_segment_path = os.path.join(t_path, f"{quspin_id}_{name[:2]}.pdf")
            qrs_segment_path = os.path.join(qrs_path, f"{quspin_id}_{name[:2]}.pdf")
            st_segment_path = os.path.join(st_path, f"{quspin_id}_{name[:2]}.pdf")

            _, t_metrics = analysis.plot_heart_vector_projection(
                original_data=sensor_data,
                segment_start_global=t_start,
                segment_end_global=t_end,
                proj_name=name,
                title_suffix="T Segment",
                show=False,
                save_path=t_segment_path,
                uncertainty_ms=40
            )

            _, qrs_metrics = analysis.plot_heart_vector_projection(
                original_data=sensor_data,
                segment_start_global=t_start_qrs,
                segment_end_global=t_end_qrs,
                proj_name=name,
                title_suffix="QRS Segment",
                show=False,
                save_path=qrs_segment_path,
                uncertainty_ms=40
            )

            _, st_metrics = analysis.plot_heart_vector_projection(
                original_data=sensor_data,
                segment_start_global=t_end_qrs + 1,
                segment_end_global=t_start,
                proj_name=name,
                title_suffix="ST Segment",
                show=False,
                save_path=st_segment_path,
                uncertainty_ms=40
            )

            out_put = np.stack((t_metrics, qrs_metrics, st_metrics), axis=0)

            row = {"patient": Patient, "run": run_id}

            for prefix, metrics in zip(["t", "qrs", "st"], [t_metrics, qrs_metrics, st_metrics]):
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        row[f"{prefix}_{k}"] = float(v.n)      # Nominal value
                        row[f"{prefix}_{k}_unc"] = float(v.s)  # Uncertainty


            # Store the metrics CSV file in the sensor base directory
            output_file = os.path.join(sensor_base_path, "result.csv")

            if os.path.exists(output_file):
                existing_data = pd.read_csv(output_file)
                updated_data = pd.concat([existing_data, pd.DataFrame([row])], ignore_index=True)
                updated_data.to_csv(output_file, index=False) 
            else:
                # Create sensor base directory if it doesn't exist
                if not os.path.exists(sensor_base_path):
                    os.makedirs(sensor_base_path)
                df = pd.DataFrame([row])
                df.to_csv(output_file, index=False)
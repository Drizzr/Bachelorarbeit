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



Patient = "P030" #"P015/P015_S01_D2024-07-06"
run = None


def load_patient_data(patient: str, run: str = None):
    """
    Load patient data from the specified file.
    """

    overview = pd.read_excel("Data/overview.xlsx")
    dir = "Data/" + patient + "/"
    number = re.findall(r'\d+', dir)
    patient_number = int(number[0]) if number else None


    if overview.loc[overview["Probanten Nr."] == patient_number, "runs"].values[0].strip() == "-":
        print(f"No runs found for patient {patient_number}. Skipping...")
        return
    
    add_filename, file_name = None, None
    
    for file in os.listdir(dir):
        if file.endswith(".tdms") and file.startswith(patient):
            if run and run not in file:
                continue
            if "addCh" in file:
                add_filename = os.path.join(dir, file)
            else:
                file_name = os.path.join(dir, file)
            
    
    log_file_path = os.path.join(dir, "QZFM_log_file.txt")
    sensor_channels_to_exclude = json.loads(overview.loc[overview["Probanten Nr."] == patient_number, "Sensors to exclude"].values[0])

    
    try:
        intervall = overview.loc[overview["Probanten Nr."] == patient_number, "Intervall"].values[0]
        intervall = intervall.split(":") if isinstance(intervall, str) else intervall
        intervall_start = int(intervall[0]) if isinstance(intervall, list) else None
        intervall_end = int(intervall[1]) if isinstance(intervall, list) else None
        print(f"Intervall: {intervall_start} - {intervall_end}")
    except Exception as e:
        print(f"Error parsing interval: {e}")
        intervall_start, intervall_end = None, None

    try:
        ica_filter = overview.loc[overview["Probanten Nr."] == patient_number, "ICA Filter (x, y, z)"].values[0].split(";")
        print(f"ICA Filter: {ica_filter}")
        ica_filter = [float(i) for i in ica_filter] if isinstance(ica_filter, list) else None
    except Exception as e:
        print(f"Error parsing ICA filter: {e}")
        ica_filter = None


    return Analyzer(
        filename=file_name,
        add_filename=add_filename,
        log_file_path=log_file_path,
        sensor_channels_to_exclude=sensor_channels_to_exclude
    ), intervall_start, intervall_end, ica_filter

analysis, intervall_start, intervall_end, ica_filter = load_patient_data(Patient, run)

for k in analysis.key_list:
    if k in ["Brustlage", "Brust", "Bauchlage", "Bauch"]:
        key = k
        break
    
(x_data, y_data, z_data), time, single_run = analysis.prepare_data(key, apply_default_filter=True, plot_alignment=True)



x_data_intervall = x_data[:, :, intervall_start:intervall_end]
y_data_intervall = y_data[:, :, intervall_start:intervall_end]
z_data_intervall = z_data[:, :, intervall_start:intervall_end]
time_intervall = time[intervall_start:intervall_end]
single_run_intervall = single_run[:, intervall_start:intervall_end]


#analysis.plot4x4(z_data[:, :, 250:1250], time[250:1250], name="z_data")

x_data_filtered, _, _, _ = analysis.ICA_filter(x_data_intervall, plot_result=True)
y_data_filtered, ica_components, _, _ = analysis.ICA_filter(y_data_intervall, plot_result=True)
z_data_filtered, _, _, _ = analysis.ICA_filter(z_data_intervall, plot_result=True)
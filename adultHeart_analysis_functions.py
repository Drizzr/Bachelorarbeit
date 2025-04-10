# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 08:52:43 2024

@author: Philipp Wunderl (philipp.wunderl@tum.de)
"""



import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import linregress
from scipy import signal #import savgol_filter,welch
from scipy.signal import find_peaks
import ast
import os
plt.rcParams['agg.path.chunksize'] = 100000000
cmap = plt.get_cmap('nipy_spectral')
cmaplist = [cmap(x) for x in np.linspace(0,1,num=40)]



def directory(path):
    # path_name = os.path.join(path, name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory'{path}' was created.")
    else:
        print(f"directory'{path}' already exists.")






def import_tdms(filename, scaling):
    data = {}
    with TdmsFile.read(filename) as tdms_file:
        for group in tdms_file.groups():
            no_of_channels = len(group.channels())
            no_of_samples = len(group.channels()[0].read_data(scaled=True))
            data_array = np.zeros((no_of_samples, no_of_channels+1))
        
            for i, channel in enumerate(group.channels()):
                # Access numpy array of data for channel:
                data_array[:, i+1] = channel.read_data(scaled=True)/scaling        
            properties = channel.properties
            data_array[:, 0] = np.linspace(0, (no_of_samples-1), no_of_samples)
                                           #* properties['wf_increment'], no_of_samples)
            data[group.name] = data_array
    return data

def find_number_layers(liste):
    if (not isinstance(liste, list)) and (not isinstance(liste, np.ndarray)):
        return 0
    if len(liste)==0:
        return 0
    return 1 + max(find_number_layers(subliste) for subliste in liste)


def save_data(data, path, name, quspin_pos_list):
    header = ", ".join(map(str,quspin_pos_list))
    filepath = path + name + ".txt"
    
    joint_list = []
    number_layers = find_number_layers(data)
    if number_layers == 1:
        max_len = len(data)
        # print(data)
        joint_list.append(data)
    elif number_layers == 2:
        max_len = max(len(sublist) for sublist in data)
        for liste in data:
            if isinstance(liste, np.ndarray):
                liste = liste.tolist()
            liste.extend([None] * (max_len - len(liste)))
            joint_list.append(liste)
        # print(f"{name} data saved")
    elif number_layers == 3:        
        max_len = max(len(sublist) for lists in data for sublist in lists)
        for liste in data:
            for sublist in liste:
                if isinstance(sublist, np.ndarray):
                    sublist = sublist.tolist()
                sublist.extend([None] * (max_len - len(sublist)))
                joint_list.append(sublist)
        # print(f"{name} data saved")
    else:
        print("Invalid Data for saving")
    print(f"{name}: {number_layers}")
    
    joint_list = np.array(joint_list, dtype=float).T
    np.savetxt(filepath, joint_list, delimiter='\t', newline='\n', header=header)
    


def remove_drift_and_offset(data, time):
    time_d = time[0:120]
    time_d = np.append(time_d,time[-60:-1])
    data_d = data[0:120]
    data_d = np.append(data_d,data[-60:-1])
    res = linregress(time_d, data_d)
    drift = res.slope * time + res.intercept
    detrended_data = data - drift
    return detrended_data

def remove_drift_and_offset_all_ch(data, time, num_ch):
    for ch in range(num_ch):    
        res = linregress(time, data[ch])
        drift = res.slope * time + res.intercept
        data[ch] = data[ch] - drift
    return np.array(data)


#Filter functions

def apply_highpass_filter(data, sampling_rate, cutoff_frequency, order=2):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    return filtered_data


def apply_lowpass_filter(data, sampling_rate, cutoff_frequency, order=2):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    return filtered_data


def apply_notch_filter(data,  sampling_rate,notch_frequency, quality_factor):
    nyquist = 0.5 * sampling_rate
    notch_frequency_normalized = notch_frequency / nyquist
    b, a = signal.iirnotch(notch_frequency_normalized, quality_factor, fs=sampling_rate)
    filtered_signal = signal.filtfilt(b, a, data)
    return filtered_signal


def bandstop_filter(data, center_frequency, bandwidth, sampling_rate, order = 4):
    nyquist = 0.5 * sampling_rate
    low_cutoff = (center_frequency - bandwidth / 2) / nyquist
    high_cutoff = (center_frequency + bandwidth / 2) / nyquist
    b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandstop', analog=False)
    filtered_signal = signal.filtfilt(b, a, data)
    return filtered_signal


def default_filter_combination(data, sampling=1000, bandstop_freq=50, lowpass_freq=95, highpass_freq=1, number_lowpass=3):
    filtered_data =  bandstop_filter(data, bandstop_freq, 2, sampling, order = 2)
    filtered_data =  bandstop_filter(filtered_data, bandstop_freq, 3, sampling, order = 3)
    filtered_data =  bandstop_filter(filtered_data, bandstop_freq*2, 3, sampling, order = 3)
    
    for i in range(number_lowpass):
        filtered_data = apply_lowpass_filter(filtered_data, sampling, lowpass_freq, order=3)

    filtered_data = apply_highpass_filter(filtered_data, sampling, highpass_freq , order=2)
    #filtered_data = filtered_data[:,sampling*2:sampling*-2]
    return filtered_data


def default_filter_combination_MSR(data, sampling=1000, lowpass_freq=65, highpass_freq=1):
    filtered_data = apply_lowpass_filter(data, sampling, lowpass_freq, order=3)
    filtered_data = apply_highpass_filter(filtered_data, sampling, highpass_freq , order=2)
    #filtered_data = filtered_data[:,sampling*2:sampling*-2]
    return filtered_data


###

# Plotting 
def plotting_time_series(data, time, num_ch, name, path = None, save = False ):
    fig, elem= plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(3.94, 3.54),dpi=100)  
    fig.suptitle(name,size='small', y=0.99)
    if num_ch>1:
        # linienstile = ['-', '--', '-.', ':']
        for i in range(num_ch):
            elem.plot(time, data[i], alpha=0.4,color=cmaplist[i], label = f"Ch {i+1}")
        elem.legend(loc='lower center',bbox_to_anchor=(0.5,-0.35),ncol=7)
        elem.grid(alpha=0.7)
        elem.minorticks_on()
        elem.grid(True,which='minor',linestyle='dotted',alpha=0.7)
        elem.set_xlabel('Time [s]')
        elem.set_ylabel('B [pT]')#("V")#
        fig.subplots_adjust(top= 0.95, bottom=0.25)
    else:
        elem.plot(time, data,"-", label = "Single Channel")
        elem.legend(loc='lower center',bbox_to_anchor=(0.5,-0.2),ncol=5)
        elem.grid(alpha=0.7)
        elem.minorticks_on()
        elem.grid(True,which='minor',linestyle='dotted',alpha=0.7)
        elem.set_xlabel('Time [s]')
        elem.set_ylabel('B [pT]')#("V")#
        fig.subplots_adjust(bottom=0.2)
    if save:
        plt.savefig(path+f'{name}_B_vs_time.png')
    plt.show()


def lsd_multichannel(data, noise_theos, freqs, name, labels, channels,  path, save = False ):
    nenbw=1.9761
    fig, elem= plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(10,5),dpi=100)  

    for ind,ch in enumerate(channels):
        f_bins,Pxx=signal.welch(data[ch], fs=freqs[ind],nperseg=int(freqs[ind])*100, window='nuttall', return_onesided=True, scaling='density')
        
        Lxx=np.sqrt(Pxx)
        enbw=f_bins[1]*nenbw #calculated from acutal f_res
        elem.loglog(f_bins,Lxx,label=f'{labels[ind]}',color=cmaplist[ind*int(len(cmaplist)/len(channels))], alpha = 0.5)
        if noise_theos[0] !=np.mean(noise_theos) and (ind<len(channels)):
            
            elem.plot(f_bins,np.full(len(f_bins),noise_theos[ind]),
                      label='Sensor theo.',color=cmaplist[ind],linestyle='-.')

    def forward(x):
        return x*np.sqrt(enbw)*np.sqrt(2)
    def inverse(x):
        return x/(np.sqrt(enbw)*np.sqrt(2))

    elem.plot(f_bins,np.full(len(f_bins),noise_theos[ind]),
              label='Sensor theo.',color=cmaplist[ind+1],linestyle='-.')
    secax = elem.secondary_yaxis('right', functions=(forward, inverse))
    secax.set_ylabel("LS (linear amplitude) [$ pT$]")
    elem.set_xlabel('Frequency [Hz]')
    elem.set_ylabel('LSD [$ pT$/$\sqrt{Hz}$]')
    fig.suptitle('Magnetic flux linear spectral density '
                 '[$ pT$/$\sqrt{Hz}$], NENBW=1.9761 bins\n '
                 '$f_s$=%d Hz, $f_{res}$=%.3f Hz \n' %(freqs[ind],f_bins[1])+name,size='small',
                 y=1.0)
    elem.set_xlim(0.1)
    elem.legend(loc='lower center',bbox_to_anchor=(0.5,-0.25),ncol=4)
    elem.grid(alpha=0.7)
    elem.minorticks_on()
    elem.grid(True,which='minor',linestyle='dotted',alpha=0.7)
    # elem.set_yticks([10,100,500, 1000, 5000, 1e4,5e4,1e5,5e5])
    plt.subplots_adjust(top = 0.9, bottom = 0.2)
    if save:
        plt.savefig(path+f'{name}_LSD.png')
    plt.show()


def plot4x4(data, time, name,path=None, save = False):
    nrows = len(data)
    ncols = len(data[0])
    
    fig, elem= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, figsize=(33/2.54,22/2.54),dpi=100)  
    fig.suptitle(f"{name}", y=0.95)
    row = 0
    col = 0

    for row in range(nrows):
        for col in range(ncols):
            d = data[row][col]          
            if len(d)>0:
                elem[row,col].plot(time, d, '-')
                #elem[row,col].axvline(window_left, color = "red", alpha= 0.3)
                elem[row,col].grid(True,linestyle='dotted',alpha=0.7)
                # elem[row,col].set_yscale("symlog")
            else:
                elem[row,col].text(0.4, 0.5,'No Data', fontsize=12, color='red', alpha=0.5, ha='center', va='center')
    fig.text(0.5, 0.08, 'time [s]', ha='center', va='center', fontsize=12)
    fig.text(0.08, 0.5, 'magnetic Field [pT]', ha='center', va='center', rotation='vertical', fontsize=12)
    if save:
        plt.savefig(path+f'{name}_4x4plot.png')
    plt.show()

def heart_vector_projections(heart_vector_data,name, save=False, path=None,  Uncertainty = 0.015):
    x = heart_vector_data[0]
    y = heart_vector_data[1]
    z = heart_vector_data[2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi = 200)
    fig.suptitle(f"Projections of 'MHV', {name}")
    axes[0].plot(x, y, label='xy-Projection')
    for i in range(0, len(x), 50):  
        arrow_start = [x[i], y[i]]
        arrow_end = [x[i + 1], y[i + 1]]
        axes[0].annotate('', arrow_end, arrow_start, arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', linewidth=2))
    # axes[0].fill_between(x, y - Uncertainty, y + Uncertainty, where =(y + Uncertainty+0.0001>=0), color='blue', alpha=0.2, label='Uncertainty')
    # axes[0].fill_between(x,  y - Uncertainty, y + Uncertainty , where =(y - Uncertainty+0.0001<=0), color='blue', alpha=0.2)
    # axes[0].fill_betweenx(y, x - Uncertainty, x + Uncertainty,where =(x + Uncertainty+0.0001>=0), color='blue', alpha=0.2)
    # axes[0].fill_betweenx(y,  x - Uncertainty, x + Uncertainty,where =(x - Uncertainty+0.0001<=0), color='blue', alpha=0.2)
    axes[0].set_xlabel('Bx [pT]')
    axes[0].set_ylabel('By [pT]')
    axes[0].legend()
    
    axes[1].plot(x, z, label='xz-Projection', color='orange')
    for i in range(0, len(x), 50):  
        arrow_start = [x[i], z[i]]
        arrow_end = [x[i + 1], z[i + 1]]
        axes[1].annotate('', arrow_end, arrow_start, arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', linewidth=2))
    # axes[1].fill_between(x, z - Uncertainty, z + Uncertainty, color='blue', alpha=0.2, label='Uncertainty')
    # axes[1].fill_betweenx(z, x - Uncertainty, x + Uncertainty, color='blue', alpha=0.2)
    axes[1].set_xlabel('Bx [pT]')
    axes[1].set_ylabel('Bz [pT]')
    axes[1].legend()
    
    axes[2].plot(y, z, label='yz-Projection', color='green')
    for i in range(0, len(x), 50):  
        arrow_start = [y[i], z[i]]
        arrow_end = [y[i + 1], z[i + 1]]
        axes[2].annotate('', arrow_end, arrow_start, arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', linewidth=2))
    # axes[2].fill_between(y, z - Uncertainty, z + Uncertainty, color='blue', alpha=0.2, label='Uncertainty')
    # axes[2].fill_betweenx(z, y - Uncertainty, y + Uncertainty, color='blue', alpha=0.2)
    axes[2].set_xlabel('By [pT]')
    axes[2].set_ylabel('Bz [pT]')
    axes[2].legend()
    
    plt.tight_layout()
    # plt.show()
    if save:
        plt.savefig(path+f'{name}_heart_vector_proj.png')

###


def avg_window(data, peak_positions, sampling, num_ch, window_left = 0.3, window_right =0.4):
    samples_left = int(window_left * sampling)
    samples_right = int(window_right * sampling)
    window_length = samples_left + samples_right

    avg_channels = []

    for ch in range(num_ch):
        windows = []
        for pos in peak_positions:
            # Sicherstellen, dass das Fenster nicht aus dem Signal rausfällt
            if pos - samples_left < 0 or pos + samples_right > data.shape[1]:
                continue  # Fenster unvollständig → überspringen
            window = data[ch, pos - samples_left : pos + samples_right]
            time = np.linspace(0, len(window) / sampling, num=len(window), endpoint=False)
            # Drift entfernen
            window_detrended = remove_drift_and_offset(window, time)
            windows.append(window_detrended)
        if windows:
            mean_window = np.mean(windows, axis=0)
        else:
            mean_window = np.zeros(window_length)  # Fallback falls kein Fenster gültig war

        avg_channels.append(mean_window)
    time_window = np.linspace(0, window_length / sampling, num=window_length, endpoint=False)

    return np.array(avg_channels), time_window



def load_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = ast.literal_eval(file.read())
        quspin_gen_dict = data.get('quspin_gen_dict', {})
        quspin_channel_dict = data.get('quspin_channel_dict', {})
        quspin_position_list =data.get('quspin_position_list', [])
    return quspin_gen_dict, quspin_channel_dict, quspin_position_list



def change_to_consistent_coordinate_system(data, quspin_gen_dict, quspin_channel_dict):
    uniform_cosy_data = []
    for quspin, ch in quspin_channel_dict.items(): 
        if "-" in str(ch):
            sign = (-1)
        else:
            sign = 1
        # sign = np.sign(ch)
        ch = abs(int(ch))
        data[ch] *= sign
        
        if "y" in quspin:
            data[ch] *=(-1)
        elif quspin_gen_dict[quspin[:-2]]==2:
            data[ch] *=(-1)
    uniform_cosy_data = data
    
    
    return uniform_cosy_data

    
def get_field_directions(quspin_channel_dict, quspin_position_list, data):
    # Initialisiere leere Listen für x, y, z Daten
    x_data = [[[],[],[],[]] for _ in range(len(quspin_position_list))]
    y_data = [[[],[],[],[]] for _ in range(len(quspin_position_list))]
    z_data = [[[],[],[],[]] for _ in range(len(quspin_position_list))]
    
    for r,row in enumerate(quspin_position_list):
        for i,quspin in enumerate(row):
        
            # print(quspin)
            for channel_suffix in ['_x', '_y', '_z']:
                channel_name = quspin + channel_suffix
                channel_index = quspin_channel_dict.get(channel_name, None) 
                if channel_index is not None:
                    channel_index = abs(int(channel_index))
                # print(f"col: {i}, row {r}")
                # print(channel_index)
                if channel_index is not None:
                    channel_data = data[channel_index]
                    
                    # Füge die 10 Werte des Kanals der entsprechenden Liste hinzu
                    if channel_suffix == '_x':
                        x_data[r][i].extend(channel_data)
                    elif channel_suffix == '_y':
                        y_data[r][i].extend(channel_data)
                    elif channel_suffix == '_z':
                        z_data[r][i].extend(channel_data)

    return x_data, y_data, z_data




def find_peak_positions_fixed_ch(data,time,ch, z_threshold = 2.5):
    meanvalue = np.median(data[ch])   
    peak_positions, peak_prop = find_peaks(data[ch], height=meanvalue*5, distance = 500)#, width = (10,60) )
    peak_heights = peak_prop["peak_heights"]

    print(f"{len(peak_positions)} peaks detected in ch {ch}; mean height: {round(np.mean(peak_heights), 2)} pT")
    
    z_scores = np.abs((peak_heights - np.median(peak_heights)) / np.std(peak_heights))
    peak_positions = peak_positions[z_scores < z_threshold]
    
    print(f"{len(peak_positions)} beats detected; average bpm: { round(len(peak_positions) / time[-1] * 60,2) }")
    # print(peak_positions)
    return peak_positions



def find_cleanest_channel(data):
    for ch in range(len(data)):
        meanvalue = np.median(data[ch])

        _, peak_prop = find_peaks(data[ch], height=meanvalue*5, distance = 500, width = (15,50) )
        peak_heights = peak_prop["peak_heights"]
        if ch == 0:
            mean = np.mean(peak_heights)
            best_mean = mean
            argmax = 0
        else:
            mean =  np.mean(peak_heights)
            if mean > best_mean:
                best_mean = mean
                argmax = ch
            
    return argmax


def make_animation(data):
    global quiver
    global ax
    def update(idx, data):
        global quiver
        x = data[0]
        y = data[1]
        z = data[2]
        quiver.remove()
        time_text.set_text(f'Time: {idx} ms')
        quiver = ax.quiver(*get_arrow(x[idx], y[idx], z[idx]))
        return []
    def get_arrow(vx, vy, vz):
        x = 0
        y = 0
        z = 0
        u = vx
        v = vy
        w = vz
        return x, y, z, u, v, w
    
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(12, 12), dpi=100)
    quiver = ax.quiver(*get_arrow(0, 0, 0),linewidths=2)
    time_text = ax.text(0.2, 0.1,0.1, s='', transform=ax.transAxes, fontsize=9)
    
    limx = abs(max(data[0]))*1.2
    
    limy = abs(max(data[1]))*1.2
    
    limz = abs(max(data[2]))*1.2
    
    ax.set_xlim(-limx, limx)
    ax.set_ylim(-limy, limy)
    ax.set_zlim(-limz, limz)
    ax.set_xlabel('Bx [pT]')
    ax.set_ylabel('By [pT]')
    ax.set_zlabel('Bz [pT]')
    ax.set_title('Vector animation')
    ani = FuncAnimation(fig, update, frames=range(len(data[0])), fargs=(data,), interval=0.01, blit=True)
    return ani

def data_single_sensor(quspin, data, quspin_channel_dict, quspin_gen_dict):
    single_data = [[],[],[]]
    for sensor in quspin_channel_dict:
        if quspin in sensor:
            ch = abs(int(quspin_channel_dict[sensor]))
            if "x" in sensor:
                single_data[0]=data[ch]
            if "y" in sensor:
                single_data[1]=data[ch]
            if "z" in sensor:
                single_data[2]=data[ch]
        # if quspin_gen_dict[quspin]==2:
        #     single_data[0]*=-1
        #     single_data[1]*=-1
    return single_data


def downsample_data(data, original_fs, target_fs):
    down_factor = int(original_fs/target_fs)
    data = [np.mean(data[i*down_factor :(i+1)*down_factor]) for i in range(int(len(data)/down_factor))]
    return data



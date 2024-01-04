import mne
import os
import numpy as np
import mne
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
import scipy.io as scio
def extract_psd_features(data, low_freq, high_freq):
    """提取 PSD 特征
    data: 数据
    """
    data_array, _ = data[:, :]  # 提取整个窗口的数据为一个numpy数组
    psds, freqs = psd_array_welch(data_array, fmin=low_freq, fmax=high_freq, n_fft=8*200, sfreq=200)
    print(psds)
    return psds

def crop_data_between_events(filtered_data, event_start, event_end):
    """
    截取event_start到 event_end之间的有用数据
    filtered_data: 大滤波之后的数据
    event_start: 开始的事件信息 16
    event_end: 结束的事件信息 64
    Returns: 截取后的数据 cropped_data
    """
    # events, _ = mne.events_from_annotations(filtered_data)
    # print(events)
    times = filtered_data.times
    # print(times)
    # print("event_start")
    # print(event_start)
    # print("event_end")
    # print(event_end)
    start_time = times[event_start]
    end_time = times[event_end]
    print("start_time")
    print(start_time)
    print("end_time")
    print(end_time)
    cropped_data = filtered_data.copy().crop(tmin=start_time, tmax=end_time)
    return cropped_data

def sliding_window(cropped_data, window_length, step_size):
    """
    cropped_data:截取后的数据
    window_length: 窗口长度（以秒为单位）
    step_size: 步进（以秒为单位）
    Returns: 滑动窗口的列表
    """
    sfreq = cropped_data.info['sfreq']  # 采样频率（以赫兹为单位）
    window_length_samples = 8
    step_size_samples = 4
    times = cropped_data.times
    print(len(times))
    num_windows = int((len(times) - (window_length_samples*200)) //( step_size_samples*200) + 1)
    print(num_windows)
    windows_start = []
    for start in np.arange(0, ((num_windows - 1) * 8), 4):
        window_start = start
        windows_start.append(window_start)
    return windows_start
    #     end = start + (num_windows)*8
    #     if end <= len(times):
    #       window = cropped_data.copy().crop(tmin=start, tmax=start + 8)
    #       windows.append(window)
    # return windows

def extract_windows_from_start_points(data, windows_start, window_length):
    """
    根据窗口的起始点列表提取滑动窗口后的数据列表
    data: 原始数据
    windows_start: 窗口的起始点列表
    window_length: 窗口长度（以秒为单位）
    Returns: 滑动窗口后的数据列表
    """
    sfreq = data.info['sfreq']  # 采样频率（以赫兹为单位）
    window_length_samples = int(window_length * sfreq)  # 将窗口长度转换为采样点数
    windows_data = []  # 初始化窗口数据列表
    for start in windows_start:
        end = start + window_length_samples
        if end <= len(data.times):
            window = data.copy().crop(tmin=start / sfreq, tmax=end / sfreq)
            windows_data.append(window)  # 将窗口数据添加到列表中
    return windows_data  # 返回滑动窗口后的数据列表



name = 'cmh'
script_directory = os.path.dirname(os.path.abspath(__file__))
print(script_directory)
folder_path = os.path.join(script_directory,'PredData',str(name))
files = os.listdir(folder_path)

all_psd_features = None

for file_name in files:
    if file_name.endswith('resample.set') and file_name.startswith(str(name)+'-'):
        file_path = os.path.join(folder_path,file_name)
        print("reading file:",file_path)
        raw = mne.io.read_raw_eeglab(file_path, preload=True)

        freq_bands = [(1,4),(4,8),(8,13),(13,30),(30,48)]
        segmented_data = {}

        for index,(low_freq, high_freq) in enumerate(freq_bands,start=1):
            filtered_data = raw.copy().filter(low_freq,high_freq,method = 'iir')
            key = f'{low_freq}-{high_freq}'
            original_index = str(file_name.split('-')[-1].split('.')[0])
            segmented_data[key] = filtered_data
            save_folder = os.path.join(script_directory,'Bigfiltered',str(name))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            filtered_data.save(os.path.join(save_folder,f'{str(name)}-{original_index}_filetered_{key}.fif'), overwrite=True)

            # print(filtered_data)
            # print(filtered_data.info)
            # print(filtered_data.ch_names)
            # print(filtered_data.info['dig'])
            events, _ = mne.events_from_annotations(filtered_data)
            event_start = events[0, 0]
            print(event_start)
            event_end = events[-1, 0]
            print(event_end)
            cropped_data = crop_data_between_events(filtered_data, event_start, event_end)


            window_length =  8
            step_size = 4
            # print(cropped_data)
            # print(cropped_data.info)
            # print(cropped_data.ch_names)
            # print(cropped_data.info['dig'])
            windows_start = sliding_window(cropped_data, window_length, step_size)
            windows_data = extract_windows_from_start_points(cropped_data, windows_start, window_length)
            # print(windows_data)
            #
            # print(windows_start)

            print("PSD features extraction")
            # all_psd_features = []
            for window_index, window in enumerate(windows_data, start=1):

                if window_index == 0:
                    num_freq_bands = len(freq_bands)
                    num_windows = len(windows_data)  # 现在可以确定窗口数量
                    num_channels = len(raw.ch_names)
                    num_features = extract_psd_features(windows_data[0], freq_bands[0][0], freq_bands[0][1]).shape[1]
                    all_psd_features = np.zeros((num_freq_bands, num_windows, num_channels, num_features))

                psd_features = extract_psd_features(window, low_freq, high_freq)
                all_psd_features[index - 1, window_index - 1, :] = psd_features
                # save_folder = "D:\RSVP\data"
                # feature_name = f'{str(name)}-{original_index}_feature_{key}.mat'
                # scio.savemat(os.path.join(save_folder,feature_name),{'psd_features': all_psd_features})
        # feature_name = f'{str(name)}_psd.mat'
        # scio.savemat(os.path.join(script_directory, feature_name),{'psd': all_psd_features})
            feature_name = f'{str(name)}_{original_index}_psd.mat'
            scio.savemat(os.path.join(script_directory, feature_name), {'psd': all_psd_features})



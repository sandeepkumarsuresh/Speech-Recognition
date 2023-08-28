import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from dtw import dtw, asymmetric, typeIVc

def create_windows(data, hann_window, window_size, hop_size):
    # create windows over the audio signal
    data_windows = sliding_window_view(data, window_size)[::hop_size,:]
    # repeat hann windows for each window in the audio signal and multiply
    hann_windows = np.repeat(hann_window.reshape(1, -1), data_windows.shape[0], axis=0)
    # Multiply hann window with audio signal
    data_windows = np.multiply(data_windows, hann_windows)
    return data_windows

def frequency2mel(freq):
    return 2595 * np.log10(1.0 + freq / 700)

def mel2frequency(mels):
    return 700 * -1 * (1.0 - 10**(mels / 2595))

def get_deltas(mfcc):
    n_frames, n_dcts = mfcc.shape
    # delta mfcc
    mfcc_d = np.zeros((n_frames, n_dcts))
    mfcc_padded = np.pad(mfcc, ((1, 1), (0, 0)), mode='edge')
    for i in range(n_frames):
        mfcc_d[i] = np.dot(np.arange(-1, 2), mfcc_padded[i:i+3])
    # delta^2 mfcc
    mfcc_dd = np.zeros((n_frames, n_dcts))
    mfcc_padded = np.pad(mfcc, ((2, 2), (0, 0)), mode='edge')
    for i in range(n_frames):
        mfcc_dd[i] = np.dot(np.arange(-2, 3), mfcc_padded[i:i+5]) / 5
    return mfcc_d, mfcc_dd
    
def get_mfcc(filepath, n_fft=512, sr=20000, hop_size=0.010, n_mels=40, n_dcts=20):
    # read file
    if isinstance(filepath, str):
        sr, data = wavfile.read(filepath)
    else:
        sr, data = sr, filepath  # pass data directly
    data = np.array(data / 32768.0, dtype=np.float32)
    data = np.pad(data, n_fft//2, mode='reflect')
    window_size = n_fft
    hop_size = int(hop_size * sr)
    # make windows
    hann_window = np.hanning(window_size)
    data_windows = create_windows(data, hann_window, window_size, hop_size)
    # compute fft
    n_frames, n_fft= data_windows.shape
    n_rfft = 1 + n_fft//2
    data_windows_T = np.transpose(data_windows)
    fft_windows = np.zeros((n_rfft, n_frames), dtype=np.complex64, order='F')
    for i in range(n_frames):
        fft_windows[:, i] = fft.fft(data_windows_T[:, i], axis=0)[:n_rfft]
    fft_windows = np.transpose(fft_windows)
    # compute power spectrum
    power_windows = np.square(np.abs(fft_windows))
    # get mels
    f_min, f_max = 0, sr / 2
    mels = np.linspace(frequency2mel(f_min), frequency2mel(f_max), n_mels+2)
    frequencies = mel2frequency(mels)
    filter_peaks = np.array(np.floor(frequencies * (n_fft + 1)/sr), dtype=int)
    filters = np.zeros((len(filter_peaks)-2, n_fft//2+1))
    for i in range(len(filter_peaks)-2):
        filters[i, filter_peaks[i + 1] : filter_peaks[i + 2]] = np.linspace(
            1, 0, filter_peaks[i + 2] - filter_peaks[i + 1])
        filters[i, filter_peaks[i] : filter_peaks[i + 1]] = np.linspace(
            0, 1, filter_peaks[i + 1] - filter_peaks[i])
    # normalization
    norm = 2.0 / (frequencies[2:n_mels+2] - frequencies[:n_mels])
    filters = filters * norm[:, np.newaxis]
    data_filters = np.dot(filters, power_windows.T)
    data_filters_log = 10.0 * np.log10(data_filters)
    # DCT
    dct_filters = np.ones((n_dcts, n_mels)) / np.sqrt(n_mels)
    vector = np.pi * np.arange(1, 2*n_mels, 2) / (2*n_mels)
    for i in range(1, n_dcts):
        dct_filters[i, :] = np.cos(i * vector)
    dct_filters *= np.sqrt(2 / n_mels)
    mfcc = np.dot(dct_filters, data_filters_log)
    # get deltas
    d_mfcc, dd_mfcc = get_deltas(mfcc)
    mfcc = np.vstack((mfcc, d_mfcc, dd_mfcc))
    return mfcc.T

def distance(ref_frame, test_frame):
    return np.sum(np.abs(ref_frame - test_frame))

# def dtw(ref, test):
#     R, T = len(ref), len(test)
#     D = np.zeros((R, T))
#     D[0][0] = distance(ref[0], test[0])
#     for i in range(1, R):
#         D[i][0] = D[i-1][0] + distance(ref[i], test[0])
#     for i in range(1, T):
#         D[0][i] = D[0][i-1] + distance(ref[0], test[i])
#     for i in range(1, R):
#         for j in range(1, T):
#             d_ij = distance(ref[i], test[j])
#             D[i][j] = min(
#                 D[i-1][j] + d_ij,
#                 D[i-1][j-1] + 2 * d_ij, 
#                 D[i][j-1] + d_ij
#             )
#     return D[-1][-1]


def get_vad_chunks(filepath):
   
    sr, data = wavfile.read(filepath)
    data = np.array(data / 32767.0, dtype=np.float32)
    print(len(data), sr)

    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    window_size = int(0.025 * sr)  # 25 ms windows
    hop_size = 1
    hann_window = np.hanning(window_size)
    data_windows = create_windows(data, hann_window, window_size, hop_size)
    n_windows = data_windows.shape[0]

    # Calculate short-term energy
    energy_windows = np.power(data_windows, 2).sum(axis=1) / window_size

    # Calculate ZCR
    zcr_windows = []
    data_raw_windows = create_windows(data, np.ones(window_size), window_size, hop_size)
    for idx in range(n_windows):
        dw = data_raw_windows[idx]
        dw = dw - np.mean(dw)
        zcr_indices, = np.nonzero(np.diff(dw > 0))
    #     print(len(zcr_indices))
        zcr_windows.append(len(zcr_indices) / window_size)
        
    # Set Thresholds
    # Estimate STE thresholds from first ~20 ms (white noise recorded)
    K = int(sr * .2)
    STE_THRESHOLD = 0.0001 #np.max(energy_windows[:K])

    # Estimate ZCR from first ~20 ms (recorded in a noisy environment)
    ZCR_THRESHOLD = np.mean(zcr_windows[:K])

    # # print(ZCR_THRESHOLD, STE_THRESHOLD)
    # plt.figure(figsize=(15, 3))
    # plt.plot(energy_windows)
    # plt.plot([STE_THRESHOLD for _ in range(len(energy_windows))],
    #         linestyle='--', label='STE Threshold')
    # plt.legend()
    # plt.xlabel('t (frames) ->')
    # plt.show()
    # plt.figure(figsize=(15, 3))
    # plt.plot(zcr_windows)
    # plt.plot([ZCR_THRESHOLD for _ in range(len(energy_windows))],
    #         linestyle='--',label='ZCR Threshold')
    # plt.legend()
    # plt.show()
    # voiced = []
    # for zcr, ste in zip(zcr_windows, energy_windows):
    #     if ste < STE_THRESHOLD and zcr > ZCR_THRESHOLD:
    #         voiced.append(0)
    #     else:
    #         voiced.append(1)
    # y = np.multiply(voiced, data[:len(voiced)])
    # plt.figure(figsize=(15, 3))
    # plt.plot(y, label='voiced', alpha=0.9)
    # plt.plot(data, label='original', alpha=0.5)
    # plt.legend()
    # plt.show()
    # wavfile.write('temp.wav', sr, y)

    idx = 0
    chunks = []
    chunk = []
    start_new_chunk = True
    while idx < len(energy_windows):
        if energy_windows[idx] > STE_THRESHOLD:
            chunk.append(data[idx])
        else:
            if len(chunk) > 50:
                chunks.append(chunk)
                chunk = []
        idx += 1
    return chunks, sr

def main():
    data_dir = 'Connected_Digits/Team-05'
    word = isolated_digit = 'Isolated_Digits/3/train/ac_3.wav'
    feat_word = get_mfcc(word)
    
    # for connected_digits in os.listdir(data_dir):
    #     filepath = os.path.join(data_dir, connected_digits)
    #     cds.append(get_mfcc(filepath))


    # for test in os.listdir(data_dir):
    #     filepath = os.path.join(data_dir, test)
    #     chunks, sr = get_vad_chunks(filepath)
    #     print(filepath)
    #     for chunk in chunks:
    #         chunk = np.array(chunk)
    #         feat_chunk = get_mfcc(chunk)
    #         print(feat_chunk.shape, feat_word.shape)
    #         if feat_chunk.shape[0] <= 1:
    #             continue
    #         alignment = dtw(feat_chunk, feat_word)
    #         print(alignment.distance)
    #         alignment.plot()
    #         plt.show()
    
        
    cd = 'Connected_Digits/Team-05/3z3a.wav'
    feat_cd = get_mfcc(cd)
    alignment = dtw(feat_cd, feat_word, keep_internals=True, step_pattern=asymmetric, open_begin=True, open_end=True)
    print(alignment.distance)
    alignment.plot()
    plt.xlabel('Test index (3z3)')
    plt.ylabel('Reference index (3)')
    plt.savefig('plots/uedtw_3z3.png')
    # alignment = dtw(feat_cd, feat_word, keep_internals=True, step_pattern=asymmetric, open_begin=False, open_end=False)
    # alignment.plot()
    # plt.show()

    cd = 'Connected_Digits/Team-05/3oa.wav'
    feat_cd = get_mfcc(cd)
    alignment = dtw(feat_cd, feat_word, keep_internals=True, step_pattern=asymmetric, open_begin=True, open_end=True)
    print(alignment.distance)
    alignment.plot()
    plt.xlabel('Test index (3oa)')
    plt.ylabel('Reference index (3)')
    plt.savefig('plots/uedtw_3oa.png')
    plt.show()
    


if __name__ == '__main__':
    main()
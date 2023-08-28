import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


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
    
def get_mfcc(filepath, n_fft=1024, sr=20000, hop_size=0.010, n_mels=40, n_dcts=13):
    # read file
    sr, data = wavfile.read(filepath)
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

def dtw(ref, test):
    R, T = len(ref), len(test)
    D = np.zeros((R, T))
    D[0][0] = distance(ref[0], test[0])
    for i in range(1, R):
        D[i][0] = D[i-1][0] + distance(ref[i], test[0])
    for i in range(1, T):
        D[0][i] = D[0][i-1] + distance(ref[0], test[i])
    for i in range(1, R):
        for j in range(1, T):
            d_ij = distance(ref[i], test[j])
            D[i][j] = min(
                D[i-1][j] + d_ij,
                D[i-1][j-1] + 2 * d_ij,
                D[i][j-1] + d_ij
            )
    return D[-1][-1]


def main():
    data_dir = 'Isolated_Digits'
    digits = ['1', '3', '4', 'o', 'z']
    refs, refs_label, tests, tests_label = [], [], [], []
    for digit in digits:
        train_dir = os.path.join(data_dir, digit, 'train')
        for file in os.listdir(train_dir):
            filepath = os.path.join(train_dir, file)
            refs.append(get_mfcc(filepath))
            refs_label.append(digit)
        
        dev_dir = os.path.join(data_dir, digit, 'dev')
        for file in os.listdir(dev_dir):
            filepath = os.path.join(dev_dir, file)
            tests.append(get_mfcc(filepath))
            tests_label.append(digit)
    
    correct = {5: 0, 10: 0, 15: 0}
    for test, test_label in tqdm(list(zip(tests, tests_label))):
        distances, pred = {}, np.inf
        # calculate all distances
        for ref, ref_label in tqdm(list(zip(refs, refs_label))):
            D = dtw(ref, test)
            # keep track of all ref distances
            if ref_label not in distances:
                distances[ref_label] = [D]
            else:
                distances[ref_label].append(D)
        # sort top-k and average for every ref_label
        for K in [5, 10, 15]:
            best_distance, pred = np.inf, -1
            for ref_label, ref_distances in distances.items():
                ref_distances_k = ref_distances[:K]
                avg_distance = sum(ref_distances_k) / len(ref_distances_k)
                if avg_distance < best_distance:
                    best_distance, pred = avg_distance, ref_label
            if (pred == test_label):
                correct[K] += 1
            print(f'K={K}: ', correct[K] / len(tests))


if __name__ == '__main__':
    main()
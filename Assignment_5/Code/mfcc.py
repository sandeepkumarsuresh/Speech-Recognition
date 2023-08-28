import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import pathlib

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
    
def get_mfcc(filepath, n_fft=512, sr=20000, hop_size=0.010, n_mels=40, n_dcts=13):
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




def main():
   
    dir = '/home/tenet/Desktop/CS22Z121/Assignment_5/Isolated Digits'
    digits = ['1' , '3' , '4' , 'o' , 'z']
    if not os.path.exists('mfcc_test'):
        os.makedirs('mfcc_test')
    mfcc_file_dir = 'mfcc_test'    
    for digit in digits:
        digit_dir = os.path.join(mfcc_file_dir,digit)
        if not os.path.exists(digit_dir):
            os.makedirs(digit_dir)
        # train_dir = os.path.join(dir,digit,'train')
        test_dir = os.path.join(dir,digit,'dev')

        for audio in os.listdir(test_dir):
            filepath = os.path.join(test_dir,audio)  
            file_prefix = pathlib.Path(audio).stem     
            print(filepath)
            mfcc_audio = get_mfcc(filepath)
            o_filename = "{}.mfcc".format(file_prefix)
            o_file_path = os.path.join(digit_dir,o_filename)
            print(o_file_path)
            # o_file = open(o_file_path,'w')
            # # o_file.write(str(mfcc_audio))
            # o_file.write(str(np.savetxt(o_file,mfcc_audio,delimiter=' ')))
            # o_file.close()
            np.savetxt(o_file_path,mfcc_audio,delimiter=' ')



if __name__ == '__main__':
    main()
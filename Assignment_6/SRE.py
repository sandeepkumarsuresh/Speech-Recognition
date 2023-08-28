import numpy as np
import pandas as pd
import os
import random
from tqdm.auto import tqdm
import scipy.io.wavfile as wav
from numpy.lib.stride_tricks import sliding_window_view
import pathlib




''''Python Program to Convert SRE-Dataset into a MFCC File'''

def STE(data_frames):
    return np.sum(np.square(data_frames))


def windowing(audio_filepath,hann_window,win_size = 512, w_hop = 0.01):

    fs,audio = wav.read(audio_filepath)
    audio = audio/np.max(np.abs(audio)) 
    hop_size = int(w_hop*fs)
    data_frame = sliding_window_view(audio,win_size)[::hop_size,:]
    hann_windows = np.repeat(hann_window.reshape(1,-1),data_frame.shape[0],axis=0)
    data_window = np.multiply(data_frame,hann_windows)

    return data_window


def VAD(filepath):
    
    
    x = windowing(filepath,hann_window)
    voiced_audio = []
    threshold = 0.0001
    for frames in x:
        if STE(frames) > threshold:
            voiced_audio.append(frames)

    return np.stack(np.asarray(voiced_audio))    

def frequency2mel(freq):
    return 2595 * np.log10(1.0 + freq / 700)

def mel2frequency(mels):
    return 700 * -1 * (1.0 - 10**(mels / 2595))

    
def get_mfcc(data_windows, n_fft=512, sr=8000, hop_size=0.010, n_mels=40, n_dcts=13):

    # compute fft
    n_frames, n_fft= data_windows.shape
    n_rfft = 1 + n_fft//2
    data_windows_T = np.transpose(data_windows)
    fft_windows = np.zeros((n_rfft, n_frames), dtype=np.complex64, order='F')
    for i in range(n_frames):
        fft_windows[:, i] = np.fft.fft(data_windows_T[:, i], axis=0)[:n_rfft]
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

    mfcc_cepstral = (mfcc - np.mean(mfcc))/np.std(mfcc)
  
    mfcc_cepstral = np.vstack((mfcc_cepstral))

    return mfcc_cepstral.T



if  __name__ == '__main__':


    SRE_dataset_path = 'SRE_Dataset'
    SRE_dataset_output_path = 'VAD_SRE_Dataset'
    hann_window = np.hanning(512)
    STE_threshold = 0.0001



    for folders in tqdm(sorted(os.listdir(SRE_dataset_path))):
        if not os.path.exists(os.path.join(SRE_dataset_output_path+folders)):  
            os.makedirs(os.path.join(SRE_dataset_output_path,folders) )
        for files in os.listdir(os.path.join(SRE_dataset_path,folders)):
            audio_filepath = os.path.join(SRE_dataset_path,folders,files)
            voice_activity = VAD(audio_filepath)
            file_prefix = pathlib.Path(files).stem
            mfcc_vad = get_mfcc(voice_activity)
            output_filename = "{}.mfcc".format(file_prefix)
            output_filepath = os.path.join(SRE_dataset_output_path,folders,output_filename)
            np.savetxt(output_filepath,mfcc_vad,delimiter=' ')



        








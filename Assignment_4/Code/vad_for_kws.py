#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile


# In[5]:


def create_windows(data, hann_window, window_size, hop_size):
    # create windows over the audio signal
    data_windows = sliding_window_view(data, window_size)[::hop_size,:]
    # repeat hann windows for each window in the audio signal and multiply
    hann_windows = np.repeat(hann_window.reshape(1, -1), data_windows.shape[0], axis=0)
    # Multiply hann window with audio signal
    data_windows = np.multiply(data_windows, hann_windows)
    data_windows.shape
    return data_windows


# In[67]:


filepath = 'Connected_Digits/Team-05/3oa.wav'  # this is me speaking with pauses
sr, data = wavfile.read(filepath)
data = np.array(data / 32767.0, dtype=np.float32)
print(len(data), sr)

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

window_size = int(0.025 * sr)  # 25 ms windows
# hop_size = window_size//2
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

# print(ZCR_THRESHOLD, STE_THRESHOLD)
plt.figure(figsize=(15, 3))
plt.plot(energy_windows)
plt.plot([STE_THRESHOLD for _ in range(len(energy_windows))],
         linestyle='--', label='STE Threshold')
plt.legend()
plt.xlabel('t (frames) ->')
plt.show()
plt.figure(figsize=(15, 3))
plt.plot(zcr_windows)
plt.plot([ZCR_THRESHOLD for _ in range(len(energy_windows))],
         linestyle='--',label='ZCR Threshold')
plt.legend()
plt.show()

voiced = []
for zcr, ste in zip(zcr_windows, energy_windows):
    if ste < STE_THRESHOLD and zcr > ZCR_THRESHOLD:
        voiced.append(0)
    else:
        voiced.append(1)

y = np.multiply(voiced, data[:len(voiced)])
plt.figure(figsize=(15, 3))
plt.plot(y, label='voiced', alpha=0.9)
plt.plot(data, label='original', alpha=0.5)
plt.legend()
plt.show()

wavfile.write('temp.wav', sr, y)


# In[68]:


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
print(len(chunks))


# In[73]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(chunks[0])
ax[1].plot(chunks[1])
plt.show()


# In[ ]:





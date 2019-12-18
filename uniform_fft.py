#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

Fs = 524288.0
Ts = 1.0/Fs
t = np.arange(0, 1, Ts) # t为0，0.00，...，1（共Fs点数）
y = np.load("ae1.npy")  # y为数据（共数据点数）
n = len(y)              # n为数据点数
k = np.arange(n)        # k为0，1，...，n（共数据点数）
T = n*Ts                # T为实际数据时长
freq = k/T              # 1.0/T为显示分辨率，freq为频谱(全部)
freq1 = freq[:int(float(n)/2)]        # 频谱只显示数据点一半（一半）
YY = 20*np.log10(abs(np.fft.fft(y)))       # fft 复数  （全部）
Y = 20*np.log10(abs(np.fft.fft(y)/n))      #  正确显示波形能量  （全部）
Y1 = Y[:int(float(n)/2)]      #  fft 幅值  （一半）

fig, ax = plt.subplots(4, 1, figsize=(10, 15))
ax[0].plot(t, y)         # 显示1s内数据波形
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')

ax[1].plot(freq, YY, 'r')
ax[1].set_xlabel('Freq(Hz)')
ax[1].set_ylabel('|Y(freq)|')

ax[2].plot(freq, Y, 'G')
ax[2].set_xlabel('Freq(Hz)')
ax[2].set_ylabel('|Y(freq)|')

ax[3].plot(freq1, Y1, 'B')
ax[3].set_xlabel('Freq(Hz)')
ax[3].set_ylabel('|Y(freq)|')

plt.show()

# %%

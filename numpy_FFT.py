# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
fs = 524288
fft_size = 512
ae1 = np.load("ae2.npy")
ae = ae1[:fft_size]
t = np.arange(0, 1.0, 1.0/fs)
np_f = np.fft.rfft(ae)/fft_size
# * 由公式可知/fft_size为了正确显示波形能量
np_freqs = np.linspace(0, fs/2, fft_size/2+1)
# * rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
# * 于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
np_fp = 20*np.log10(np.clip(np.abs(np_f), 1e-20, 1e100))
# * 最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。
# * 为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理
print(np_freqs)
print(np_f.shape)
print(np_fp.shape)
plt.figure(figsize=(8, 4))
plt.subplot(2, 1, 1)
plt.plot(t[:fft_size], ae)
plt.subplot(2, 1, 2)
plt.plot(np_freqs, np_fp)
plt.show()

# %%
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

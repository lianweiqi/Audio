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

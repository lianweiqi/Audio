# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import soundfile as sf 
# %%
sample_rate, signal = scipy.io.wavfile.read('test.wav')
print(sample_rate, signal, signal.shape)
# %%
y, sr = librosa.load("test.wav", sr=None)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f}beats per minute'.format(tempo))
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
y_harmonic, y_percussive = librosa.effects.hpss(y)
print(sr)
print(y.shape)
print(y.dtype)
fig, ax = plt.subplots(2, 1, figsize=(10, 16))
# * librosa.display.waveplot 波形幅值包络线
librosa.display.waveplot(y[1200:1500], sr, ax=ax[0])
ax[1].plot(y[1200:1500])
plt.show()
# %%
print(beat_times.shape)
print(beat_times)
# %%
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.power_to_db(melspec)
print(logmelspec.shape)
plt.figure()
librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
plt.title('Beat wavform')
plt.show()

# %%
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
print(mfcc.shape)
plt.figure()
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# %%
y = np.load("ae1.npy")
sr = 524288
plt.figure(1)
librosa.display.waveplot(y, sr)
plt.figure(2)
plt.plot(y)
plt.show()
# %%
m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
fig, axes = plt.subplots(2, 1, figsize=(8, 12))
librosa.display.specshow(m_slaney, x_axis='time', ax=axes[0])
axes[0].set_title('RASTAMAT / Auditory toolbox (dct_type=2)')
librosa.display.specshow(m_htk, x_axis='time')
plt.colorbar(ax=axes[0])
axes[1].set_title('HTK-style (dct_type=3)')
plt.colorbar(ax=axes[1])
plt.tight_layout()
plt.show()



# %%
# * 特征融合
# 设置梅尔滤波器组参数，并设置分帧参数n_fft--帧长，hop_length--帧移
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=60, n_fft=1024, hop_length=512, fmax=16000) 
mfcc = librosa.feature.mfcc(y, sr, S=librosa.power_to_db(S), n_mfcc=40) # 提取mfcc系数
stft_coff = abs(librosa.stft(y, 1024, 512, 1024)) 
# ! 分帧然后求短时傅里叶变换，分帧参数与对数能量梅尔滤波器组参数设置要相同
energy = np.sum(np.square(stft_coff), 0) # 每一帧的平均能量
MFCC_Energy = np.vstack((mfcc, energy)) # 将每一帧的MFCC与短时能量拼接在一起
# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# %%
y, sr = librosa.load("test.wav", sr=None)
"""
部分函数
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f}beats per minute'.format(tempo))
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
"""
print(sr)
print(y.shape)
print(y.dtype)
fig, ax = plt.subplots(2, 1, figsize=(10, 16))
# * librosa.display.waveplot 波形幅值包络线
librosa.display.waveplot(y[1200:1500], sr, ax=ax[0])
ax[1].plot(y[1200:1500])
plt.show()

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

# %%
y = np.load("ae1.npy")
sr = 524288
plt.figure(1)
librosa.display.waveplot(y, sr)
plt.figure(2)
plt.plot(y)
plt.show()

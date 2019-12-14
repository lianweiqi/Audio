# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt
filename = librosa.util.example_audio_file()
y, sr = librosa.load(filename, sr=None)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f}beats per minute'.format(tempo))
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
plt.figure()
librosa.display.waveplot(y, sr)
plt.show()

# %%
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.power_to_db(melspec)
logmelspec.shape
plt.figure()
librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
plt.title('Beat wavform')
plt.show()

# %%
import numpy as np
import librosa
y, sr = librosa.load(librosa.util.example_audio_file())
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc.shape

# %%

#%%
import torch
import torchaudio
import matplotlib.pyplot as plt

#%%
filename = "./_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
waveform, sample_rate = torchaudio.load(filename)
print("shape of waveform: {}".format(waveform.size()))
print("sample rate of waveform: {}".format(sample_rate))
plt.figure()
plt.plot(waveform.t().numpy())
#%%
specgram = torchaudio.transforms.Spectrogram()(waveform)
print("shape of spectrogram: {}".format(specgram.size()))
plt.figure()
plt.imshow(specgram.log2()[1, :, :].numpy(), cmap='gray')
plt.show()
#%%
specgram = torchaudio.transforms.MelSpectrogram()(waveform)
print("shape of spectrogram:{}".format(specgram.size()))
plt.figure()
p = plt.imshow(specgram.log2()[0, :, :].detach().numpy(), cmap='gray')

# %%
new_sample_rate = sample_rate/10
channel = 0
transformed = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform[channel, :].view(1, -1))
print("shape of transform waveform: {}".format(transformed.size()))
plt.figure()
plt.plot(transformed[0, :].numpy())

# %%
print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))

# %%
def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/tensor_minusmean.abs().max()


# %%
transformed = torchaudio.transforms.MuLawEncoding()(waveform)
print("shape of mulawencoding waveform: {}".format(transformed.size()))
plt.figure()
plt.plot(transformed[0, :].numpy())

# %%
reconstructed = torchaudio.transforms.MuLawDecoding()(transformed)
print("shape of recovered waveform: {}".format(reconstructed.size()))
plt.figure()
plt.plot(reconstructed[0,:].numpy())

# %%
error = ((waveform - reconstructed).abs()/waveform.abs()).median()
print(error)

# %%

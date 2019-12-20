# %%
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
import numpy as np
# %%
Fs = 524288
x = np.load("ae1.npy")
F, f_names = ShortTermFeatures.feature_extraction(
    x, Fs, 0.010*Fs, 0.005*Fs )
plt.subplot(211)
plt.plot(F[2, :])
plt.xlabel('Frame no')
plt.ylabel(f_names[2])
plt.subplot(212)
plt.plot(F[3, :])
plt.xlabel('Frame no')
plt.ylabel(f_names[3])
plt.show()

# %%
plt.plot(x)

# %%

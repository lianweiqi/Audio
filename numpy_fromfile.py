# %%
import numpy as np
import matplotlib.pyplot as plt
with open('3.ae', 'rb') as file:
    file.seek(4, 0)
    A = np.fromfile(file, '>i2', -1)
    C = A.reshape((-1, 3))
    # print(C)
    a, b, c = np.hsplit(C, 3)
    print(a,'\n', b,'\n', c)
    plt.subplot(3, 1, 1)
    plt.plot(20*np.log10(np.abs(a[:150])))
    plt.subplot(3, 1, 2)
    plt.plot(20*np.log10(np.abs(b[:150])))
    plt.subplot(3, 1, 3)
    plt.plot(20*np.log10(np.abs(c[:150])))

# %%
import numpy as np
import matplotlib.pyplot as plt
with open('3.tev', 'rb') as f:
    f.seek(4, 0)
    tev = np.fromfile(f, '>i2', -1)
    print(tev[:150])
    plt.plot(tev[:150])
    m = 20*np.log10(np.abs(tev[:150]))
    plt.plot(m)
# %%

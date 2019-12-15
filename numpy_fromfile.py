# %%
import numpy as np
import matplotlib.pyplot as plt
with open('3.ae', 'rb') as ae:
    ae.seek(6, 0)
    A = np.fromfile(ae, '<u2', -1)
    # print(A)
    B = A.reshape((-1, 4))
    # print(B)
    a, _, b, c = np.hsplit(B, 4)
    # print(a,'\n', b,'\n', c)
    a, b, c = a.astype('int32') - 32768, b.astype('int32') - 32768, c.astype('int32') - 32768
    # print(a,'\n', b,'\n', c)
    
    plt.subplot(3, 1, 1)
    plt.plot(a[:150])
    plt.subplot(3, 1, 2)
    plt.plot(b[:150])
    plt.subplot(3, 1, 3)
    plt.plot(c[:150])
    plt.figure(2)
    plt.plot(a[:150])
    plt.plot(b[:150])
    plt.plot(c[:150])
# %%
print(a.shape)
print(b.shape)
print(c.shape)
# %%
# import numpy as np
# import matplotlib.pyplot as plt
with open('3.tev', 'rb') as tev:
    tev.seek(4, 0)
    tev_data = np.fromfile(tev, '<u2', 25690112)
    tev_data = tev_data.astype('int32') - 32768
    print(tev_data)
    plt.plot(tev_data[17250:17350])
# %%
print(tev_data.shape)


# %%

# %%
import numpy as np
import matplotlib.pyplot as plt
with open('3.ae', 'rb') as ae:
    ae.seek(10, 0)
    A = np.fromfile(ae, '<u2', -1)
    print(A)
    B = A.reshape((-1, 3))
    print(B)
    # print(C)
    a, b, c = np.hsplit(B, 3)
    print(a,'\n', b,'\n', c)
    
    plt.subplot(3, 1, 1)
    plt.plot(20*np.log10(np.abs(a[:150])))
    plt.subplot(3, 1, 2)
    plt.plot(20*np.log10(np.abs(b[:150])))
    plt.subplot(3, 1, 3)
    plt.plot(20*np.log10(np.abs(c[:150])))
# %%
print(a.shape)
print(b.shape)
# %%
import numpy as np
import matplotlib.pyplot as plt
with open('3.tev', 'rb') as tev:
    tev.seek(4, 0)
    tev_data = np.fromfile(tev, '<u2', -1)
    a = tev_data.astype('int32')
    test_data = a - 32768
    print(tev_data)
    print(a)
    print(test_data)
    # plt.plot(tev_data)
    # print(test_data)
    # plt.plot(tev_data)
    # print(test_data)
    # plt.plot(test_data)
    m = test_data[:150]
    plt.plot(m)
    plt.show()
# %%
print(tev_data)
print(tev_data.shape)


# %%

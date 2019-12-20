# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
# %%
with open('3.ae', 'rb') as ae:
    ae.seek(6, 0)
    ae_data = np.fromfile(ae, '<u2', -1)
    ae_data = ae_data.reshape((-1, 4))
    ae1, _, ae3, ae2 = np.hsplit(ae_data, 4)
    (ae1, ae2, ae3) = ((ae1.astype('int32') - 32768).astype('float16'), 
                       (ae2.astype('int32') - 32768).astype('float16'), 
                       (ae3.astype('int32') - 32768).astype('float16'))
    # * 归一化处理
    (ae1, ae2, ae3) = (ae1/32767, ae2/32767, ae3/32767)
    # * 数字采样值转mV
    # TODO 修改放大器倍数（变为uV）
    # (ae1, ae2, ae3) = (ae1*1250//32767, ae2*1250/8/32767, ae3*1250/8/32767)
    # * mV转dB
    # (ae1, ae2, ae3) = (20*np.log10(ae1), 20*np.log10(ae2), 20*np.log10(ae3))
    # * 二维数组转一维
    (ae1, ae2, ae3) = (ae1.reshape((-1, )), ae2.reshape((-1, )), ae3.reshape((-1, )))
# %%
np.save("ae1.npy", ae1)
np.save("ae2.npy", ae2)
np.save("ae3.npy", ae3)
    
# %%
print(ae1.shape)
print(sys.getsizeof(ae1))
print(ae1,'\n', ae2,'\n', ae3)
fig, ax = plt.subplots(5, figsize=(7, 20))
ax[0].plot(ae1)
ax[0].plot(ae2)
ax[0].plot(ae3)
ax[1].plot(ae1[:150])
ax[2].plot(ae2[:150])
ax[3].plot(ae3[:150])
ax[4].plot(ae1[:150])
ax[4].plot(ae2[:150])
ax[4].plot(ae3[:150])
# plt.savefig("ae.pdf", format='pdf')
# plt.savefig('ae.svg', format='svg')
# %%

# * 用于求log，将0变原数组中不为0的最小值

def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data

with open('3.tev', 'rb') as tev:
    tev.seek(4, 0)
    tev_data = np.fromfile(tev, '<u2', 25690112)
    tev_data = (tev_data.astype('int32') - 32768).astype('float16')
    # * 转mV
    tev_data = tev_data*1250/8/32767
    
    # * 转dB
    # tev_data = np.abs(tev_data)
    # tev_data = replaceZeroes(tev_data)
    # tev_data = 20*np.log10(tev_data)
    # tev_data = np.where(tev_data > 0.0000000001, 20*np.log10(tev_data), -10)
    # * 数据为一维数组
    print(tev_data.shape)
    print(tev_data)
# %%    
np.save("tev.npy", tev_data)    
# %%
plt.figure(1)
plt.plot(tev_data[15000:17000])
plt.figure(2)
plt.plot(tev_data[23000:25000])
# plt.savefig('tev.pdf', format='pdf')
# %%

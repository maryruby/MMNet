import matplotlib.pyplot as plt

mmnet_iid = [0.049309684435526524, 0.02485708196957903, 0.010341980457305944, 0.003775419394174917, 0.0013111472129822, 0.0005320831139886684]
mmnet_iid_last_loss = [0.04490489522616048, 0.02066322922706587, 0.0077814575036365685, 0.0025555169582366366, 0.0008141620953878714, 0.00029010136922180685]
mmse = [0.07856885115305556, 0.052920626401901316, 0.03281208197275809, 0.01842927098274283, 0.009156667788823403, 0.003991039594014367]
featurous = [0.04776395718256654, 0.023489480813344343, 0.009306668043137067, 0.0031977089246114643, 0.0010092715422315157, 0.0003134361902873417]
feat_last = [0.04303864677747071, 0.01967145959536243, 0.007438854773839099, 0.002288436094919799, 0.0006148922443388605, 0.00015874942143734305]
featurous_last_loss = [0.05033020853996317, 0.026566874583562106, 0.011827814976373752, 0.004443852901458545, 0.001389168898264792, 0.00038072586059600955]
y = [11, 12, 13, 14, 15, 16]

fig, ax = plt.subplots()
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
ax.set(yscale = 'log')
ax.set_xlabel('SNR, дБ')
ax.set_ylabel('SER')
ax.plot(y, mmnet_iid, color = 'purple', label = 'MMNet-iid')
ax.plot(y, mmnet_iid_last_loss, color = 'purple', linestyle = "--", label = 'Modified loss')
ax.plot(y, featurous, color = 'blue', linestyle = "-", label = 'Modified σ')
ax.plot(y, feat_last, color = 'blue', linestyle = "--", label = 'Modified σ and loss')
ax.plot(y, mmse, color = 'red', label = 'MMSE')
ax.legend()
plt.show()
#fig.savefig("./graphs/PRESENT_disp.png")


import matplotlib.pyplot as plt

mmnet_iid = [0.049309684435526524, 0.02485708196957903, 0.010341980457305944, 0.003775419394174917, 0.0013111472129822, 0.0005320831139886684]
modified = [0.04776395718256654, 0.023489480813344343, 0.009306668043137067, 0.0031977089246114643, 0.0010092715422315157, 0.0003134361902873417];
mmse = [0.07856885115305556, 0.052920626401901316, 0.03281208197275809, 0.01842927098274283, 0.009156667788823403, 0.003991039594014367];
y = [11, 12, 13, 14, 15, 16];

fig, ax = plt.subplots()
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
ax.set(yscale = 'log')
ax.plot(y, mmnet_iid, color = 'purple', label = 'MMNet-iid')
ax.plot(y, modified, color = 'blue', label = 'Modified')
ax.plot(y, mmse, color = 'red', label = 'MMSE')
ax.legend()
plt.show()
fig.savefig("./graphs/PRESENT_disp.png")
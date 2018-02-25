import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np 

# lr = [0.2, 0.15, 0.1, 0.05, 0.02, 0.015, 0.01, 0.005, 0.002, 0.001]
lr = [0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]
lr.sort()
with open('plots/snrvslrh_code.pkl', "rb") as fp:
        fin_snr_train = pkl.load(fp)
        fin_snr_test = pkl.load(fp)
print(fin_snr_train)
print(fin_snr_test)

# for higher learning rate
# fin_snr_train [0.31336870470495787, 0.3223824229748729, 0.3395009623532802, 4.374002340275863e+22, nan, nan, nan]
# fin_snr_test [0.4861161589229294, 0.48751392731093085, 0.49253908473004704, 6.290648390314647e+22, nan, nan, nan]

fig, ax = plt.subplots()
ax.plot(lr, fin_snr_train, linewidth=4.0, label = 'NMSE of training data')
ax.plot(lr, fin_snr_test, linewidth=4.0, label = 'NMSE of testing data')
start, end = ax.get_ylim()
#ax.yaxis.set_ticks(np.arange(0.1, 0.5, 0.02))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
fig.suptitle('NMSE vs Learning Rate (higher range) for the LMS model', fontsize=36)
plt.ylabel('NMSE', fontsize=24)
plt.xlabel('Learning Rate (higher range)', fontsize=24)
for tick in ax.xaxis.get_major_ticks():
	tick.label.set_fontsize(20)
for tick in ax.yaxis.get_major_ticks():
	tick.label.set_fontsize(20)
plt.legend(loc='upper right')
plt.show()

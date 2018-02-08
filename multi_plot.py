#!/usr/bin/env python3
from pathlib import Path
import soundfile as sf
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

'''print('Reading clean data...')
rootdir = Path('clean/')
filelist = [f for f in rootdir.glob('**/*') if f.is_file()]
filelist = np.repeat(filelist,3)
filelist = sorted(filelist)
k=0
print(type(float(sys.argv[1])))'''

'''import soundfile as sf

data,samplerate = sf.read('noisy/sp01_airport_sn0.wav')
print(data.shape,samplerate)'''

'''from scipy.io import wavfile

print(data.shape,samplerate)'''
rnn_plt = np.load('rnn10.npy')
lms_plt = np.load('lms1000.npy')

fig, ax = plt.subplots()
#lms_line = ax.plot(lms_plt,'bs',linewidth=4.0, label='SNR while training LMS Model')
rnn_line = ax.plot(rnn_plt,'r--',linewidth=4.0, label='SNR while training RNN Model')
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, 1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
fig.suptitle('SNR vs Number of Iterations while training the model', fontsize=36)
plt.ylabel('SNR (in dB)', fontsize=34)
plt.xlabel('Number of iterations', fontsize=34)
#red_patch = mlines.Line2D(rnn_plt,[], color='red', marker='*', label='SNR while training RNN Model')
#blue_patch = mlines.Line2D(lms_plt,[], color='blue', marker='s', label='SNR while training LMS Model')
plt.legend(fontsize = 34)
for tick in ax.xaxis.get_major_ticks():
	tick.label.set_fontsize(30) 
for tick in ax.yaxis.get_major_ticks():
	tick.label.set_fontsize(30) 
plt.show()
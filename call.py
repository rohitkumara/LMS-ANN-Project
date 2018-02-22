from speech_noise_removal_final import RNNFilter
from LMS_tf import LMSFilter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pickle

#for learning rate
lr = [0.2, 0.15, 0.1, 0.05, 0.02, 0.015, 0.01, 0.005, 0.002, 0.001] #0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25,
#lr = [0.002]
lr.sort()
fin_snr_train = []
fin_snr_test = []

def l_plotting(train_data, test_data):
    fig, ax = plt.subplots()
    ax.plot(lr, train_data, 'r', linewidth=4.0, label='Final NMSE of output for training data')
    ax.plot(lr, test_data, 'b', linewidth=4.0, label='NMSE of output for testing data')
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 0.5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    fig.suptitle('Final NMSE vs Learning Rate (lower range) for RNN Filter', fontsize=36)
    plt.ylabel('NMSE', fontsize=24)
    plt.xlabel('Learning Rate', fontsize=24)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    #plt.show()
    plt.savefig('plots/snrvslrl_code.png')
    with open('plots/snrvslrl_code.pkl', "wb") as fp:
        pickle.dump(fig, fp, protocol=4)

for i in range(len(lr)):
    filter = LMSFilter('Mockingbird.wav', epoch = 10, lr = lr[i])
    print('For learning rate:', lr[i])
    snr_train, snr_test = filter.driver()
    fin_snr_train.append(snr_train)
    fin_snr_test.append(snr_test)
l_plotting(fin_snr_train, fin_snr_test)


#for higher learning rate
lr = [0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25]
#lr = [0.002]
lr.sort()
fin_snr_train = []
fin_snr_test = []

def h_plotting(train_data, test_data):
    fig, ax = plt.subplots()
    ax.plot(lr, train_data, 'r', linewidth=4.0, label='Final NMSE of output for training data')
    ax.plot(lr, test_data, 'b', linewidth=4.0, label='NMSE of output for testing data')
    start, end = ax.get_ylim()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    fig.suptitle('Final NMSE vs Learning Rate (higher range) for RNN Filter', fontsize=36)
    plt.ylabel('NMSE', fontsize=24)
    plt.xlabel('Learning Rate', fontsize=24)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    #plt.show()
    plt.savefig('plots/snrvslrh_code.png')
    with open('plots/snrvslrh_code.pkl', "wb") as fp:
        pickle.dump(fig, fp, protocol=4)

for i in range(len(lr)):
    filter = LMSFilter('Mockingbird.wav', epoch = 10, lr = lr[i])
    print('For learning rate:', lr[i])
    snr_train, snr_test = filter.driver()
    fin_snr_train.append(snr_train)
    fin_snr_test.append(snr_test)
h_plotting(fin_snr_train, fin_snr_test)

#for filter length
fl = [512, 256, 128, 64, 32, 16, 8, 4]
#fl = [4]
fl.sort()
fin_snr_train = []
fin_snr_test = []

def fl_plotting(train_data, test_data):
    fig, ax = plt.subplots()
    ax.plot(fl, train_data, 'r', linewidth=4.0, label='Final NMSE of output for training data')
    ax.plot(fl, test_data, 'b', linewidth=4.0, label='NMSE of output for testing data')
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 0.1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    fig.suptitle('Final NMSE vs Filter Length for RNN Filter', fontsize=36)
    plt.ylabel('NMSE', fontsize=24)
    plt.xlabel('Filter Length', fontsize=24)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    #plt.show()
    plt.savefig('plots/snrvsfl_code.png')
    with open('plots/snrvsfl_code.pkl', "wb") as fp:
        pickle.dump(fig, fp, protocol=4)

for i in range(len(fl)):
    filter = LMSFilter('Mockingbird.wav', epoch = 10, tap = fl[i])
    print('For filter length:', fl[i])
    snr_train, snr_test = filter.driver()
    fin_snr_train.append(snr_train)
    fin_snr_test.append(snr_test)
fl_plotting(fin_snr_train, fin_snr_test)

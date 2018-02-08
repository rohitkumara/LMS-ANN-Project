#!/usr/bin/env python3

import numpy as np
import soundfile as sf
import sounddevice as sd 
from scipy.signal import butter,lfilter,freqs
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from pathlib import Path

tap = 32
epoch = 10
p=0.9

def input_from_history(data, n):
	y = np.size(data)-n
	ret = np.zeros([y,n])
	for i in range(y):
		ret[i,:] = data[i:i+n]
	return ret

def read_speech(p):
	print('Reading clean data...')
	rootdir = Path('clean/')
	filelist = [f for f in rootdir.glob('**/*') if f.is_file()]
	filelist = sorted(filelist)
	data, Fs = read_wav(str(filelist[0]))
	for i in range(len(filelist)):
		if i == 0:
			continue
		temp, Fs = read_wav(str(filelist[i]))
		data = np.append(data, temp)
	l = int(p*data.size)
	train_data = data[0:l]
	l = data.size - l
	test_data = data[l:-1]
	return train_data,test_data,Fs

def read_wav(FILE_NAME):
    data,samplerate = sf.read(FILE_NAME)
    return data, samplerate

def butter_lowpass(data, cutoff, Fs, order = 4):
	normalCutoff = cutoff/(Fs/2)
	b,a = butter(order, normalCutoff, btype='low', analog = False)
	y = lfilter(b,a, data,axis = 0)
	return y

def get_data(filename):
	data,Fs= read_wav(filename)
	return data, Fs

def data_equalization(data, noise):
	noise_len = noise.shape[0]
	data_len = data.shape[0]
	data = (np.amax(noise)/np.amax(data))*data
	n = int(data_len/noise_len)+1
	noi_temp = np.tile(noise,n)
	trainX = noi_temp[0:data_len] + data
	trainY = noi_temp[0:data_len]
	print(trainX.shape,trainY.shape)
	return trainX, trainY

def data_preprocessing(trainX, trainY):
	trainX = trainX/np.amax(trainX)
	trainY = trainY/np.amax(trainY)
	trainX_use = input_from_history(trainX,tap)
	trainX_use = trainX_use.reshape((trainX_use.shape[0],tap,1))
	trainY_use = trainY[tap-1:-1].reshape((trainY.size-tap,1))
	return trainX_use, trainY_use

def save_file(filename,data,Fs):
	sf.write(filename,data,Fs)

def play_file(data,Fs):
	try:
		ti = np.shape(data)[0]/Fs
		print('Time in sec:', ti)
		sd.play(data, Fs)
		time.sleep(ti)
		sd.stop()
	except:
		sd.stop()

def measure_snr(noisy, noise):
	data = noisy - noise
	pwr_noise = (np.sum(noise**2))/noise.size
	pwr_data = (np.sum(data**2))/data.size
	snr = pwr_data/pwr_noise
	return 10*np.log10(snr)

def main():
	#data preprocessing step
	noise, Fs = get_data('Mockingbird.wav')
	data,test_data, Fs = read_speech(p)
	print(data.shape,noise.shape)
	trainX_o, trainY_o = data_equalization(data, noise)
	trainX, trainY = data_preprocessing(trainX_o,trainY_o)
	print(trainX.shape, trainY.shape)
	init_snr = measure_snr(trainX_o,trainY_o) # data = noisy- noise
	print('INIT SNR:',init_snr)
	#Neural Network Model
	model = Sequential()
	#model.add(GRU(16, return_sequences = True, input_shape=(tap, 1)))
	#model.add(Dropout(0.5))
	model.add(GRU(16, input_shape=(tap, 1)))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	#opt = RMSprop(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=opt)
	snr_plt = []
	strt = time.time()
	for i in range(epoch):
		hist = model.fit(trainX, trainY, epochs=1, batch_size=1500)
		yhat = model.predict(trainX, batch_size = 1500, verbose = 0)
		snr = measure_snr(yhat.reshape([yhat.size,1]),trainY_o[tap-1:-1].reshape([data.size-tap,1]))
		snr_plt.append(snr)
		print('Epoch: {}/{}'.format((i+1), epoch),'loss:',hist.history['loss'],'SNR:',snr)
	end = time.time()
	print('Time taken',(end-strt))
	sav_file = 'rnn{}.npy'.format(epoch)
	np.save(sav_file,snr_plt)
	fig, ax = plt.subplots()
	ax.plot(snr_plt, linewidth=4.0)
	start, end = ax.get_ylim()
	ax.yaxis.set_ticks(np.arange(start, end, 0.5))
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
	fig.suptitle('SNR vs Number of Iterations while training the RNN model', fontsize=36)
	plt.ylabel('SNR (in dB)', fontsize=24)
	plt.xlabel('Number of iterations', fontsize=24)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(20) 
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(20) 
	plt.show()
	predict = yhat.reshape([yhat.size,1])
	print('SNR of INPUT data:', init_snr)
	#play_file(trainX_o, Fs)
	print('SNR of OUTPUT data:', snr)
	#play_file(predict, Fs)
	save_file('new_version_training_out.wav',predict,Fs)

	trainX_o, trainY_o = data_equalization(test_data, noise)
	start = time.time()
	trainX, trainY = data_preprocessing(trainX_o, trainY_o)
	yhat = model.predict(trainX, batch_size = 1000, verbose = 0)
	end = time.time()
	predict = yhat.reshape([yhat.size,1])
	print('Time Taken:', (end-start))
	print('SNR of INPUT:', measure_snr(trainX_o,trainY_o))
	#play_file(trainX_o, Fs)
	print('SNR of OUTPUT:', measure_snr(predict,trainY_o[tap-1:-1].reshape([predict.size,1])))
	#play_file(predict,Fs)
	save_file('new_version_testing_out.wav',predict, Fs)


main()	
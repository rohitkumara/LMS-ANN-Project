#!/usr/bin/env python3

import numpy as np
import soundfile as sf
import sounddevice as sd 
from scipy.signal import butter,lfilter,freqs
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

tap = 32
epoch = 30

def input_from_history_mod(data, n):
	ex = n-(np.size(data)%n)
	temp = np.zeros([ex,])
	print(data.shape, temp.shape, n)
	data = np.concatenate((data, temp))
	print(data.shape)
	y = int(data.size/n)
	ret = np.zeros([y,n])
	for i in range(y):
		ret[i,:] = data[i*n:(i+1)*n]
	return ret

def read_wav(FILE_NAME):
    data,samplerate = sf.read(FILE_NAME)
    return data, samplerate

def butter_lowpass(data, cutoff, Fs, order = 4):
	normalCutoff = cutoff/(Fs/2)
	b,a = butter(order, normalCutoff, btype='low', analog = False)
	y = lfilter(b,a,data,axis = 0)
	return y

def get_data(filename):
	data,Fs= read_wav(filename)
	return data, Fs

def data_equalization(data, noise):
	noise_len = noise.shape[0]
	data_len = data.shape[0]
	n = int(noise_len/data_len)+1
	dat_temp = np.tile(data,n)
	trainX = dat_temp[0:noise_len] + noise
	trainY = dat_temp[0:noise_len]
	print(trainX.shape,trainY.shape)
	return trainX, trainY

def data_preprocessing(trainX, trainY):
	trainX = trainX/np.amax(trainX)
	trainY = trainY/np.amax(trainY)
	trainX_use = input_from_history_mod(trainX,tap)
	trainY_use = input_from_history_mod(trainY,tap)
	trainX_use = trainX_use.reshape((trainX_use.shape[0],tap,1))
	trainY_use = trainY_use.reshape((trainY_use.shape[0],tap,1))
	return trainX_use, trainY_use

def data_postprocessing(data):
	ret = np.zeros([data.size,1])
	n = data.shape[1]
	for i in range(data.shape[0]):
		ret[i*n:(i+1)*n] = data[i,:]
	return ret

def save_file(filename,data,Fs):
	sf.write(filename,data,Fs)

def play_file(data,Fs):
	ti = np.shape(data)[0]/Fs
	print('Time in sec:',ti)
	sd.play(data, Fs)
	time.sleep(ti)
	sd.stop()

def measure_snr(noisy, data):
	noise = noisy - data
	pwr_noise = (np.sum(np.abs(noise)**2))/noise.size
	pwr_data = (np.sum(np.abs(data)**2))/data.size
	snr = pwr_data/pwr_noise
	return 10*np.log10(snr)

def main():
	#data preprocessing step
	temp1_1, Fs = get_data('landing.wav')
	temp1_2, Fs = get_data('takeoff.wav')
	temp2, Fs = get_data('Mockingbird.wav')
	data = temp2
	noise = np.concatenate((temp1_1,temp1_2))
	print(data.shape,noise.shape)
	trainX_o, trainY_o = data_equalization(data, noise)
	trainX, trainY = data_preprocessing(trainX_o,trainY_o)
	print(trainX.shape, trainY.shape)
	print('playing noisy data')
	#play_file(trainX_o, Fs)
	init_snr = measure_snr(trainX_o,trainY_o)
	print(init_snr)
	#Neural Network Model
	model = Sequential()
	model.add(LSTM(128,return_sequences = True, input_shape=(tap, 1)))
	model.add(Dropout(0.5))
	model.add(LSTM(128,return_sequences = True))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	#opt = RMSprop(lr=0.001)
	model.compile(loss='mean_squared_error', optimizer=opt)
	snr_plt = []
	strt = time.time()
	for i in range(epoch):
		hist = model.fit(trainX, trainY, epochs=1, batch_size=500)
		yhat = model.predict(trainX, batch_size = 500, verbose = 0)
		yhat = data_postprocessing(yhat)
		snr = measure_snr(yhat[:trainY_o.size].reshape([trainY_o.size,1]),trainY_o.reshape([trainY_o.size,1]))
		snr_plt.append(snr)
		print('Epoch:',i+1,'loss:',hist.history['loss'],'SNR:',snr)
	end = time.time()
	print('Time taken',(end-strt))
	fig, ax = plt.subplots()
	ax.plot(snr_plt)
	start, end = ax.get_ylim()
	ax.yaxis.set_ticks(np.arange(start, end, 0.5))
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
	plt.ylabel('SNR (in dB)')
	plt.xlabel('Number of iterations')
	plt.show()
	predict = yhat[:trainY_o.size].reshape([trainY_o.size,1])
	#play_file(trainX_o, Fs)
	print('playing input noisy data')
	print(init_snr)
	play_file(trainX_o, Fs)
	print('playing output')
	print(snr)
	play_file(predict, Fs)
	save_file('output_training_data_50.wav',predict,Fs)

	temp1, Fs = get_data('rocket.wav')
	temp2, Fs = get_data('Mockingbird.wav')
	le = min(temp1.shape[0],temp2.shape[0])
	trainX_o = temp1[0:le] + temp2[0:le]
	trainY_o = temp2[0:le]
	print('Playing noisy data')
	print(measure_snr(trainX_o,trainY_o))
	play_file(trainX_o, Fs)
	trainX, trainY = data_preprocessing(trainX_o, trainY_o)
	yhat = model.predict(trainX, batch_size = 500, verbose = 0)
	yhat = data_postprocessing(yhat)
	predict = yhat[:trainY_o.size].reshape([trainY_o.size,1])
	print('Output of neural network')
	print(measure_snr(predict,trainY_o.reshape([trainY_o.size,1])))
	play_file(predict,Fs)
	save_file('output_testing_data_50.wav',predict, Fs)


main()	
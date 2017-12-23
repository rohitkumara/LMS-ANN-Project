import numpy as np
import soundfile as sf
import sounddevice as sd 
import noise_gen as ng 
import padasip as pd 
from scipy.signal import butter,lfilter,freqs
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import time

tap = 16
epoch = 10

def input_from_history(data, n):
	y = np.size(data)-n
	ret = np.zeros([y,n])
	for i in range(y):
		ret[i,:] = data[i:i+n]
	return ret

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
	'''
	noise,_ = ng.pink_noise(np.shape(data)[0])
	white_noise,fft_white = ng.white_noise(np.shape(data)[0])
	pink_noise,fft_pink = ng.pink_noise(np.shape(data)[0])
	print(Fs)

	bandlimit_white_noise = butter_lowpass(white_noise, 20000,Fs,order = 10)
	scaled_bandlimit_white_noise = (np.max(data)/np.max(bandlimit_white_noise))*bandlimit_white_noise

	bandlimit_pink_noise = butter_lowpass(pink_noise, 20000,Fs,order = 10)
	scaled_bandlimit_pink_noise = (np.max(data)/np.max(bandlimit_pink_noise))*bandlimit_pink_noise

	data_noise = data + scaled_bandlimit_pink_noise
	'''
	return data, Fs

def data_preprocessing(trainX, trainY):
	trainX = trainX/np.amax(trainX)
	trainY = trainY/np.amax(trainY)
	trainX_use = input_from_history(trainX,tap)
	trainY_use = input_from_history(trainY,tap)
	trainX_use = trainX_use.reshape((trainX_use.shape[0],tap,1))
	trainY_use = trainY_use.reshape((trainY_use.shape[0],tap))
	return trainX_use, trainY_use

def save_file(filename,data,Fs):
	sf.write(filename,data,Fs)

def play_file(data,Fs):
	ti = np.shape(data)[0]/Fs
	print(ti)
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
	temp1, Fs = get_data('aircraft027.wav')
	temp2, Fs = get_data('Mockingbird.wav')
	le = min(temp1.shape[0],temp2.shape[0])
	print(le,Fs)
	trainX_o = temp1[0:le] + temp2[0:le]
	trainY_o = temp2[0:le]
	print('Playing noisy data')
	print(measure_snr(trainX_o,trainY_o))
	print('Playing actual data')
	#play_file(trainY_o, Fs)
	trainX, trainY = data_preprocessing(trainX_o, trainY_o)
	print(trainX.shape, trainY.shape)
	#Neural Network Model
	model = Sequential()
	model.add(LSTM(8, input_shape=(tap, 1)))
	model.add(Dense(tap))
	model.compile(loss='mean_squared_error', optimizer='adam')
	snr_plt = []
	strt = time.time()
	for i in range(epoch):
		hist = model.fit(trainX, trainY, epochs=1, batch_size=1000, verbose=0)
		yhat = model.predict(trainX, batch_size = 1000, verbose = 0)
		snr = measure_snr(yhat[:,tap-1].reshape([yhat.shape[0],1]),trainY_o[tap-1:-1].reshape([trainY_o.size-tap,1]))
		snr_plt.append(snr)
		print('Epoch:',i+1,'loss:',hist.history['loss'],'SNR:',snr)
	end = time.time()
	print('Time taken',(end-strt))
	plt.plot(snr_plt)
	plt.show()
	predict = yhat[:,tap-1].reshape([yhat.shape[0],1])
	#play_file(trainX_o, Fs)
	print('Output of neural network')
	print(measure_snr(predict,trainY_o[tap-1:-1].reshape([trainY_o.size-tap,1])))
	#play_file(predict,Fs)
	save_file('output_training_data_50.wav',predict,Fs)

	temp1, Fs = get_data('aircraft027.wav')
	temp2, Fs = get_data('Australian.wav')
	le = min(temp1.shape[0],temp2.shape[0])
	trainX_o = temp1[0:le] + temp2[0:le]
	trainY_o = temp2[0:le]
	print('Playing noisy data')
	print(measure_snr(trainX_o,trainY_o))
	#play_file(trainX_o, Fs)
	print('Playing actual data')
	#play_file(trainY_o, Fs)
	trainX, trainY = data_preprocessing(trainX_o, trainY_o)
	yhat = model.predict(trainX, batch_size = 1000, verbose = 0)
	predict = yhat[:,tap-1].reshape([yhat.shape[0],1])
	print('Output of neural network')
	print(measure_snr(predict,trainY_o[tap-1:-1].reshape([trainY_o.size-tap,1])))
	#play_file(predict,Fs)
	save_file('output_testing_data_50.wav',predict, Fs)


main()	
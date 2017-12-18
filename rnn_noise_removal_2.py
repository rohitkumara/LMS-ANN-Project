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
	noise,_ = ng.pink_noise(np.shape(data)[0])
	white_noise,fft_white = ng.white_noise(np.shape(data)[0])
	pink_noise,fft_pink = ng.pink_noise(np.shape(data)[0])
	print(Fs)

	bandlimit_white_noise = butter_lowpass(white_noise, 20000,Fs,order = 10)
	scaled_bandlimit_white_noise = (np.max(data)/np.max(bandlimit_white_noise))*bandlimit_white_noise

	bandlimit_pink_noise = butter_lowpass(pink_noise, 20000,Fs,order = 10)
	scaled_bandlimit_pink_noise = (np.max(data)/np.max(bandlimit_pink_noise))*bandlimit_pink_noise
	data = data[:,0]
	data_noise = data + np.abs(scaled_bandlimit_pink_noise)
	return data_noise, data, Fs

def data_preprocessing(trainX, trainY):
	trainX = trainX/np.amax(trainX)
	trainY = trainY/np.amax(trainY)
	trainX_use = pd.input_from_history(trainX,tap)
	trainY_use = pd.input_from_history(trainY,tap)
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

def main():
	#data preprocessing step
	trainX_o,trainY_o, Fs = get_data('output.wav')
	print('Playing noisy data')
	play_file(trainX_o, Fs)
	print('Playing actual data')
	play_file(trainY_o, Fs)
	trainX, trainY = data_preprocessing(trainX_o, trainY_o)
	print(trainX.shape, trainY.shape)
	#Neural Network Model
	model = Sequential()
	model.add(LSTM(8, input_shape=(tap, 1)))
	model.add(Dense(tap))
	model.compile(loss='mean_squared_error', optimizer='adam')
	hist = model.fit(trainX, trainY, epochs=20, batch_size=1000, verbose=2)
	loss = list(hist.history['loss'])
	plt.plot(loss)
	plt.show()
	yhat = model.predict(trainX, batch_size = 1000, verbose = 0)
	a_yhat = yhat*(np.amax(trainY_o/np.amax(trainY_o))/np.amax(yhat))*np.amax(trainY_o)
	temp1 = a_yhat[0,0:-1].tolist()
	temp2 = a_yhat[:,tap-1].tolist()
	predict = temp1 + temp2
	predict = np.array(predict)
	print('Output of neural network')
	play_file(predict,Fs)
	save_file('output_training_data_50.wav',predict,Fs)

	trainX_o,trainY_o, Fs = get_data('output_test.wav')
	print('Playing noisy data')
	play_file(trainX_o, Fs)
	print('Playing actual data')
	play_file(trainY_o, Fs)
	trainX, trainY = data_preprocessing(trainX_o, trainY_o)
	yhat = model.predict(trainX, batch_size = 1000, verbose = 0)
	a_yhat = yhat*(np.amax(trainY_o/np.amax(trainY_o))/np.amax(yhat))*np.amax(trainY_o)
	temp1 = a_yhat[0,0:-1].tolist()
	temp2 = a_yhat[:,tap-1].tolist()
	predict = temp1 + temp2
	predict = np.array(predict)
	print('Output of neural network')
	play_file(predict,Fs)
	save_file('output_testing_data_50.wav',predict, Fs)


main()	
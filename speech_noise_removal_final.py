#!/usr/bin/env python3

import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
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


class RNNFilter:
	def __init__(self, noise_fn, tap=32, epoch=10, p=0.9, lr=0.001, dropout = 0.5, plot = False):
		self.noise_fn = noise_fn
		self.tap = tap
		self.epoch = epoch
		self.p = p
		self.lr = lr
		self.plot = plot
		self.dropout = dropout

	def input_from_history(self,data):
		y = np.size(data) - self.tap
		ret = np.zeros([y,self.tap])
		for i in range(y):
			ret[i,:] = data[i:i+self.tap]
		return ret

	def read_speech(self,data):
		print('Reading clean data...')
		rootdir = Path('clean/')
		filelist = [f for f in rootdir.glob('**/*') if f.is_file()]
		filelist = sorted(filelist)
		data, Fs = self.read_wav(str(filelist[0]))
		for i in range(len(filelist)):
			if i == 0:
				continue
			temp, Fs = self.read_wav(str(filelist[i]))
			data = np.append(data, temp)
		l = int((self.p)*data.size)
		train_data = data[0:l]
		test_data = data[l:-1]
		return train_data,test_data,Fs

	def read_wav(self, FILE_NAME):
		self.data,samplerate = sf.read(FILE_NAME)
		return self.data, samplerate

	def get_data(self,filename):
		data,Fs= self.read_wav(filename)
		return data, Fs

	def data_equalization(self,data, noise):
		noise_len = noise.shape[0]
		data_len = data.shape[0]
		data = (np.amax(noise)/np.amax(data))*data
		n = int(data_len/noise_len)+1
		noi_temp = np.tile(noise,n)
		trainX = noi_temp[0:data_len] + data
		trainY = noi_temp[0:data_len]
		#print(trainX.shape,trainY.shape)
		return trainX, trainY

	def data_preprocessing(self,trainX, trainY):
		trainX = trainX/np.amax(trainX)
		trainY = trainY/np.amax(trainY)
		trainX_use = self.input_from_history(trainX)
		trainX_use = trainX_use.reshape((trainX_use.shape[0],self.tap,1))
		trainY_use = trainY[self.tap-1:-1].reshape((trainY.size-self.tap,1))
		return trainX_use, trainY_use

	def save_file(self,filename,data,Fs):
		sf.write(filename,data,Fs)

	def play_file(self,data,Fs):
		try:
			ti = np.shape(data)[0]/Fs
			print('Time in sec:', ti)
			sd.play(data, Fs)
			time.sleep(ti)
			sd.stop()
		except:
			sd.stop()

	def measure_snr(self,noisy, noise):
		data = noisy - noise
		pwr_noise = (np.sum(noise**2))/noise.size
		pwr_data = (np.sum(data**2))/data.size
		snr = pwr_data/pwr_noise
		return snr

	def driver(self):
		#data preprocessing step
		noise, Fs = self.get_data(self.noise_fn)
		data,test_data, Fs = self.read_speech(self.p)
		#print(data.shape,noise.shape)
		trainX_o, trainY_o = self.data_equalization(data, noise)
		trainX, trainY = self.data_preprocessing(trainX_o,trainY_o)
		#print(trainX.shape, trainY.shape)
		init_snr = self.measure_snr(trainX_o,trainY_o) # data = noisy- noise
		print('INIT SNR:',init_snr)
		#Neural Network Model
		model = Sequential()
		#model.add(GRU(16, return_sequences = True, input_shape=(self.tap, 1)))
		#model.add(Dropout(0.5))
		model.add(GRU(16, input_shape=(self.tap, 1)))
		model.add(Dropout(self.dropout))
		model.add(Dense(1))
		opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		#opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
		#opt = RMSprop(lr=0.001)
		model.compile(loss='mean_squared_error', optimizer=opt)
		snr_plt = []
		strt = time.time()
		for i in range(self.epoch):
			hist = model.fit(trainX, trainY, epochs=1, batch_size=1000)
			yhat = model.predict(trainX, batch_size = 1000, verbose = 0)
			snr = self.measure_snr(yhat.reshape([yhat.size,1]),trainY_o[self.tap-1:-1].reshape([data.size-self.tap,1]))
			snr_plt.append(snr)
			print('Epoch: {}/{}'.format((i+1), self.epoch),'loss:',hist.history['loss'],'SNR:',snr)
		end = time.time()
		print(hist.history['loss'])
		print('Time taken',(end-strt))
		sav_file = 'rnn{}.npy'.format(self.epoch)
		np.save(sav_file,snr_plt)
		if(self.plot == True):
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
		self.save_file('new_version_training_out.wav',predict,Fs)

		trainX_o, trainY_o = self.data_equalization(test_data, noise)
		start = time.time()
		trainX, trainY = self.data_preprocessing(trainX_o, trainY_o)
		yhat = model.predict(trainX, batch_size = 1000, verbose = 0)
		end = time.time()
		predict = yhat.reshape([yhat.size,1])
		print('Time Taken:', (end-start))
		print('SNR of INPUT:', self.measure_snr(trainX_o,trainY_o))
		#play_file(trainX_o, Fs)
		snr_test = self.measure_snr(predict,trainY_o[self.tap-1:-1].reshape([predict.size,1]))
		print('SNR of OUTPUT:', snr_test)
		#play_file(predict,Fs)
		self.save_file('new_version_testing_out.wav',predict, Fs)
		return snr, snr_test

def main():
	rnnfil = RNNFilter('Mockingbird.wav', epoch = 2)
	_,_ = rnnfil.driver()

if __name__ == "__main__": main()

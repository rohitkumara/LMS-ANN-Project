import tensorflow as tf
import numpy as np
import soundfile as sf
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

class LMSFilter:
	def __init__(self, noise_fn, tap=32, epoch=10, p=0.9, lr=0.001, batch_size = 10, plot = False):
		self.noise_fn = noise_fn
		self.tap = tap
		self.epoch = epoch
		self.p = p
		self.lr = lr
		self.plot = plot
		self.batch_size = batch_size

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
		trainX_use = trainX_use.reshape((trainX_use.shape[0],self.tap))
		trainY_use = trainY[self.tap-1:-1].reshape((trainY.size-self.tap))
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
		print(trainX.shape, trainY.shape)
		X = tf.placeholder(tf.float32, [None, self.tap])
		Y = tf.placeholder(tf.float32, [None, 1])
		W = tf.Variable(tf.random_normal([1, self.tap], stddev=0.1))
		#LMS Algorithm
		out = tf.matmul(X,tf.transpose(W))
		yhat = out
		err = Y - yhat
		err = tf.reduce_mean(tf.square(err))
		opt = tf.train.GradientDescentOptimizer(self.lr).minimize(err)
		init_snr = self.measure_snr(trainX_o,trainY_o)
		print('INIT SNR:', init_snr)
		init_all = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init_all)
		j=0
		av_cost = np.inf
		strt = time.time()
		snr_plt = []
		for j in range(self.epoch):
			av_cost = 0
			for i in range(int(trainY.shape[0]/self.batch_size)):
				batch_X = trainX[i:i+self.batch_size,:].reshape([self.batch_size,self.tap])
				batch_Y = trainY[i:i+self.batch_size].reshape([self.batch_size,1])
				sess.run(opt, feed_dict = {X:batch_X, Y:batch_Y})
				av_cost += sess.run(err, feed_dict = {X:batch_X, Y:batch_Y})
			yout = sess.run(yhat, feed_dict = {X:trainX})
			snr = self.measure_snr(yout,trainY_o[self.tap-1:-1].reshape([yout.size,1]))
			snr_plt.append(snr)
			print('Epoch:',j, 'Sq. Error:', av_cost,'SNR:',snr)
		end = time.time()
		print('Time taken',(end-strt))
		sav_file = 'lms{}.npy'.format(self.epoch)
		np.save(sav_file,snr_plt)
		if(self.plot == True):
			fig, ax = plt.subplots()
			ax.plot(snr_plt, linewidth=4.0)
			start, end = ax.get_ylim()
			ax.yaxis.set_ticks(np.arange(start, end, 0.5))
			ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
			fig.suptitle('SNR vs Number of Iterations while training the LMS model', fontsize=26)
			plt.ylabel('SNR (in dB)', fontsize=24)
			plt.xlabel('Number of iterations', fontsize=24)
			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_fontsize(20)
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_fontsize(20)
			plt.show()
		predict = yout
		print('SNR of INPUT:', init_snr)
		#play_file(trainX_o,Fs)
		print('SNR of OUTPUT:', self.measure_snr(predict,trainY_o[self.tap-1:-1].reshape([yout.size,1])))
		#play_file(predict,Fs)

		print('')
		print('')

		trainX_o, trainY_o = self.data_equalization(test_data, noise)
		start = time.time()
		trainX, trainY = self.data_preprocessing(trainX_o, trainY_o)
		yout = sess.run(yhat, feed_dict = {X:trainX})
		predict = yout
		end = time.time()
		print('Time Taken:', (end-start))
		snr_test = self.measure_snr(predict ,trainY_o[self.tap-1:-1].reshape([yout.size,1]))
		print('SNR of INPUT:', self.measure_snr(trainX_o, trainY_o))
		#play_file(trainX_o,Fs)
		print('SNR of OUTPUT:', snr_test)
		#play_file(predict,Fs)
		return snr, snr_test

def main():
	rnnfil = RNNFilter('Mockingbird.wav', epoch = 2)
	_,_ = rnnfil.driver()

if __name__ == "__main__": main()

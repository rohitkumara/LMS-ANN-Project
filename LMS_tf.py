import tensorflow as tf
import numpy as np
import soundfile as sf
import sounddevice as sd
import time
import matplotlib.pyplot as plt

mu = 0.01
e = 0.05
tap = 16
batch_size = 1000
epoch = 1000

def input_from_history(data, n):
	y = np.size(data)-n
	ret = np.zeros([y,n])
	for i in range(y):
		ret[i,:] = data[i:i+n]
	return ret

def read_wav(FILE_NAME):
    data,samplerate = sf.read(FILE_NAME)
    return data, samplerate

def get_data(filename):
	data,Fs= read_wav(filename)
	return data, Fs

def data_preprocessing(trainX, trainY):
	trainX = trainX/np.amax(trainX)
	trainY = trainY/np.amax(trainY)
	trainX_use = input_from_history(trainX,tap)
	trainY_use = input_from_history(trainY,tap)
	trainX_use = trainX_use.reshape((trainX_use.shape[0],tap))
	trainY_use = trainY_use.reshape((trainY_use.shape[0],tap))
	return trainX_use, trainY_use

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
	temp1, Fs = get_data('aircraft027.wav')
	temp2, Fs = get_data('Mockingbird.wav')
	le = min(temp1.shape[0],temp2.shape[0])
	print(le,Fs)
	trainX_o = temp1[0:le] + temp2[0:le]
	trainY_o = temp2[0:le]
	print('Playing noisy data')
	print(measure_snr(trainX_o,trainY_o))
	#play_file(trainX_o, Fs)
	print('Playing actual data')
	#play_file(trainY_o, Fs)
	trainX, trainY = data_preprocessing(trainX_o, trainY_o)
	trainY = trainY[:,-1]
	trainY = trainY.reshape([trainY.shape[0],1])
	print(trainX.shape, trainY.shape)
	X = tf.placeholder(tf.float32, [None, tap])
	Y = tf.placeholder(tf.float32, [None, 1])
	W = tf.Variable(tf.random_normal([1, tap], stddev=0.1))
	#LMS Algorithm
	out = tf.matmul(X,tf.transpose(W))
	yhat = out
	err = Y - yhat
	err = tf.reduce_mean(tf.square(err))
	opt = tf.train.GradientDescentOptimizer(mu).minimize(err)

	init_all = tf.global_variables_initializer()
	pre_snr = 0
	with tf.Session() as sess:
		sess.run(init_all)
		j=0
		av_cost = np.inf
		strt = time.time()
		snr_plt = []
		for j in range(epoch):
			av_cost = 0
			for i in range(int(trainY.shape[0]/batch_size)):
				batch_X = trainX[i:i+batch_size,:].reshape([batch_size,tap])
				batch_Y = trainY[i:i+batch_size].reshape([batch_size,1])
				sess.run(opt, feed_dict = {X:batch_X, Y:batch_Y})
				av_cost += sess.run(err, feed_dict = {X:batch_X, Y:batch_Y})
			yout = sess.run(yhat, feed_dict = {X:trainX})
			snr = measure_snr(yout,trainY_o[tap-1:-1].reshape([yout.size,1]))
			snr_plt.append(snr)
			print('Epoch:',j, 'Sq. Error:', av_cost,'SNR:',snr)
			if(pre_snr > snr):
				break
			pre_snr = snr
		end = time.time()
	print('Time taken',(end-strt))
	print('noisy signal')
	#play_file(trainX_o,Fs)
	print('after noise removal')
	#play_file(yout,Fs)
	plt.plot(snr_plt)
	plt.show()
main()
#data = np.array([2,3,4,5,6,7,8,9])
#print(input_from_history(data, 2))
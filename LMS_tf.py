import tensorflow as tf
import numpy as np
import soundfile as sf
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

mu = 0.01
e = 0.05
tap = 16
batch_size = 1500
epoch = 100

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

def data_equalization(data, noise):
	noise_len = noise.shape[0]
	data_len = data.shape[0]
	n = int(noise_len/data_len)+1
	dat_temp = np.tile(data,n)
	trainX = dat_temp[0:noise_len] + noise
	trainY = dat_temp[0:noise_len]
	print(trainX.shape,trainY.shape)
	return trainX, trainY

def save_file(filename,data,Fs):
	sf.write(filename,data,Fs)

def play_file(data,Fs):
	try:
		ti = np.shape(data)[0]/Fs
		print('Time in sec:',ti)
		sd.play(data, Fs)
		time.sleep(ti)
		sd.stop()
	except:
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
	sess = tf.Session()
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
	end = time.time()
	print('Time taken',(end-strt))
	np.save('lms25.npy',snr_plt)
	fig, ax = plt.subplots()
	ax.plot(snr_plt, linewidth=4.0)
	start, end = ax.get_ylim()
	ax.yaxis.set_ticks(np.arange(start, end, 0.5))
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
	fig.suptitle('SNR vs Number of Iterations while training the LMS model', fontsize=20)
	plt.ylabel('SNR (in dB)', fontsize=18)
	plt.xlabel('Number of iterations', fontsize=18)
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14) 
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14) 
	plt.show()
	print('SNR of INPUT:', measure_snr(trainX_o,trainY_o))
	#play_file(trainX_o,Fs)
	print('SNR of OUTPUT:', measure_snr(yout,trainY_o[tap-1:-1].reshape([yout.size,1])))
	#play_file(yout,Fs)

	print('')
	print('')

	temp1, Fs = get_data('rocket.wav')
	temp2, Fs = get_data('Mockingbird.wav')
	start = time.time()
	le = min(temp1.shape[0],temp2.shape[0])
	trainX_o = temp1[0:le] + temp2[0:le]
	trainY_o = temp2[0:le]
	trainX, trainY = data_preprocessing(trainX_o, trainY_o)
	yout = sess.run(yhat, feed_dict = {X:trainX})
	end = time.time()
	print('Time Taken:', (end-start))
	snr = measure_snr(yout,trainY_o[tap-1:-1].reshape([yout.size,1]))
	print('SNR of INPUT:', measure_snr(trainX_o,trainY_o))
	#play_file(trainX_o,Fs)
	print('SNR of OUTPUT:', snr)
	#play_file(yout,Fs)

main()
#data = np.array([2,3,4,5,6,7,8,9])
#print(input_from_history(data, 2))
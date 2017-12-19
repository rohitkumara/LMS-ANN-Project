import tensorflow as tf
import numpy as np

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
	pwr_noise = (np.max(noise)**2)/noise.size
	pwr_data = (np.max(data)**2)/data.size
	snr = pwr_data/pwr_noise
	return snr

def main():
	pass


#data = np.array([2,3,4,5,6,7,8,9,])
#print(input_from_history(data, 2))
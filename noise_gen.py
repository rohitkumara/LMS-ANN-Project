#generating PINK & WHITE noise  

import numpy as np 
import matplotlib.pyplot as plt 
import sounddevice as sd
import time

def pink_noise(n):
	gauss = np.random.normal(size = n)
	arr = np.sqrt(np.arange(1,n,dtype = np.int32))
	fft_gauss = np.fft.fft(gauss)
	fft_gauss[0] = 0
	fft_gauss[1:] = fft_gauss[1:]/arr
	pink = np.fft.ifft(fft_gauss) 
	return pink, fft_gauss


def white_noise(n):
	gauss = np.random.normal(size = n)
	fft_gauss = np.fft.fft(gauss)
	return gauss, fft_gauss


'''
val = np.log(np.abs(pink_noise(100000)))
print(np.max(val))
sd.play(val)
time.sleep(10)
sd.stop()

plt.plot(val)

val_white = np.log(np.abs(white_noise(100000)))
print(np.max(val_white))
plt.plot(val_white)
plt.show()
'''
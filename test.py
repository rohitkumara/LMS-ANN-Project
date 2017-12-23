import tensorflow as tf
import numpy as np

X = tf.Variable(10, name='X',dtype=tf.float32)

x_sq = tf.square(X)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(x_sq)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	print('10')
	for step in range(100):
		sess.run(train)
		print("step", step, "x:", sess.run(X), "x^2:", sess.run(x_sq))
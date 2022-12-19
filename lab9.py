import tensorflow as tf
import numpy as numpy

tf1a = tf.range(0.0, 100.0, 2)
# tf2 = tf.Variable([1,1,1,2,2,2,3,3,3])
tf1b = tf.reshape(tf.Variable([1,1,1,2,2,2,3,3,3]), [3, 3]).numpy()
tf1c = tf.fill([3, 5], 55)
tf1d = tf.random.uniform(shape=[5,4,3], minval=0., maxval=1.)


tf2 = tf.random.uniform(shape=[10,20,3]) 
# tf2 = tf.reshape(tf2, (300,2)) # goal == 600
# tf2 = tf.reshape(tf2, (300,1,1,2)) # goal == 600
# tf2 = tf.reshape(tf2, (300,2,2,1)) # amount of data doesn't match a reshaped goal == 1200
# tf2 = tf.reshape(tf2, (60,3,3)) # amount of data doesn't match a reshaped goal == 540
# print(tf2)

tf3 = tf.random.uniform(shape=[10,20,3]) 
# tf3 = tf.reshape(tf3, (-1300)) # no negative except -1 !
# tf3 = tf.reshape(tf3, (300,1,1,1,-1)) # 
# tf3 = tf.reshape(tf3, (100,-1,1)) # 
# tf3 = tf.reshape(tf3, (50,2,-1,3)) # 
# print(tf3)
#############################################################3

# a = tf.Variable([1])
# a = 1

# def f(x):
#     return tf.pow(x, 1) + tf.pow(x, 2) + tf.pow(x, 3)
# with tf.GradientTape() as tape:
#     y = f(a)

# print("AAAAAAAAAAAAAAAAAAA",tape.gradient(y, a))

# x = tf.constant(-1.)
# with tf.GradientTape() as g:
#   g.watch(x)
#   y = x + (x*x) + (x*x*x)
# dy_dx = g.gradient(y, x)
# print(dy_dx.numpy())
# print(dy_dx * dy_dx)

#############################################################

x1 = tf.Variable([1.,2.,3.])
x2 = tf.Variable([2.,0.,2.])

with tf.GradientTape() as g1:
  g1.watch(x1)
  y1 = tf.reduce_sum(x1 * x1)

with tf.GradientTape() as g2:
  g2.watch(x2)
  y2 = tf.reduce_sum(x2 * x2)

dy_dx1 = g1.gradient(y1, x1)
dy_dx2 = g2.gradient(y2, x2)
print(dy_dx1.numpy())
print(dy_dx2.numpy())
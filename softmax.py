from __future__ import print_function
import tensorflow as tf
import time

c_num = 1000
c_len = 128000

v_data = tf.Variable(tf.zeros([c_num,c_len], dtype=tf.float32), name='v_data')
v_res = tf.Variable(tf.zeros([c_num,c_len], dtype=tf.float32), name='v_res')
op_data_set = tf.assign(v_data, tf.random_normal(shape=[c_num,c_len], mean=0.0, stddev=0.1, dtype=tf.float32), name='op_data_set')
op_res_set = tf.assign(v_res, tf.nn.softmax(v_data))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(op_data_set)
start = time.clock()
sess.run(op_res_set)
end = time.clock()
sess.run(v_res)
sess.close()

print(end - start, 's')



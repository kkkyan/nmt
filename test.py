import tensorflow as tf 


a = tf.Variable(5)
b = a+a
op = a.assign(10)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([a,b]))
    print(sess.run(a))

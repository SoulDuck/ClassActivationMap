import tensorflow as tf
import numpy as np
import data



image_height , image_width , image_color_ch , n_classes ,train_imgs , train_labs , test_imgs, test_labs = data.eye_64x64()
sample_img=test_imgs[0]
sample_img=sample_img.reshape([1,64,64,3])



saver = tf.train.import_meta_graph('./model/best_acc.ckpt.meta')
sess=tf.Session()
saver.restore(sess, './model/best_acc.ckpt')

softmax=tf.get_default_graph().get_tensor_by_name('softmax:0')
x_=tf.get_default_graph().get_tensor_by_name('x_:0')
y_=tf.get_default_graph().get_tensor_by_name('y_:0')

result=sess.run(softmax , feed_dict={x_:sample_img})
print result


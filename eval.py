import data
from CAM import *



image_height , image_width , image_color_ch , n_classes ,train_imgs , train_labs , test_imgs, test_labs = data.eye_64x64()
sample_img=test_imgs[0]
sample_img=sample_img.reshape([1,64,64,3])
tmp_label=np.array([[1,0]])
print np.shape(tmp_label)

saver = tf.train.import_meta_graph('./model/best_acc.ckpt.meta')
sess=tf.Session()
saver.restore(sess, './model/best_acc.ckpt')
softmax=tf.get_default_graph().get_tensor_by_name('softmax:0')
top_conv=tf.get_default_graph().get_tensor_by_name('top_conv/relu:0')
x_=tf.get_default_graph().get_tensor_by_name('x_:0')
y_=tf.get_default_graph().get_tensor_by_name('y_:0')
cam=tf.get_default_graph().get_tensor_by_name('classmap_reshape:0')
y_conv_tensor=tf.get_default_graph().get_tensor_by_name('y_conv:0')
y_conv,result=sess.run([y_conv_tensor,softmax], feed_dict={x_:sample_img })
cam_abnormal , cam_normal=eval_inspect_cam(sess,cam,top_conv, sample_img , 1,x_,y_,y_conv_tensor)

if __debug__ ==True:
    print np.shape(cam_abnormal)
    plt.imshow(cam_abnormal)
    plt.show()
    print np.shape(cam_normal)
    plt.imshow(cam_normal)
    plt.show()

print result


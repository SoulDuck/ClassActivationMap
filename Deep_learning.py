import tensorflow as tf
from CAM import get_class_map
from CAM import inspect_cam
import random
import numpy as np
#import matplotlib.pyplot as plt
import os , sys , glob
from tensorflow.examples.tutorials.mnist import input_data

def convolution2d(name,x,out_ch,k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        in_ch=x.get_shape()[-1]
        filter=tf.get_variable("w" , [k,k,in_ch , out_ch] , initializer=tf.contrib.layers.xavier_initializer())
        bias=tf.Variable(tf.constant(0.1) , out_ch)
        layer=tf.nn.conv2d(x , filter ,[1,s,s,1] , padding)+bias
        layer=tf.nn.relu(layer)
        if __debug__ == True:
            print 'layer shape : ' ,layer.shape

        return layer

def max_pool(x , k=3 , s=2 , padding='SAME'):

    if __debug__ ==True:
        print 'layer name :'
        print 'layer shape :',layer.shape
    return tf.nn.max_pool(x , ksize=[1,k,k,1] , strides=[1,s,s,1] , padding=padding)


def affine(name,x,out_ch ,keep_prob):
    with tf.variable_scope(name) as scope:
        if len(x.get_shape())==4:
            batch, height , width , in_ch=x.get_shape().as_list()
            w_fc=tf.get_variable('w' , [height*width*in_ch ,out_ch] , initializer= tf.contrib.layers.xavier_initializer())
            x = tf.reshape(x, (-1, height * width * in_ch))
        elif len(x.get_shape())==2:
            batch, in_ch = x.get_shape().as_list()
            w_fc=tf.get_variable('w' ,[in_ch ,out_ch] ,initializer=tf.contrib.layers.xavier_initializer())

        b_fc=tf.Variable(tf.constant(0.1 ), out_ch)
        layer=tf.matmul(x , w_fc) + b_fc

        layer=tf.nn.relu(layer)
        layer=tf.nn.dropout(layer , keep_prob)
        print 'layer name :'
        print 'layer shape :',layer.get_shape
        print 'layer dropout rate :',keep_prob
        return layer
def gap(name,x , n_classes ):
    in_ch=x.get_shape()[-1]
    gap_x=tf.reduce_mean(x, (1,2))
    with tf.variable_scope(name) as scope:
        gap_w=tf.get_variable('w' , shape=[in_ch , n_classes] , initializer=tf.random_normal_initializer(0,0.01))
    y_conv=tf.matmul(gap_x, gap_w)
    return y_conv

def algorithm(y_conv , y_ , learning_rate):
    """

    :param y_conv: logits
    :param y_: labels
    :param learning_rate: learning rate
    :return:  pred,pred_cls , cost , correct_pred ,accuracy
    """
    if __debug__ ==True:
        print y_conv.get_shape()
        print y_.get_shape()

    pred=tf.nn.softmax(y_conv)
    pred_cls=tf.argmax(pred , axis=1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv , labels=y_))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    correct_pred=tf.equal(tf.argmax(y_conv , 1) , tf.argmax(y_ , 1))
    accuracy =  tf.reduce_mean(tf.cast(correct_pred , dtype=tf.float32))
    return pred,pred_cls , cost , train_op,correct_pred ,accuracy
def next_batch(imgs, labs , batch_size):
    indices=random.sample(range(np.shape(imgs)[0]) , batch_size)
    batch_xs=imgs[indices]
    batch_ys=labs[indices]
    return batch_xs , batch_ys

if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_DATA_SET', one_hot=True)
    mnist_train_imgs=np.reshape(mnist.train.images , (55000 ,28,28,1))
    mnist_train_labs=mnist.train.labels
    mnist_test_imgs = np.reshape(mnist.test.images, (10000, 28, 28, 1))
    mnist_test_labs = mnist.test.labels

    print mnist_test_imgs.shape , mnist_train_imgs.shape , mnist_train_labs.shape , mnist_test_labs.shape

    ####
    image_height = 28
    image_width = 28
    image_color_ch = 1
    n_classes = 10
    #####

    x_ = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, image_color_ch])
    y_ = tf.placeholder(dtype=tf.int32, shape=[None, n_classes])
    layer = convolution2d('conv1',x_,64 )
    layer = max_pool(layer)
    top_conv = convolution2d('conv2', x_, 128)
    layer = max_pool(top_conv)
    y_conv   = gap('gap' ,layer,n_classes)
    cam=get_class_map('gap',top_conv,0,im_width=image_width)
    pred, pred_cls, cost,train_op, correct_pred, accuracy=algorithm(y_conv , y_ ,0.005)

    sess=tf.Session()
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    check_point=1000
    for step in range(50000):
        if step % check_point==0:
            inspect_cam(sess, cam , top_conv , mnist_test_imgs , mnist_test_labs ,step , 50 , x_,y_,y_conv)
        batch_xs , batch_ys=next_batch(mnist_train_imgs , mnist_train_labs , batch_size=60)
        train_acc, _ =sess.run([accuracy,train_op] , feed_dict={x_:batch_xs , y_:batch_ys})
        print train_acc

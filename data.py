
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def mnist_28x28():

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
    train_imgs=mnist_train_imgs
    train_labs=mnist_train_labs
    test_imgs=mnist_test_imgs
    test_labs=mnist_test_labs
    return image_height , image_width , image_color_ch , n_classes, train_imgs , train_labs , test_imgs, test_labs
    #####
def eye_64x64():
    folder_paths='/Users/seongjungkim/PycharmProjects/ClassActivationMap/data/'
    train_imgs=np.load(folder_paths+'train_eye_64_imgs.npy')
    train_labs=np.load(folder_paths+'train_eye_64_labs.npy')
    val_imgs = np.load(folder_paths+'val_eye_64_imgs.npy')
    val_labs = np.load(folder_paths+'val_eye_64_labs.npy')
    test_imgs=np.load(folder_paths+'test_eye_64_imgs.npy')
    test_labs = np.load(folder_paths+'test_eye_64_labs.npy')
    test_imgs=np.concatenate((test_imgs , val_imgs) , axis=0)
    test_labs=np.concatenate((test_labs, val_labs), axis=0)

    ####
    image_height = 64
    image_width = 64
    image_color_ch = 3
    n_classes = 2
    if (np.max(train_imgs) >1  ):
        train_imgs=train_imgs/255.
    if (np.max(test_imgs) > 1):
        test_imgs = test_imgs /255.

    return image_height , image_width , image_color_ch , n_classes ,train_imgs , train_labs , test_imgs, test_labs


def concatenate():
    folder_path='/Users/seongjungkim/Desktop/npy_64/'
    train_0_img=np.load(folder_path+'train_0_img.npy')
    train_1_img =np.load(folder_path+'train_1_img.npy')
    train_2_img =np.load(folder_path+'train_2_img.npy')
    train_3_img =np.load(folder_path+'train_3_img.npy')
    train_4_img =np.load(folder_path+'train_4_img.npy')
    train_5_img =np.load(folder_path+'train_5_img.npy')
    train_6_img =np.load(folder_path+'train_6_img.npy')
    train_7_img =np.load(folder_path+'train_7_img.npy')

    train_imgs=np.concatenate((train_0_img,train_1_img,train_2_img,train_3_img,train_4_img,train_5_img,train_6_img,train_7_img ), axis=0)

    train_0_lab = np.load(folder_path+'train_0_lab.npy')
    train_1_lab = np.load(folder_path+'train_1_lab.npy')
    train_2_lab = np.load(folder_path+'train_2_lab.npy')
    train_3_lab = np.load(folder_path+'train_3_lab.npy')
    train_4_lab = np.load(folder_path+'train_4_lab.npy')
    train_5_lab = np.load(folder_path+'train_5_lab.npy')
    train_6_lab = np.load(folder_path+'train_6_lab.npy')
    train_7_lab = np.load(folder_path+'train_7_lab.npy')

    train_labs = np.concatenate((train_0_lab, train_1_lab, train_2_lab, train_3_lab, train_4_lab, train_5_lab,
                                train_6_lab,
                                train_7_lab), axis=0)
    np.save('./data/train_eye_64_imgs' , train_imgs)
    np.save('./data/train_eye_64_labs' , train_labs)

if __name__ == '__main__':
    #concatenate()
    eye_64x64()
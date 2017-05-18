import numpy as np
import matplotlib
import tensorflow as tf
import sys
import os
import pickle as pickle
from six.moves import urllib
import tarfile
import scipy.stats.mstats
import scipy.io as sio
import keras
from keras.utils import to_categorical
from keras import backend as K 
import multiprocessing as mp
from subprocess import call
import warnings
from tqdm import tqdm
###############################################################Command Line Args 
try:
    num_to_make = int(sys.argv[1])
    print('Number of foolers to generate:', num_to_make)
except:
    print('Defaulted to making one fooling image')
    num_to_make = 1

try:
    mode = sys.argv[2]       # 'normal', 'mix', or 'fast'
    print('Chosen mode:', mode)
except:
    print('Defaulted to normal mode since no mode given through command line')
    mode = 'normal'


############################################################### Load DATA
if not os.path.isfile("../data/svhn_train.mat"):
    print('Downloading SVHN train set...')
    call(
        "curl -o data/svhn_train.mat "
        "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
        shell=True
    )
if not os.path.isfile("../data/svhn_test.mat"):
    print('Downloading SVHN test set...')
    call(
        "curl -o data/svhn_test.mat "
        "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
        shell=True
    )
train = sio.loadmat('../data/svhn_train.mat')
test = sio.loadmat('../data/svhn_test.mat')

x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# reshape (n_samples, 1) to (n_samples,) and change 1-index
# to 0-index
y_train = np.reshape(train['y'], (-1,)) 
y_test = np.reshape(test['y'], (-1,))

y_train[y_train == 10] = 0 
y_test[y_test == 10] = 0 

y_train_labels = y_train 
y_test_labels = y_test

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
####################################################################
####################################################################
sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(1) #keras hack for the moment 
model = keras.models.load_model('../data/model_svhn.h5') #load the pretrained model


#function below assumes this is true 
x_input = tf.placeholder(tf.float32, shape=(None,) + x_train.shape[1:])
y = tf.placeholder(tf.float32, shape=(None,) + y_train.shape[1:])
predictions = model(x_input)
logits, = predictions.op.inputs 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


#currently broken, not sure why 
def make_fooling_image_fast(image, target_label, reg=1e-3, step=10/255.):
    fooling_image = image.copy()
    target = to_categorical(target_label, 10)

    dFool, probs = sess.run([tf.gradients(loss, x_input)[0], predictions], feed_dict={x_input: fooling_image, y: target})
    fooling_image[0] -= step * np.sign(np.squeeze(dFool[0]) + reg * (fooling_image[0] - image[0]))
    fooling_image[0] = np.clip(fooling_image[0], 0, 1)

    return fooling_image, probs 



def make_fooling_image_it(image, target_label, reg=1e-3, step=1/255., max_iters=1000, confidence_thresh=0.9):
    fooling_image = image.copy()
    
    target = to_categorical(target_label, 10)
    for j in range(max_iters):
        dFool, probs = sess.run([tf.gradients(loss + reg*tf.reduce_mean(tf.square(image[0] -
                                                                                  fooling_image[0]))/2., x_input)[0],
                                 predictions],
                                 feed_dict={x_input: fooling_image, y: target})
        fooling_image[0] -= step * (np.squeeze(dFool[0]))
        fooling_image[0] = np.clip(fooling_image[0], 0, 1)  # poor man's box constraints
        fool_prob = probs[0, target_label]

        if j % 10 == 0:
            print('Fooling Image Probability Percent (iter %s): %s' % (j, 100.*fool_prob))
            
        if fool_prob > confidence_thresh:
            print('Final fooled prob percent:', 100*fool_prob)
            break

    return fooling_image, probs 


def make_fooling_image_constrained(image, target_label, reg=0, step=50/255., max_iters=1000, confidence_thresh=0.8):
    fooling_image = image.copy()
    target = to_categorical(target_label, 10)
    pc_num = 50
    
    for i in range(max_iters):
        
        dFool, probs = sess.run([tf.gradients(loss + reg*tf.reduce_sum(tf.abs(image[0] - fooling_image[0])), 
                                              x_input)[0],
                                 predictions], feed_dict={x_input: fooling_image, y: target})
        
        filtered_gradient = (np.squeeze(dFool[0]).reshape((1,-1)).dot(u[:,:pc_num])).dot(u[:,:pc_num].T)
        fooling_image[0] -= step * filtered_gradient.reshape((32,32,3))
        fooling_image[0] = np.clip(fooling_image[0], 0, 1)
        
        fool_prob = probs[0, target_label]

        if i % 10 == 0:
            print('Fooling Image Probability Percent (iter %s): %s' % (i, 100.*fool_prob))

        if fool_prob > confidence_thresh:
            print('Final fooled prob percent:', 100*fool_prob)
            break

    return fooling_image, probs




l1_distances = []
l2_distances = []
linf_distances = []



try:
    history = pickle.load(open("svhn_data/" + mode + "_foolers.p", "rb"))
except:
    history = {}

if not os.path.exists('svhn_data'):
    os.makedirs('svhn_data')

if not os.path.exists('svhn_data/normal'):
    os.makedirs('svhn_data/normal')

if not os.path.exists('svhn_data/mix'):
    os.makedirs('svhn_data/mix')

if not os.path.exists('svhn_data/fast'):
    os.makedirs('svhn_data/fast')

for i in range(num_to_make):
    # choose source image from which to generate a fooling image
    rand_int = np.random.randint(10000, size=1)[0]
    # rand_int = examples[i]
    image, true_y = x_test[rand_int:rand_int+1], y_test_labels[rand_int]

    # ensure the network gets our current example correct
    while True:
        p = sess.run(predictions, feed_dict={x_input: image})
        # it's not interesting to do a source-target attack when the net doesn't even understand the source image
        if  p[0].argmax() == true_y:
            break
        rand_int = np.random.randint(10000, size=1)[0]
        image, true_y = x_test[rand_int:rand_int+1], y_test_labels[rand_int]

    target_y = np.random.choice(10)
    while target_y == true_y:
        target_y = np.random.choice(10)

    print('Rand int:', rand_int)

    if mode == 'normal':
        fooling_image, probs  = make_fooling_image_it(image, target_y)
    elif mode == 'fast':
        fooling_image, probs  = make_fooling_image_fast(image, target_y)
        
    guess = probs[0].argmax()
    
    if guess == true_y:
        fooled = 'not_fooled'
        print('Network is NOT fooled!')
    else:
        fooled = 'fooled'
        print('Network is fooled!')


    if fooled == 'fooled':
        l2 = np.sum(np.square(image - fooling_image))
        l1 = np.sum(np.abs(image - fooling_image))
        linf = np.sum(np.max(np.abs(image - fooling_image)))

        l2_distances.append(l2)
        l1_distances.append(l1)
        linf_distances.append(linf)

        history[str(rand_int)] = [fooled, true_y, target_y, fooling_image, image, l2, l1]

    print('Number of fooling examples collected:', len(l2_distances))
    print('L1 mean:', np.mean(np.array(l1_distances)))
    print('L2 mean:', np.mean(np.array(l2_distances)))
    print('LInf mean:', np.mean(np.array(linf_distances)))

    pickle.dump(history, open("svhn_data/" + mode + "_foolers.p", "wb"))

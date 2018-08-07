import tensorflow as tf
from rbm4cf import *
from dataset import load_movielens

sess = tf.Session()

base_dir = r'../input/ml-100k/'
inp, mask, inp_test, mask_test = load_movielens('u1.base','u1.test',base_dir)
# tf.reset_default_graph()
sess = tf.Session()
rbm = RBM4CF(num_visible=inp.shape[1], num_hidden=100, k=5, session=sess)
rbm.fit(inp,mask,validation_data=(inp_test,mask_test),epochs=50,batch_size=64)
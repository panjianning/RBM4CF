import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

def rmse(y_true,y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

class RBM4CF(object):
    
    def __init__(self, num_visible, num_hidden, k, session):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self._create_placeholders()
        self._create_variables()
        self.train_op = self._train_op()
        self.sess = session

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _create_placeholders(self):
        self.input = tf.placeholder(shape=[None, self.num_visible], dtype=tf.float32, 
                                    name="input")
        self.mask = tf.placeholder(shape=[None, self.num_visible], dtype=tf.float32, 
                                   name="mask")
        
    def _create_variables(self):
        with tf.variable_scope("model"):
            self.weights = tf.get_variable(shape=[self.num_visible, self.num_hidden],
                                           dtype=tf.float32, name="weights")
            self.hbias = tf.get_variable(shape=[self.num_hidden], dtype=tf.float32,
                                         name="hbias")
            self.vbias = tf.get_variable(shape=[self.num_visible], dtype=tf.float32, 
                                         name="vbias")
            
            self.prev_dw = tf.get_variable(shape=[self.num_visible, self.num_hidden],
                                           dtype=tf.float32, name="prev_dw")
            self.prev_d_hbias = tf.get_variable(shape=[self.num_hidden], dtype=tf.float32,
                                         name="prev_d_hbias")
            self.prev_d_vbias = tf.get_variable(shape=[self.num_visible], dtype=tf.float32, 
                                         name="prev_d_vbias")
            
    def _train_op(self, w_lr = 0.001, v_lr=0.001, h_lr=0.001, decay=0.000, 
              T=5, momentum=0.9):
        v1, h1, h1a, v2, v2a, h2, h2a = self._contrastive_divergence(self.input)
        for _ in range(T-1):
            v1, h1, h1a, v2, v2a, h2, h2a = self._contrastive_divergence(self.input)
        dw, d_vbias, d_hbias = self._gradient(v1, h1a, v2, h2a, self.mask)
        if decay:
            dw -= decay * self.weights
        update_w = tf.assign(self.weights, 
                             self.weights + momentum*self.prev_dw + w_lr * dw)
        update_v_bias = tf.assign(self.vbias, self.vbias + 
                                  momentum*self.prev_d_vbias + v_lr * d_vbias)
        update_h_bias = tf.assign(self.hbias, self.hbias + 
                                  momentum*self.prev_d_hbias + h_lr * d_hbias)
        update_prev_dw = tf.assign(self.prev_dw, dw)
        update_prev_dvb = tf.assign(self.prev_d_vbias, d_vbias)
        update_prev_dhb = tf.assign(self.prev_d_hbias, d_hbias)
        op = (update_w, update_v_bias, update_h_bias, 
                    update_prev_dw, update_prev_dvb, update_prev_dhb)
        return op
    
    def fit(self, inp, inp_mask, validation_data=None, epochs=1, batch_size=32):
        train_idx = mask.reshape(-1,5).max(axis=1).astype(bool)
        y_train_true = inp.reshape(-1,5).argmax(axis=1)[train_idx]
            
        print('[Baseline] train rmse %.4f' % (rmse(y_train_true,np.mean(y_train_true))))
        
        if validation_data:
            inp_test = validation_data[0]
            mask_test = validation_data[1]
            test_idx = mask_test.reshape(-1,5).max(axis=1).astype(bool)
            y_test_true = inp_test.reshape(-1,5).argmax(axis=1)[test_idx]
            print('[Baseline] validation rmse %.4f' % (rmse(y_test_true,np.mean(y_test_true))))
            
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(1,epochs+1):
            print('[Epoch %03d] ' % epoch, end='')
            for pos in range(0, inp.shape[0], batch_size):
                batch_inp = inp[pos:pos+batch_size]
                batch_mask = inp_mask[pos:pos+batch_size]
                _ = self.sess.run(self.train_op, feed_dict={
                    self.input:batch_inp,self.mask:batch_mask})
                
            y_pred = rbm.predict(inp)
            y_pred_train = y_pred.reshape(-1,5).argmax(axis=1)[train_idx]
            print('train rmse: %.4f' % rmse(y_pred_train, y_train_true),end=' ')

            if validation_data:
                y_pred_test = y_pred.reshape(-1,5).argmax(axis=1)[test_idx]
                print('validation rmse: %.4f' % rmse(y_pred_test,y_test_true))
            else:
                print()
                
    def predict(self, v1):
        h1, _ = self._sample_hidden(tf.constant(v1,dtype=tf.float32))
        v2, v2a = self._sample_visible(h1)
        return self.sess.run(v2a)

    def get_weights(self):
        weights = {}
        a,b,c = self.sess.run([self.weights,self.vbias,self.hbias])
        weights['weights'] = a
        weights['v_bias'] = b
        weights['h_bias'] = c
        return weights    

    def _sample_hidden(self, vis):
        activations = tf.nn.sigmoid(tf.matmul(vis,self.weights) + self.hbias)
        h1_sample = tf.nn.relu(tf.sign(activations - tf.random_uniform(
            tf.shape(activations))))
        return h1_sample, activations
    
    def _sample_visible(self, hid):
        logits = tf.matmul(hid, tf.transpose(self.weights)) + self.vbias
        logits = tf.reshape(logits, shape=(-1, self.k))
        activations = tf.nn.softmax(logits)
        activations = tf.reshape(activations, shape=(-1,self.num_visible))
        v1_sample = tf.nn.relu(tf.sign(activations - tf.random_uniform(
            tf.shape(activations))))
        return v1_sample, activations
    
    def _contrastive_divergence(self, v1):
        h1, h1a = self._sample_hidden(v1)
        v2, v2a = self._sample_visible(h1)
        h2, h2a = self._sample_hidden(v2)
        return [v1, h1, h1a, v2, v2a, h2, h2a]
    
    def _gradient(self, v1, h1, v2, h2a, masks):
        vh_mask = self._outer(masks, h1)
        dw = tf.reduce_mean(self._outer(v1,h1)*vh_mask - 
                            self._outer(v2,h2a) * vh_mask, axis=0)
        d_vbias = tf.reduce_mean((v1 * masks) - (v2 * masks), axis=0)
        d_hbias = tf.reduce_mean(h1 - h2a, axis=0)
        return [dw,d_vbias,d_hbias]
    
    def _outer(self, v, h):
        return tf.expand_dims(v,2) * tf.expand_dims(h,1)
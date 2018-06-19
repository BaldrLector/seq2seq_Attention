# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import HyperParams as hp
from model import *
import tensorflow as tf
import tensorflow.contrib as contrib
import tensorflow.contrib.seq2seq as seq2seq


def encoder(inputs, is_training=True, scope="encoder", reuse=None):
    '''
    Args:
      inputs: A 2d tensor with shape of [N, T_x, E], with dtype of int32. Encoder inputs.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs = bn(inputs,scope='bn_encoder_input')
        memory, cell, cell_bw = lstm(inputs, num_units=hp.num_hidden, bidirection=True)  # (N, T, 2*num_hidden)
        memory = tf.nn.relu(memory)
    return memory, cell, cell_bw


def decoder(inputs, memory, is_training=True, scope="decoder", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T_y, D = 51].
      memory: A 3d tensor with shape of [N, T_x, D_num_hidden].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        memory = bn(memory,scope='bn_decoder_memory')
        inputs = bn(inputs,scope='bn_decoder_input')
        outputs, state, attention_mechanism, decoder_cell, cell_with_attention = attention_decoder(inputs, memory,

                                                                                                   num_units=hp.output_dims)
        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(), [1, 2, 0])
        # Decoder RNNs
        # outputs += lstm(outputs, hp.output_dims, bidirection=False, scope="decoder_gru1")  # (N, T_y/r, E)
        # outputs += lstm(outputs, hp.output_dims, bidirection=False, scope="decoder_gru1")  # (N, T_y/r, E)

        outputs = tf.nn.sigmoid(outputs) * 100

    return outputs, state, attention_mechanism, decoder_cell, cell_with_attention, alignments


class Graph:
    def __init__(self, mode='train'):
        is_training = True if mode == 'train' else False

        # Graph
        # Data Feeding
        # x: Text. (N, Tx, n_mfcc)
        # y: (N, T_y, D=51)
        # if mode == 'train' :
        #     self.x, self.y = data_loader.__next__()
        # elif mode == 'eval':
        #     self.x = tf.placeholder(tf.float32, shape=(None, None,hp.n_mfcc))
        #     self.y = tf.placeholder(tf.float32,shape=(None,None,hp.output_dims))
        # else:
        #     self.x, self.y = None, None

        self.x = tf.placeholder(tf.float32, shape=(None, None, hp.n_mfcc))
        self.y = tf.placeholder(tf.float32, shape=(None, None, hp.output_dims))

        self.encoder_inputs = self.x  # (N, T_x, n_mfcc)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.y[:, :1, :]), self.y[:, :-1, :]),
                                        1)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('net'):
            self.memory, self.encoder_lstmfwcell, self.encoder_lstmbwcell = encoder(self.encoder_inputs,
                                                                                    is_training=is_training)  # (N, Tx, n_mfcc)
            # tf.summary.histogram('{}/encoder_lstmfwcell.weights[0]'.format(mode), self.encoder_lstmfwcell.weights[0])
            # tf.summary.histogram('{}/encoder_lstmfwcell.weights[1]'.format(mode), self.encoder_lstmfwcell.weights[1])
            #
            # tf.summary.histogram('{}/encoder_lstmbwcell.weights[0]'.format(mode), self.encoder_lstmbwcell.weights[0])
            # tf.summary.histogram('{}/encoder_lstmbwcell.weights[1]'.format(mode), self.encoder_lstmbwcell.weights[1])
            #
            # tf.summary.histogram('{}/memory'.format(mode), self.memory)

            self.y_hat, _, self.attention_mechanism, self.decoder_cell, self.cell_with_attention, self.alignments = decoder(
                self.decoder_inputs, self.memory, is_training=is_training)

            # tf.summary.histogram('{}/attention_mechanism'.format(mode), self.attention_mechanism)
            # tf.summary.histogram('{}/decoder_cell'.format(mode), self.decoder_cell)
            # tf.summary.histogram('{}/cell_with_attention'.format(mode), self.cell_with_attention)

            tf.summary.histogram('{}/y_hat'.format(mode), self.y_hat)
            tf.summary.histogram('{}/y'.format(mode), self.y)

        if mode in ('train', 'eval'):
            self.loss = tf.losses.mean_squared_error(self.y, self.y_hat)
            # training scheme
            # self.lr = learning_rate_decay(hp.lr, global_step=self.global_step)
            # self.lr = tf.train.exponential_decay(hp.lr,global_step=self.global_step,decay_rate=0.9,decay_steps=10)
            self.lr = hp.lr
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads_vars = self.optimizer.compute_gradients(self.loss)

            for grad, var in grads_vars:
                tf.summary.histogram(var.name + '/grad', grad)
                tf.summary.histogram(var.name, var)
            # gradient clipping
            pass

            self.train_op = self.optimizer.apply_gradients(grads_vars, global_step=self.global_step)

            # summary
            tf.summary.scalar('{}/loss'.format(mode), self.loss)
            tf.summary.scalar('{}/lr'.format(mode), self.lr)

            self.merged = tf.summary.merge_all()

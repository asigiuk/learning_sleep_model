import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def encoder(input, is_training, feature_dim):

    end_points ={}
    input = tf.expand_dims(input,-1)

    enc = tf.layers.conv2d(input ,filters = 32, kernel_size = (5, 5), strides=(2, 2), padding='same', name= 'enc_conv_layer_1', reuse= tf.AUTO_REUSE)
    enc = tf.layers.batch_normalization(enc, training=is_training, name='BN_1')
    enc = tf.nn.leaky_relu(enc ,alpha = 0.2)
    end_points['layer_1'] = enc
    enc = tf.layers.dropout(enc, rate = 0.3, training= is_training, name = 'dropout_layer_1')


    enc = tf.layers.conv2d(enc ,filters = 128, kernel_size = (5, 5), strides=(2, 2), padding='same', name= 'enc_conv_layer_2', reuse= tf.AUTO_REUSE)
    enc = tf.layers.batch_normalization(enc, training=is_training, name='BN_2')
    end_points['layer_2'] = enc
    enc = tf.nn.leaky_relu(enc ,alpha = 0.2)

    enc = tf.layers.dropout(enc, rate = 0.3, training= is_training, name = 'dropout_layer_2')

    enc = tf.layers.flatten(enc)
    enc = tf.layers.dense(enc, feature_dim , name = 'enc_dense_layer_3', reuse= tf.AUTO_REUSE)
    enc = tf.layers.batch_normalization(enc, axis=-1, renorm= False , training = is_training, name = 'BN_3')
    end_points['layer_3'] = enc
    enc = tf.nn.leaky_relu(enc ,alpha = 0.2)
    # disc = tf.layers.dropout(disc, rate = 0.3, training= is_training, name = 'DO_1')


    return enc, end_points

def discriminator(predictor_output, encoder_output, is_training):
    # concatonate predictor and encoder
    predictor_output = tf.squeeze(predictor_output, axis=1)
    disc = tf.concat([predictor_output, encoder_output], 1)

    disc = tf.layers.dense(disc, 16 , name = 'FC_1')
    disc = tf.layers.batch_normalization( disc, axis=-1, renorm= False , training = is_training, name = 'BN_1')
    disc = tf.nn.leaky_relu(disc ,alpha = 0.2)
    disc = tf.layers.dropout(disc, rate = 0.3, training= is_training, name = 'DO_1')

    disc = tf.layers.dense(disc, 2 , name = 'FC_2')

    logits = tf.expand_dims(disc, axis=1)
    # print('##########', logits.shape)
    # disc_loss = tf.losses.softmax_cross_entropy(multi_class_labels=tf.one_hot(label, 2), logits=logits)

    return logits

def predictor(encoder_output, is_training):
    #
    pred = tf.layers.dense(encoder_output, 16 , name = 'FC_1')
    pred = tf.layers.batch_normalization( pred, axis=-1, renorm= False , training = is_training, name = 'BN_1')
    pred = tf.nn.leaky_relu(pred ,alpha = 0.2)
    pred = tf.layers.dropout(pred, rate = 0.3, training= is_training, name = 'DO_1')

    pred = tf.layers.dense(pred, 2 , name = 'FC_2')
    logits = tf.expand_dims(pred, axis=1)
    # pred_loss = tf.losses.softmax_cross_entropy(multi_class_labels=tf.one_hot(label, 2), logits=logits)
    # print('##########', logits.shape)

    return logits

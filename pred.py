#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os, getopt
import glob
import time
import cv2
import os
import sys

from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from skimage import io
from model import *
from sys import argv
import argparse

p = argparse.ArgumentParser()

p.add_argument('--input_dir', type=str, default='./project_dir/final', help='Directory where checkpoints and event logs are written to.')

p.add_argument('--batch_size', type=int, default=1, help='The Number of gradients to collect before updating params.')

p.add_argument('--encoder_feature_size', type=int, default=10, help='The Number of gradients to collect before updating params.')


FLAGS = p.parse_args()
#tf.enable_eager_execution()

def create_train(image_path, sleep_label, source_label):
    array_lst = list()
    images = list()
    # read file and convert to numpy array
    with open(image_path.decode(), 'r') as file:
        for row in file.readlines()[1:]:
            row_lst = list()
            for element in row.split(','):
                row_lst.append(float(element))
            array_lst.append(row_lst)
        image_np = np.array(array_lst)
        image_np = 2 *((image_np - image_np.min())/(image_np.max() - image_np.min())) -1
    # convert datatype
    image_np = np.float32(image_np)
    sleep_label = np.int16(sleep_label)
    source_label = np.int16(source_label)
    return image_np, sleep_label, source_label

def main():
    # check input/output directory
    if not os.path.exists(FLAGS.input_dir):
        raise ValueError('Output directory does not exists {}'.format(FLAGS.input_dir))
    # chekc and set checkpoint file path
    pretrained_checkpoint_path = os.path.join(FLAGS.input_dir, 'snapshot_1.ckpt')
    if not os.path.isfile(pretrained_checkpoint_path+'.meta'):
        raise ValueError('checkpoint file does not exiist {}'.format(pretrained_checkpoint_path))
    # chekc output file
    prediction_file = os.path.join(FLAGS.input_dir,'predictions.txt')
    # read input files from test_list
    test_path = os.path.join(FLAGS.input_dir, "test_list.txt")
    if os.path.isfile(test_path):
        source_label_lst = list()
        sleep_label_lst = list()
        test_filenames = list()
        with open(test_path, 'r') as file:
            filenames = file.readlines()
            for file_name in filenames:
                test_filenames.append(file_name.strip())
                file_name = file_name.split('/')[-1]
                label = file_name.split('_')[1]
                if label == 's1':
                    source_label_lst.append(0)
                elif label == 's2':
                    source_label_lst.append(1)
                if 'F' in file_name:
                    sleep_label_lst.append(0)
                if 'MW' in file_name:
                    sleep_label_lst.append(1)
    else:
        raise ValueError('Could not find test_list.txt file {}'.format(test_path))

    test_dataset_size = len(test_filenames)
    n_batch = int(np.ceil(test_dataset_size / FLAGS.batch_size))
    # prepare input for prediction model
    filenames_lst = test_filenames
    image_path_np = np.array(test_filenames)
    sleep_label_np = np.array(sleep_label_lst)
    source_label_np = np.array(source_label_lst)
    # print('##### input dataset shape',image_path_np, sleep_label_np, source_label_np)

    ##### Create place holder
    tf_input_images = tf.placeholder(tf.float32, shape = (None, 64, 8192), name = 'tf_input_images')
    tf_source_label = tf.placeholder(tf.int32, shape = (None), name = 'tf_source_label')
    tf_sleep_label = tf.placeholder(tf.int32, shape = (None), name = 'tf_sleep_label')
    tf_is_training = tf.placeholder(tf.bool, name = 'tf_is_training')
    tf_filenames = tf.placeholder(tf.string, shape = (None), name = 'tf_filenames')

    # Input pipeline
    with tf.variable_scope(None, 'Data_input_pipeline'):
        # import data and apply mapping function
        test_dataset = tf.data.Dataset.from_tensor_slices((image_path_np, sleep_label_np, source_label_np))

        test_dataset = test_dataset.map((lambda images, sleep_label, source_label : tf.py_func(create_train, [images, sleep_label, source_label], [np.float32, np.int16, np.int16])), num_parallel_calls= 8).batch(FLAGS.batch_size)

        test_iterator = test_dataset.make_initializable_iterator()
        test_next_element = test_iterator.get_next()

    # define encoder model
    with tf.variable_scope(None, 'encoder'):
        features,_ = encoder(tf_input_images, tf_is_training, FLAGS.encoder_feature_size)
    tf.summary.image('input_images', tf.expand_dims(tf_input_images,-1), max_outputs=3)

    # define predictor model
    with tf.variable_scope(None, 'predictor'):
        pred_logits = predictor(features, tf_is_training)
        # define prediction loss function
        pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(tf_sleep_label, 2), logits=pred_logits)
        predict_sleep = tf.nn.softmax(pred_logits)
        sleep_prediction = tf.argmax(predict_sleep, axis=1)

    with tf.variable_scope(None, 'discriminator'):
        disc_logits = discriminator(predict_sleep , features, tf_is_training)
        # define discriminator loss function
        disc_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(tf_source_label, 2), logits=disc_logits)
        predict_source = tf.nn.softmax(disc_logits)
        source_prediction = tf.argmax(predict_source, axis=1)

    # define checkpoint varaibles
    trained_include = ['encoder', 'predictor', 'discriminator', 'global_step']
    trained_exclude = []

    trained_vars = tf.contrib.framework.get_variables_to_restore(
        include = trained_include,
        exclude = trained_exclude)
    # print('saved_varaiables ------------------------- ', trained_vars)
    tf_saver = tf.train.Saver(trained_vars, name="saver")

    # initialize variables
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        # initialize variables
        sess.run(init)
        # load variables from checkpoint
        print("Loading from checkpoint ", pretrained_checkpoint_path)
        tf_saver.restore(sess, pretrained_checkpoint_path)
        sess.run(test_iterator.initializer)
        # init output lists
        sleep_prediction_lst = list()
        source_prediction_lst = list()

        for batch in range(n_batch):
            batch_images, batch_sleep_label, batch_source_label = sess.run(test_next_element)

            sleep_pred_np, source_pred_np = sess.run([predict_sleep, predict_source], feed_dict={tf_input_images: batch_images, tf_source_label: batch_source_label,  tf_sleep_label: batch_sleep_label, tf_is_training: False})

            sleep_prediction_lst.append(np.argmax(sleep_pred_np))
            source_prediction_lst.append(np.argmax(source_pred_np))

    # save predictions
    with open(prediction_file, 'w') as fout:
      fout.write('file_name, sleep_prediction, sleep_label, source_prediction, source_label\n')
      for file_name, sleep_pred, sleep_label, source_pred, source_label in zip(filenames_lst, sleep_prediction_lst, sleep_label_lst, source_prediction_lst, source_label_lst):
          output_line = ",".join([file_name, str(sleep_pred), str(sleep_label), str(source_pred), str(source_label), '\n'])
          fout.write(output_line)

    # Evaluate predictor
    y_true = sleep_label_lst
    y_pred = sleep_prediction_lst
    conf_mat_output = confusion_matrix(y_true, y_pred)
    output_acc = accuracy_score(y_true, y_pred)
    print("\n\n\n==================== Predictor evaluation Result Summary ====================")
    print("Accuracy score : {}".format(output_acc))
    # print("F1 score: {}".format(output_f1))
    print(classification_report(y_true, y_pred, digits=7))
    print("===================================================================")

    # Evaluate discriminator
    y_true = source_label_lst
    y_pred = source_prediction_lst
    conf_mat_output = confusion_matrix(y_true, y_pred)
    output_acc = accuracy_score(y_true, y_pred)
    print("\n\n\n==================== Discriminator evaluation Result Summary ====================")
    print("Accuracy score : {}".format(output_acc))
    print(classification_report(y_true, y_pred, digits=7))
    print("===================================================================")

if __name__ == "__main__":
    main()

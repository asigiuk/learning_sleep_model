#!/usr/bin/python

import tensorflow as tf
import numpy as np
import os, getopt
import glob
import time
import cv2
import os
import sys

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from skimage import io
from model import *
from sys import argv
import argparse

p = argparse.ArgumentParser()

p.add_argument('--dataset_dir', type=str, default='./data/intermediate/', help='Input dataset.')

p.add_argument('--test_percentage', type=int, default=10, help='Percentage of images to use as a test set.')

p.add_argument('--output_dir', type=str, default='./project_dir/experiment_final', help='Directory where checkpoints and event logs are written to.')

p.add_argument('--checkpoint_path', type=str, default='', help='Path to pretrained model checkpoint')

p.add_argument('--restore_model', type=bool, default=False, help='Restore weights from checkpoint')

p.add_argument('--max_epochs', type=int, default=3, help='Number of epochs for encoder predictor training loop')

p.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')

p.add_argument('--lambda_value', type=float, default=1e-5, help='Coefficient applied to discriminator loss.')

p.add_argument('--batch_size', type=int, default=32, help='The number of samples in each batch.')

p.add_argument('--encoder_feature_size', type=int, default=10, help='The Number output feature from encoder.')

FLAGS = p.parse_args()

def create_train(image_path, sleep_label, source_label):
    array_lst = list()
    images = list()
    # load file and convert to numpy array
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
    # saved model checkpoint file path
    save_checkpoint_path = os.path.join(FLAGS.output_dir, 'snapshot_1.ckpt')
    # restore model from pretrained checkpoint path
    pretrained_checkpoint_path = os.path.join(FLAGS.output_dir, 'snapshot_1.ckpt')
    if FLAGS.checkpoint_path:
        pretrained_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    # create output file
    output_file = os.path.join(FLAGS.output_dir,'log.txt')

    # create directory
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    # create train/test split and images and label lists
    if os.path.isdir(FLAGS.dataset_dir):
        # read images and generate image list
        image_path_lst = list()
        source_label_lst = list()
        sleep_label_lst = list()
        for file_name in os.listdir(FLAGS.dataset_dir):
            if file_name.endswith('.txt'):
                # print(file_name)
                file_path = os.path.join(FLAGS.dataset_dir, file_name)
                image_path_lst.append(file_path)
        training_filenames, test_filenames = train_test_split(image_path_lst, test_size=FLAGS.test_percentage/100, random_state=13)
        for file_name in training_filenames:
            file_name = file_name.split('/')[-1]
            # print(file_name)
            label = file_name.split('_')[1]
            if label == 's1':
                source_label_lst.append(0.)
            elif label == 's2':
                source_label_lst.append(1.)
            if 'F' in file_name:
                sleep_label_lst.append(0.)
            if 'MW' in file_name:
                sleep_label_lst.append(1.)

        # save file name list
        train_path = os.path.join(FLAGS.output_dir, "train_list.txt")
        with open(train_path, 'w') as file:
            file.write('\n'.join(training_filenames))
        test_path = os.path.join(FLAGS.output_dir, "test_list.txt")
        with open(test_path, 'w') as file:
            file.write('\n'.join(test_filenames))
    # read train images and label lists from file
    else:
        train_path = os.path.join(FLAGS.output_dir, "train_list.txt")
        source_label_lst = list()
        sleep_label_lst = list()
        training_filenames = list()
        with open(train_path, 'r') as file:
            filenames = file.readlines()
            for file_name in filenames:
                training_filenames.append(file_name.strip())
                file_name = file_name.split('/')[-1]
                label = file_name.split('_')[1]
                if label == 's1':
                    source_label_lst.append(0.)
                elif label == 's2':
                    source_label_lst.append(1.)
                if 'F' in file_name:
                    sleep_label_lst.append(0.)
                if 'MW' in file_name:
                    sleep_label_lst.append(1.)

    image_path_np = np.array(training_filenames)
    sleep_label_np = np.array(sleep_label_lst)
    source_label_np = np.array(source_label_lst)

    ##### Create place holder
    input_images = tf.placeholder(tf.float32, shape = (None, 64, 8192), name = 'input_images')
    source_label = tf.placeholder(tf.int32, shape = (None), name = 'source_label')
    sleep_label = tf.placeholder(tf.int32, shape = (None), name = 'sleep_label')
    tf_is_training = tf.placeholder(tf.bool, name = 'tf_is_training')


    train_dataset_size = len(training_filenames)
    n_batch = int(np.ceil(train_dataset_size / FLAGS.batch_size))
    # Input pipeline
    with tf.variable_scope(None, 'Data_input_pipeline'):
        # import data and apply mapping function
        train_dataset = tf.data.Dataset.from_tensor_slices((image_path_np, sleep_label_np, source_label_np )).shuffle(FLAGS.batch_size*2)
        train_dataset = train_dataset.map((lambda images, sleep_label, source_label: tf.py_func(create_train, [images, sleep_label, source_label], [np.float32, np.int16, np.int16])), num_parallel_calls= 8).batch(FLAGS.batch_size).repeat()

        train_iterator = train_dataset.make_initializable_iterator()
        train_next_element = train_iterator.get_next()

    # define encoder model
    with tf.variable_scope(None, 'encoder'):
        features,_ = encoder(input_images, tf_is_training, FLAGS.encoder_feature_size)
    # tf.summary.image('input_images', tf.expand_dims(input_images,-1), max_outputs=3)

    # define predictor model
    with tf.variable_scope(None, 'predictor'):
        pred_logits = predictor(features, tf_is_training)
        # define prediction loss function
        pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(sleep_label, 2), logits=pred_logits)
        predict_sleep = tf.nn.softmax(pred_logits)

    # define discriminator model
    with tf.variable_scope(None, 'discriminator'):
        disc_logits = discriminator(predict_sleep , features, tf_is_training)
        # define discriminator loss function
        disc_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(source_label, 2), logits=disc_logits)

    # define value function
    value_loss = pred_loss - FLAGS.lambda_value * disc_loss
    # add summaries
    tf.summary.scalar('lambda value', FLAGS.lambda_value)
    tf.summary.scalar('discriminator loss', disc_loss)
    tf.summary.scalar('dredictor loss', pred_loss)
    tf.summary.scalar('value loss', value_loss)
    # define trainable variables for each model
    train_encoder = [var for var in tf.trainable_variables() if ('encoder' in var.name)]
    train_predictor = [var for var in tf.trainable_variables() if ('predictor' in var.name)]
    train_discriminator = [var for var in tf.trainable_variables() if ('discriminator' in var.name)]

    # add trainable variables to summary
    for var in train_encoder:
        print('encoder training parameters -------------------------------',var)
        tf.summary.histogram(str(var.name),var)

    for var in train_predictor:
        print('predictor training parameters -------------------------------',var)
        tf.summary.histogram(str(var.name),var)

    for var in train_discriminator:
        print('discriminator training parameters -------------------------------',var)
        tf.summary.histogram(str(var.name),var)

    # create global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    increment_global_step_op = tf.assign(global_step, global_step+1)

    # define exponential decay learning rate
    decay_steps = int(train_dataset_size/
                      FLAGS.batch_size)
    decay_steps = 10
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      0.94,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    # define encoder optimizer
    with tf.variable_scope('encoder_opt') as scope:
        tf_optimizer_enc = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'encoder')
        with tf.control_dependencies(update_ops):
            optimizer_enc = tf_optimizer_enc.minimize(value_loss, var_list = train_encoder, name="train_predictor")
    # define perdictor optimizer
    with tf.variable_scope('predictor_opt') as scope:
        tf_optimizer_pred = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'predictor')
        with tf.control_dependencies(update_ops):
            optimizer_pred = tf_optimizer_pred.minimize(value_loss, var_list = train_predictor, name="train_predictor")
    # define discriminator optimizer
    with tf.variable_scope('discriminator_opt') as scope:
        tf_optimizer_disc = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'discriminator')
        with tf.control_dependencies(update_ops):
            optimizer_disc = tf_optimizer_disc.minimize(value_loss, var_list = train_discriminator, name="train_discriminator")

    # define checkpoint varaibles
    trained_include = ['encoder', 'predictor', 'discriminator', 'global_step']
    trained_exclude = []

    trained_vars = tf.contrib.framework.get_variables_to_restore(
        include = trained_include,
        exclude = trained_exclude)
    print('saved_varaiables ------------------------- ', trained_vars)
    tf_saver = tf.train.Saver(trained_vars, name="saver")

    # initialize, configure and start session
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        sess.run(init)
        # restoring the latest checkpoint in checkpoint_dir
        if FLAGS.restore_model:
            print("Loading from checkpoint ", pretrained_checkpoint_path)
            tf_saver.restore(sess, pretrained_checkpoint_path)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.output_dir + '/train', sess.graph)
        sess.run(train_iterator.initializer)

        disc_loss_np = 0
        while disc_loss_np < 2.8:
            for epoch in range(FLAGS.max_epochs):
                # f = open(output_file, 'a')
                global_step_np = sess.run(increment_global_step_op)
                # train predictor
                cum_pred_loss = 0.0
                # cum_val_loss = 0.0
                for batch in range(n_batch):
                    batch_images, batch_sleep_label, batch_source_label = sess.run(train_next_element)
                    # encoder training
                    _, _, val_loss_np, pred_loss_np, disc_loss_np, summary_train = sess.run([optimizer_enc, optimizer_pred, value_loss, pred_loss, disc_loss, merged], feed_dict={input_images: batch_images, source_label:batch_source_label,  sleep_label:batch_sleep_label, tf_is_training: True})
                    sys.stdout.write('\r>> training predictor: loss {},  batch {}/{}, ephocs {}/{}    '.format(pred_loss_np, batch+1, n_batch, epoch+1, FLAGS.max_epochs))
                    sys.stdout.flush()
                    cum_pred_loss += pred_loss_np
                # add predition loss to summary
                epoch_pred_loss = cum_pred_loss / (n_batch *FLAGS.batch_size)
                train_writer.add_summary(summary_train, global_step_np)
                summary = tf.Summary()
                summary.value.add(tag='predictor loss ' , simple_value= epoch_pred_loss)
                train_writer.add_summary(summary, global_step_np)
            print('\nPredictor training complete')

            # train discriminator
            cum_val_loss = 0.0
            cum_disc_loss = 0.0
            for batch in range(n_batch):
                batch_images, batch_sleep_label, batch_source_label = sess.run(train_next_element)
                # predictor training
                _, val_loss_np, pred_loss_np, disc_loss_np, summary_train = sess.run([optimizer_disc, value_loss, pred_loss, disc_loss, merged], feed_dict={input_images: batch_images, source_label:batch_source_label,  sleep_label:batch_sleep_label, tf_is_training: True})
                sys.stdout.write('\r>> training discriminator: loss {}, batch {}/{}    '.format(disc_loss_np, batch+1, n_batch))
                sys.stdout.flush()
                cum_disc_loss += disc_loss_np
                cum_val_loss += val_loss_np
            print('\nDiscriminator training complete\n')
            epoch_val_loss = cum_val_loss / (n_batch *FLAGS.batch_size)
            epoch_disc_loss = cum_disc_loss / (n_batch *FLAGS.batch_size)

            # basic evaluation
            pred_sleep, gt_label = sess.run([predict_sleep, sleep_label], feed_dict={input_images: batch_images, source_label:batch_source_label,  sleep_label:batch_sleep_label, tf_is_training: False})
            #
            print('step number {} >>>> predictor loss {}, discriminator loss {}, value loss {}\n'.format(global_step_np,  epoch_pred_loss, epoch_disc_loss ,epoch_val_loss))
            with open(output_file, 'a') as file:
                file.write('predictor loss: {}, discriminator loss: {}, value loss: {} for step number {}'.format( epoch_pred_loss, epoch_disc_loss ,epoch_val_loss , global_step_np))

            # Add to summary
            summary = tf.Summary()
            summary.value.add(tag=' value loss ' , simple_value= epoch_val_loss)
            summary.value.add(tag='discriminator loss ' , simple_value= epoch_disc_loss)
            train_writer.add_summary(summary, global_step_np)

            # saving checkpoint
            print('save model')
            tf_saver.save(sess, save_checkpoint_path)
            with open(output_file, 'a') as file:
                file.write('predictor: softmax output {}, ground truth label: {}\n'.format(pred_sleep, gt_label))


if __name__ == "__main__":
    main()

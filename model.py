import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def create_train(good_train_images, bad_train_images):
    #print(image_pet_dataset_path)
    shape =(299,299,3)

    good_image_aug = image_augment(good_train_images)
    bad_image_aug = image_augment(bad_train_images)


    dataset_features = np.float32(dataset_features)
    dataset_labels = np.int16(dataset_labels)

    return

def image_augment(image):
    augment_img = iaa.Sequential([iaa.OneOf([iaa.Flipud(0.5),iaa.Fliplr(0.5)]),
                                  iaa.OneOf([iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
                                             , iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))])]
                                 , random_order=True)

    image_aug = augment_img.augment_image(image)

    return image_aug

def make_generator_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 32)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None,112 , 112, 16)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    #model.add(tf.keras.layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #assert model.output_shape == (None,224 , 224, 16)
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 224, 224, 3)

    return model

def generator(input, is_training):

    end_points = {}
    # layer 1
    dec = tf.layers.dense(input, 7*7*256, use_bias=False , name = 'dec_dense_layer_1')

    dec = tf.layers.batch_normalization(dec ,axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_1')
    end_points['layer_1'] = dec
    dec = tf.nn.leaky_relu(dec ,alpha = 0.2)


    dec = tf.reshape(dec,[-1,7, 7, 256])

    dec = tf.layers.conv2d_transpose(dec, filters = 128, kernel_size= (5, 5), strides=(2, 2), padding='same', use_bias=False, name = 'dec_conv_layer_2')
    end_points['layer_2'] = dec
    dec = tf.layers.batch_normalization(dec ,axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_2')
    dec = tf.nn.leaky_relu(dec ,alpha = 0.2)
    #end_points['layer_2'] = dec

    dec = tf.layers.conv2d_transpose(dec, filters = 64, kernel_size= (5, 5), strides=(2, 2), padding='same', use_bias=False, name = 'dec_conv_layer_3')
    dec = tf.layers.batch_normalization(dec ,axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_3')
    dec = tf.nn.leaky_relu(dec ,alpha = 0.2)
    end_points['layer_3'] = dec

    dec = tf.layers.conv2d_transpose(dec, filters = 32, kernel_size= (5, 5), strides=(2, 2), padding='same', use_bias=False, name = 'dec_conv_layer_4')
    dec = tf.layers.batch_normalization(dec ,axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_4')
    dec = tf.nn.leaky_relu(dec ,alpha = 0.2)
    end_points['layer_4'] = dec


    dec = tf.layers.conv2d_transpose(dec, filters = 16, kernel_size= (5, 5), strides=(2, 2), padding='same', use_bias=False, name = 'dec_conv_layer_5')
    dec = tf.layers.batch_normalization(dec ,axis=-1, renorm= False , training = is_training, name = 'batch_norm_layer_5')
    dec = tf.nn.leaky_relu(dec ,alpha = 0.2)
    end_points['layer_5'] = dec

    dec = tf.layers.conv2d_transpose(dec, filters = 3, kernel_size= (5, 5), strides=(2, 2), padding='same', activation= tf.nn.tanh, use_bias=False, name = 'dec_conv_layer_6')
    end_points['layer_6'] = dec

    return dec, end_points


def make_discriminator_model():
    model = tf.keras.Sequential()

    #model.add(tf.keras.layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same'))
    #model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    #model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    #model.add(tf.keras.layers.LeakyReLU())
    #model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

# def discriminator(input, is_training):
#
#     end_points ={}
#
#     enc = tf.layers.conv2d(input ,filters = 32, kernel_size = (5, 5), strides=(2, 2), padding='same', name= 'enc_conv_layer_1', reuse= tf.AUTO_REUSE)
#     enc = tf.nn.leaky_relu(enc ,alpha = 0.2)
#     end_points['layer_1'] = enc
#     enc = tf.layers.dropout(enc, rate = 0.3, training= is_training, name = 'dropout_layer_1')
#
#
#     enc = tf.layers.conv2d(enc ,filters = 128, kernel_size = (5, 5), strides=(2, 2), padding='same', name= 'enc_conv_layer_2', reuse= tf.AUTO_REUSE)
#     end_points['layer_2'] = enc
#     enc = tf.nn.leaky_relu(enc ,alpha = 0.2)
#
#     enc = tf.layers.dropout(enc, rate = 0.3, training= is_training, name = 'dropout_layer_2')
#
#     enc = tf.layers.flatten(enc)
#     enc = tf.layers.dense(enc, 1 , name = 'enc_dense_layer_3', reuse= tf.AUTO_REUSE)
#     end_points['layer_3'] = enc
#
#     return enc, end_points

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

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)
    print(predictions.shape)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

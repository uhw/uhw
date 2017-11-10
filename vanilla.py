# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 12:41:47 2017

@author: Pavitrakumar
"""

import os
import numpy as np
import time

from PIL import Image
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Input, merge, Conv2D, Conv2DTranspose, LeakyReLU
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import keras.backend as K
from keras.initializers import RandomNormal
K.set_image_dim_ordering('tf')

np.random.seed(42)


def get_gen_normal(noise_shape):
    noise_shape = noise_shape
    """
    Changing padding = 'same' in the first layer makes a lot fo difference!!!!
    """
    kernel_init = 'glorot_uniform'

    gen_input = Input(shape = noise_shape) #if want to directly use with conv layer next

    generator = Conv2DTranspose(filters = 512, kernel_size = (4,4), strides = (1,1), padding = "valid", data_format = "channels_last", kernel_initializer = kernel_init)(gen_input)
    generator = BatchNormalization(momentum = 0.5)(generator, training=1)
    generator = LeakyReLU(0.2)(generator)

    generator = Conv2DTranspose(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator, training=1)
    generator = LeakyReLU(0.2)(generator)

    generator = Conv2DTranspose(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator, training=1)
    generator = LeakyReLU(0.2)(generator)

    generator = Conv2DTranspose(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator, training=1)
    generator = LeakyReLU(0.2)(generator)

    generator = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = BatchNormalization(momentum = 0.5)(generator, training=1)
    generator = LeakyReLU(0.2)(generator)


    generator = Conv2DTranspose(filters = 3, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(generator)
    generator = Activation('tanh')(generator)

    # gen_opt = Adam(lr=0.00015, beta_1=0.5)
    generator_model = Model(inputs=gen_input, outputs=generator)
    # generator_model.compile(loss='binary_crossentropy', optimizer=gen_opt, metrics=['accuracy'])
    # generator_model.summary()

    return generator_model

def get_disc_normal(image_shape=(64,64,3)):
    image_shape = image_shape

    dropout_prob = 0.4

    kernel_init = 'glorot_uniform'

    dis_input = Input(shape = image_shape)

    discriminator = Conv2D(filters = 64, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(dis_input)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Conv2D(filters = 128, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator, training=1)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Conv2D(filters = 256, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator, training=1)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Conv2D(filters = 512, kernel_size = (4,4), strides = (2,2), padding = "same", data_format = "channels_last", kernel_initializer = kernel_init)(discriminator)
    discriminator = BatchNormalization(momentum = 0.5)(discriminator, training=1)
    discriminator = LeakyReLU(0.2)(discriminator)

    discriminator = Flatten()(discriminator)

    discriminator = Dense(1)(discriminator)
    discriminator = Activation("sigmoid")(discriminator)

    # dis_opt = Adam(lr=0.0002, beta_1=0.5)
    discriminator_model = Model(inputs=dis_input, outputs=discriminator)
    # discriminator_model.compile(loss="binary_crossentropy", optimizer=dis_opt, metrics=["accuracy"])
    # discriminator_model.summary()
    return discriminator_model

def setup():
    if not os.path.isdir("./images"):
        os.makedirs("./images")
        print("[ INFO ] images directory created")
    if not os.path.isdir("./saved_models"):
        os.makedirs("./saved_models")
        print("[ INFO ] saved_models directory created")

def data_generator(batch_size, train_data_directory):
    train_datagen = ImageDataGenerator(shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(train_data_directory,
                                                        target_size=(64, 64),
                                                        batch_size=batch_size,
                                                        class_mode="binary")
    return train_generator


def build_network(input_shape, noise_size, lambda_param):
    print("[ INFO ] building network...", end="")
    noise_shape = (1, 1, noise_size)
    discriminator = get_disc_normal(input_shape)
    generator = get_gen_normal(noise_shape)

    # Compute Wasserstein Loss and Gradient Penalty
    d_real_input = Input(shape=input_shape)
    noise = Input(shape=noise_shape) # Ex: shape = (128, )
    d_fake_input = generator(noise)

    epsilon_input = K.placeholder(shape=(None, ) + input_shape)
    d_mixed_input = Input(shape=input_shape, tensor=d_real_input + epsilon_input)

    loss_real = K.mean(discriminator(d_real_input))
    loss_fake = K.mean(discriminator(d_fake_input))

    gradient_mixed = K.gradients(discriminator(d_mixed_input), [d_mixed_input])[0]
    normalized_gradient_mixed = K.sqrt(K.sum(K.square(gradient_mixed), axis=[1, 2, 3]))
    gradient_penalty = K.mean(K.square(normalized_gradient_mixed - 1))

    loss = loss_fake - loss_real + (lambda_param * gradient_penalty)

    # Discriminator
    training_updates = Adam(lr=0.0002).get_updates(discriminator.trainable_weights, [], loss)
    discriminator_train = K.function([d_real_input, noise, epsilon_input],
                                     [loss_real, loss_fake],
                                     training_updates)
    # Generator
    loss = -loss_fake
    training_updates = Adam(lr=0.0002).get_updates(generator.trainable_weights, [], loss)
    generator_train = K.function([noise], [loss], training_updates)

    print("done")
    return (discriminator, generator, discriminator_train, generator_train)

# Save the generator and discriminator networks (and weights) for later use
def save_models(epoch, generator, discriminator):
    print("[ INFO ] saving model...", end="")
    generator.save("./saved_models/gan_generator_epoch_{0}.h5".format(epoch))
    discriminator.save("./saved_models/gan_discriminator_epoch_{0}.h5".format(epoch))
    print("done")

def save_images(images, epoch, generation):
    print("[ INFO ] saving images...", end="")
    for i in range(images.shape[0]):
        g_image = array_to_img(images[i])
        g_image.save("./images/generated_image_{0}_{1}_{2}.png".format(epoch, generation, i))
    print("done")

def execute2(epochs, batch_size, noise_size, train_generator, networks):
    discriminator, generator, discriminator_train, generator_train = networks
    generation = 0
    noise_shape = (batch_size, 1, 1, noise_size)
    fixed_noise = np.random.normal(size=noise_shape).astype("float32")
    batches = train_generator.samples // batch_size

    g_error = 0
    epoch = 0
    batch_counter = 0
    while epoch < epochs:
        for batch, label in train_generator:
            batch_counter += 1

            # Leave after finishing all batches or executing for 100 cycles.
            if batch.shape[0] != batch_size:
                print("[ INFO ] Epoch Completed {0}".format(epoch))
                batch_counter = 0
                epoch += 1
                break
            elif batch_counter % 100 == 0:
                print("[ INFO ] Trained on {0} Batches".format(batch_counter))
                break

            real_data = batch
            noise = np.random.normal(size=noise_shape)
            epsilon = real_data.std() * np.random.uniform(-0.5, 0.5, size=real_data.shape)
            epsilon *= np.random.uniform(size=(batch_size, 1, 1, 1))

            d_real_error, d_fake_error = discriminator_train([real_data, noise, epsilon])
            d_error = d_real_error - d_fake_error

        noise = np.random.normal(size=noise_shape)
        g_error, = generator_train([noise])

        print("[{0}/{1}][{2}/{3}] generation: {4} d_loss: {5} g_loss: {6} d_real: {7}, d_fake: {8}".format(epoch, epochs, batch_counter, batches, generation, d_error, g_error, d_real_error, d_fake_error))

        if generation % 10 == 0:
            # print("[{0}/{1}][{2}/{3}] d_loss: {4} g_loss: {5} d_real: {6}, d_fake: {7}".format(epoch, epochs, batch_counter, batches, d_error, g_error, d_real_error, d_fake_error))
            fake = generator.predict(fixed_noise)
            save_images(fake, epoch, generation)
            save_models(epoch, generator, discriminator)

        generation += 1

def train2(epochs, batch_size, input_shape, noise_size, lambda_param, train_data_directory):
    setup()
    train_generator = data_generator(batch_size, train_data_directory)
    # Outputs Keras functions that takes inputs:
    # discriminator(real_input, noise, epsilon)
    # Inputs:
    #       - real_input = same as input_shape
    #       - noise = (batch_size, noise_size)
    #       - epsilon = (None, batch_size, ) + input_shape -- a 5D tensor (really 4D)
    # Outputs:
    #       - (discriminator_real_error, discriminator_fake_error)
    #
    # generator(noise)
    # Inputs:
    #       - noise = same as for discriminator
    # Outputs:
    #       - generator_error
    networks = build_network(input_shape, noise_size, lambda_param)
    execute2(epochs, batch_size, noise_size, train_generator, networks)

def main():
    print("[ INFO ] initialized")
    train2(100000, 32, (64, 64, 3), 128, 15, "/data/shibberu/dataset-download/faces")

if __name__ == "__main__":
    main()

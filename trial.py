import tensorflow as tf
from keras.layers import BatchNormalization, LeakyReLU, Add
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense, Activation, Reshape
from keras.models import Input, Model
from keras.optimizers import Adam

import tensorflow as tf
from keras import backend as K  # needed for mixing TensorFlow and Keras commands
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
K.set_session(sess)

import os

if not os.path.isdir("./collected_data"):
    os.makedirs("./collected_data")
    print("made collected_data dir")

if not os.path.isdir("./collected_models"):
    os.makedirs("./collected_models")
    print("made collected_models dir")

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image as IM
from keras.preprocessing.image import ImageDataGenerator, array_to_img


class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding="same",
                 data_format=None,
                 strides=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r * r * filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        # Handling Dimension(None) type for undefined batch dim
        bsize = K.shape(I)[0]
        # bsize, a, b, c/(r*r), r, r
        X = K.reshape(I, [bsize, a, b, int(c / (r * r)), r, r])
        # bsize, a, b, r, r, c/(r*r)
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))
        # Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:, i, :, :, :, :]
             for i in range(a)]  # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:, i, :, :, :] for i in range(b)]  # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], int(self.r * unshifted[1]), int(self.r * unshifted[2]), int(unshifted[3] / (self.r * self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop("rank")
        config.pop("dilation_rate")
        config["filters"] /= int(self.r * self.r)
        config["r"] = self.r
        return config


class AnimeGeneratorFactory():
    def build(self, input_shape):
        """
            Returns a generator Model described here: https://arxiv.org/pdf/1708.05509.pdf

            Args:
                input_same: A 3 length tuple describing (width, height, channel)

            Output:
                Keras Model
        """
        MOMENTUM = 0.9
        DIM = 16
        DEPTH = 64
        NUM_RESIDUAL = 16
        NUM_SUBPIXEL = 2
        FINAL_FILTERS = 3
        #FINAL_FILTERS = 1
        INITIAL_FILTERS = 64

        def residual_block(layer, filters, momentum):
            """
                Residual Block consisting of
                    Conv2D -> Batch Normalization -> relu -> Conv2D -> Batch Normalization -> Residual Addition

                Args:
                    layer:   Keras Layer
                    filters: output size as an integer
                    momentum: variable for batch normalization

                Returns:
                    Keras layer
            """
            shortcut = layer
            layer = Conv2DTranspose(filters=filters, kernel_size=(
                3, 3), strides=(1, 1), padding="same")(layer)
            layer = BatchNormalization(momentum=momentum)(layer)
            layer = Activation("relu")(layer)
            layer = Conv2DTranspose(filters=filters, kernel_size=(
                3, 3), strides=(1, 1), padding="same")(layer)
            layer = BatchNormalization(momentum=momentum)(layer)

            layer = Add()([layer, shortcut])
            return layer

        def residual_layer(layer, number, filters, momentum):
            """
                Facade for residual block.

                Creates Residual layer with specified number of residual blocks

                Args:
                    layer:   Keras layer
                    number:  number of residual blocks in layer
                    filters: output size as an integer
                    momentum: variable for batch normalization

                Returns:
                    Keras layer
            """
            for _ in range(number):
                layer = residual_block(layer, filters, momentum)
            return layer

        def subpixel_block(layer, filters, momentum):
            """
                sub-pixel block consisting of
                    Conv2D -> pixel shuffler x 2 -> Batch Normalization -> Relu

                the code of subpixel layer is based on https://github.com/Tetrachrome/subpixel

                Args:
                    layer:   Keras Layer
                    filters: output size as an integer
                    momentum: variable for batch normalization

                Returns:
                    Keras layer
            """

            layer = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=(1, 1), padding="same")(layer)
            layer = Subpixel(filters, (3, 3), 2)(layer)

            layer = BatchNormalization(momentum=momentum)(layer)
            layer = Activation("relu")(layer)
            return layer

        def subpixel_layer(layer, number, filters, momentum):
            """
                Facade for subpixel block.

                Creates subpixel layer with specified number of subpixel blocks

                Args:
                    layer:   Keras layer
                    number:  number of subpixel blocks in layer
                    filters: output size as an integer
                    momentum: variable for batch normalization

                Returns:
                    Keras layer
            """
            for _ in range(number):
                layer = subpixel_block(layer, filters, momentum)
            return layer

        inputs = Input(shape=input_shape)
        filters = INITIAL_FILTERS
        layer = Dense(DEPTH * DIM * DIM)(inputs)

        layer = BatchNormalization(momentum=MOMENTUM)(layer)
        layer = Activation("relu")(layer)
        layer = Reshape((DIM, DIM, DEPTH))(layer)
        old = layer

        # 16 residual layers
        layer = residual_layer(layer, NUM_RESIDUAL, filters, MOMENTUM)

        layer = BatchNormalization(momentum=MOMENTUM)(layer)
        layer = Activation("relu")(layer)
        layer = Add()([layer, old])

        filters *= 4
        # 3 sub-pixel layers
        layer = subpixel_layer(layer, NUM_SUBPIXEL, filters, MOMENTUM)

        layer = Conv2D(filters=FINAL_FILTERS, kernel_size=(
            9, 9), strides=(1, 1), padding="same")(layer)
        layer = Activation("tanh")(layer)

        model = Model(inputs=inputs, outputs=layer)
        optimizer = Adam(lr=0.00015, beta_1=0.5)
        model.compile(loss="binary_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model


class AnimeDiscriminatorFactory(object):
    """
        Discriminator Factory Class that creates the model described here:
        https://arxiv.org/pdf/1708.05509.pdf
    """

    def build(self, input_shape):
        """
            Returns a Model described here: https://arxiv.org/pdf/1708.05509.pdf
            Args:
                input_same: A 3 length tuple describing (width, height, channel)
            Output:
                Keras Model
        """
        RESIDUAL_BLOCKS_PER_LAYER = 2
        LEAKY_RELU_ALPHA = 0.2
        MODULES = 5

        def intermediate_layer(layer, filters, kernel_size):
            """
                Create the intermediate layers between residual layers.
                Args:
                    layer:       Keras layer
                    filters:     output size as an integer
                    kernel_size: length 2 tuple
                Returns:
                    Keras layer
            """
            layer = Conv2D(filters=filters, kernel_size=kernel_size,
                           strides=(2, 2), padding="same")(layer)
            layer = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(layer)
            return layer

        def initial_layer(input_layer):
            """
                Facade for intermediate_layer for the first layer of the network.
                Args:
                    input_layer: Keras Input Layer
                Returns:
                    Keras layer
            """
            INITIAL_LAYER_FILTER = 32
            INITIAL_KERNEL_SIZE = (4, 4)
            return intermediate_layer(input_layer, INITIAL_LAYER_FILTER, INITIAL_KERNEL_SIZE)

        def residual_block(layer, filters):
            """
                Residual Block consisting of
                    Conv2D -> LeakyReLU -> Conv2D -> LeakyReLU -> Residual Addition
                Args:
                    layer:   Keras Layer
                    filters: output size as an integer
                Returns:
                    Keras layer
            """
            shortcut = layer
            layer = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=(1, 1), padding="same")(layer)
            layer = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(layer)
            layer = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=(1, 1), padding="same")(layer)

            layer = Add()([layer, shortcut])
            layer = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(layer)
            return layer

        def residual_layer(layer, number, filters):
            """
                Facade for residual block.
                Creates Residual layer with specified number of residual blocks
                Args:
                    layer:   Keras layer
                    number:  number of residual blocks in layer
                    filters: output size as an integer
                Returns:
                    Keras layer
            """
            for _ in range(number):
                layer = residual_block(layer, filters)
            return layer

        # NOTE: notation kxnysz
        # - k specifies that the convolution layer has kernel_size x
        # - n specifies that the convolution layer has y feature maps
        # - s specifies that the convolution layer has stride z

        inputs = Input(shape=input_shape)

        filters = 32
        # initial layer k4n32s2
        layer = initial_layer(inputs)
        for i in range(MODULES):
            layer = residual_layer(layer, RESIDUAL_BLOCKS_PER_LAYER, filters)
            filters *= 2

            intermediate_kernel_size = (3, 3)
            if i < 2:
                intermediate_kernel_size = (4, 4)
            layer = intermediate_layer(
                layer, filters, intermediate_kernel_size)

        outputs = Dense(1, activation="sigmoid")(layer)

        reshaped_output = Reshape((1,))(outputs)

        model = Model(inputs=inputs, outputs=reshaped_output)

        optimizer = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])

        return model


def data_generator(batch_size, train_data_directory):
    train_datagen = ImageDataGenerator(shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(train_data_directory,
                                                        target_size=(64, 64),
                                                        batch_size=batch_size,
                                                        class_mode="binary")
    return train_generator


def build_network(input_shape, noise_shape):
    discriminator = AnimeDiscriminatorFactory().build(input_shape)
    generator = AnimeGeneratorFactory().build(noise_shape)

    gan_inputs = Input(noise_shape)
    generator_outputs = generator(gan_inputs)
    gan_outputs = discriminator(generator_outputs)
    gan = Model(inputs=gan_inputs, outputs=gan_outputs)
    optimizer = Adam(lr=0.00015, beta_1=0.5)
    gan.compile(loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])
    return (discriminator, generator, gan)

# Save the generator and discriminator networks (and weights) for later use


def save_models(epoch, generator, discriminator, gan):
    print("saving model...", end="")
    generator.save(
        "./collected_models/gan_generator_epoch_{0}.h5".format(epoch))
    discriminator.save(
        "./collected_models/gan_discriminator_epoch_{0}.h5".format(epoch))
    gan.save("./collected_models/gan_core_epoch_{0}".format(epoch))
    print("done")


def execute(epochs, batch_size, input_shape, noise_shape, train_generator, discriminator, generator, gan):
    e = 1
    d_losses_real = []
    d_losses_fake = []
    g_losses = []
    stagnation_counter = 0
    for batch, label in train_generator:
        dloss = 0
        gloss = 0

        # print("-" * 15, "Epoch %d" % e, "-" * 15)

        # Get a random set of input noise and images
        real_data = batch
        noise = np.random.normal(0, 1, size=(batch_size, ) + noise_shape)
        fake_data = generator.predict(noise)

        # discriminator_images = np.concatenate([real_data, fake_data])

        # Labels for generated and real data
        real_label = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        fake_label = np.random.normal(batch_size) * 0.2
        # discriminator_labels = np.concatenate((real_label, fake_label))

        # Train discriminator
        # print("discriminator start...", end="")
        discriminator.trainable = True
        generator.trainable = False
        d_loss_real, d_acc_real = discriminator.train_on_batch(
            real_data, real_label)
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(
            fake_data, fake_label)
        # d_loss, d_acc = discriminator.train_on_batch(discriminator_images,
        #                                              discriminator_labels)
        # print("done")

        # Train generator
        # print("generator start...", end="")
        discriminator.trainable = False
        generator.trainable = True
        gan_noise = np.random.normal(0, 1, size=(batch_size, ) + noise_shape)
        gan_label = real_label
        g_loss, g_acc = gan.train_on_batch(gan_noise, gan_label)
        # print("done")

        # Store loss of most recent batch from this epoch
        d_losses_real.append(d_loss_real)
        d_losses_fake.append(d_loss_fake)
        g_losses.append(g_loss)

        # According to this Github (https://github.com/forcecore/Keras-GAN-Animeface-Character),
        # the network will simply fail if any of these start at 15.
        if d_loss_real >= 15 or d_loss_fake >= 15 or g_loss >= 15:
            stagnation_counter += 1

            if stagnation_counter >= 10 and e < 100:
                return False

        # Log some data. Accuracy will probably be 0. We want the loss to decrease though.
        # print("d real: ", (d_loss_real, d_acc_real))
        # print("d loss: ", (d_loss_fake, d_acc_real))
        # print("g loss: ", (g_loss, g_acc))

        if e % 100 == 0:
            # print("saving generated image...", end="")
            for i in range(fake_data.shape[0]):
                g_image = array_to_img(fake_data[i])
                g_image.save(
                    "./collected_data/gan_generated_image_epoch_{0}_{1}.png".format(e, i))
            # print("done")

        if e == 1 or e % 20 == 0:
            # print("saving model...", end="")
            save_models(e, generator, discriminator, gan)
            print("d real: ", (d_loss_real, d_acc_real))
            print("d loss: ", (d_loss_fake, d_acc_real))
            print("g loss: ", (g_loss, g_acc))
            # print("done")

        if e >= epochs:
            break
        e += 1

    return True


def train(epochs, batch_size, input_shape, noise_shape, train_data_directory):
    train_generator = data_generator(batch_size, train_data_directory)
    discriminator, generator, gan = build_network(input_shape, noise_shape)

    run = True
    while run:
        try:
            status = execute(epochs, batch_size, input_shape, noise_shape, train_generator,
                              discriminator, generator, gan)
            if status:
                print("[INFO] Model Completed")
                run = False
            else:
                print("[INFO] stagnant GAN")
                run = True
                train_generator = data_generator(batch_size, train_data_directory)
                discriminator, generator, gan = build_network(input_shape, noise_shape)
        except Exception as e:
            print(e)
            run = True
            train_generator = data_generator(batch_size, train_data_directory)
            discriminator, generator, gan = build_network(input_shape, noise_shape)
            pass


train(100000, 32, (64, 64, 3), (1, 1, 128),
      "/data/shibberu/dataset-download/faces")

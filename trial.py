import os
from keras.layers import BatchNormalization, LeakyReLU, Add, Flatten, Conv2D, Conv2DTranspose, Dense, Activation, Reshape
from keras.models import Input, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.initializers import RandomNormal
from keras.losses import binary_crossentropy

import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#### --- for hinton
import tensorflow as tf
from keras import backend as K  # needed for mixing TensorFlow and Keras commands
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
K.set_session(sess)
####

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
        GAMMA_INITIALIZER = RandomNormal(1., 0.02)

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
            layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
            layer = BatchNormalization(momentum=momentum, gamma_initializer=GAMMA_INITIALIZER)(layer, training=1)
            layer = Activation("relu")(layer)
            layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
            layer = BatchNormalization(momentum=momentum, gamma_initializer=GAMMA_INITIALIZER)(layer, training=1)

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
            # r = upscale factor
            layer = Subpixel(filters=filters, kernel_size=(3, 3), r=2, padding="same")(layer)

            layer = BatchNormalization(momentum=momentum, gamma_initializer=GAMMA_INITIALIZER)(layer, training=1)
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

        MOMENTUM = 0.9
        DIM = 16
        DEPTH = 64
        NUM_RESIDUAL = 16
        NUM_SUBPIXEL = 2
        FINAL_FILTERS = 3
        INITIAL_FILTERS = 64

        inputs = Input(shape=input_shape)
        filters = INITIAL_FILTERS # 64

        layer = Dense(DEPTH * DIM * DIM)(inputs)
        layer = BatchNormalization(momentum=MOMENTUM, gamma_initializer=GAMMA_INITIALIZER)(layer, training=1)
        layer = Activation("relu")(layer)
        layer = Reshape((DIM, DIM, DEPTH))(layer)

        old = layer

        # 16 residual layers
        layer = residual_layer(layer, NUM_RESIDUAL, filters, MOMENTUM)

        layer = BatchNormalization(momentum=MOMENTUM, gamma_initializer=GAMMA_INITIALIZER)(layer, training=1)
        layer = Activation("relu")(layer)
        layer = Add()([layer, old])

        filters *= 4
        # 3 sub-pixel layers
        layer = subpixel_layer(layer, NUM_SUBPIXEL, filters, MOMENTUM)

        layer = Conv2D(filters=FINAL_FILTERS, kernel_size=(
            9, 9), strides=(1, 1), padding="same")(layer)
        layer = Activation("tanh")(layer)

        model = Model(inputs=inputs, outputs=layer)
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

        KERNEL_INITIALIZER = RandomNormal(0, 0.02)

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
                           kernel_initializer=KERNEL_INITIALIZER, strides=(2, 2),
                           padding="same")(layer)
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
            layer = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer=KERNEL_INITIALIZER,
                           strides=(1, 1), padding="same")(layer)
            layer = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(layer)
            layer = Conv2D(filters=filters, kernel_size=(3, 3), kernel_initializer=KERNEL_INITIALIZER,
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
            layer = intermediate_layer(layer, filters, intermediate_kernel_size)

        layer = Dense(1, activation="sigmoid")(layer)
        outputs = Reshape((1, ))(layer)

        model = Model(inputs=inputs, outputs=outputs)
        return model

def setup():
    if not os.path.isdir("./images"):
        os.makedirs("./images")
        print("[ INFO ] images directory created")
    if not os.path.isdir("./saved_models"):
        os.makedirs("./saved_models")
        print("[ INFO ] saved_models directory created")
    if not os.path.isdir("./saved_losses"):
        os.makedirs("./saved_losses")
        print("[ INFO ] saved_losses directory created")

def data_generator(batch_size, train_data_directory):
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(train_data_directory,
                                                        target_size=(64, 64),
                                                        batch_size=batch_size,
                                                        class_mode="binary")
    return train_generator


def build_network(input_shape, noise_size, lambda_param):
    print("[ INFO ] building network...", end="")
    discriminator = AnimeDiscriminatorFactory().build(input_shape)
    generator = AnimeGeneratorFactory().build((noise_size, ))

    # Compute Wasserstein Loss and Gradient Penalty
    d_real_input = Input(shape=input_shape)
    noise = Input(shape=(noise_size, ))
    d_fake_input = generator(noise)

    epsilon_input = K.placeholder(shape=(None, ) + input_shape)
    d_mixed_input = Input(shape=input_shape, tensor=(d_real_input + epsilon_input))

    d_real = discriminator(d_real_input)
    d_fake = discriminator(d_fake_input)
    d_loss_real = K.mean(binary_crossentropy(K.ones_like(d_real), d_real))
    d_loss_fake = K.mean(binary_crossentropy(K.zeros_like(d_fake), d_fake))
    g_loss = K.mean(binary_crossentropy(K.ones_like(d_fake), d_fake))

    gradient_mixed = K.gradients(discriminator(d_mixed_input), [d_mixed_input])[0]
    normalized_gradient_mixed = K.sqrt(K.sum(K.square(gradient_mixed), axis=1))
    gradient_penalty = K.mean(K.square(normalized_gradient_mixed - 1))

    d_loss = d_loss_real + d_loss_fake + (lambda_param * gradient_penalty)

    # Discriminator
    training_updates = Adam(lr=0.000001, beta_1=0.5, beta_2=0.9).get_updates(discriminator.trainable_weights, [], d_loss)
    discriminator_train = K.function([d_real_input, noise, epsilon_input],
                                     [d_loss, d_loss_real, d_loss_fake],
                                     training_updates)
    # Generator
    training_updates = Adam(lr=0.000001, beta_1=0.5, beta_2=0.9).get_updates(generator.trainable_weights, [], g_loss)
    generator_train = K.function([noise], [g_loss], training_updates)

    print("done")
    return (discriminator, generator, discriminator_train, generator_train)

def load_images():
    print("[ INFO ] loading images...", end="")
    images = np.load("./normalized_images.npy")
    print("done")
    return images

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

def save_errors(d_real_errors, d_fake_errors, g_errors):
    np.save("./saved_losses/d_real_errors.npy", np.array(d_real_errors))
    np.save("./saved_losses/d_fake_errors.npy", np.array(d_fake_errors))
    np.save("./saved_losses/g_errors.npy", np.array(g_errors))

def execute(epochs, batch_size, noise_size, images, networks):
    discriminator, generator, discriminator_train, generator_train = networks
    generation = 0
    noise_shape = (batch_size, noise_size)
    fixed_noise = np.random.normal(size=noise_shape).astype("float32")
    batches = images.shape[0] // batch_size

    g_error = 0
    d_real_errors = []
    d_fake_errors = []
    g_errors = []
    for epoch in range(epochs):
        np.random.shuffle(images)
        batch_counter = 0

        # Execute all mini-batches.
        while batch_counter < batches:
            # Determine the number of Discriminator Training between Generator Trains.
            d_iteration = 5
            # Pretrain discriminator for the first few generations heavily.
            if generation < 20 or generation % 500 == 0:
                d_iteration = 100

            # Train the discriminator 5 or 100 times or finish the batch (whichever comes first).
            while d_iteration > 0 and batch_counter < batches:
                # Grab batch (real_data) from images.
                real_data = images[batch_counter * batch_size : (batch_counter + 1) * batch_size]

                # Get noises (noise = generator noise, epsilon = perturb image noise).
                noise = np.random.normal(size=noise_shape)
                epsilon = real_data.std() * np.random.uniform(-0.5, 0.5, size=real_data.shape)
                epsilon *= np.random.uniform(size=(batch_size, 1, 1, 1))

                # Compute errors.
                d_error, d_real_error, d_fake_error = discriminator_train([real_data, noise, epsilon])

                # Store errors.
                d_real_errors.append(d_real_error)
                d_fake_errors.append(d_fake_error)

                # Increment/Decrement counter values.
                batch_counter += 1 # mini_batch completed
                d_iteration -= 1   # discriminator cycle decrement

            noise = np.random.normal(size=noise_shape)
            g_error, = generator_train([noise])
            g_errors.append(g_error)

            # Log every 10 generations.
            if generation % 10 == 0:
                print("[{0}/{1}][{2}/{3}] generation: {4} d_loss: {5} g_loss: {6} d_real: {7}, d_fake: {8}".format(epoch, epochs, batch_counter, batches, generation, d_error, g_error, d_real_error, d_fake_error))

            # Save every 500 generations.
            if generation % 500 == 0:
                fake = generator.predict(fixed_noise)
                save_images(fake, epoch, generation)
                save_models(epoch, generator, discriminator)
                save_errors(d_real_errors, d_fake_errors, g_errors)

            generation += 1

def train(epochs, batch_size, input_shape, noise_size, lambda_param):
    setup()
    # train_generator = data_generator(batch_size, train_data_directory)
    images = load_images()
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
    print("[ INFO ] training started")
    execute(epochs, batch_size, noise_size, images, networks)


def main():
    print("[ INFO ] initialized")
    train(100000, 32, (64, 64, 3), 128, 20)

if __name__ == "__main__":
    main()

import os
from keras.layers import BatchNormalization, LeakyReLU, Add, Flatten, Conv2D, Conv2DTranspose, Dense, Activation, Reshape
from keras.models import Input, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

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
            layer = BatchNormalization(momentum=momentum)(layer, training=1)
            layer = Activation("relu")(layer)
            layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
            layer = BatchNormalization(momentum=momentum)(layer, training=1)

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

            layer = BatchNormalization(momentum=momentum)(layer, training=1)
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
        layer = BatchNormalization(momentum=MOMENTUM)(layer, training=1)
        layer = Activation("relu")(layer)
        layer = Reshape((DIM, DIM, DEPTH))(layer)

        old = layer

        # 16 residual layers
        layer = residual_layer(layer, NUM_RESIDUAL, filters, MOMENTUM)

        layer = BatchNormalization(momentum=MOMENTUM)(layer, training=1)
        layer = Activation("relu")(layer)
        layer = Add()([layer, old])

        filters *= 4
        # 3 sub-pixel layers
        layer = subpixel_layer(layer, NUM_SUBPIXEL, filters, MOMENTUM)

        layer = Conv2D(filters=FINAL_FILTERS, kernel_size=(
            9, 9), strides=(1, 1), padding="same")(layer)
        layer = Activation("tanh")(layer)

        model = Model(inputs=inputs, outputs=layer)
        # optimizer = Adam(lr=0.0002, beta_1=0.5)
        # model.compile(loss="binary_crossentropy",
        #               optimizer=optimizer,
        #               metrics=["accuracy"])
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
    discriminator = AnimeDiscriminatorFactory().build(input_shape)
    generator = AnimeGeneratorFactory().build((noise_size, ))

    # Compute Wasserstein Loss and Gradient Penalty
    d_real_input = Input(shape=input_shape)
    noise = Input(shape=(noise_size, )) # Ex: shape = (128, )
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
    training_updates = Adam(lr=0.000001).get_updates(discriminator.trainable_weights, [], loss)
    discriminator_train = K.function([d_real_input, noise, epsilon_input],
                                     [loss_real, loss_fake],
                                     training_updates)
    # Generator
    loss = -loss_fake
    training_updates = Adam(lr=0.000001).get_updates(generator.trainable_weights, [], loss)
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
    noise_shape = (batch_size, noise_size)
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
    train2(100000, 32, (64, 64, 3), 128, 10, "/data/shibberu/dataset-download/faces")

if __name__ == "__main__":
    main()

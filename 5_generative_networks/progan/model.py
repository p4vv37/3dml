# add a discriminator block
from keras import Sequential, Model, Input
from keras.constraints import max_norm
from keras.initializers import RandomNormal
from keras.layers import LeakyReLU, Reshape, Dense, AveragePooling2D, UpSampling2D, Flatten
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from tensorflow import device

from .config import PARAMETERS
from .layers import PixelNormalization, WeightedSum, MinibatchStdev, EqualizedConv2D
from .misc import wasserstein_loss

def add_discriminator_block(old_model, n_input_layers=3, filters=128, gpus=1):
    # weight initialization
    init = RandomNormal(stddev=1.0)
    # weight constraint
    const = max_norm(1)
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-2].value * 2, in_shape[-2].value * 2, in_shape[-1].value)
    in_image = Input(shape=input_shape)
    # define new input processing layer
    d = EqualizedConv2D(filters, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # define new block
    d = EqualizedConv2D(filters, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = EqualizedConv2D(filters, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D()(d)
    block_new = d
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model

    if gpus <= 1:
        print("[INFO] training with 1 GPU...")
        model1 = Model(in_image, d)
    else:
        with device("/cpu:0"):
            model = Model(in_image, d)
        model1 = multi_gpu_model(model, gpus=gpus)
    # compile model
    model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=PARAMETERS.learning_rate,
                                                         beta_1=PARAMETERS.adam_beta1,
                                                         beta_2=PARAMETERS.adam_beta2,
                                                         epsilon=PARAMETERS.adam_epsilon))
    # downsample the new larger image
    downsample = AveragePooling2D()(in_image)
    # connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fade in output of old model input layer with new input
    d = WeightedSum()([block_old, block_new])
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    if gpus <= 1:
        print("[INFO] training with 1 GPU...")
        model2 = Model(in_image, d)
    else:
        with device("/cpu:0"):
            model = Model(in_image, d)
        model2 = multi_gpu_model(model, gpus=gpus)
    # compile model
    model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=PARAMETERS.learning_rate,
                                                         beta_1=PARAMETERS.adam_beta1,
                                                         beta_2=PARAMETERS.adam_beta2,
                                                         epsilon=PARAMETERS.adam_epsilon))
    return [model1, model2]


# define the discriminator models for each image resolution
def define_discriminator(n_blocks, input_shape=(4, 4, 3), gpus=1):
    # weight initialization
    init = RandomNormal(stddev=1.0)
    # weight constraint
    const = max_norm(1)
    model_list = list()
    # base model input
    in_image = Input(shape=input_shape)
    # conv 1x1
    d = EqualizedConv2D(128, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = MinibatchStdev()(d)
    d = EqualizedConv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = EqualizedConv2D(128, (4, 4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = Flatten()(d)
    out_class = Dense(1)(d)
    # define model
    if gpus <= 1:
        print("[INFO] training with 1 GPU...")
        model = Model(in_image, out_class)
    else:
        with device("/cpu:0"):
            model = Model(in_image, out_class)
        model = multi_gpu_model(model, gpus=gpus)
    # compile model
    model.compile(loss=wasserstein_loss, optimizer=Adam(lr=PARAMETERS.learning_rate,
                                                        beta_1=PARAMETERS.adam_beta1,
                                                        beta_2=PARAMETERS.adam_beta2,
                                                        epsilon=PARAMETERS.adam_epsilon))
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        filters = 2**(10-i)
        if filters > 512:
            filters = 512
        if filters < 16:
            filters = 16
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_discriminator_block(old_model, gpus=gpus)
        # store model
        model_list.append(models)
    return model_list


# add a generator block
def add_generator_block(old_model, filters=256, gpus=1):
    # weight initialization
    init = RandomNormal(stddev=1.0)
    # weight constraint
    const = max_norm(1)
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = UpSampling2D()(block_end)
    g = EqualizedConv2D(filters, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = EqualizedConv2D(filters, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # add new output layer
    out_image = EqualizedConv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    # define model
    if gpus <= 1:
        print("[INFO] training with 1 GPU...")
        model1 = Model(old_model.input, out_image)
    else:
        with device("/cpu:0"):
            model = Model(old_model.input, out_image)
        model1 = multi_gpu_model(model, gpus=gpus)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    # define model
    if gpus <= 1:
        print("[INFO] training with 1 GPU...")
        model2 = Model(old_model.input, merged)
    else:
        with device("/cpu:0"):
            model = Model(old_model.input, merged)
        model2 = multi_gpu_model(model, gpus=gpus)
    return [model1, model2]


# define generator models
def define_generator(latent_dim, n_blocks, in_dim=4, gpus=1):
    # weight initialization
    init = RandomNormal(stddev=1.0)
    # weight constraint
    const = max_norm(1)
    model_list = list()
    # base model latent input
    in_latent = Input(shape=(latent_dim,))
    # linear scale up to activation maps
    g = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = Reshape((in_dim, in_dim, 128))(g)
    # conv 4x4, input block
    # noinspection DuplicatedCode
    g = EqualizedConv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # conv 3x3
    g = EqualizedConv2D(128, (3, 3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    # conv 1x1, output block
    out_image = EqualizedConv2D(3, (1, 1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    # define model
    if gpus <= 1:
        print("[INFO] training with 1 GPU...")
        model = Model(in_latent, out_image)
    else:
        with device("/cpu:0"):
            model = Model(in_latent, out_image)
        model = multi_gpu_model(model, gpus=gpus)
    # store model
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        filters = 2**(10-i)
        if filters > 512:
            filters = 512
        if filters < 16:
            filters = 16
        # get prior model without the fade-on
        old_model = model_list[i - 1][0]
        # create new model for next resolution
        models = add_generator_block(old_model, gpus=gpus)
        # store model
        model_list.append(models)
    return model_list


# define composite models for training generators via discriminators
def define_composite(discriminators, generators, gpus=1):
    model_list = list()
    # create composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]
        # straight-through model
        d_models[0].trainable = False

        if gpus <= 1:
            print("[INFO] training with 1 GPU...")
            model1 = Sequential()
        else:
            with device("/cpu:0"):
                model = Sequential()
            model1 = multi_gpu_model(model, gpus=gpus)
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=PARAMETERS.learning_rate,
                                                             beta_1=PARAMETERS.adam_beta1,
                                                             beta_2=PARAMETERS.adam_beta2,
                                                             epsilon=PARAMETERS.adam_epsilon))
        # fade-in model
        d_models[1].trainable = False

        if gpus <= 1:
            print("[INFO] training with 1 GPU...")
            model2 = Sequential()
        else:
            with device("/cpu:0"):
                model = Sequential()
            model2 = multi_gpu_model(model, gpus=gpus)
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=PARAMETERS.learning_rate,
                                                             beta_1=PARAMETERS.adam_beta1,
                                                             beta_2=PARAMETERS.adam_beta2,
                                                             epsilon=PARAMETERS.adam_epsilon))
        # store
        model_list.append([model1, model2])
    return model_list

import matplotlib.pyplot as plt
import numpy as np
from keras import backend
from matplotlib import pyplot

from .layers import WeightedSum


def deprocess(x):
    x = np.clip(x, -1, 1)
    return (x + 1) / 2.0


def load_models():
    pass


# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25, save_models=True):
    # devise name
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    # X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(np.sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis('off')
        img = deprocess(X[i])
        img = np.clip(img, 0, 1)
        pyplot.imshow(img)
    # save plot to file
    filename1 = 'plot_%s.png' % (name)
    pyplot.savefig(filename1)
    pyplot.close()
    if save_models:
        # save the generator model
        filename2 = 'model_%s.h5' % (name)
        g_model.save(F"models/{filename2}")
        print('>Saved: %s and %s' % (filename1, filename2))


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = -np.ones((n_samples, 1))
    return X, y


# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


def show_images(generated_images, suffix=""):
    n_images = len(generated_images)
    rows = 4
    cols = n_images // rows

    plt.figure(figsize=(cols, rows))
    for i in range(n_images):
        img = deprocess(generated_images[i])
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(F'prev_{suffix}.png')
    plt.show()

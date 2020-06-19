import numpy as np
import time

from .misc import generate_fake_samples, generate_latent_points, update_fadein, \
    summarize_performance, show_images
from .dataset import load_dataset


def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, latent_dim, fadein=False):
    # calculate the number of batches per training epoch
    bat_per_epo = len(dataset)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    t0 =time.time()
    for i in range(n_steps):
        # update alpha for all WeightedSum layers when fading in new blocks
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)
        # prepare real and fake samples
        X_real, y_real = dataset.next()
        if len(y_real) != half_batch:
            dataset.on_epoch_end()
            dataset.reset()
            X_real, y_real = dataset.next()

        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update the generator via the discriminator's error
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        # summarize loss on this batch
        print(F'\r>{i + 1}/{n_steps}, d1=%.3f, d2=%.3f g=%.3f' % (d_loss1, d_loss2, g_loss), end="")
        if time.time() - t0 > 30:
            # save preview of images every 30 s.
            t0 = time.time()
            summarize_performance('fresh_batch_preview', g_model, latent_dim, save_models=False)
    print("")


# rain the generator and discriminator
def train(g_models, d_models, gan_models, dataset_dir, latent_dim, e_norm, e_fadein, n_batch):
    # fit the baseline model
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
    # scale dataset to appropriate size
    gen_shape = g_normal.output_shape
    dataset, _ = load_dataset(dataset_dir, gen_shape[1], n_batch[0]//2)
    print('Scaled Data', gen_shape)
    # train normal or straight-through models
    train_epochs(g_normal, d_normal, gan_normal, dataset, e_norm[0], n_batch[0], latent_dim)
    summarize_performance('tuned', g_normal, latent_dim)

    latent_for_prev = generate_latent_points(latent_dim, 4)
    show_images(g_normal.predict(latent_for_prev), suffix=0)

    # process each level of growth
    for i in range(1, len(g_models)):
        # retrieve models for this level of growth
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        dataset, _ = load_dataset(dataset_dir, gen_shape[1], n_batch[i]//2)
        print('Scaled Data', gen_shape)
        # train fade-in models for next level of growth
        train_epochs(g_fadein, d_fadein, gan_fadein, dataset, e_fadein[i], n_batch[i], latent_dim, True)
        summarize_performance('faded', g_fadein, latent_dim)
        # train normal or straight-through models
        train_epochs(g_normal, d_normal, gan_normal, dataset, e_norm[i], n_batch[i], latent_dim)
        summarize_performance('tuned', g_normal, latent_dim)
        show_images(g_normal.predict(latent_for_prev), suffix=i)

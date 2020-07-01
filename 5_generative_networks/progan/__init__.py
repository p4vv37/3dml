import os

import tensorflow as tf

from .config import PARAMETERS
from .model import define_discriminator, define_generator, define_composite
from .train import train


def main(runs=1, gpus=1, dataset_name=None, n_epochs=None, n_batch=None, learning_rate=None, load_models=False,
         n_blocks=7):
    print(F"-------------\n\n\nGPU:{tf.test.is_gpu_available(cuda_only=True)}\n\n\n-------------")
    print(F"Num GPUs: {gpus}")

    # size of the latent space
    latent_dim = 128
    if not load_models:
        # define models
        d_models = define_discriminator(n_blocks, gpus=gpus)
        # define models
        g_models = define_generator(latent_dim, n_blocks, gpus=gpus)
        # define composite models
        gan_models = define_composite(d_models, g_models, gpus=gpus)
        # load image data
    else:
        d_models, g_models, gan_models = load_models()

    if dataset_name is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "all_age_faces_dataset")
    elif dataset_name == "af":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "AF_dataset", "data")
    elif dataset_name == "faces":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "all_age_faces_dataset")
    elif dataset_name == "flowers":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "flowers")
    elif dataset_name == "pokemon":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pokemon")
    else:
        raise ValueError(F"Unknown dataset_name {dataset_name}")

    if n_epochs is not None:
        PARAMETERS.n_epochs = n_epochs
    if n_batch is not None:
        PARAMETERS.n_batch = n_batch
    if learning_rate is not None:
        PARAMETERS.learning_rate = learning_rate
    for r in range(runs):
        train(g_models, d_models, gan_models, data_dir, latent_dim,
              PARAMETERS.n_epochs, PARAMETERS.n_epochs, PARAMETERS.n_batch)
    return g_models, d_models, gan_models


if __name__ == "__main__":
    main(4, 1, dataset_name="pokemon", n_epochs=[0, 0, 0, 0, 0, 1, 128], n_batch=[16, 16, 8, 8, 4, 4, 4],
         learning_rate=0.0005)

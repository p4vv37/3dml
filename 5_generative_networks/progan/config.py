class PARAMETERS(object):
    n_batch = [16, 16, 16, 8, 4, 4, 4]
    # n_epochs = [8, 8, 64, 64, 128, 128, 256]
    # n_epochs = [8, 8, 8, 8, 16, 32, 32]
    n_epochs = [256, 512, 1024, 1024, 1024, 1024, 2048]
    # n_epochs = [1, 1, 1, 1, 1, 1, 1]
    learning_rate = 0.001
    adam_beta1 = 0.0
    adam_beta2 = 0.99
    adam_epsilon = 1e-8
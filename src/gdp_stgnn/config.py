"""Training hyper-parameters shared by the neural models.

The number of epochs is overridden by the training scripts (300 epochs, best-validation checkpoint).
"""

temporalonly_hyperparams = {
    "learning_rate": 0.001,
    "n_epochs": 1000,
    "save_loss_interval": 1,
    "print_interval": 50,
    "batch_size": 32,
    "patience": 50,
}

spatial_only_hyperparams = {
    "batch_size": 4,
    "save_loss_interval": 1,
    "print_interval": 50,
    "n_epochs": 1000,
    "learning_rate": 0.001,
}

stmodels_hyperparams = {
    "batch_size": 4,
    "save_loss_interval": 1,
    "print_interval": 50,
    "n_epochs": 1000,
    "learning_rate": 0.001,
}

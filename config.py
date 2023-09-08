class Config:

    epochs = 100
    batch_size = 32

    input_dim = 29 * 29

    n_l1 = 1000
    n_l2 = 1000

    z_dim = 10
    n_labels = 2

    keep_prob = 0.15

    beta1 = 0.5
    beta2 = 0.9
    beta1_sup = 0.9

    supervised_lr = 0.0001
    reconstruction_lr = 0.0001
    regularization_lr = 0.0001

    data_root = '../dataset/CHD/id_image_29/'
    save_path = '.'

    labeled_percentage = 0.1

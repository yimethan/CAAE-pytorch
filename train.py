import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader

from config import Config
from caae import *
from load_dataset import *

writer = SummaryWriter()


def gradient_penalty(real_samples, g_samples, discriminator):
    """
    calculates the gradient penalty term used in the Wasserstein GAN with Gradient Penalty (WGAN-GP) loss
    :param real_samples: Real data samples from the dataset
    :param g_samples: Generated samples produced by the generator
    :param discriminator: The discriminator network
    :return: The calculated gradient penalty is returned as the output of the function
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_samples.size(1))  # Make alpha the same size as real_samples
    print(real_samples.shape, g_samples.shape)
    interpolates = alpha * real_samples + ((1 - alpha) * g_samples)
    # a linear combination of real_samples and g_samples using the alpha factor
    # represents the points between real and generated data in the input space

    interpolates.requires_grad_(True)  # Enable gradient computation
    d_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(d_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
    g_penalty = torch.mean((slopes - 1.) ** 2)

    return g_penalty


encoder = Encoder()
decoder = Decoder()
discriminator_g = Discriminator('g')
discriminator_c = Discriminator('c')

autoencoder_optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()),
                             lr=Config.reconstruction_lr, betas=(Config.beta1, Config.beta2))
discriminator_g_optimizer = Adam(discriminator_g.parameters(), lr=Config.regularization_lr,
                                 betas=(Config.beta1, Config.beta2))
discriminator_c_optimizer = Adam(discriminator_c.parameters(), lr=Config.regularization_lr,
                                 betas=(Config.beta1, Config.beta2))
generator_optimizer = Adam(encoder.parameters(), lr=Config.regularization_lr, betas=(Config.beta1, Config.beta2))
supervised_encoder_optimizer = Adam(encoder.parameters(), lr=Config.supervised_lr,
                                    betas=(Config.beta1_sup, Config.beta2))

reconstruction_scheduler = lr_scheduler.StepLR(autoencoder_optimizer, step_size=50, gamma=0.9)
supervised_scheduler = lr_scheduler.StepLR(supervised_encoder_optimizer, step_size=50, gamma=0.9)
disc_g_scheduler = lr_scheduler.StepLR(discriminator_g_optimizer, step_size=50, gamma=0.9)
disc_c_scheduler = lr_scheduler.StepLR(discriminator_c_optimizer, step_size=50, gamma=0.9)

dataset = GetDataset()

train_len = int(len(dataset) * 0.7)
train_labeled_size = int(train_len * Config.labeled_percentage)
train_unlabeled_size = train_len - train_labeled_size

test_len = len(dataset) - train_len

# unlabeled_batch_size = int(Config.batch_size * (train_labeled_size / train_unlabeled_size))

train_labeled_data, train_unlabeled_data, test_data = random_split(dataset,
                                                                   [train_labeled_size, train_unlabeled_size, test_len])

train_labeled_loader = DataLoader(train_labeled_data, batch_size=Config.batch_size, shuffle=False)
train_unlabeled_loader = DataLoader(train_unlabeled_data, batch_size=Config.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=Config.batch_size, shuffle=False)


def evaluate(y_true, y_pred, epoch):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (tp + fn)
    err = (fn + fp) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = 1 - fnr
    f1score = (2 * precision * recall) / (precision + recall)

    print('---Test results-------')

    print(tp, fn)
    print(fp, tn)
    print('False negative rate: ', fnr)
    print('Error rate: ', err)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 score: ', f1score)

    writer.add_scalar('test/False_negative_rate', fnr, epoch)
    writer.add_scalar('test/Error_rate', err, epoch)
    writer.add_scalar('test/Precision', precision, epoch)
    writer.add_scalar('test/Recall', recall, epoch)
    writer.add_scalar('test/F1_score', f1score, epoch)


def train():
    for epoch in range(Config.epochs):

        encoder.train()
        decoder.train()
        discriminator_g.train()
        discriminator_c.train()

        print("------------------ Epoch {}/{} ------------------".format(epoch, Config.epochs))

        for batch_idx, labeled in enumerate(train_labeled_loader):
            unlabeled = next(iter(train_unlabeled_loader))

            autoencoder_optimizer.zero_grad()
            discriminator_g_optimizer.zero_grad()
            discriminator_c_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            supervised_encoder_optimizer.zero_grad()

            x_labeled, y_labeled = labeled['input'], labeled['label']
            x_unlabeled, y_unlabeled = unlabeled['input'], unlabeled['label']

            z_real_dist = torch.randn(Config.batch_size, Config.z_dim) * 5.
            real_cat_dist = torch.randint(low=0, high=2, size=(Config.batch_size,))
            real_cat_dist = torch.eye(Config.n_labels)[real_cat_dist]  # one-hot encoded

            encoder_output_label, encoder_output_latent = encoder(x_unlabeled)
            decoder_input = torch.cat((encoder_output_label, encoder_output_latent), dim=1)
            decoder_output = decoder(decoder_input)

            autoencoder_loss = F.mse_loss(decoder_output, x_unlabeled)

            autoencoder_loss.backward()
            autoencoder_optimizer.step()

            # dis for gaussian
            d_g_real = discriminator_g(z_real_dist)
            d_g_fake = discriminator_g(encoder_output_latent)

            # print('real cat dist', real_cat_dist.shape)  # 32, 2

            d_c_real = discriminator_c(real_cat_dist)
            d_c_fake = discriminator_c(encoder_output_label)

            # wgan-gp
            real_penalty = gradient_penalty(z_real_dist, encoder_output_latent, discriminator_g)
            dc_g_loss = -torch.mean(d_g_real) + torch.mean(d_g_fake) + 10.0 * real_penalty
            fake_penalty = gradient_penalty(real_cat_dist, encoder_output_label, discriminator_c)
            dc_c_loss = -torch.mean(d_c_real) + torch.mean(d_c_fake) + 10.0 * fake_penalty

            dc_g_loss.backward()
            discriminator_g_optimizer.step()

            dc_c_loss.backward()
            discriminator_c_optimizer.step()

            # generator
            generator_loss = -torch.mean(d_g_fake) - torch.mean(d_c_fake)

            generator_loss.backward()
            generator_optimizer.step()

            # Semi-Supervised Classification Phase
            encoder_output_label_, encoder_output_latent_ = encoder(x_labeled, supervised=True)

            # Classification accuracy of encoder
            output_label = torch.argmax(encoder_output_label_, dim=1)
            correct_pred = output_label.eq(torch.argmax(y_labeled, dim=1))
            accuracy = torch.mean(correct_pred.float())

            supervised_encoder_loss = F.cross_entropy(encoder_output_label_, y_labeled)

            supervised_encoder_loss.backward()
            supervised_encoder_optimizer.step()

        writer.add_scalar('train/loss/autoencoder_loss', autoencoder_loss, epoch)
        writer.add_scalar('train/loss/dc_g_loss', dc_g_loss, epoch)
        writer.add_scalar('train/loss/dc_c_loss', dc_c_loss, epoch)
        writer.add_scalar('train/loss/generator_loss', generator_loss, epoch)
        writer.add_scalar('train/loss/supervised_encoder_loss', supervised_encoder_loss, epoch)

        writer.add_scalar('train/encoder_accuracy', accuracy, epoch)

        writer.add_image('train/decoder_output', decoder_output[0], epoch)

        reconstruction_scheduler.step()
        supervised_scheduler.step()
        disc_g_scheduler.step()
        disc_c_scheduler.step()

        test(epoch)


def test(epoch):
    y_true = []
    y_pred = []
    # total_prob = []
    # total_latent = []

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        discriminator_g.eval()
        discriminator_c.eval()

        for batch_idx, (inputs, labels) in enumerate(test_loader):

            batch_pred, batch_latent = encoder(inputs)
            # total_latent.append(batch_latent.cpu().numpy())

            batch_label = labels.argmax(dim=1).cpu().numpy()
            prob = batch_pred.max(dim=1).cpu().numpy()
            batch_pred = batch_pred.argmax(dim=1).cpu().numpy()

            y_pred.extend(batch_pred.tolist())
            y_true.extend(batch_label.tolist())
            # total_prob.extend(prob.tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # total_prob = np.array(total_prob)
        # total_latent = np.concatenate(total_latent, axis=0)

        evaluate(y_true, y_pred, epoch)


if __name__ == '__main__':
    train()
    writer.close()

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
import torchsummary

from config import Config
from caae import *
from load_dataset import *

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter()


def gradient_penalty(real_samples, fake_samples, discriminator):
    # Generate random epsilon for interpolation
    epsilon = torch.rand(real_samples.size(0), 1)
    epsilon = epsilon.expand_as(real_samples).to(device)

    # Interpolate between real and fake samples
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True).to(device)

    # Calculate discriminator scores on the interpolated samples
    d_interpolated = discriminator(interpolated).to(device)

    # Compute gradients of the scores with respect to the interpolated samples
    gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(d_interpolated.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    # Compute the gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    g_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # Penalty term

    return g_penalty


def evaluate(y_true, y_pred, epoch):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (tp + fn)
    err = (fn + fp) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = 1 - fnr
    f1score = (2 * precision * recall) / (precision + recall)

    print('---Test results-------')

    print('TP:', tp, 'FN:', fn)
    print('FP:', fp, 'TN:', tn)
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


encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator_g = Disgauss().to(device)
discriminator_c = Discateg().to(device)

autoencoder_optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()),
                             lr=Config.reconstruction_lr, betas=(Config.beta1, Config.beta2))
discriminator_g_optimizer = Adam(discriminator_g.parameters(), lr=Config.regularization_lr,
                                 betas=(Config.beta1, Config.beta2))
discriminator_c_optimizer = Adam(discriminator_c.parameters(), lr=Config.regularization_lr,
                                 betas=(Config.beta1, Config.beta2))
generator_optimizer = Adam(encoder.parameters(), lr=Config.regularization_lr, betas=(Config.beta1, Config.beta2))
supervised_encoder_optimizer = Adam(encoder.parameters(), lr=Config.supervised_lr,
                                    betas=(Config.beta1_sup, Config.beta2))

reconstruction_scheduler = lr_scheduler.StepLR(autoencoder_optimizer, step_size=Config.step_size, gamma=Config.gamma)
supervised_scheduler = lr_scheduler.StepLR(supervised_encoder_optimizer, step_size=Config.step_size, gamma=Config.gamma)
disc_g_scheduler = lr_scheduler.StepLR(discriminator_g_optimizer, step_size=Config.step_size, gamma=Config.gamma)
disc_c_scheduler = lr_scheduler.StepLR(discriminator_c_optimizer, step_size=Config.step_size, gamma=Config.gamma)
generator_scheduler = lr_scheduler.StepLR(generator_optimizer, step_size=Config.step_size, gamma=0.9)

dataset = GetDataset()

train_len = int(len(dataset) * 0.7)
train_labeled_size = int(train_len * Config.labeled_percentage)
train_unlabeled_size = train_len - train_labeled_size

test_len = len(dataset) - train_len

unlabeled_batch_size = int(Config.batch_size * (train_labeled_size / train_unlabeled_size))

train_labeled_data, train_unlabeled_data, test_data = random_split(dataset,
                                                                   [train_labeled_size, train_unlabeled_size, test_len])

# train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42, shuffle=False)
# train_labeled_data, train_unlabeled_data = train_test_split(train_data, test_size=0.9, shuffle=False)

train_labeled_loader = DataLoader(train_labeled_data, batch_size=Config.batch_size)
train_unlabeled_loader = DataLoader(train_unlabeled_data, batch_size=Config.batch_size)
test_loader = DataLoader(test_data, batch_size=Config.batch_size, shuffle=False)

torchsummary.summary(encoder, (1, 29, 29))
torchsummary.summary(decoder, (12,))
torchsummary.summary(discriminator_g, (10,))
torchsummary.summary(discriminator_c, (2,))

criterion = nn.BCEWithLogitsLoss()


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
            generator_optimizer.zero_grad()
            supervised_encoder_optimizer.zero_grad()

            x_labeled, y_labeled = labeled['input'].to(device), labeled['label'].to(device)
            x_unlabeled, y_unlabeled = unlabeled['input'].to(device), unlabeled['label'].to(device)

            z_real_dist = torch.randn(Config.batch_size, Config.z_dim) * 5.
            real_cat_dist = torch.randint(low=0, high=Config.n_labels, size=(Config.batch_size,))
            real_cat_dist = torch.eye(Config.n_labels)[real_cat_dist]  # one-hot encoded

            z_real_dist = z_real_dist.to(device)
            real_cat_dist = real_cat_dist.to(device)

            encoder_output_label, encoder_output_latent = encoder(x_unlabeled)
            decoder_input = torch.cat((encoder_output_label, encoder_output_latent), dim=1)
            decoder_output = decoder(decoder_input).to(device)

            autoencoder_loss = F.mse_loss(decoder_output, x_unlabeled)

            autoencoder_loss.backward()
            autoencoder_optimizer.step()

            # # wgan-gp
            # for _ in range(5):
            #     encoder_output_label, encoder_output_latent = encoder(x_unlabeled)
            #
            #     d_g_real = discriminator_g(z_real_dist)
            #     d_g_fake = discriminator_g(encoder_output_latent)
            #
            #     real_penalty = gradient_penalty(z_real_dist, encoder_output_latent, discriminator_g).to(device)
            #     dc_g_loss = -torch.mean(d_g_real) + torch.mean(d_g_fake) + 10.0 * real_penalty
            #
            #     discriminator_g_optimizer.zero_grad()
            #
            #     dc_g_loss.backward()
            #     discriminator_g_optimizer.step()
            #
            #     encoder_output_label, encoder_output_latent = encoder(x_unlabeled)
            #
            #     d_c_real = discriminator_c(real_cat_dist)
            #     d_c_fake = discriminator_c(encoder_output_label)
            #
            #     fake_penalty = gradient_penalty(real_cat_dist, encoder_output_label, discriminator_c).to(device)
            #     dc_c_loss = -torch.mean(d_c_real) + torch.mean(d_c_fake) + 10.0 * fake_penalty
            #
            #     discriminator_c_optimizer.zero_grad()
            #
            #     dc_c_loss.backward()
            #     discriminator_c_optimizer.step()

            for _ in range(5):
                encoder_output_label, encoder_output_latent = encoder(x_unlabeled)

                d_g_real = discriminator_g(z_real_dist)
                d_g_fake = discriminator_g(encoder_output_latent)

                target_real = torch.ones_like(d_g_real)
                target_fake = torch.zeros_like(d_g_fake)

                dc_g_real_loss = torch.mean(criterion(d_g_real, target_real))
                dc_g_fake_loss = torch.mean(criterion(d_g_fake, target_fake))

                dc_g_loss = dc_g_real_loss + dc_g_fake_loss

                discriminator_g_optimizer.zero_grad()
                dc_g_loss.backward()
                discriminator_g_optimizer.step()

                encoder_output_label, encoder_output_latent = encoder(x_unlabeled)

                d_c_real = discriminator_c(real_cat_dist)
                d_c_fake = discriminator_c(encoder_output_latent)

                dc_c_real_loss = torch.mean(criterion(d_c_real, target_real))
                dc_c_fake_loss = torch.mean(criterion(d_c_fake, target_fake))

                dc_c_loss = dc_c_real_loss + dc_c_fake_loss

                discriminator_c_optimizer.zero_grad()
                dc_c_loss.backward()
                discriminator_c_optimizer.step()

            # generator
            _encoder_output_label, _encoder_output_latent = encoder(x_unlabeled)
            d_g_fake = discriminator_g(_encoder_output_latent).to(device)
            d_c_fake = discriminator_c(_encoder_output_label).to(device)

            generator_g_loss = -torch.mean(torch.log(d_g_fake + 1e-8))
            generator_c_loss = -torch.mean(torch.log(d_c_fake + 1e-8))

            # generator_loss = -torch.mean(d_g_fake) - torch.mean(d_c_fake)
            generator_loss = generator_g_loss + generator_c_loss

            generator_loss.backward()
            generator_optimizer.step()

            # Semi-Supervised Classification Phase
            encoder_output_label_, encoder_output_latent_ = encoder(x_labeled, supervised=True)

            # Classification accuracy of encoder
            output_label = torch.argmax(encoder_output_label_, dim=1)
            correct_pred = output_label.eq(y_labeled)
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

        writer.add_image('train/x_unlabeled', x_unlabeled[0], epoch)
        writer.add_image('train/decoder_output', decoder_output[0], epoch)

        reconstruction_scheduler.step()
        supervised_scheduler.step()
        disc_g_scheduler.step()
        disc_c_scheduler.step()
        generator_scheduler.step()

        test(epoch)


def test(epoch):
    y_true = []
    y_pred = []
    # total_prob = []
    # total_latent = []

    encoder.eval()
    decoder.eval()
    discriminator_g.eval()
    discriminator_c.eval()

    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):

            x = inputs['input'].to(device)
            y = inputs['label'].to(device)

            batch_pred, batch_latent = encoder(x)
            # total_latent.append(batch_latent.cpu().numpy())

            # TODO: every sample in a batch has same values

            batch_label = y.cpu().numpy()
            batch_pred = batch_pred.argmax(dim=1).cpu().numpy()

            # print('batch_pred', batch_pred)

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

    torch.autograd.set_detect_anomaly(False)
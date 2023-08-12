import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models.model import Model


def gradient_penalty(self, real_samples, g_samples, discriminator):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_samples.size(1))  # Make alpha the same size as real_samples
    interpolates = alpha * real_samples + ((1 - alpha) * g_samples)

    interpolates.requires_grad_(True)  # Enable gradient computation
    d_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(d_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    slopes = torch.sqrt(torch.sum(gradients**2, dim=1))
    gradient_penalty = torch.mean((slopes - 1.)**2)
    return gradient_penalty


import torch
import torch.nn.functional as F
from torch.optim import Adam


def build(self):
    self.is_build = True
    self.x_input = torch.placeholder(dtype=torch.float32, shape=[None, self.input_dim])
    self.x_input_l = torch.placeholder(dtype=torch.float32, shape=[None, self.input_dim])
    self.y_input = torch.placeholder(dtype=torch.float32, shape=[None, self.n_labels])
    self.x_target = torch.placeholder(dtype=torch.float32, shape=[None, self.input_dim])
    self.real_distribution = torch.placeholder(dtype=torch.float32, shape=[None, self.z_dim])
    self.categorial_distribution = torch.placeholder(dtype=torch.float32, shape=[None, self.n_labels])
    self.manual_decoder_input = torch.placeholder(dtype=torch.float32, shape=[1, self.z_dim + self.n_labels])
    self.learning_rate = torch.placeholder(torch.float32, shape=[])
    self.keep_prob = torch.placeholder(torch.float32, shape=[])

    # Reconstruction Phase
    self.encoder_output_label, self.encoder_output_latent = self.model.encoder(self.x_input, self.keep_prob)
    decoder_input = torch.cat((self.encoder_output_label, self.encoder_output_latent), 1)
    decoder_output = self.model.decoder(decoder_input)

    self.autoencoder_loss = F.mse_loss(self.x_target, decoder_output)
    autoencoder_optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
    autoencoder_optimizer.zero_grad()
    self.autoencoder_loss.backward()
    autoencoder_optimizer.step()

    # Regularization Phase
    d_g_real = self.model.discriminator_gauss(self.real_distribution)
    d_g_fake = self.model.discriminator_gauss(self.encoder_output_latent)

    d_c_real = self.model.discriminator_categorical(self.categorial_distribution)
    d_c_fake = self.model.discriminator_categorical(self.encoder_output_label)

    self.dc_g_loss = -torch.mean(d_g_real) + torch.mean(d_g_fake) \
                     + 10.0 * self.gradient_penalty(self.real_distribution, self.encoder_output_latent,
                                                    self.model.discriminator_gauss)

    self.dc_c_loss = -torch.mean(d_c_real) + torch.mean(d_c_fake) \
                     + 10.0 * self.gradient_penalty(self.categorial_distribution, self.encoder_output_label,
                                                    self.model.discriminator_categorical)

    dc_g_optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
    dc_c_optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
    dc_g_optimizer.zero_grad()
    dc_c_optimizer.zero_grad()
    self.dc_g_loss.backward()
    self.dc_c_loss.backward()
    dc_g_optimizer.step()
    dc_c_optimizer.step()

    self.generator_loss = -torch.mean(d_g_fake) - torch.mean(d_c_fake)

    en_var = [var for var in self.model.parameters() if 'e_' in var.name]
    generator_optimizer = Adam(en_var, lr=self.learning_rate, betas=(self.beta1, self.beta2))
    generator_optimizer.zero_grad()
    self.generator_loss.backward()
    generator_optimizer.step()

    # Semi-Supervised Classification Phase
    self.encoder_output_label_, self.encoder_output_latent_ = self.model.encoder(self.x_input_l, self.keep_prob,
                                                                                 supervised=True)

    self.output_label = torch.argmax(self.encoder_output_label_, 1)
    correct_pred = torch.eq(self.output_label, torch.argmax(self.y_input, 1))
    self.accuracy = torch.mean(correct_pred.float())

    self.supervised_encoder_loss = F.cross_entropy(self.encoder_output_label_, torch.argmax(self.y_input, 1))
    supervised_encoder_optimizer = Adam(en_var, lr=self.learning_rate, betas=(self.beta1, self.beta1_sup))
    supervised_encoder_optimizer.zero_grad()
    self.supervised_encoder_loss.backward()
    supervised_encoder_optimizer.step()


def get_val_acc(self, val_size, batch_size, tfdata, sess):
    acc = 0
    y_true, y_pred = [], []
    num_batches = int(val_size / batch_size)

    for j in tqdm.tqdm(range(num_batches)):
        batch_x_l, batch_y_l = data_stream(tfdata, sess)
        batch_x_l = torch.from_numpy(batch_x_l)
        batch_y_l = torch.from_numpy(batch_y_l)
        batch_pred = self.output_label(batch_x_l, batch_y_l, 1.0).argmax(dim=1).cpu().numpy()

        batch_label = np.argmax(batch_y_l, axis=1)
        y_pred += batch_pred.tolist()
        y_true += batch_label.tolist()

    avg_acc = np.equal(y_true, y_pred).mean()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (tp + fn)
    err = (fn + fp) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = 1 - fnr
    f1 = (2 * precision * recall) / (precision + recall)

    return avg_acc, precision, recall, f1


def train(self):
    train_unlabel, train_label, validation = self.construct_data_flow()
    if not self.is_build:
        self.build()
    all_variables = list(self.parameters())
    init = torch.global_variables_initializer()

    # Tensorboard visualization
    writer = SummaryWriter(log_dir=tensorboard_path)
    writer.add_scalar('Autoencoder Loss', self.autoencoder_loss)
    writer.add_scalar('Discriminator gauss Loss', self.dc_g_loss)
    writer.add_scalar('Discriminator categorical Loss', self.dc_c_loss)
    writer.add_scalar('Generator Loss', self.generator_loss)
    writer.add_scalar('Supervised Encoder Loss', self.supervised_encoder_loss)
    writer.add_scalar('Supervised Encoder Accuracy', self.accuracy)

    accuracies = []
    # Saving the model
    step = 0
    # Early stopping
    best_sess = None
    best_f1 = 0.0
    stop = False
    last_improvement = 0
    require_improvement = 20

    accs = {
        'known': [],
        'unknown': []
    }
    f1s = {
        'known': [],
        'unknown': []
    }

    tensorboard_path, saved_model_path, log_path = form_results(
        self.model_name, self.results_path, self.z_dim, self.supervised_lr, self.batch_size, self.n_epochs, self.beta1)

    with torch.Session() as sess:
        sess.run(init)
        for epoch in range(self.n_epochs):
            if epoch == 50:
                self.supervised_lr /= 10
                self.reconstruction_lr /= 10
                self.regularization_lr /= 10

            n_batches = int(self.n_labeled / self.batch_size)
            num_normal = 0
            num_attack = 0

            print("------------------Epoch {}/{}------------------".format(epoch, self.n_epochs))

            for b in tqdm.tqdm(range(1, n_batches + 1)):
                z_real_dist = np.random.randn(self.batch_size, self.z_dim) * 5.
                real_cat_dist = np.random.randint(low=0, high=2, size=self.batch_size)
                real_cat_dist = np.eye(self.n_labels)[real_cat_dist]

                batch_x_ul, batch_y_ul = data_stream(train_unlabel, sess)
                batch_x_l, batch_y_l = data_stream(train_label, sess)
                batch_x_ul = torch.from_numpy(batch_x_ul)
                batch_y_ul = torch.from_numpy(batch_y_ul)
                batch_x_l = torch.from_numpy(batch_x_l)
                batch_y_l = torch.from_numpy(batch_y_l)

                num_normal += (batch_y_ul.argmax(dim=1) == 0).sum()
                num_attack += (batch_y_ul.argmax(dim=1) == 1).sum()

                self.autoencoder_optimizer.zero_grad()
                self.discriminator_g_optimizer.zero_grad()
                self.discriminator_c_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                self.supervised_encoder_optimizer.zero_grad()

                self.autoencoder_optimizer.step()
                self.discriminator_g_optimizer.step()
                self.discriminator_c_optimizer.step()
                self.generator_optimizer.step()
                self.supervised_encoder_optimizer.step()

                if b % 10 == 0:
                    a_loss, d_g_loss, d_c_loss, g_loss, s_loss, summary = (
                        self.autoencoder_loss.item(), self.dc_g_loss.item(), self.dc_c_loss.item(),
                        self.generator_loss.item(), self.supervised_encoder_loss.item(), self.summary_op)

                    writer.add_scalar('Autoencoder Loss', a_loss, global_step=step)
                    writer.add_scalar('Discriminator gauss Loss', d_g_loss, global_step=step)
                    writer.add_scalar('Discriminator categorical Loss', d_c_loss, global_step=step)
                    writer.add_scalar('Generator Loss', g_loss, global_step=step)
                    writer.add_scalar('Supervised Encoder Loss', s_loss, global_step=step)

                    with open(log_path + '/log.txt', 'a') as log:
                        log.write("Epoch: {}, iteration: {}\n".format(epoch, b))
                        log.write("Autoencoder Loss: {}\n".format(a_loss))
                        log.write("Discriminator Gauss Loss: {}".format(d_g_loss))
                        log.write("Discriminator Categorical Loss: {}".format(d_c_loss))
                        log.write("Generator Loss: {}\n".format(g_loss))
                        log.write("Supervised Loss: {}".format(s_loss))

                    step += 1

            print('Num normal: ', num_normal)
            print('Num attack: ', num_attack)

            if (epoch + 1) % 2 == 0:
                print("Runing on validation...----------------")
                acc_known, precision_known, recall_known, f1_known = (
                    self.get_val_acc(self.validation_size, self.batch_size, validation, sess))

                print("Accuracy on Known attack: {}".format(acc_known))
                print("Precision on Known attack: {}".format(precision_known))
                print("Recall on Known attack: {}".format(recall_known))
                print("F1 on Known attack: {}".format(f1_known))

                accs['known'].append(acc_known)
                f1s['known'].append(f1_known)

                if f1_known > best_f1:
                    best_sess = sess
                elif (epoch + 1) == self.n_epochs:
                    sess = best_sess

                if self.unknown_attack is not None:
                    acc_unknown, precision_unknown, recall_unknown, f1_unknown = (
                        self.get_val_acc(self.validation_unknown_size, self.batch_size, self.validation_unknown, sess))

                    print("Accuracy on unKnown attack: {}".format(acc_unknown))
                    print("Precision on unKnown attack: {}".format(precision_unknown))
                    print("Recall on unKnown attack: {}".format(recall_unknown))
                    print("F1 on unKnown attack: {}".format(f1_unknown))

                    accs['unknown'].append(acc_unknown)
                    f1s['unknown'].append(f1_unknown)

                print('Save model')
                saver.save(sess, save_path=saved_model_path, global_step=step)

            with open(log_path + '/sum_val.txt', 'w') as summary:
                summary.write(json.dumps(accs))
                summary.write(json.dumps(f1s))

            with open(log_path + '/sum_val.txt', 'w') as summary:
                summary.write(json.dumps(accs))
                summary.write(json.dumps(f1s))


def test(self, results_path, unknown_test):
    if not self.is_build:
        self.build()
    init = torch.global_variables_initializer()
    saver = torch.load(f"{results_path}/Saved_models/model.pth")
    if unknown_test:
        data_path = [f"{self.data_dir}/{a}/" for a in [self.unknown_attack, 'Normal']]
        test_size = 0
        for f in [f"{p}/datainfo.txt" for p in data_path]:
            data_read = json.load(open(f))
            test_size += data_read['test']
    else:
        test_size = self.data_info['test']
        data_path = [f"{self.data_dir}/{a}/" for a in self.labels if a != self.unknown_attack]

    print('Test data: ', data_path)
    with torch.Session() as sess:
        saver.load_state_dict(torch.load(f"{results_path}/Saved_models/model.pth"))
        saver.eval()
        test = data_from_tfrecord(tf_filepath=[f"{p}test" for p in data_path], batch_size=self.batch_size,
                                  repeat_time=1, shuffle=False)
        num_batches = int(test_size / self.batch_size)
        y_true = np.empty((0), int)
        y_pred = np.empty((0), int)
        raw_pred = np.empty((0, self.n_labels), int)
        total_prob = np.empty((0), float)
        total_latent = np.empty((0, self.z_dim), float)

        for _ in tqdm.tqdm(range(num_batches)):
            x_test, y_test = data_stream(test, sess)
            batch_raw_pred, batch_pred, batch_latent = sess.run(
                [self.encoder_output_label_, self.output_label, self.encoder_output_latent_],
                feed_dict={self.x_input_l: x_test, self.keep_prob: 1.0})
            total_latent = np.append(total_latent, batch_latent, axis=0)
            batch_label = np.argmax(y_test, axis=1).reshape((self.batch_size))
            raw_pred = np.append(raw_pred, batch_raw_pred, axis=0)
            y_pred = np.append(y_pred, batch_pred, axis=0)
            y_true = np.append(y_true, batch_label, axis=0)

    evaluate(y_true, y_pred)
    return raw_pred, y_pred, y_true


def ensemble_predict(self, model_dir, unknown_test):
    model_paths = [f for f in os.listdir(model_dir) if not f.startswith('.')]
    ensemble_pred = []
    for model_path in model_paths:
        print(model_path)
        pred, _, y_true = self.test(model_dir + model_path, unknown_test=unknown_test)
        ensemble_pred.append(pred)
    ensemble_pred = np.mean(ensemble_pred, axis=0)
    ensemble_pred = np.argmax(ensemble_pred, axis=1)
    evaluate(y_true, ensemble_pred)
    return ensemble_pred, y_true


def timing(self, x, model_path, num_loop=100, use_gpu=False):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if not self.is_build:
        self.build()
    self.to(device)
    checkpoint = torch.load(os.path.join(model_path, "best_model.pth"), map_location=device)
    self.load_state_dict(checkpoint['model_state_dict'])
    self.eval()

    # For warm-up
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_pred = self.forward_label(x_tensor).cpu().numpy()

    time = []
    with torch.no_grad():
        for _ in range(num_loop):
            start = timeit.default_timer()
            y_pred = self.forward_label(x_tensor).cpu().numpy()
            _ = np.array(y_pred)
            end = timeit.default_timer()
            time.append(end - start)
    return time



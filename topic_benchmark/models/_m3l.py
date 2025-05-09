# Code adapted from https://github.com/ezosa/M3L-topic-model
import datetime
import multiprocessing as mp
import os
import warnings
from collections import OrderedDict, defaultdict
from typing import Optional

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from scipy.special import softmax
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class M3LDataset(Dataset):

    def __init__(
        self,
        dtm: scipy.sparse.csr.csr_matrix,
        embeddings: Optional[np.ndarray] = None,
        image_embeddings: Optional[np.ndarray] = None,
    ):
        self.dtm = dtm
        self.embeddings = embeddings
        self.image_embeddings = image_embeddings

    def __len__(self):
        return self.dtm.shape[0]

    def __getitem__(self, i):
        res_dict = {}
        res_dict["X_bow"] = torch.FloatTensor(
            np.squeeze(self.dtm[i, :].todense())
        )
        if self.embeddings is not None:
            res_dict["X_contextual"] = torch.FloatTensor(self.embeddings[i, :])
            if self.image_embeddings is not None:
                res_dict["X_image"] = torch.FloatTensor(
                    self.image_embeddings[i, :]
                )
        if self.embeddings is None and self.image_embeddings is not None:
            res_dict["X_contextual"] = torch.FloatTensor(
                self.image_embeddings[i, :]
            )
        return res_dict


class ContextualInferenceNetwork(nn.Module):
    """Inference Network."""

    def __init__(
        self,
        input_size,
        bert_size,
        output_size,
        hidden_sizes,
        activation="softplus",
        dropout=0.2,
        label_size=0,
    ):
        super(ContextualInferenceNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            self.activation = nn.ReLU()
        self.input_layer = nn.Linear(bert_size + label_size, hidden_sizes[0])
        self.hiddens = nn.Sequential(
            OrderedDict(
                [
                    (
                        "l_{}".format(i),
                        nn.Sequential(nn.Linear(h_in, h_out), self.activation),
                    )
                    for i, (h_in, h_out) in enumerate(
                        zip(hidden_sizes[:-1], hidden_sizes[1:])
                    )
                ]
            )
        )
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""
        x = x_bert
        if labels:
            x = torch.cat((x_bert, labels), 1)
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        return mu, log_sigma


class CombinedInferenceNetwork(nn.Module):
    """Inference Network."""

    def __init__(
        self,
        input_size,
        bert_size,
        output_size,
        hidden_sizes,
        activation="softplus",
        dropout=0.2,
        label_size=0,
    ):
        super(CombinedInferenceNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            self.activation = nn.ReLU()
        self.adapt_bert = nn.Linear(bert_size, input_size)
        self.input_layer = nn.Linear(
            input_size + input_size + label_size, hidden_sizes[0]
        )
        self.hiddens = nn.Sequential(
            OrderedDict(
                [
                    (
                        "l_{}".format(i),
                        nn.Sequential(nn.Linear(h_in, h_out), self.activation),
                    )
                    for i, (h_in, h_out) in enumerate(
                        zip(hidden_sizes[:-1], hidden_sizes[1:])
                    )
                ]
            )
        )
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""
        x_bert = self.adapt_bert(x_bert)
        x = torch.cat((x, x_bert), 1)
        if labels is not None:
            x = torch.cat((x, labels), 1)
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        return mu, log_sigma


# ----- Multimodal and Multingual (M3L) -----
class ContrastiveM3LDecoderNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        bert_sizes,
        n_components=10,
        hidden_sizes=(100, 100),
        activation="softplus",
        dropout=0.2,
        learn_priors=True,
        label_size=0,
    ):
        super(ContrastiveM3LDecoderNetwork, self).__init__()
        self.input_size = input_size
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
        topic_prior_variance = 1.0 - (1.0 / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components
        )
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)
        self.inf_net1 = ContextualInferenceNetwork(
            input_size, bert_sizes[0], n_components, hidden_sizes, activation
        )
        self.inf_net3 = ContextualInferenceNetwork(
            input_size, bert_sizes[1], n_components, hidden_sizes, activation
        )
        # topic_word_matrix is K x V, where L = no. of languages
        self.topic_word_matrix = None
        # beta dimensions remains the same for multimodal because images have no BOW reconstruction
        # beta is L x K x V where L = no. of languages
        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x_bow, x_bert, x_image):
        """Forward pass."""
        # x_bert: batch_size x L x bert_dim
        # pass language1 x_bert to inference net1 (input is batch_size x bert_dim)
        posterior_mu1, posterior_log_sigma1 = self.inf_net1(x_bow, x_bert)
        posterior_sigma1 = torch.exp(posterior_log_sigma1)
        # pass encoded image to inference net3 (input is batch_size x image_enc_dim)
        # x_bow does not matter, inference net will not use it anyway
        posterior_mu3, posterior_log_sigma3 = self.inf_net3(x_bow, x_image)
        posterior_sigma3 = torch.exp(posterior_log_sigma3)
        # generate separate thetas for each language
        z1 = self.reparameterize(posterior_mu1, posterior_log_sigma1)
        z3 = self.reparameterize(posterior_mu3, posterior_log_sigma3)
        theta1 = F.softmax(z1, dim=1)
        theta3 = F.softmax(z3, dim=1)
        thetas_no_drop = torch.stack([theta1, theta3])
        theta1 = self.drop_theta(theta1)
        theta3 = self.drop_theta(theta3)
        thetas = torch.stack([theta1, theta3])
        word_dist = F.softmax(
            self.beta_batchnorm(torch.matmul(thetas[0], self.beta)),
            dim=1,
        )
        self.topic_word_matrix = self.beta
        return (
            self.prior_mean,
            self.prior_variance,
            posterior_mu1,
            posterior_sigma1,
            posterior_log_sigma1,
            posterior_mu3,
            posterior_sigma3,
            posterior_log_sigma3,
            word_dist,
            thetas_no_drop,
        )

    def get_theta(self, x, x_bert, lang_index=0):
        with torch.no_grad():
            if lang_index == 0:
                posterior_mu, posterior_log_sigma = self.inf_net1(x, x_bert)
            else:
                posterior_mu, posterior_log_sigma = self.inf_net3(x, x_bert)
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1
            )
            return theta


class ContrastiveDecoderNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        bert_size,
        n_components=10,
        hidden_sizes=(100, 100),
        activation="softplus",
        dropout=0.2,
        learn_priors=True,
        label_size=0,
    ):
        super(ContrastiveDecoderNetwork, self).__init__()
        # input_size: same as vocab size
        self.input_size = input_size
        # n_components: no. of topics
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
        topic_prior_variance = 1.0 - (1.0 / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components
        )
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)
        self.inf_net1 = ContextualInferenceNetwork(
            input_size, bert_size, n_components, hidden_sizes, activation
        )
        self.topic_word_matrix = None
        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_bert):
        """Forward pass."""
        posterior_mu1, posterior_log_sigma1 = self.inf_net1(x, x_bert)
        posterior_sigma1 = torch.exp(posterior_log_sigma1)
        z1 = self.reparameterize(posterior_mu1, posterior_log_sigma1)
        theta1 = F.softmax(z1, dim=1)
        thetas_no_drop = torch.stack([theta1])
        z_no_drop = torch.stack([z1])
        theta1 = self.drop_theta(theta1)
        thetas = torch.stack([theta1])
        word_dist = F.softmax(
            self.beta_batchnorm(torch.matmul(thetas[0], self.beta)),
            dim=1,
        )
        self.topic_word_matrix = self.beta
        return (
            self.prior_mean,
            self.prior_variance,
            posterior_mu1,
            posterior_sigma1,
            posterior_log_sigma1,
            word_dist,
            thetas_no_drop,
            z_no_drop,
        )

    def get_theta(self, x, x_bert):
        with torch.no_grad():
            posterior_mu, posterior_log_sigma = self.inf_net1(x, x_bert)
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1
            )
            return theta


class DecoderNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        bert_size,
        infnet,
        n_components=10,
        hidden_sizes=(100, 100),
        activation="softplus",
        dropout=0.2,
        learn_priors=True,
        label_size=0,
    ):
        super(DecoderNetwork, self).__init__()
        self.input_size = input_size
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None
        if infnet == "zeroshot":
            self.inf_net = ContextualInferenceNetwork(
                input_size,
                bert_size,
                n_components,
                hidden_sizes,
                activation,
                label_size=label_size,
            )
        elif infnet == "combined":
            self.inf_net = CombinedInferenceNetwork(
                input_size,
                bert_size,
                n_components,
                hidden_sizes,
                activation,
                label_size=label_size,
            )
        else:
            raise Exception(
                "Missing infnet parameter, options are zeroshot and combined"
            )
        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
        topic_prior_variance = 1.0 - (1.0 / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components
        )
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)
        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
        posterior_sigma = torch.exp(posterior_log_sigma)
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1
        )
        theta = self.drop_theta(theta)
        word_dist = F.softmax(
            self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1
        )
        self.topic_word_matrix = self.beta
        estimated_labels = None
        if labels is not None:
            estimated_labels = self.label_classification(theta)
        return (
            self.prior_mean,
            self.prior_variance,
            posterior_mu,
            posterior_sigma,
            posterior_log_sigma,
            word_dist,
            estimated_labels,
        )

    def get_theta(self, x, x_bert, labels=None):
        with torch.no_grad():
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1
            )
            return theta


class MultimodalContrastiveTM:
    """Class to train the contextualized topic model. This is the more general class that we are keeping to
    avoid braking code, users should use the two subclasses ZeroShotTM and CombinedTm to do topic modeling.

    :param bow_size: int, dimension of input
    :param contextual_size: int, dimension of input that comes from BERT embeddings
    :param n_components: int, number of topic components, (default 10)
    :param hidden_sizes: tuple, length = n_layers, (default (100, 100))
    :param activation: string, 'softplus', 'relu', (default 'softplus')
    :param dropout: float, dropout to use (default 0.2)
    :param learn_priors: bool, make priors a learnable parameter (default True)
    :param batch_size: int, size of batch to use for training (default 64)
    :param lr: float, learning rate to use for training (default 2e-3)
    :param momentum: float, momentum to use for training (default 0.99)
    :param solver: string, optimizer 'adam' or 'sgd' (default 'adam')
    :param num_epochs: int, number of epochs to train for, (default 100)
    :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
    :param num_data_loader_workers: int, number of data loader workers (default cpu_count). set it to 0 if you are using Windows
    :param label_size: int, number of total labels (default: 0)
    :param loss_weights: dict, it contains the name of the weight parameter (key) and the weight (value) for each loss.
    It supports only the weight parameter beta for now. If None, then the weights are set to 1 (default: None).

    """

    def __init__(
        self,
        bow_size,
        contextual_sizes,
        n_components=10,
        hidden_sizes=(100, 100),
        activation="softplus",
        dropout=0.2,
        learn_priors=True,
        batch_size=16,
        lr=2e-3,
        momentum=0.99,
        solver="adam",
        num_epochs=100,
        reduce_on_plateau=False,
        num_data_loader_workers=mp.cpu_count(),
        label_size=0,
        loss_weights=None,
    ):
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # bow_size is an array of size n_languages; one bow_size for each language
        self.bow_size = bow_size
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.contextual_sizes = contextual_sizes
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        if loss_weights:
            self.weights = loss_weights
        else:
            self.weights = {"KL": 1, "CL": 100}
        self.model = ContrastiveM3LDecoderNetwork(
            bow_size,
            self.contextual_sizes,
            n_components,
            hidden_sizes,
            activation,
            dropout,
            learn_priors,
            label_size=label_size,
        )
        if self.solver == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99)
            )
        elif self.solver == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum
            )
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)
        self.best_loss_train = float("inf")
        self.train_data = None
        self.nn_epoch = None
        # best_components: n_components x vocab_size
        self.best_components = None
        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False
        self.model = self.model.to(self.device)

    def _infoNCE_loss(self, embeddings1, embeddings2, temperature=0.07):
        batch_size = embeddings1.shape[0]
        labels = torch.arange(batch_size)
        labels = torch.cat([labels, labels])
        embeddings_cat = torch.cat([embeddings1, embeddings2])
        loss_func = NTXentLoss()
        infonce_loss = loss_func(embeddings_cat, labels)
        return infonce_loss

    def _kl_loss1(self, thetas1, thetas2):
        theta_kld = F.kl_div(thetas1.log(), thetas2, reduction="sum")
        return theta_kld

    def _kl_loss2(
        self,
        prior_mean,
        prior_variance,
        posterior_mean,
        posterior_variance,
        posterior_log_variance,
    ):
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1
        )
        logvar_det_division = (
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        )
        KL = 0.5 * (
            var_division + diff_term - self.n_components + logvar_det_division
        )
        return KL

    def _rl_loss(self, true_word_dists, pred_word_dists):
        print("True word dist: ", true_word_dists.size())
        print("Pred word dist: ", pred_word_dists.size())
        RL = -torch.sum(
            true_word_dists * torch.log(pred_word_dists + 1e-10), dim=1
        )
        return RL

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            X_bow = batch_samples["X_bow"]
            X_bow = X_bow[:, 0, :]
            X_contextual = batch_samples["X_contextual"]
            X_image = batch_samples["X_image"]
            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()
                X_image = X_image.cuda()
            self.model.zero_grad()
            (
                prior_mean,
                prior_variance,
                posterior_mean1,
                posterior_variance1,
                posterior_log_variance1,
                posterior_mean3,
                posterior_variance3,
                posterior_log_variance3,
                word_dist,
                thetas,
            ) = self.model(X_bow, X_contextual, X_image)
            # Recon loss for lang1 and lang2 (no recon loss for image)
            rl_loss1 = self._rl_loss(X_bow, word_dist)
            # rl_loss2 = self._rl_loss(X_bow, word_dists[1])

            # KL losses between posterior distributions and a prior distribution
            kl_en_prior = self._kl_loss2(
                prior_mean,
                prior_variance,
                posterior_mean1,
                posterior_variance1,
                posterior_log_variance1,
            )
            kl_image_prior = self._kl_loss2(
                prior_mean,
                prior_variance,
                posterior_mean3,
                posterior_variance3,
                posterior_log_variance3,
            )
            # KL loss between posterior distributions of paired languages/modalities
            kl_en_image = self._kl_loss2(
                posterior_mean1,
                posterior_variance1,
                posterior_mean3,
                posterior_variance3,
                posterior_log_variance3,
            )
            # InfoNCE loss/NTXentLoss
            infoNCE_en_image = self._infoNCE_loss(thetas[0], thetas[1])
            loss = (
                rl_loss1
                + self.weights["KL"] * kl_en_prior
                + self.weights["KL"] * kl_image_prior
                + self.weights["CL"] * infoNCE_en_image
            )
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
            # compute train loss
            samples_processed += X_bow.size()[0]
            train_loss += loss.item()
        train_loss /= samples_processed
        return samples_processed, train_loss

    def fit(
        self,
        train_dataset,
        verbose=False,
        patience=5,
        delta=0,
    ):
        """
        Train the CTM model.

        :param train_dataset: PyTorch Dataset class for training data.
        :param verbose: verbose
        :param patience: How long to wait after last time validation loss improved. Default: 5
        :param delta: Minimum change in the monitored quantity to qualify as an improvement. Default: 0

        """
        self.train_data = train_dataset
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_loader_workers,
        )
        # init training variables
        train_loss = 0
        samples_processed = 0
        # train loop
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()
            pbar.update(1)
            self.best_components = self.model.beta
            pbar.set_description(
                "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch + 1,
                    self.num_epochs,
                    samples_processed,
                    len(self.train_data) * self.num_epochs,
                    train_loss,
                    e - s,
                )
            )
        pbar.close()

    def _validation(self, loader):
        """Validation epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            # batch_size x L x vocab_size
            X_bow = batch_samples["X_bow"]
            X_bow = X_bow.squeeze(dim=2)
            # batch_size x L x bert_size
            X_contextual = batch_samples["X_contextual"]
            # batch_size x image_enc_size
            X_image = batch_samples["X_image"]
            # forward pass
            self.model.zero_grad()
            (
                prior_mean,
                prior_variance,
                posterior_mean1,
                posterior_variance1,
                posterior_log_variance1,
                posterior_mean2,
                posterior_variance2,
                posterior_log_variance2,
                posterior_mean3,
                posterior_variance3,
                posterior_log_variance3,
                word_dist,
                thetas,
            ) = self.model(X_bow, X_contextual, X_image)
            # Recon loss for lang1 and lang2 (no recon loss for image)
            rl_loss1 = self._rl_loss(X_bow, word_dist)
            # rl_loss2 = self._rl_loss(X_bow, word_dists[1])
            # KL loss between posterior distributions of paired languages/modalities
            kl_en_de = self._kl_loss(
                posterior_mean1,
                posterior_variance1,
                posterior_mean2,
                posterior_variance2,
                posterior_log_variance2,
            )
            kl_en_image = self._kl_loss(
                posterior_mean1,
                posterior_variance1,
                posterior_mean3,
                posterior_variance3,
                posterior_log_variance3,
            )
            kl_de_image = self._kl_loss(
                posterior_mean2,
                posterior_variance2,
                posterior_mean3,
                posterior_variance3,
                posterior_log_variance3,
            )
            # InfoNCE loss/NTXentLoss
            infoNCE_en_de = self._infoNCE_loss(thetas[0], thetas[1])
            infoNCE_en_image = self._infoNCE_loss(thetas[0], thetas[2])
            loss = (
                rl_loss1
                + rl_loss2
                + self.weights["CL"] * infoNCE_en_de
                + self.weights["CL"] * infoNCE_en_image
                + self.weights["KL"] * kl_en_de
                + self.weights["KL"] * kl_en_image
                + self.weights["KL"] * kl_de_image
            )
            loss = loss.sum()
            # compute train loss
            # samples_processed += X_bow.size()[0]
            samples_processed += X_bow.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def get_doc_topic_distribution(self, dataset, n_samples=20, lang_index=0):
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_data_loader_workers,
        )
        pbar = tqdm(n_samples, position=0, leave=True)
        final_thetas = []
        for sample_index in range(n_samples):
            with torch.no_grad():
                collect_theta = []
                for batch_samples in loader:
                    # batch_size x L x bert_size
                    X_contextual = batch_samples["X_contextual"]
                    if self.USE_CUDA:
                        X_contextual = X_contextual.cuda()
                    # forward pass
                    self.model.zero_grad()
                    thetas = self.model.get_theta(
                        x=None, x_bert=X_contextual, lang_index=lang_index
                    )
                    collect_theta.extend(thetas.detach().cpu().numpy())
                pbar.update(1)
                pbar.set_description(
                    "Sampling: [{}/{}]".format(sample_index + 1, n_samples)
                )
                final_thetas.append(np.array(collect_theta))
        pbar.close()
        return np.sum(final_thetas, axis=0) / n_samples

    @property
    def components_(self):
        return self.model.topic_word_matrix.cpu().detach().numpy()

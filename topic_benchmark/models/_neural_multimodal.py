# Code adapted from https://github.com/ezosa/M3L-topic-model
import multiprocessing as mp
from collections import OrderedDict, defaultdict
from typing import Optional, Sequence

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange


class MultimodalDataset(Dataset):

    def __init__(
        self,
        bow: scipy.sparse.csr.csr_matrix,
        text_embeddings: Optional[np.ndarray] = None,
        image_embeddings: Optional[np.ndarray] = None,
    ):
        self.bow = bow
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings

    def __len__(self):
        return self.bow.shape[0]

    def __getitem__(self, i):
        res_dict = {}
        res_dict["bow"] = torch.FloatTensor(
            np.squeeze(self.bow[i, :].todense())
        )
        if self.text_embeddings is not None:
            res_dict["text_embeddings"] = torch.FloatTensor(
                self.text_embeddings[i, :]
            )
        if self.image_embeddings is not None:
            res_dict["image_embeddings"] = torch.FloatTensor(
                self.image_embeddings[i, :]
            )
        return res_dict


class ZeroShotEncoderNetwork(nn.Module):
    def __init__(
        self,
        n_dimensions: int,
        n_latent: int,
        hidden_sizes: Sequence[int] = (100, 100),
        dropout=0.2,
        device: str = "cpu",
    ):
        super(ZeroShotEncoderNetwork, self).__init__()
        self.device = device
        self.output_size = n_latent
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.activation = nn.Softplus()
        self.input_layer = nn.Linear(n_dimensions, hidden_sizes[0])
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
        self.f_mu = nn.Linear(hidden_sizes[-1], n_latent)
        self.f_mu_batchnorm = nn.BatchNorm1d(n_latent, affine=False)
        self.f_sigma = nn.Linear(hidden_sizes[-1], n_latent)
        self.f_sigma_batchnorm = nn.BatchNorm1d(n_latent, affine=False)
        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        return mu, log_sigma


class ContrastiveMultimodalDecoderNetwork(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_dimensions: int,
        n_components: int,
        hidden_sizes=(100, 100),
        dropout=0.2,
        device: str = "cpu",
    ):
        super(ContrastiveMultimodalDecoderNetwork, self).__init__()
        self.device = device
        self.n_vocab = n_vocab
        self.n_components = n_components
        self.n_dimensions = n_dimensions
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        self.prior_mean = self.prior_mean.to(self.device)
        self.prior_mean = nn.Parameter(self.prior_mean)
        topic_prior_variance = 1.0 - (1.0 / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components
        )
        self.prior_variance = self.prior_variance.to(self.device)
        self.prior_variance = nn.Parameter(self.prior_variance)
        self.text_encoder = ZeroShotEncoderNetwork(
            n_dimensions=n_dimensions,
            n_latent=n_components,
            hidden_sizes=hidden_sizes,
        )
        self.image_encoder = ZeroShotEncoderNetwork(
            n_dimensions=n_dimensions,
            n_latent=n_components,
            hidden_sizes=hidden_sizes,
        )
        self.topic_word_matrix = None
        self.beta = torch.Tensor(n_components, n_vocab)
        self.beta = self.beta.to(self.device)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(n_vocab, affine=False)
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, text_embeddings, image_embeddings):
        text_mu, text_log_sigma = self.text_encoder(text_embeddings)
        text_sigma = torch.exp(text_log_sigma)
        image_mu, image_log_sigma = self.image_encoder(image_embeddings)
        image_sigma = torch.exp(image_log_sigma)
        z_text = self.reparameterize(text_mu, text_log_sigma)
        z_image = self.reparameterize(image_mu, image_log_sigma)
        # Document-topic proportions
        text_theta = F.softmax(z_text, dim=1)
        image_theta = F.softmax(z_image, dim=1)
        thetas_no_drop = torch.stack([text_theta, image_theta])
        text_theta = self.drop_theta(text_theta)
        image_theta = self.drop_theta(image_theta)
        word_dist = F.softmax(
            self.beta_batchnorm(torch.matmul(text_theta, self.beta)),
            dim=1,
        )
        self.topic_word_matrix = self.beta
        return (
            self.prior_mean,
            self.prior_variance,
            text_mu,
            text_sigma,
            text_log_sigma,
            image_mu,
            image_sigma,
            image_log_sigma,
            word_dist,
            thetas_no_drop,
        )

    def get_theta(self, embeddings, is_image: bool = False):
        with torch.no_grad():
            if not is_image:
                mu, sigma = self.text_encoder(embeddings)
            else:
                mu, sigma = self.image_encoder(embeddings)
            theta = F.softmax(self.reparameterize(mu, sigma), dim=1)
            return theta


class MultimodalDecoderNetwork(nn.Module):

    def __init__(
        self,
        n_vocab: int,
        n_dimensions: int,
        n_components: int,
        hidden_sizes=(100, 100),
        dropout=0.2,
        device: str = "cpu",
    ):
        super(MultimodalDecoderNetwork, self).__init__()
        self.device = device
        self.n_vocab = n_vocab
        self.n_components = n_components
        self.n_dimensions = n_dimensions
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        self.prior_mean = self.prior_mean.to(self.device)
        self.prior_mean = nn.Parameter(self.prior_mean)
        topic_prior_variance = 1.0 - (1.0 / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components
        )
        self.prior_variance = self.prior_variance.to(self.device)
        self.prior_variance = nn.Parameter(self.prior_variance)
        self.encoder = ZeroShotEncoderNetwork(
            # We pass the two embeddings concatenated to the encoder
            n_dimensions=n_dimensions * 2,
            n_latent=n_components,
            hidden_sizes=hidden_sizes,
        )
        self.topic_word_matrix = None
        self.beta = torch.Tensor(n_components, n_vocab)
        self.image_beta = torch.Tensor(n_components, self.n_dimensions)
        self.text_beta = torch.Tensor(n_components, self.n_dimensions)
        self.beta = self.beta.to(self.device)
        self.image_beta = self.image_beta.to(self.device)
        self.text_beta = self.text_beta.to(self.device)
        self.beta = nn.Parameter(self.beta)
        self.image_beta = nn.Parameter(self.image_beta)
        self.text_beta = nn.Parameter(self.text_beta)
        nn.init.xavier_uniform_(self.beta)
        nn.init.xavier_uniform_(self.image_beta)
        nn.init.xavier_uniform_(self.text_beta)
        self.beta_batchnorm = nn.BatchNorm1d(n_vocab, affine=False)
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, embeddings):
        mu, log_sigma = self.encoder(embeddings)
        sigma = torch.exp(log_sigma)
        z = self.reparameterize(mu, log_sigma)
        # Document-topic proportions
        theta = F.softmax(z, dim=1)
        thetas_no_drop = theta
        theta = self.drop_theta(theta)
        bow_reconstruction = F.softmax(
            self.beta_batchnorm(torch.matmul(theta, self.beta)),
            dim=1,
        )
        image_reconstruction = torch.matmul(theta, self.image_beta)
        text_reconstruction = torch.matmul(theta, self.text_beta)
        self.topic_word_matrix = self.beta
        return (
            self.prior_mean,
            self.prior_variance,
            mu,
            sigma,
            log_sigma,
            bow_reconstruction,
            image_reconstruction,
            text_reconstruction,
            thetas_no_drop,
        )

    def get_theta(self, embeddings):
        with torch.no_grad():
            mu, log_sigma = self.encoder(embeddings)
            theta = F.softmax(self.reparameterize(mu, log_sigma), dim=1)
            return theta


class MultimodalZeroShotTM:
    def __init__(
        self,
        n_vocab: int,
        n_dimensions: int,
        n_components: int,
        hidden_sizes: Sequence[int] = (100, 100),
        dropout=0.2,
        batch_size=16,
        lr=2e-3,
        momentum=0.99,
        num_epochs=100,
        num_data_loader_workers=mp.cpu_count(),
        device: str = "cpu",
    ):
        self.device = device
        self.n_vocab = n_vocab
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.n_dimensions = n_dimensions
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.num_data_loader_workers = num_data_loader_workers
        self.model = MultimodalDecoderNetwork(
            n_vocab=n_vocab,
            # Concatenated image and text embeddings
            n_dimensions=n_dimensions,
            n_components=n_components,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(self.momentum, 0.99)
        )
        self.best_loss_train = float("inf")
        self.train_data = None
        self.nn_epoch = None
        self.model = self.model.to(self.device)

    def _cosine_loss(self, true_image_embeddings, pred_image_embeddings):
        cos = nn.CosineSimilarity(dim=1)
        return 1 - cos(true_image_embeddings, pred_image_embeddings)

    def _kl_divergence(
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

    def _reconstruction_loss(self, true_word_dists, pred_word_dists):
        RL = -torch.sum(
            true_word_dists * torch.log(pred_word_dists + 1e-10), dim=1
        )
        return RL

    def _step(self, bow, text_embeddings, image_embeddings):
        self.model.zero_grad()
        # Concatenating image and text features for the encoder network
        embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        (
            prior_mean,
            prior_variance,
            mu,
            sigma,
            log_sigma,
            bow_reconstruction,
            image_reconstruction,
            text_reconstruction,
            thetas_no_drop,
        ) = self.model(embeddings)
        bow_loss = self._reconstruction_loss(bow, bow_reconstruction)
        img_loss = self._cosine_loss(image_embeddings, image_reconstruction)
        text_loss = self._cosine_loss(text_embeddings, text_reconstruction)
        # print("sigma: ", sigma.size())
        # print("log_sigma: ", log_sigma.size())
        # print("prior_variance:", prior_variance.size())
        # print("mu:", mu.size())
        # print("prior_mean:", prior_mean.size())
        # print("thetas_no_drop:", thetas_no_drop.size())
        kl_prior = self._kl_divergence(
            prior_mean,
            prior_variance,
            mu,
            sigma,
            log_sigma,
        )
        loss = bow_loss + img_loss + text_loss + kl_prior
        loss = loss.sum()
        loss.backward()
        self.optimizer.step()
        return loss

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            bow = batch_samples["bow"]
            bow = bow[:, 0, :]
            text_embeddings = batch_samples["text_embeddings"]
            image_embeddings = batch_samples["image_embeddings"]
            bow = bow.to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            image_embeddings = image_embeddings.to(self.device)
            loss = self._step(bow, text_embeddings, image_embeddings)
            samples_processed += bow.size()[0]
            train_loss += loss.item()
        train_loss /= samples_processed
        return samples_processed, train_loss

    def fit(
        self,
        bow: scipy.sparse.csr.csr_matrix,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
    ):
        self.train_data = MultimodalDataset(
            bow=bow,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_loader_workers,
        )
        train_loss = 0
        for epoch in trange(
            self.num_epochs, desc="Training epochs for Multimodal ZeroshotTM."
        ):
            sp, train_loss = self._train_epoch(train_loader)

    def _get_theta(self, embeddings: np.ndarray, n_samples: int) -> np.ndarray:
        final_thetas = []
        for sample_index in trange(
            n_samples, desc="Estimating topic distribution for samples."
        ):
            with torch.no_grad():
                collect_theta = []
                for batch_start in range(
                    0, embeddings.shape[0], self.batch_size
                ):
                    embedding_batch = embeddings[
                        batch_start : batch_start + self.batch_size
                    ]
                    embedding_batch = embedding_batch.to(self.device)
                    self.model.zero_grad()
                    thetas = self.model.get_theta(embedding_batch)
                    collect_theta.extend(thetas.detach().cpu().numpy())
                final_thetas.append(np.array(collect_theta))
        return np.sum(final_thetas, axis=0) / n_samples

    def get_doc_topic_distribution(
        self,
        text_embeddings: Optional[np.ndarray] = None,
        image_embeddings: Optional[np.ndarray] = None,
        n_samples: int = 20,
    ):
        text_embeddings = (
            torch.FloatTensor(text_embeddings)
            if text_embeddings is not None
            else None
        )
        image_embeddings = (
            torch.FloatTensor(image_embeddings)
            if image_embeddings is not None
            else None
        )
        if (text_embeddings is not None) and (image_embeddings is not None):
            doc_topic_dist = self._get_theta(
                torch.cat((text_embeddings, image_embeddings), dim=1),
                n_samples,
            )
            return doc_topic_dist
        elif text_embeddings is not None:
            # Concatenating embeddings with themselves when the other is missing
            return self._get_theta(
                torch.cat((text_embeddings, text_embeddings), dim=1), n_samples
            )
        elif image_embeddings is not None:
            return self._get_theta(
                torch.cat((image_embeddings, image_embeddings), dim=1),
                n_samples,
            )
        else:
            raise TypeError(
                "Image and text embeddings are both None, can't calculate doc-topic proportions."
            )

    @property
    def components_(self):
        return self.model.topic_word_matrix.cpu().detach().numpy()


class MultimodalContrastiveTM:
    def __init__(
        self,
        n_vocab: int,
        n_dimensions: int,
        n_components: int,
        hidden_sizes: Sequence[int] = (100, 100),
        dropout=0.2,
        batch_size=16,
        lr=2e-3,
        momentum=0.99,
        num_epochs=100,
        num_data_loader_workers=mp.cpu_count(),
        device: str = "cpu",
    ):
        self.device = device
        self.n_vocab = n_vocab
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.n_dimensions = n_dimensions
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.num_data_loader_workers = num_data_loader_workers
        self.model = ContrastiveMultimodalDecoderNetwork(
            n_vocab=n_vocab,
            n_dimensions=n_dimensions,
            n_components=n_components,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, betas=(self.momentum, 0.99)
        )
        self.best_loss_train = float("inf")
        self.train_data = None
        self.nn_epoch = None
        self.model = self.model.to(self.device)

    def _kl_divergence(
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

    def _reconstruction_loss(self, true_word_dists, pred_word_dists):
        RL = -torch.sum(
            true_word_dists * torch.log(pred_word_dists + 1e-10), dim=1
        )
        return RL

    def _infoNCE_loss(self, embeddings1, embeddings2, temperature=0.07):
        batch_size = embeddings1.shape[0]
        labels = torch.arange(batch_size)
        labels = torch.cat([labels, labels])
        embeddings_cat = torch.cat([embeddings1, embeddings2])
        loss_func = NTXentLoss()
        infonce_loss = loss_func(embeddings_cat, labels)
        return infonce_loss

    def _step(self, bow, text_embeddings, image_embeddings):
        self.model.zero_grad()
        (
            prior_mean,
            prior_variance,
            text_mu,
            text_sigma,
            text_log_sigma,
            image_mu,
            image_sigma,
            image_log_sigma,
            word_dist,
            thetas,
        ) = self.model(text_embeddings, image_embeddings)
        bow_loss = self._reconstruction_loss(bow, word_dist)
        kl_text_prior = self._kl_divergence(
            prior_mean,
            prior_variance,
            text_mu,
            text_sigma,
            text_log_sigma,
        )
        kl_image_prior = self._kl_divergence(
            prior_mean,
            prior_variance,
            image_mu,
            image_sigma,
            image_log_sigma,
        )
        kl_text_image = self._kl_divergence(
            text_mu,
            text_sigma,
            image_mu,
            image_sigma,
            image_log_sigma,
        )
        # Contrastive loss between image-topic and text-topic distributions
        infoNCE_en_image = self._infoNCE_loss(thetas[0], thetas[1])
        loss = (
            bow_loss
            + kl_text_prior
            + kl_image_prior
            + kl_text_image
            + 100 * infoNCE_en_image
        )
        loss = loss.sum()
        loss.backward()
        self.optimizer.step()
        return loss

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            bow = batch_samples["bow"]
            bow = bow[:, 0, :]
            text_embeddings = batch_samples["text_embeddings"]
            image_embeddings = batch_samples["image_embeddings"]
            bow = bow.to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            image_embeddings = image_embeddings.to(self.device)
            loss = self._step(bow, text_embeddings, image_embeddings)
            samples_processed += bow.size()[0]
            train_loss += loss.item()
        train_loss /= samples_processed
        return samples_processed, train_loss

    def fit(
        self,
        bow: scipy.sparse.csr.csr_matrix,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
    ):
        self.train_data = MultimodalDataset(
            bow=bow,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_loader_workers,
        )
        train_loss = 0
        for epoch in trange(self.num_epochs, desc="Training epochs for M3L."):
            sp, train_loss = self._train_epoch(train_loader)

    def _get_theta(
        self, embeddings: np.ndarray, n_samples: int, is_image: bool = False
    ) -> np.ndarray:
        final_thetas = []
        for sample_index in trange(
            n_samples,
            desc="Estimating {}-topic distribution for samples.".format(
                "image" if is_image else "document"
            ),
        ):
            with torch.no_grad():
                collect_theta = []
                for batch_start in range(
                    0, embeddings.shape[0], self.batch_size
                ):
                    embedding_batch = embeddings[
                        batch_start : batch_start + self.batch_size
                    ]
                    embedding_batch = embedding_batch.to(self.device)
                    self.model.zero_grad()
                    thetas = self.model.get_theta(
                        embedding_batch, is_image=is_image
                    )
                    collect_theta.extend(thetas.detach().cpu().numpy())
                final_thetas.append(np.array(collect_theta))
        return np.sum(final_thetas, axis=0) / n_samples

    def get_doc_topic_distribution(
        self,
        text_embeddings: Optional[np.ndarray] = None,
        image_embeddings: Optional[np.ndarray] = None,
        n_samples: int = 20,
    ):
        text_embeddings = (
            torch.FloatTensor(text_embeddings)
            if text_embeddings is not None
            else None
        )
        image_embeddings = (
            torch.FloatTensor(image_embeddings)
            if image_embeddings is not None
            else None
        )
        if (text_embeddings is not None) and (image_embeddings is not None):
            text_topic_dist = self._get_theta(
                text_embeddings, n_samples, is_image=False
            )
            image_topic_dist = self._get_theta(
                image_embeddings, n_samples, is_image=True
            )
            return (text_topic_dist + image_topic_dist) / 2
        elif text_embeddings is not None:
            return self._get_theta(text_embeddings, n_samples, is_image=False)
        elif image_embeddings is not None:
            return self._get_theta(image_embeddings, n_samples, is_image=True)
        else:
            raise TypeError(
                "Image and text embeddings are both None, can't calculate doc-topic proportions."
            )

    @property
    def components_(self):
        return self.model.topic_word_matrix.cpu().detach().numpy()

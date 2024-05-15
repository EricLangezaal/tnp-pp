import itertools
import math
import random
from abc import ABC
from functools import partial
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
import gpytorch
import torch
import torch.distributions as td
import pdb

from linear_operator import to_linear_operator
from linear_operator.operators import KroneckerProductLinearOperator

from icicl.networks.kernels import GibbsKernel

from .base import GroundTruthPredictor
from .synthetic import (
    SyntheticGeneratorBimodalInput,
    SyntheticGeneratorUniformInput,
    SyntheticGeneratorUniformInputRandomOffset,
    SyntheticBatch
)

from .on_off_grid import OOTGBatch

KERNEL_TYPES = [
    "eq",
    "matern12",
    "matern32",
    "matern52",
    "noisy_mixture",
    "weakly_periodic",
    "periodic",
    "noisy_periodic_mixture",
    "gibbs_switch",
    "gibbs_random_switch",
    "gibbs_random_switch_and_direction",
]


class GPGeneratorBase(ABC):
    noisy_mixture_long_lengthscale: float = 1.0
    weakly_periodic_period: float = 1.0

    def __init__(
        self,
        *,
        kernel_type: Union[List[str], str],
        min_log10_lengthscale: float,
        max_log10_lengthscale: float,
        noise_std: float,
        out_dim: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_type = kernel_type
        self.min_log10_lengthscale = torch.as_tensor(
            min_log10_lengthscale, dtype=torch.float64
        )
        self.max_log10_lengthscale = torch.as_tensor(
            max_log10_lengthscale, dtype=torch.float64
        )
        self.noise_std = noise_std
        self.out_dim = out_dim

    def set_up_gp(self) -> GroundTruthPredictor:
        # Sample lengthscale
        log10_lengthscale = (
            torch.rand(()) * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )
        lengthscale = 10.0**log10_lengthscale

        if isinstance(self.kernel_type, str):
            kernel_type = self.kernel_type
        else:
            kernel_type = random.choice(self.kernel_type)

        if kernel_type == "eq":
            kernel = gpytorch.kernels.RBFKernel()
            kernel.lengthscale = lengthscale
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "matern12":
            kernel = gpytorch.kernels.MaternKernel(nu=0.5)
            kernel.lengthscale = lengthscale
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "matern32":
            kernel = gpytorch.kernels.MaternKernel(nu=1.5)
            kernel.lengthscale = lengthscale
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "matern52":
            kernel = gpytorch.kernels.MaternKernel(nu=2.5)
            kernel.lengthscale = lengthscale
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "noisy_mixture":
            kernel1 = gpytorch.kernels.RBFKernel()
            kernel1.lengthscale = lengthscale
            kernel2 = gpytorch.kernels.RBFKernel()
            kernel2.lengthscale = self.noisy_mixture_long_lengthscale

            kernel = kernel1 + kernel2
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "weakly_periodic":
            kernel1 = gpytorch.kernels.RBFKernel()
            kernel1.lengthscale = lengthscale
            kernel2 = gpytorch.kernels.PeriodicKernel()
            kernel2.period_length = self.weakly_periodic_period
            kernel = kernel1 + kernel2
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "periodic":
            kernel = gpytorch.kernels.PeriodicKernel()
            kernel.period_length = lengthscale
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "noisy_periodic_mixture":
            kernel1 = gpytorch.kernels.PeriodicKernel()
            kernel1.period_length = lengthscale
            kernel2 = gpytorch.kernels.RBFKernel()
            kernel2.lengthscale = self.noisy_mixture_long_lengthscale
            kernel = kernel1 + kernel2
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "gibbs_switch":
            lengthscale_fn = partial(switching_lengthscale_fn, torch.as_tensor(0.0))
            kernel = GibbsKernel(lengthscale_fn=lengthscale_fn)
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std)

        elif kernel_type == "gibbs_random_switch":
            kernels = [
                GibbsKernel(
                    lengthscale_fn=partial(
                        switching_lengthscale_fn, torch.as_tensor(x0)
                    )
                )
                for x0 in [-1.0, 0.0, 1.0]
            ]
            gt_pred = MixtureGPGroundTruthPredictor(
                kernels=kernels, noise_std=self.noise_std
            )
        elif kernel_type == "gibbs_random_switch_and_direction":
            kernels = [
                GibbsKernel(
                    lengthscale_fn=partial(
                        switching_lengthscale_and_direction_fn,
                        torch.as_tensor(x0),
                        direction,
                    )
                )
                for (x0, direction) in itertools.product(
                    [-1.0, 0.0, 1.0], [True, False]
                )
            ]
            gt_pred = MixtureGPGroundTruthPredictor(
                kernels=kernels, noise_std=self.noise_std
            )        

        else:
            raise ValueError("Unknown kernel type.")
        
        if self.out_dim > 1:
            assert isinstance(gt_pred, GPGroundTruthPredictor)
            gt_pred = GPGroundTruthPredictor(kernel=kernel, noise_std=self.noise_std, out_dim=self.out_dim)

        return gt_pred


class GPGenerator(GPGeneratorBase):
    def sample_outputs(
        self,
        x: torch.Tensor,
        num_offtg: Optional[int] = None 
    ) -> Tuple[torch.Tensor, GroundTruthPredictor]:
        """Sample context and target outputs, given the inputs `x`.

        Arguments:
            x: Tensor of shape (batch_size, num_ctx + num_trg, dim) containing
                the context and target inputs.

        Returns:
            y: Tensor of shape (batch_size, num_ctx + num_trg, 1) containing
                the context and target outputs.
        """

        # Set up GP kernel
        gt_pred = self.set_up_gp()
        if num_offtg is not None:
            y = gt_pred.sample_outputs(x, num_offtg=num_offtg)
        else:
            y = gt_pred.sample_outputs(x)

        return y.to(torch.float32), gt_pred


class RandomScaleGPGenerator(GPGenerator, SyntheticGeneratorUniformInput):
    pass


class RandomScaleGPGeneratorRandomOffset(
    GPGenerator, SyntheticGeneratorUniformInputRandomOffset
):
    pass


class RandomScaleGPGeneratorBimodalInput(GPGenerator, SyntheticGeneratorBimodalInput):
    pass


class GPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(self, kernel: gpytorch.kernels.Kernel, noise_std: float, out_dim: int = 1):
        self.kernel = kernel
        self.noise_std = noise_std
        self.out_dim = out_dim

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
        batch: SyntheticBatch = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        dtype = xc.dtype

        xc = xc.to(torch.float64)
        yc = yc.to(torch.float64)
        xt = xt.to(torch.float64)
        num_ctx = xc.shape[-2]

        if self.out_dim > 1:
            return self.__call_mult_dim(batch=batch, xc=xc, yc=yc, xt=xt, yt=yt)

        x = torch.cat((xc, xt), dim=-2)
        with torch.no_grad():
            kxx = self.kernel.to(x.device)(x).evaluate()

        kxx += (
            torch.eye(x.shape[-2], dtype=torch.float64).to(x.device)
            * self.noise_std**2.0
        )

        kcc = kxx[:, :num_ctx, :num_ctx]
        kct = kxx[:, :num_ctx, num_ctx:]
        ktc = kxx[:, num_ctx:, :num_ctx]
        ktt = kxx[:, num_ctx:, num_ctx:]

        mean = (ktc @ torch.linalg.solve(kcc, yc))[  # pylint: disable=not-callable
            ..., 0
        ]
        cov = ktt - ktc @ torch.linalg.solve(kcc, kct)  # pylint: disable=not-callable
        std = torch.diagonal(cov, dim1=-2, dim2=-1).sqrt()

        if yt is not None:
            yt = yt.to(torch.float64)
            gt_loglik = td.Normal(loc=mean, scale=std).log_prob(yt[..., 0])
            gt_loglik = gt_loglik.sum(-1)
            gt_loglik = gt_loglik.to(dtype)

        else:
            gt_loglik = None

        mean = mean.to(dtype)[:, :, None]
        std = std.to(dtype)[:, :, None]

        return mean, std, gt_loglik      
    
    def __call_mult_dim(self, batch: OOTGBatch, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor,yt: Optional[torch.Tensor] = None):
        xc, yc, xt = xc.squeeze(), yc.squeeze(), xt.squeeze()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self.noise_std ** 2

        # labelling technically doesn't matter, as long as off grid context and target have same label.
        xc_labels = torch.concat((torch.zeros(batch.xc_off_grid.shape[-2], 1), torch.ones(batch.xc_on_grid.shape[-2], 1))).to(torch.long)
        model = MultitaskGPModel((xc, xc_labels), yc, self.kernel, likelihood, out_dim=self.out_dim)
        model.eval()
        
        outputDist = likelihood(model(xt, torch.zeros(len(xt), 1, dtype=torch.long)))

        gt_loglik = None
        if yt is not None:
            gt_loglik = outputDist.log_prob(yt.squeeze()).sum().to(xt.dtype)

        return outputDist.mean.unsqueeze(0), outputDist.stddev.unsqueeze(0), gt_loglik
    

    def sample_outputs(self, x: torch.Tensor, num_offtg: Optional[int] = None) -> torch.Tensor:
        kernel = self.kernel.to(x).to(torch.float64)

        with torch.no_grad():
            kxx = kernel(x.to(torch.float64)).evaluate()

        if self.out_dim > 1:
            x_labels = torch.zeros(x.shape[-2], 1, dtype=torch.long)
            x_labels[num_offtg:] = 1
            with torch.no_grad():
                covar_i = gpytorch.kernels.IndexKernel(num_tasks=self.out_dim)(x_labels)
                kxx = kxx.mul(covar_i).evaluate()
        
        kxx += torch.eye(kxx.shape[-1], dtype=torch.float64) * self.noise_std ** 2.0
        if kxx.numel() > 0:
            py = td.MultivariateNormal(
                loc=torch.zeros(kxx.shape[:-1], dtype=torch.float64),
                covariance_matrix=kxx,
            )
            y = py.sample().unsqueeze(-1)
        else:
            y = torch.zeros((*kxx.shape[:-1], 1), dtype=torch.float64)

        return y.to(torch.float32)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood, out_dim):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.task_module = gpytorch.kernels.IndexKernel(num_tasks=out_dim)

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_module(i)
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


class MixtureGPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(self, kernels: gpytorch.kernels.Kernel, noise_std: float):
        self.kernels = kernels
        self.noise_std = noise_std

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        dtype = xc.dtype

        xc = xc.to(torch.float64)
        yc = yc.to(torch.float64)
        xt = xt.to(torch.float64)
        num_ctx = xc.shape[-2]

        x = torch.cat((xc, xt), dim=-2)

        pyts = []
        marginal_logliks = []
        gt_logliks = []
        for kernel in self.kernels:
            with torch.no_grad():
                kxx = kernel.to(x.device)(x).evaluate()

            kxx += (
                torch.eye(x.shape[-2], dtype=torch.float64).to(x.device)
                * self.noise_std**2.0
            )

            kcc = kxx[:, :num_ctx, :num_ctx]
            kct = kxx[:, :num_ctx, num_ctx:]
            ktc = kxx[:, num_ctx:, :num_ctx]
            ktt = kxx[:, num_ctx:, num_ctx:]

            # First compute marginal log-likelihoods of context set.
            kcc_inv_yc = torch.linalg.solve(kcc, yc)  # pylint: disable=not-callable
            kcc_logdet = torch.logdet(kcc)
            yc_kcc_inv_yc = (yc.transpose(-1, -2) @ kcc_inv_yc)[..., 0, 0]

            marginal_loglik = -0.5 * (
                yc_kcc_inv_yc + kcc_logdet + num_ctx * math.log(2 * math.pi)
            )

            # Now compute predictive distributions.
            mean = (ktc @ kcc_inv_yc)[..., 0]
            cov = ktt - ktc @ torch.linalg.solve(  # pylint: disable=not-callable
                kcc, kct
            )
            std = torch.diagonal(cov, dim1=-2, dim2=-1).sqrt()
            pyt = td.Normal(loc=mean, scale=std)

            if yt is not None:
                yt = yt.to(torch.float64)
                gt_loglik = pyt.log_prob(yt[..., 0])
                gt_loglik = gt_loglik.sum(-1)
                gt_loglik = gt_loglik.to(dtype)

            else:
                gt_loglik = None

            pyts.append(pyt)
            marginal_logliks.append(marginal_loglik)
            gt_logliks.append(gt_loglik)

        # Now compute marginal predictions and gt_logliks.
        posterior_probs = torch.stack(marginal_logliks, dim=-1).softmax(dim=-1)
        if yt is not None:
            gt_loglik = torch.logsumexp(
                (torch.stack(gt_logliks, dim=-1) + posterior_probs.log()), dim=-1
            )
        else:
            gt_loglik = None
        means = torch.stack([pyt.mean for pyt in pyts], dim=-1)
        variances = torch.stack([pyt.variance for pyt in pyts], dim=-1)
        marginal_mean = (means * posterior_probs[..., None, :]).sum(-1)
        marginal_var = ((variances + means**2) * posterior_probs[..., None, :]).sum(
            -1
        ) - marginal_mean**2

        marginal_mean = marginal_mean.to(dtype)[:, :, None]
        marginal_var = marginal_var.to(dtype)[:, :, None]

        return marginal_mean, marginal_var**0.5, gt_loglik

    def sample_outputs(self, x: torch.Tensor) -> torch.Tensor:
        # First, sample kernel.
        kernel = random.choice(self.kernels).to(x)

        # Set up covariance at input locations
        with torch.no_grad():
            kxx = kernel(x.to(torch.float64)).evaluate()

        kxx += torch.eye(kxx.shape[-1], dtype=torch.float64) * self.noise_std**2.0

        # Sample from GP with zero mean and covariance kxx.
        py = td.MultivariateNormal(
            loc=torch.zeros(kxx.shape[:-1], dtype=torch.float64), covariance_matrix=kxx
        )
        y = py.sample().unsqueeze(-1)

        return y.to(torch.float32)


def switching_lengthscale_fn(x0: torch.Tensor, x: torch.Tensor):
    return torch.where(
        x[..., :1] < x0.to(x),
        torch.ones(*x[..., :1].shape).to(x) * 4.0,
        torch.ones(*x[..., :1].shape).to(x) * 0.1,
    )


def switching_lengthscale_and_direction_fn(
    x0: torch.Tensor, direction: bool, x: torch.Tensor
):
    if direction:
        return torch.where(
            x[..., :1] < x0.to(x),
            torch.ones(*x[..., :1].shape).to(x) * 4.0,
            torch.ones(*x[..., :1].shape).to(x) * 0.1,
        )
    return torch.where(
        x[..., :1] > x0.to(x),
        torch.ones(*x[..., :1].shape).to(x) * 4.0,
        torch.ones(*x[..., :1].shape).to(x) * 0.1,
    )

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

from icicl.networks.kernels import GibbsKernel

from .base import GroundTruthPredictor
from .synthetic import (
    SyntheticGeneratorBimodalInput,
    SyntheticGeneratorUniformInput,
    SyntheticGeneratorUniformInputRandomOffset,
)

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
            multi_kernel = gpytorch.kernels.MultitaskKernel(
                kernel, num_tasks=self.out_dim, rank=1
            )
            gt_pred = GPGroundTruthPredictor(multi_kernel, noise_std=self.noise_std)

        return gt_pred


class GPGenerator(GPGeneratorBase):
    def sample_outputs(
        self,
        x: torch.Tensor,
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
    def __init__(self, kernel: gpytorch.kernels.Kernel, noise_std: float):
        self.kernel = kernel
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

        if isinstance(self.kernel, gpytorch.kernels.MultitaskKernel):
            return self.__call_mult_dim(xc=xc, yc=yc, xt=xt, yt=yt)

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
    
    def __call_mult_dim(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor,yt: Optional[torch.Tensor] = None):
        target_len = xt.shape[-2]

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.kernel.num_tasks)

        model = MultitaskGPModel(xc, yc, self.kernel, likelihood)

        # with torch.enable_grad():
        #     model.train()
        #     likelihood.train()
        #     # Use the adam optimizer
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        #     # "Loss" for GPs - the marginal log likelihood
        #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        #     for _ in tqdm(range(50)):
        #         optimizer.zero_grad()
        #         output = model(xc)
        #         loss = -mll(output, yc)
        #         loss.backward()
        #         optimizer.step()

        model.eval()
        outputDist = model(xt)

        gt_loglik = None
        if yt is not None:
            # only first dimension of GP is related to targets since off grid
            gt_loglik = outputDist.log_prob(yt)[..., 0].sum(-1).to(xt.dtype)

        return outputDist.loc[..., :target_len], outputDist.stddev[..., 0], gt_loglik
    

    def sample_outputs(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(x).to(torch.float64)

        # Set up covariance at input locations
        with torch.no_grad():
            kxx = kernel(x.to(torch.float64)).evaluate()

        kxx += torch.eye(kxx.shape[-1], dtype=torch.float64) * self.noise_std**2.0

        # Sample from GP with zero mean and covariance kxx.
        if kxx.numel() > 0:
            if isinstance(kernel, gpytorch.kernels.MultitaskKernel):
                y = gpytorch.distributions.MultitaskMultivariateNormal(
                    mean = torch.zeros(x.shape[:-1] + (kernel.num_tasks,), dtype=torch.float64),
                    covariance_matrix=kxx,
                ).sample()
            else:
                py = td.MultivariateNormal(
                    loc=torch.zeros(kxx.shape[:-1], dtype=torch.float64),
                    covariance_matrix=kxx,
                )
                y = py.sample().unsqueeze(-1)
        else:
            y = torch.zeros((*kxx.shape[:-1], 1), dtype=torch.float64)

        return y.to(torch.float32)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=kernel.num_tasks
        )
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


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

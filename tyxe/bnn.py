from collections import defaultdict
import itertools
from operator import itemgetter

import torch

import pyro.nn as pynn
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC, JitTrace_ELBO, JitTraceMeanField_ELBO


from . import util


__all__ = ["PytorchBNN", "VariationalBNN", "MCMC_BNN"]


def _empty_guide(*args, **kwargs):
    return {}


def _as_tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return x,


def _to(x, device):
    return map(lambda t: t.to(device) if device is not None else t, _as_tuple(x))


class _BNN(pynn.PyroModule):
    """BNN base class that takes an nn.Module, turns it into a PyroModule and applies a prior to it, i.e. replaces
    nn.Parameter attributes by PyroSamples according to the specification in the prior. The forward method wraps the
    forward pass of the net and samples weights from the prior distributions.

    :param nn.Module net: pytorch neural network to be turned into a BNN.
    :param prior tyxe.priors.Prior: prior object that specifies over which parameters we want uncertainty.
    :param str name: base name for the BNN PyroModule."""

    def __init__(self, net, prior, name=""):
        super().__init__(name)

        self.net = net
        pynn.module.to_pyro_module_(self.net)
        self.prior = prior
        self.prior.apply_(self.net)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def update_prior(self, new_prior):
        """Uppdates the prior of the network, i.e. calls its update_ method on the net.

        :param tyxe.priors.Prior new_prior: Prior for replacing the previous prior, i.e. substituting the PyroSample
            attributes of the net."""
        self.prior = new_prior
        self.prior.update_(self.net)


class PytorchBNN(_BNN):
    """Low-level variational BNN class that can serve as a drop-in replacement for an nn.Module.

    :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO as the loss, i.e. calculate KL
        divergences in closed form or via a Monte Carlo approximate of the difference of log densities between
        variational posterior and prior."""

    def __init__(self, net, prior, guide_builder=None, name="", closed_form_kl=True):
        super().__init__(net, prior, name=name)
        self.net_guide = guide_builder(self.net) if guide_builder is not None else _empty_guide
        self.cached_output = None
        self.cached_kl_loss = None
        self._loss = TraceMeanField_ELBO() if closed_form_kl else Trace_ELBO()

    def named_pytorch_parameters(self, *input_data):
        """Equivalent of the named_parameters method of an nn.Module. Ensures that prior and guide are run once to
        initialize all pyro parameters. Those are then collected and returned via the trace poutine."""
        model_trace = poutine.trace(self.net, param_only=True).get_trace(*input_data)
        guide_trace = poutine.trace(self.net_guide, param_only=True).get_trace(*input_data)
        for name, msg in itertools.chain(model_trace.nodes.items(), guide_trace.nodes.items()):
            yield name, msg["value"].unconstrained()

    def pytorch_parameters(self, input_data_or_fwd_fn):
        yield from map(itemgetter(1), self.named_pytorch_parameters(input_data_or_fwd_fn))

    def cached_forward(self, *args, **kwargs):
        # cache the output of forward to make it effectful, so that we can access the output when running forward with
        # posterior rather than prior samples
        self.cached_output = super().forward(*args, **kwargs)
        return self.cached_output

    def forward(self, *args, **kwargs):
        self.cached_kl_loss = self._loss.differentiable_loss(self.cached_forward, self.net_guide, *args, **kwargs)
        return self.cached_output


class _SupervisedBNN(_BNN):
    """
    Base class for supervised BNNs that defines
    the interface of the predict method and implements evaluate.
    Importantly, this class is agnostic to the kind of inference performed.

    :param tyxe.likelihoods.Likelihood likelihood: Likelihood object that implements a forward method including
        a pyro.sample statement for labelled data given neural network predictions and implements logic for aggregating
        multiple predictions and evaluating them."""

    def __init__(self, net, prior, likelihood, name=""):
        super().__init__(net, prior, name=name)
        self.likelihood = likelihood

    def model(self, x, obs=None, anneal_factor: float=1.0):
        assert anneal_factor > 0.0, anneal_factor
        with poutine.scale(scale=anneal_factor):
            predictions = self(*_as_tuple(x))
        predictions = self.likelihood(predictions, obs)
        return predictions

    def evaluate(self, input_data, y, num_predictions=1, aggregate=True, reduction="sum", return_type=tuple):
        """"
        Utility method for evaluation. Calculates a likelihood-dependent errors measure, e.g.
        squared errors, in case self.likelihood is Gaussian (Regression)
        classifications errors, in case self.likelihood is Discrete (Classification)

        :param input_data: Inputs to the neural net. Must be a tuple of more than one.
        :param y: observations, e.g. class labels in case of classification or N scalars in case of regression.
        :param int num_predictions: number of forward passes with different weight samples to do.
        :param bool aggregate: whether to aggregate the outputs of the forward passes before evaluating.
        :param str reduction: "sum", "mean" or "none". How to process the tensor of errors. "sum" adds them up,
            "mean" averages them and "none" simply returns the tensor.
        """
        self.eval()
        with torch.no_grad():
                # only forward through net
                net_predictions = self.predict(
                    *_as_tuple(input_data),
                    num_predictions=num_predictions,
                    aggregate=aggregate,
                    net_only=True
                )
                aleatoric_uncertainty = self.likelihood.aleatoric_uncertainty(net_predictions)
                epistemic_uncertainty = self.likelihood.epistemic_uncertainty(net_predictions)

                # predictive variance should be = aleatoric + epistemic uncertainty
                predictive_variance = self.likelihood.aggregate_predictions(net_predictions)[-1]

                nll = - self.likelihood.log_likelihood(net_predictions, y.unsqueeze(0), reduction=reduction)

                # now forward through likelihood for error
                sampled_predictions = torch.stack([self.likelihood_fwd(pred) for pred in net_predictions])

                error = self.likelihood.error(sampled_predictions, y.unsqueeze(0), reduction=reduction, sample=False)

        self.train()

        if return_type==dict:
            return {
                "error": error,
                "nll": nll,
                "predictive_variance": predictive_variance,
                "aleatoric_uncertainty": aleatoric_uncertainty,
                "epistemic_uncertainty": epistemic_uncertainty,
            }
        else:
            return error, nll, predictive_variance, aleatoric_uncertainty, epistemic_uncertainty



    def likelihood_fwd(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *input_data, num_predictions=1, aggregate=True, net_only=False):
        """Makes predictions on the input data

        :param input_data: inputs to the neural net, e.g. torch.Tensors
        :param int num_predictions: number of forward passes through the net
        :param bool aggregate: whether to aggregate the predictions depending on the likelihood, e.g. averaging them."""
        raise NotImplementedError


class VariationalBNN(_SupervisedBNN):
    """
    Variational BNN class for supervised problems. Requires a likelihood that describes the data noise and an
    optional guide builder for it should it contain any variables that need to be inferred. Provides high-level utility
    method such as fit, predict and

    :param callable guide_builder: callable that takes a probabilistic pyro function with sample statements and returns
        an object that helps with inference, i.e. a callable guide function that samples from an approximate posterior
        for variational BNNs or an MCMC kernel for MCMC-based BNNs. May be None for maximum likelihood inference if
        the prior leaves all parameters of the net as such.
    :param callable net_guide_builder: pyro.infer.autoguide.AutoCallable style class that given a pyro function
        constructs a variational posterior that sample the same unobserved sites from distributions with learnable
        parameters.
    :param callable likelihood_guide_builder: optional callable that constructs a guide for the likelihood if it
        contains any unknown variable, such as the precision/scale of a Gaussian.
    """
    def __init__(self, net, prior, likelihood, net_guide_builder=None, likelihood_guide_builder=None, name=""):
        super().__init__(net, prior, likelihood, name=name)
        self.net_guide = net_guide_builder(self.net) if net_guide_builder is not None else _empty_guide
        self.likelihood_fwd = self.guided_likelihood

        weight_sample_sites = list(util.pyro_sample_sites(self.net))
        if likelihood_guide_builder is not None:
            # self.likelihood_guide = likelihood_guide_builder(poutine.block(
            #     self.model, hide=weight_sample_sites + [self.likelihood.data_name]))
            self.likelihood_guide = likelihood_guide_builder(poutine.block(
                    self.likelihood, hide=[self.likelihood.data_name]
            ))
        else:
            self.likelihood_guide = _empty_guide

    def guide(self, x, obs=None, anneal_factor: float=1.0):
        assert anneal_factor > 0.0, anneal_factor
        with poutine.scale(scale=anneal_factor):
            result = self.net_guide(*_as_tuple(x)) or {}
            result.update(self.likelihood_guide(*_as_tuple(x), obs) or {})
        return result

    def guided_forward(self, *args, guide_tr=None, **kwargs):
        if guide_tr is None:
            guide_tr = poutine.trace(self.net_guide).get_trace(*args, **kwargs)
        return poutine.replay(self.net, trace=guide_tr)(*args, **kwargs)

    def fit(
            self,
            data_loader,
            optim,
            num_epochs,
            callback=None,
            num_particles=1,
            closed_form_kl=True,
            device=None,
            kl_schedule=lambda step_n: 1.0,
            jit=False,
            **loss_kwargs,
        ):
        """Optimizes the variational parameters on data from data_loader using optim for num_epochs.

        :param Iterable data_loader: iterable over batches of data, e.g. a torch.utils.data.DataLoader. Assumes that
            each element consists of a length two tuple of list, with the first element either containing a single
            object or a list of objects, e.g. torch.Tensors, that are the inputs to the neural network. The second
            element is a single torch.Tensor e.g. of class labels.
        :param optim: pyro optimizer to be used for constructing an SVI object, e.g. pyro.optim.Adam({"lr": 1e-3}).
        :param int num_epochs: number of passes over data_loader.
        :param callable callback: optional function to invoke after every training epoch. Receives the BNN object,
            the epoch number and the average value of the ELBO over the epoch. May return True to terminate
            optimization before num_epochs, e.g. if it finds that a validation log likelihood saturates.
        :param int num_particles: number of MC samples for estimating the ELBO.
        :param bool closed_form_kl: whether to use TraceMeanField_ELBO or Trace_ELBO, i.e. calculate KL divergence
            between approximate posterior and prior in closed form or via a Monte Carlo estimate.
        :param torch.device device: optional device to send the data to.
        :param loss_kwargs: kwargs to give to loss during init
        :param kl_schedule: function from natural numbers to [0.0, 1.0] that implements a KL annealing schedule.
            Output is multiplied onto guide loss. Input is step number, starting at 1.
        :param jit: use jit version of loss?
        """
        old_training_state = self.net.training
        self.net.train(True)

        loss = (JitTraceMeanField_ELBO if jit else TraceMeanField_ELBO) \
            if closed_form_kl else \
            (JitTrace_ELBO if jit else Trace_ELBO)
        svi = SVI(self.model, self.guide, optim, loss=loss(num_particles=num_particles, **loss_kwargs))

        step = 1
        for i in range(num_epochs):
            elbo = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                elbo += svi.step(tuple(_to(input_data, device)), tuple(_to(observation_data, device))[0],
                        anneal_factor=kl_schedule(step))
                step += 1

            # the callback can stop training by returning True
            if callback is not None and callback(self, i, elbo / num_batch):
                break

        self.net.train(old_training_state)
        return svi

    def predict(self, *input_data, num_predictions=1, aggregate=True, guide_traces=None, net_only=False):
        """
        Args:
            net_only: only forward guided net, not likelihood
        """
        if guide_traces is None:
            guide_traces = [None] * num_predictions

        preds = []
        self.net.eval() # not doing monte carlo dropout
        with torch.no_grad():
            for trace in guide_traces:
                """
                def guided_forward(self, *args, guide_tr=None, **kwargs):
                    if guide_tr is None:
                        guide_tr = poutine.trace(self.net_guide).get_trace(*args, **kwargs)
                    return poutine.replay(self.net, trace=guide_tr)(*args, **kwargs)
                """
                pred = self.guided_forward(*input_data, guide_tr=trace)
                if not aggregate and not net_only:
                    pred = self.guided_likelihood(pred)
                preds.append(pred)
        self.net.train()
        predictions = torch.stack(preds)

        """
        def guide(self, x, obs=None, anneal_factor: float=1.0):
            assert anneal_factor > 0.0, anneal_factor
            with poutine.scale(scale=anneal_factor):
                result = self.net_guide(*_as_tuple(x)) or {}
                result.update(self.likelihood_guide(*_as_tuple(x), obs) or {})
            return result
        """

        return self.likelihood.aggregate_predictions(predictions) if \
            (aggregate and not net_only) else predictions

    def guided_likelihood(self, *args, guide_tr=None, **kwargs):
        if guide_tr is None:
            guide_tr = poutine.trace(self.likelihood_guide).get_trace(*args, **kwargs)
        return poutine.replay(self.likelihood, trace=guide_tr)(*args, **kwargs)



class MCMC_BNN(_SupervisedBNN):
    """
    Supervised BNN class with an interface to pyro's MCMC that is unified with the VariationalBNN class.
    """

    def __init__(self, net, prior, likelihood, name=""):
        super().__init__(net, prior, likelihood, name=name)
        self.likelihood = likelihood
        self._mcmc = None
        self.likelihood_fwd = self.likelihood

    def fit(self, data_loader, num_samples, kernel_builder, device=None, batch_data=False, **mcmc_kwargs):
        """Runs MCMC on the data from data loader using the kernel that was used to instantiate the class.

        :param data_loader: iterable or list of batched inputs to the net. If iterable treated like the data_loader
            of VariationalBNN and all network inputs are concatenated via torch.cat. Otherwise must be a tuple of
            a single or list of network inputs and a tensor for the targets.
        :param int num_samples: number of MCMC samples to draw.
        :param callable kernel_builder: function or class that returns an object that will accepted as kernel by
        pyro.infer.mcmc.MCMC, e.g. pyro.infer.mcmc.HMC or NUTS. Will be called with the entire model, i.e. also
        infer variables in the likelihood.
        :param torch.device device: optional device to send the data to.
        :param batch_data: whether to treat data_loader as a full batch of data or an iterable over mini-batches.
        :param dict mcmc_kwargs: keyword arguments for initializing the pyro.infer.mcmc.MCMC object."""

        self.kernel = kernel_builder(self.model)

        if batch_data:
            input_data, observation_data = data_loader
        else:
            input_data_lists = defaultdict(list)
            observation_data_list = []
            for in_data, obs_data in iter(data_loader):
                for i, data in enumerate(_as_tuple(in_data)):
                    input_data_lists[i].append(data.to(device))
                observation_data_list.append(obs_data.to(device))
            input_data = tuple(torch.cat(input_data_lists[i]) for i in range(len(input_data_lists)))
            observation_data = torch.cat(observation_data_list)

        self._mcmc = MCMC(self.kernel, num_samples, **mcmc_kwargs)
        self._mcmc.run(input_data, observation_data)

        return self._mcmc

    def predict(self, *input_data, num_predictions=1, aggregate=True, net_only=False):
        """
        Args:
            net_only: only forward guided net, not likelihood
        """
        if self._mcmc is None:
            raise RuntimeError("Call .fit to run MCMC and obtain samples from the posterior first.")

        preds = []
        weight_samples = self._mcmc.get_samples(num_samples=num_predictions)
        self.net.eval() # not doing monte carlo dropout
        with torch.no_grad():
            for i in range(num_predictions):
                weights = {name: sample[i] for name, sample in weight_samples.items()}
                pred = poutine.condition(self, weights)(*input_data)
                if not aggregate and not net_only:
                    pred = self.likelihood(pred)
                preds.append(pred)

        predictions = torch.stack(preds)
        self.net.train()
        return self.likelihood.aggregate_predictions(predictions) if \
            (aggregate and not net_only) else predictions



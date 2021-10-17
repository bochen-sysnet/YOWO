import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import EntropyModel,GaussianConditional
from compressai.models import CompressionModel
import sys, os, math
sys.path.append('..')
import codec.arithmeticcoding as arithmeticcoding

class RecEntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.
    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
    for an introduction.
    """

    _offset: Tensor

    def __init__(
        self,
        channels,
        tail_mass = 1e-9,
        init_scale = 10,
        filters = (3, 3, 3, 3),
    ):
        super().__init__()

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))
        
        self.lstm_matrix = nn.ModuleList() 
        self.lstm_bias = nn.ModuleList() 
        self.lstm_factor = nn.ModuleList() 
        self.model_states = []
        for i in range(len(self.filters) + 1):
            # build rnn
            kernel = filters[i + 1] * filters[i]
            self.lstm_matrix.append(nn.LSTM(kernel,kernel,1))
            kernel = filters[i + 1]
            self.lstm_bias.append(nn.LSTM(kernel,kernel,1))
            self.lstm_factor.append(nn.LSTM(kernel,kernel,1))
            # initial states
            m_state = torch.zeros(1,channels*2,filters[i + 1] * filters[i]).cuda()
            b_state = torch.zeros(1,channels*2,filters[i + 1]).cuda()
            f_state = torch.zeros(1,channels*2,filters[i + 1]).cuda()
            layer_states = [m_state,b_state,f_state]
            self.model_states.append(layer_states)
        
    def init_state(self):
        return self.model_states

    def _get_medians(self):
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, state, force = False, stopGradient = True):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False
            
        # update matrix, bias and factor with RNN
        filters = (1,) + self.filters + (1,)
        for i in range(len(self.filters) + 1):
            continue
            m_state,b_state,f_state = state[i]
            matrix = getattr(self, f"_matrix{i:d}")
            matrix = matrix.view(1,self.channels,-1)
            if stopGradient:
                matrix = matrix.detach()
            matrix, m_state = self.lstm_matrix[i](matrix, torch.split(m_state.to(matrix.device),self.channels,dim=1)) 
            matrix = matrix.view(self.channels, filters[i + 1], filters[i])
            setattr(self, f"_matrix{i:d}", nn.Parameter(matrix))
            m_state = torch.cat(m_state,dim=1)
            if stopGradient:
                m_state = m_state.detach()

            bias = getattr(self, f"_bias{i:d}")
            bias = bias.view(1,self.channels,-1)
            if stopGradient:
                bias = bias.detach()
            bias, b_state = self.lstm_bias[i](bias, torch.split(b_state.to(bias.device),self.channels,dim=1))
            bias = bias.view(self.channels, filters[i + 1], 1)
            setattr(self, f"_bias{i:d}", nn.Parameter(bias))
            b_state = torch.cat(b_state,dim=1)
            if stopGradient:
                b_state = b_state.detach()

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                factor = factor.view(1,self.channels,-1)
                if stopGradient:
                    factor = factor.detach()
                factor, f_state = self.lstm_factor[i](factor, torch.split(f_state.to(factor.device),self.channels,dim=1)) 
                factor = factor.view(self.channels, filters[i + 1], 1)
                setattr(self, f"_factor{i:d}", nn.Parameter(factor))
                f_state = torch.cat(f_state,dim=1)
                if stopGradient:
                    f_state = f_state.detach()
                    
            state[i] = [m_state,b_state,f_state]

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return state, True

    def loss(self):
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs, stop_gradient):
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
                
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs):
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        likelihood = torch.clip(likelihood, min=self.tail_mass, max=1 - self.tail_mass)
        return likelihood

    def forward(
        self, x, training = None, stopGradient = True
    ):
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            perm = (1, 2, 3, 0)
            inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize

        outputs = self.quantize(
            values, "noise" if training else "dequantize", self._get_medians()
        )

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            # TorchScript not yet supported
            likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)
        
    def get_actual_bits(self, x):
        string = self.compress(x)
        bits_act = torch.FloatTensor([len(b''.join(string))*8]).squeeze(0)
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        return bits_est
        
class RecProbModel(CompressionModel):

    def __init__(self, channels=128):
        super().__init__(channels)

        self.h1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.h2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.lstm = ConvLSTM(channels)
        
        self.gaussian_conditional = GaussianConditional(None)
        
        self.channels = channels
        
        h = w = 224
        self.model_states = torch.zeros(1,self.channels*2,h//16,w//16).cuda()
        
        self.prior_latent = None

    def forward(self, x, hidden, RPM_flag, training):
        if not RPM_flag:
            x_hat,likelihoods = self.entropy_bottleneck(x,training=training)
            return x_hat,likelihoods,hidden.detach()
        assert self.prior_latent is not None, 'prior latent is none!'
        x_hat = self.h1(self.prior_latent)
        x_hat, hidden = self.lstm(x_hat, hidden.to(x.device))
        gaussian_params = self.h2(x_hat)
        scales_hat, means_hat = torch.split(gaussian_params, self.channels, dim=1)
        self.scales_hat,self.means_hat = scales_hat, means_hat
        x, likelihoods = self.gaussian_conditional(x, scales_hat, means=means_hat, training=training)
        tiny = 1e-10
        likelihoods = torch.clip(likelihoods, min=tiny, max=1 - tiny)
        return x, likelihoods, hidden.detach()

    def compress(self, x, RPM_flag):
        if not RPM_flag:
            x_string = self.entropy_bottleneck.compress(x)
            return x_string
        assert self.prior_latent is not None, 'prior latent is none!'
        indexes = self.gaussian_conditional.build_indexes(self.scales_hat)
        x_string = self.gaussian_conditional.compress(x, indexes, means=self.means_hat)
        return x_string
        
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
        
    def init_state(self):
        return self.model_states
        
    def loss(self, RPM_flag):
        if RPM_flag:
            return torch.FloatTensor([0]).squeeze(0).cuda(0)
        return self.aux_loss()
        
    def memorize(self, x_hat):
        self.prior_latent = x_hat.detach()
        
    def get_actual_bits(self, x, RPM_flag):
        x_string = self.compress(x, RPM_flag)
        bits_act = torch.FloatTensor([len(b''.join(x_string))*8]).squeeze(0)
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        return bits_est
        
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
    
# Gaussian Conditional
class RecGaussianConditional(CompressionModel):

    def __init__(self, channels=128):
        super().__init__(channels)

        self.h_a1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.h_a2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.h_s1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.h_s2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.enc_lstm = ConvLSTM(channels)
        self.dec_lstm = ConvLSTM(channels)
        
        self.gaussian_conditional = GaussianConditional(None)
        
        self.channels = channels
        
        h = w = 224
        self.model_states = torch.zeros(1,self.channels*4,h//16,w//16).cuda()

    def forward(self, x, hidden, training):
        state_enc, state_dec = torch.split(hidden.to(x.device),self.channels*2,dim=1)
        z = self.h_a1(x)
        z, state_enc = self.enc_lstm(z, state_enc)
        z = self.h_a2(z)
        z_hat, z_likelihoods = self.entropy_bottleneck(z, training=training)
        z_hat = self.h_s1(z_hat)
        z_hat, state_dec = self.dec_lstm(z_hat, state_dec)
        gaussian_params = self.h_s2(z_hat)
        scales_hat, means_hat = torch.split(gaussian_params, self.channels, dim=1)
        x_hat, x_likelihoods = self.gaussian_conditional(x, scales_hat, means=means_hat, training=training)
        hidden = torch.cat((state_enc, state_dec),dim=1).detach()
        return x_hat, (x_likelihoods,z_likelihoods), hidden

    def compress(self, x, hidden):
        state_enc, state_dec = torch.split(hidden.to(x.device),self.channels*2,dim=1)
        z = self.h_a1(x)
        z, state_enc = self.enc_lstm(z, state_enc)
        z = self.h_a2(z)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.h_s1(z_hat)
        z_hat, state_dec = self.dec_lstm(z_hat, state_dec)
        gaussian_params = self.h_s2(z_hat)
        scales_hat, means_hat = torch.split(gaussian_params, self.channels, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        x_string = self.gaussian_conditional.compress(x, indexes, means=means_hat)
        return (x_string,z_string)
        
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
        
    def init_state(self):
        return self.model_states
        
    def loss(self):
        return torch.FloatTensor([0]).squeeze(0).cuda(0)
        
    def get_actual_bits(self, x, hidden):
        x_string,z_string = self.compress(x, hidden)
        x_bits_act = torch.FloatTensor([len(b''.join(x_string))*8]).squeeze(0)
        z_bits_act = torch.FloatTensor([len(b''.join(z_string))*8]).squeeze(0)
        bits_act = x_bits_act + z_bits_act
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        x_likelihoods,z_likelihoods = likelihoods
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(x_likelihoods.device)
        x_bits_est = torch.sum(torch.log(x_likelihoods)) / (-log2)
        z_bits_est = torch.sum(torch.log(z_likelihoods)) / (-log2)
        bits_est = x_bits_est + z_bits_est
        return bits_est
        
class ConvLSTM(nn.Module):
    def __init__(self, channels=128, forget_bias=1.0, activation=F.relu):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Conv2d(2*channels, 4*channels, kernel_size=3, stride=1, padding=1)
        self._forget_bias = forget_bias
        self._activation = activation
        self._channels = channels

    def forward(self, x, state):
        c, h = torch.split(state,self._channels,dim=1)
        x = torch.cat((x, h), dim=1)
        y = self.conv(x)
        j, i, f, o = torch.split(y, self._channels, dim=1)
        f = torch.sigmoid(f + self._forget_bias)
        i = torch.sigmoid(i)
        c = c * f + i * self._activation(j)
        o = torch.sigmoid(o)
        h = o * self._activation(c)

        return h, torch.cat((c, h),dim=1)
        
class RecProbModel2(EntropyModel):

    _offset: Tensor

    def __init__(
        self,
        channels,
        tail_mass = 1e-9,
        init_scale = 10,
        filters = (3, 3, 3, 3),
    ):
        super().__init__()

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))
        
        self.sigma = self.mu = self.prior_latent = None
        self.RPM = RPM(channels)
        h = w = 224
        self.model_states = torch.zeros(1,self.channels*2,h//16,w//16).cuda()
        
    def init_state(self):
        return self.model_states

    def _get_medians(self):
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force = False, stopGradient = True):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self, RPM_flag):
        if RPM_flag:
            return torch.FloatTensor([0]).squeeze(0).cuda(0)
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs, stop_gradient):
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
                
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs):
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        likelihood = torch.clip(likelihood, min=self.tail_mass, max=1 - self.tail_mass)
        return likelihood

    def forward(
        self, x, rpm_hidden, RPM_flag, training = None
    ):
        if RPM_flag:
            assert self.prior_latent is not None, 'prior latent is none!'
            likelihood, rpm_hidden, self.sigma, self.mu = self.RPM(self.prior_latent, torch.round(x), rpm_hidden)
            self.prior_latent = torch.round(x).detach()
            rpm_hidden = rpm_hidden.detach()
            return self.prior_latent, likelihood, rpm_hidden
        self.prior_latent = torch.round(x).detach()
            
        #if RPM_flag:
        #    assert self.prior_latent is not None, 'prior latent is none!'
        #    likelihood, rpm_hidden, self.sigma, self.mu = self.RPM(self.prior_latent, torch.round(x), rpm_hidden)
        #    return torch.round(x).detach(), likelihood, rpm_hidden.detach()
            
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            perm = (1, 2, 3, 0)
            inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize

        outputs = self.quantize(
            values, "noise" if training else "dequantize", self._get_medians()
        )

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            # TorchScript not yet supported
            likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood, rpm_hidden

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)
        
    def get_actual_bits(self, x, RPM_flag):
        string = self.compress(x)
        bits_act = torch.FloatTensor([len(b''.join(string))*8]).squeeze(0)
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        return bits_est
        
# conditional probability
# predict y_t based on parameters computed from y_t-1
class RPM(nn.Module):
    def __init__(self, channels=128, act=torch.tanh):
        super(RPM, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.lstm = ConvLSTM(channels)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(channels, 2*channels, kernel_size=3, stride=1, padding=1)
        self.channels = channels

    def forward(self, x, x_target, hidden):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #x, hidden = self.lstm(x, hidden.to(x.device))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        sigma_mu = F.relu(self.conv8(x))
        likelihood, sigma, mu = rpm_likelihood(x_target, sigma_mu, self.channels)
        return likelihood, hidden, sigma, mu
        
def rpm_likelihood(x_target, sigma_mu, channels=128, tiny=1e-10):

    sigma, mu = torch.split(sigma_mu, channels, dim=1)

    half = torch.FloatTensor([0.5]).to(x_target.device)

    upper = x_target + half
    lower = x_target - half

    sig = torch.maximum(sigma, torch.FloatTensor([-7.0]).to(x_target.device))
    upper_l = torch.sigmoid((upper - mu))# * (torch.exp(-sig) + tiny))
    lower_l = torch.sigmoid((lower - mu))# * (torch.exp(-sig) + tiny))
    p_element = upper_l - lower_l
    p_element = torch.clip(p_element, min=tiny, max=1 - tiny)

    return p_element, sigma, mu

def test_RPM():
    channels = 128
    net = RecProbModel2(channels)
    #for n, p in net.named_parameters():
    #    print(n,p.size())
    x = torch.rand(1, channels, 14, 14)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in net.named_parameters())
    optimizer = optim.Adam(parameters, lr=1e-4)
    rpm_hidden = net.init_state()
    rpm_flag = True
    x_hat, likelihoods, rpm_hidden = net(x,rpm_hidden,False,training=True)
    train_iter = tqdm(range(0,10000))
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()

        x_hat, likelihoods, rpm_hidden = net(x,rpm_hidden,rpm_flag,training=True)
        
        loss = net.get_estimate_bits(likelihoods)
        mse = torch.mean(torch.pow(x-x_hat,2))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1)
        optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"likelihood: {float(torch.mean(likelihoods)):.4f}. "
            f"loss: {float(loss):.2f}. "
            f"MSE: {float(mse):.2f}. ")
            
def test_EB():
    channels = 128
    from compressai.entropy_models import EntropyBottleneck
    def get_estimate_bits(self, likelihoods):
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        return bits_est
    EntropyBottleneck.get_estimate_bits = get_estimate_bits
    net = EntropyBottleneck(channels)
    x = torch.rand(1, channels, 14, 14)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in net.named_parameters() if "quantiles" not in n)
    optimizer = optim.Adam(parameters, lr=1e-4)
    aux_parameters = set(p for n, p in net.named_parameters() if "quantiles" in n)
    aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
    train_iter = tqdm(range(0,10000))
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()

        x_hat, likelihoods = net(x,training=True)
        
        loss = net.get_estimate_bits(likelihoods)
        mse = torch.mean(torch.pow(x-x_hat,2))

        loss.backward()
        optimizer.step()
        
        aux_loss = net.loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"likelihood: {float(torch.mean(likelihoods)):.4f}. "
            f"loss: {float(loss):.2f}. "
            f"aux_loss: {float(aux_loss):.2f}. "
            f"MSE: {float(mse):.2f}. ")
            
def test_REB():
    channels = 128
    net = RecEntropyBottleneck(channels,'test')
    x = torch.rand(1, channels, 14, 14)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in net.named_parameters() if "quantiles" not in n)
    optimizer = optim.Adam(parameters, lr=1e-4)
    aux_parameters = set(p for n, p in net.named_parameters() if "quantiles" in n)
    aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
    rpm_hidden = net.init_state()
    train_iter = tqdm(range(0,10000))
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()
        
        rpm_hidden,_ = net.update(rpm_hidden, force=True)

        x_hat, likelihoods = net(x,training=True)
        
        loss = net.get_estimate_bits(likelihoods)
        mse = torch.mean(torch.pow(x-x_hat,2))

        loss.backward()
        optimizer.step()
        
        aux_loss = net.loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"likelihood: {float(torch.mean(likelihoods)):.4f}. "
            f"loss: {float(loss):.2f}. "
            f"aux_loss: {float(aux_loss):.2f}. "
            f"MSE: {float(mse):.2f}. ")
            
def test_RGC():
    channels = 128
    net = RecGaussianConditional(channels)
    x = torch.rand(1, channels, 14, 14)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in net.named_parameters() if "quantiles" not in n)
    optimizer = optim.Adam(parameters, lr=1e-4)
    rpm_hidden = net.init_state()
    train_iter = tqdm(range(0,10000))
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()

        x_hat, likelihoods, rpm_hidden = net(x,rpm_hidden,training=True)
        
        loss = net.get_estimate_bits(likelihoods)
        mse = torch.mean(torch.pow(x-x_hat,2))

        loss.backward()
        optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"likelihood1: {float(torch.mean(likelihoods[0])):.4f}. "
            f"likelihood2: {float(torch.mean(likelihoods[1])):.4f}. "
            f"loss: {float(loss):.2f}. "
            f"MSE: {float(mse):.2f}. ")
        
if __name__ == '__main__':
    test_RPM()
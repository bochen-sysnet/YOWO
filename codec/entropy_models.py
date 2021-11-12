import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import EntropyModel,GaussianConditional,EntropyBottleneck
from compressai.models import CompressionModel
import sys, os, math, time
sys.path.append('..')
import codec.arithmeticcoding as arithmeticcoding
        
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
    
# each module should include encoding/decoding time
class RecProbModel(CompressionModel):

    def __init__(
        self,
        channels,
    ):
        super().__init__(channels)

        self.channels = int(channels)
        
        self.sigma = self.mu = self.prior_latent = None
        self.RPM = RPM(channels)
        h = w = 224
        self.gaussian_conditional = GaussianConditional(None)
        
    def set_RPM(self, RPM_flag):
        self.RPM_flag = RPM_flag
        
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def loss(self):
        if self.RPM_flag:
            return torch.FloatTensor([0]).squeeze(0).cuda(0)
        return self.aux_loss()

    def forward(
        self, x, rpm_hidden, training = None
    ):
        if self.RPM_flag:
            assert self.prior_latent is not None, 'prior latent is none!'
            rpm_in = self.prior_latent
            self.sigma, self.mu, rpm_hidden = self.RPM(rpm_in, rpm_hidden.to(x.device))
            self.sigma = torch.maximum(self.sigma, torch.FloatTensor([-7.0]).to(x.device))
            self.sigma = torch.exp(self.sigma)/10
            x_hat,likelihood = self.gaussian_conditional(x, self.sigma, means=self.mu, training=training)
            rpm_hidden = rpm_hidden.detach()
        else:
            x_hat,likelihood = self.entropy_bottleneck(x,training=training)
        # self.prior_latent = torch.round(x).detach()
        return x_hat, likelihood, rpm_hidden
        
    def get_actual_bits(self, string):
        bits_act = torch.FloatTensor([len(b''.join(string))*8]).squeeze(0)
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
        bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        return bits_est
        
    def compress(self, x):
        if self.RPM_flag:
            indexes = self.gaussian_conditional.build_indexes(self.sigma)
            string = self.gaussian_conditional.compress(x, indexes, means=self.mu)
        else:
            string = self.entropy_bottleneck.compress(x)
        return string

    def decompress(self, string, shape):
        if self.RPM_flag:
            indexes = self.gaussian_conditional.build_indexes(self.sigma)
            x_hat = self.gaussian_conditional.decompress(string, indexes, means=self.mu)
        else:
            x_hat = self.entropy_bottleneck.decompress(string, shape)
        return x_hat
        
    # there should be a validattion for speed
    # RPM will be executed twice on both encoder and decoder
    def set_prior(self, x):
        if x is not None:
            self.prior_latent = torch.round(x).detach()
        else:
            self.prior_latent = None
        
    # we should only use one hidden from compression or decompression
    def compress_slow(self, x, rpm_hidden):
        # shouldnt be used together with forward()
        # otherwise rpm_hidden will be messed up
        t_0 = time.perf_counter()
        if self.RPM_flag:
            assert self.prior_latent is not None, 'prior latent is none!'
            sigma, mu, rpm_hidden = self.RPM(self.prior_latent, rpm_hidden.to(self.prior_latent.device))
            sigma = torch.maximum(sigma, torch.FloatTensor([-7.0]).to(sigma.device))
            sigma = torch.exp(sigma)/10
            indexes = self.gaussian_conditional.build_indexes(sigma)
            string = self.gaussian_conditional.compress(x, indexes, means=mu)
        else:
            string = self.entropy_bottleneck.compress(x)
        duration = time.perf_counter() - t_0
        return string, rpm_hidden, duration
        
    def decompress_slow(self, string, shape, rpm_hidden):
        t_0 = time.perf_counter()
        if self.RPM_flag:
            assert self.prior_latent is not None, 'prior latent is none!'
            sigma, mu, rpm_hidden = self.RPM(self.prior_latent, rpm_hidden.to(self.prior_latent.device))
            sigma = torch.maximum(sigma, torch.FloatTensor([-7.0]).to(sigma.device))
            sigma = torch.exp(sigma)/10
            indexes = self.gaussian_conditional.build_indexes(sigma)
            x_hat = self.gaussian_conditional.decompress(string, indexes, means=mu)
        else:
            x_hat = self.entropy_bottleneck.decompress(string, shape)
        duration = time.perf_counter() - t_0
        return x_hat, rpm_hidden, duration
        
class MeanScaleHyperPriors(CompressionModel):

    def __init__(
        self,
        channels,
        useAttention=False,
    ):
        super().__init__(channels)

        self.channels = int(channels)
        
        self.sigma = self.mu = self.z_string = None
        h = w = 224
        self.gaussian_conditional = GaussianConditional(None)
        
        lite = True
        
        if lite:
            self.h_a1 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
            )
            
            self.h_a2 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            )
            
            self.h_s1 = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(inplace=True),
            )
            
            self.h_s2 = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.h_a1 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
            )
            
            self.h_a2 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            )
            
            self.h_s1 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
            )
            
            self.h_s2 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1),
            )
        
        self.useAttention = useAttention
        
        if self.useAttention:
            self.s_attn_a = Attention(channels)
            self.t_attn_a = Attention(channels)
            self.s_attn_s = Attention(channels)
            self.t_attn_s = Attention(channels)
        
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def loss(self):
        return self.aux_loss()

    def forward(
        self, x, training = None
    ):
        z = self.h_a1(x)
        if self.useAttention:
            # use attention
            B,C,H,W = z.size()
            z = z.view(B,C,-1).transpose(1,2).contiguous() # [B,HW,C]
            z = self.s_attn_a(z,z,z)
            z = z.transpose(0,1).contiguous() #[HW,B,C]
            z = self.t_attn_a(z,z,z)
            z = z.permute(1,2,0).view(B,C,H,W).contiguous()
        z = self.h_a2(z)
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        self.z = z # for fast compression
        
        g = self.h_s1(z_hat)
        if self.useAttention:
            # use attention
            B,C,H,W = z.size()
            g = g.view(B,C,-1).transpose(1,2).contiguous() # [B,HW,C]
            g = self.s_attn_s(g,g,g)
            g = g.transpose(0,1).contiguous() #[HW,B,C]
            g = self.t_attn_s(g,g,g)
            g = g.permute(1,2,0).view(B,C,H,W).contiguous()
        gaussian_params = self.h_s2(g)
            
        self.sigma, self.mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        # post-process sigma to stablize training
        self.sigma = torch.exp(self.sigma)/10
        x_hat,x_likelihood = self.gaussian_conditional(x, self.sigma, means=self.mu, training=training)
        return x_hat, (x_likelihood,z_likelihood)
        
    def get_actual_bits(self, string):
        (x_string,z_string) = string
        bits_act = torch.FloatTensor([len(b''.join(x_string))*8 + len(b''.join(z_string))*8]).squeeze(0)
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        (x_likelihood,z_likelihood) = likelihoods
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(x_likelihood.device)
        bits_est = torch.sum(torch.log(x_likelihood)) / (-log2) + torch.sum(torch.log(z_likelihood)) / (-log2)
        return bits_est
        
    def compress(self, x):
        # a fast implementation of compression
        z_string = self.entropy_bottleneck.compress(self.z)
        indexes = self.gaussian_conditional.build_indexes(self.sigma)
        x_string = self.gaussian_conditional.compress(x, indexes, means=self.mu)
        return (x_string,z_string)

    def decompress(self, string, shape):
        indexes = self.gaussian_conditional.build_indexes(self.sigma)
        x_hat = self.gaussian_conditional.decompress(string[0], indexes, means=self.mu)
        return x_hat
        
    # we should only use one hidden from compression or decompression
    def compress_slow(self, x):
        # shouldnt be used together with forward()
        t_0 = time.perf_counter()
        B,C,H,W = x.size()
        z = self.h_a1(x)
        if self.useAttention:
            # use attention
            z = z.view(B,C,-1).transpose(1,2).contiguous() # [B,HW,C]
            z = self.s_attn_a(z,z,z)
            z = z.transpose(0,1).contiguous() #[HW,B,C]
            z = self.t_attn_a(z,z,z)
            z = z.permute(1,2,0).view(B,C,H,W).contiguous()
        z = self.h_a2(z)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])
        
        g = self.h_s1(z_hat)
        if self.useAttention:
            # use attention
            g = g.view(B,C,-1).transpose(1,2).contiguous() # [B,HW,C]
            g = self.s_attn_s(g,g,g)
            g = g.transpose(0,1).contiguous() #[HW,B,C]
            g = self.t_attn_s(g,g,g)
            g = g.permute(1,2,0).view(B,C,H,W).contiguous()
        gaussian_params = self.h_s2(g)
        
        sigma, mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        sigma = torch.exp(sigma)/10
        indexes = self.gaussian_conditional.build_indexes(sigma)
        x_string = self.gaussian_conditional.compress(x, indexes, means=mu)
        duration = time.perf_counter() - t_0
        return (x_string, z_string), x.size()[-2:], duration
        
    def decompress_slow(self, string, shape):
        t_0 = time.perf_counter()
        B,C,H,W = x.size()
        z_hat = self.entropy_bottleneck.decompress(string[1], shape)
        g = self.h_s1(z_hat)
        if self.useAttention:
            # use attention
            g = g.view(B,C,-1).transpose(1,2).contiguous() # [B,HW,C]
            g = self.s_attn_s(g,g,g)
            g = g.transpose(0,1).contiguous() #[HW,B,C]
            g = self.t_attn_s(g,g,g)
            g = g.permute(1,2,0).view(B,C,H,W).contiguous()
        gaussian_params = self.h_s2(g)
        
        sigma, mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        sigma = torch.exp(sigma)/10
        indexes = self.gaussian_conditional.build_indexes(sigma)
        x_hat = self.gaussian_conditional.decompress(string[0], indexes, means=mu)
        duration = time.perf_counter() - t_0
        return x_hat, duration
        
class JointAutoregressiveHierarchicalPriors(CompressionModel):

    def __init__(
        self,
        channels,
        useAttention=False,
    ):
        super().__init__(channels)

        self.channels = int(channels)
        
        self.sigma = self.mu = self.z_string = None
        h = w = 224
        self.gaussian_conditional = GaussianConditional(None)
        
        self.h_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2),
        )

        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels * 3 // 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels * 3 // 2, channels, kernel_size=3, stride=1, padding=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 1),
        )
        
        self.useAttention = useAttention
        
        if self.useAttention:
            self.s_attn = Attention(channels)
            self.t_attn = Attention(channels)
        
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def loss(self):
        return self.aux_loss()

    def forward(
        self, x, ctx_params, training = None
    ):
        bs,c,h,w = x.size()
        z = self.h_a(x)
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        self.z = z # for fast compression
        params = self.h_s(z_hat)
        g = self.conv1(torch.cat((params, ctx_params), dim=1))
        if self.useAttention:
            # use attention
            B,C,H,W = g.size()
            g = g.view(B,C,-1).transpose(1,2).contiguous() # [B,HW,C]
            g = self.s_attn(g,g,g)
            g = g.transpose(0,1).contiguous() #[HW,B,C]
            g = self.t_attn(g,g,g)
            g = g.permute(1,2,0).view(B,C,H,W).contiguous()
        gaussian_params = self.conv2(g)
        self.sigma, self.mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        # post-process sigma to stablize training
        self.sigma = torch.exp(self.sigma)/10
        x_hat,x_likelihood = self.gaussian_conditional(x, self.sigma, means=self.mu, training=training)
        return x_hat, (x_likelihood,z_likelihood)
        
    def get_actual_bits(self, string):
        (x_string,z_string) = string
        bits_act = torch.FloatTensor([len(b''.join(x_string))*8 + len(b''.join(z_string))*8]).squeeze(0)
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        (x_likelihood,z_likelihood) = likelihoods
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(x_likelihood.device)
        bits_est = torch.sum(torch.log(x_likelihood)) / (-log2) + torch.sum(torch.log(z_likelihood)) / (-log2)
        return bits_est
        
    def compress(self, x):
        # a fast implementation of compression
        z_string = self.entropy_bottleneck.compress(self.z)
        indexes = self.gaussian_conditional.build_indexes(self.sigma)
        x_string = self.gaussian_conditional.compress(x, indexes, means=self.mu)
        return (x_string,z_string)

    def decompress(self, string, shape):
        indexes = self.gaussian_conditional.build_indexes(self.sigma)
        x_hat = self.gaussian_conditional.decompress(string[0], indexes, means=self.mu)
        return x_hat
        
    # we should only use one hidden from compression or decompression
    def compress_slow(self, x, ctx_params):
        # shouldnt be used together with forward()
        t_0 = time.perf_counter()
        bs,c,h,w = x.size()
        z = self.h_a(x)
        z_string = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_string, z.size()[-2:])
        params = self.h_s(z_hat)
        g = self.conv1(torch.cat((params, ctx_params), dim=1))
        if self.useAttention:
            # use attention
            B,C,H,W = g.size()
            g = g.view(B,C,-1).transpose(1,2).contiguous() # [B,HW,C]
            g = self.s_attn(g,g,g)
            g = g.transpose(0,1).contiguous() #[HW,B,C]
            g = self.t_attn(g,g,g)
            g = g.permute(1,2,0).view(B,C,H,W).contiguous()
        gaussian_params = self.conv2(g)
        sigma, mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        sigma = torch.exp(sigma)/10
        indexes = self.gaussian_conditional.build_indexes(sigma)
        x_string = self.gaussian_conditional.compress(x, indexes, means=mu)
        duration = time.perf_counter() - t_0
        return (x_string, z_string), x.size(), duration
        
    def decompress_slow(self, string, shape, ctx_params):
        t_0 = time.perf_counter()
        bs,c,h,w = shape
        z_hat = self.entropy_bottleneck.decompress(string[1], [4,4])
        params = self.h_s(z_hat)
        g = self.conv1(torch.cat((params, ctx_params), dim=1))
        if self.useAttention:
            # use attention
            B,C,H,W = g.size()
            g = g.view(B,C,-1).transpose(1,2).contiguous() # [B,HW,C]
            g = self.s_attn(g,g,g)
            g = g.transpose(0,1).contiguous() #[HW,B,C]
            g = self.t_attn(g,g,g)
            g = g.permute(1,2,0).view(B,C,H,W).contiguous()
        gaussian_params = self.conv2(g)
        sigma, mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        sigma = torch.exp(sigma)/10
        indexes = self.gaussian_conditional.build_indexes(sigma)
        x_hat = self.gaussian_conditional.decompress(string[0], indexes, means=mu)
        duration = time.perf_counter() - t_0
        return x_hat, duration
        
def attention(q, k, v, d_model, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_model)
        
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output
        
class Attention(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        
        bs = q.size(0)
        
        # perform linear operation
        
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_model, self.dropout)
        
        output = self.out(scores) # bs * sl * d_model
    
        return output
        
# conditional probability
# predict y_t based on parameters computed from y_t-1
class RPM(nn.Module):
    def __init__(self, channels=128, act=torch.tanh):
        super(RPM, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(channels, 2*channels, kernel_size=3, stride=1, padding=1)
        self.channels = channels
        self.lstm = ConvLSTM(channels)

    def forward(self, x, hidden):
        # [B,C,H//16,W//16]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x, hidden = self.lstm(x, hidden.to(x.device))
            
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        sigma_mu = F.relu(self.conv8(x))
        sigma, mu = torch.split(sigma_mu, self.channels, dim=1)
        return sigma, mu, hidden
        
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
        
def test(name = 'Joint'):
    channels = 128
    if name =='RPM':
        net = RecProbModel(channels)
    elif name == 'Joint':
        net = JointAutoregressiveHierarchicalPriors(channels,useAttention=True)
    else:
        net = MeanScaleHyperPriors(channels,useAttention=True)
    x = torch.rand(4, channels, 14, 14)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in net.named_parameters())
    optimizer = optim.Adam(parameters, lr=1e-4)
    rpm_hidden = torch.zeros(1,channels*2,14,14)
    isTrain = True
    rpm_flag = True
    if name == 'RPM':
        net.set_prior(x)
            
    train_iter = tqdm(range(0,10000))
    duration_e = duration_d = bits_est = 0
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()
        
        net.update(force=True)

        if name == 'RPM':
            net.set_RPM(rpm_flag)
            if isTrain:
                x_hat, likelihoods, rpm_hidden = net(x,rpm_hidden,training=True)
                string = net.compress(x)
            else:
                x_q, _, _ = net(x,rpm_hidden,training=False)
                string, _, duration_e = net.compress_slow(x,rpm_hidden)
                x_hat, rpm_hidden, duration_d = net.decompress_slow(string, x.size()[-2:], rpm_hidden)
                net.set_prior(x)
                mse2 = torch.mean(torch.pow(x_hat-x_q,2))
        elif name == 'Joint':
            if isTrain:
                x_hat, likelihoods = net(x,x,training=True)
                string = net.compress(x)
            else:
                x_q, _ = net(x,x,training=False)
                string, shape, duration_e = net.compress_slow(x, x)
                x_hat, duration_d = net.decompress_slow(string, shape, x)
                mse2 = torch.mean(torch.pow(x_hat-x_q,2))
        else:
            if isTrain:
                x_hat, likelihoods = net(x,training=True)
                string = net.compress(x)
            else:
                x_q,_ = net(x,training=False)
                string, shape, duration_e = net.compress_slow(x)
                x_hat, duration_d = net.decompress_slow(string, shape)
                mse2 = torch.mean(torch.pow(x_hat-x_q,2))
            
        bits_act = net.get_actual_bits(string)
        mse = torch.mean(torch.pow(x-x_hat,2))*1024
        
        if isTrain:
            bits_est = net.get_estimate_bits(likelihoods)
            loss = bits_est + net.loss() + mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1)
            optimizer.step()
        
            train_iter.set_description(
                f"Batch: {i:4}. "
                f"loss: {float(loss):.2f}. "
                f"bits_est: {float(bits_est):.2f}. "
                f"bits_act: {float(bits_act):.2f}. "
                f"MSE: {float(mse):.2f}. "
                f"ENC: {float(duration_e):.3f}. "
                f"DEC: {float(duration_d):.3f}. ")
        else:
            train_iter.set_description(
                f"Batch: {i:4}. "
                f"bits_act: {float(bits_act):.2f}. "
                f"MSE: {float(mse):.2f}. "
                f"MSE2: {float(mse2):.4f}. "
                f"ENC: {float(duration_e):.3f}. "
                f"DEC: {float(duration_d):.3f}. ")
    
if __name__ == '__main__':
    test()
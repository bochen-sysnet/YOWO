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
        
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
    
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
            self.sigma, self.mu, rpm_hidden = self.RPM(self.prior_latent, rpm_hidden)
            self.sigma = torch.maximum(self.sigma, torch.FloatTensor([-7.0]).to(self.sigma.device))
            self.sigma = torch.exp(self.sigma)/10
            x_hat,likelihood = self.gaussian_conditional(x, self.sigma, means=self.mu, training=training)
            rpm_hidden = rpm_hidden.detach()
        else:
            x_hat,likelihood = self.entropy_bottleneck(x,training=training)
        self.prior_latent = torch.round(x).detach()
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
        
class JointAutoregressiveHierarchicalPriors(CompressionModel):

    def __init__(
        self,
        channels,
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

        self.h_s = nn.ModuleList([
            nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(channels, channels * 3 // 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels * 3 // 2, channels, kernel_size=3, stride=1, padding=1)]
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 5 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels * 5 // 3, channels * 4 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels * 4 // 3, channels * 2, 1),
        )
        
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
        
        params = z_hat
        for i,m in enumerate(self.h_s):
            if i in [0,2]:
                sz = torch.Size([bs,c,h//2,w//2]) if i==0 else torch.Size([bs,c,h,w])
                params = m(params,output_size=sz)
            else:
                params = m(params)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        self.sigma, self.mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
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
    else:
        net = JointAutoregressiveHierarchicalPriors(channels)
    x = torch.rand(1, channels, 14, 14)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in net.named_parameters())
    optimizer = optim.Adam(parameters, lr=1e-4)
    rpm_hidden = torch.zeros(1,self.channels*2,h//16,w//16).cuda()
    rpm_flag = True
    if name == 'RPM':
        net.set_RPM(False)
        x_hat, likelihoods, rpm_hidden = net(x,rpm_hidden,training=False)
    train_iter = tqdm(range(0,10000))
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()

        net.update(force=True)
        if name == 'RPM':
            net.set_RPM(rpm_flag)
            x_hat, likelihoods, rpm_hidden = net(x,rpm_hidden,training=True)
        else:
            x_hat, likelihoods = net(x,x,training=True)
            
        string = net.compress(x)
        bits_act = net.get_actual_bits(string)
        x_hat2 = net.decompress(string, x.size()[-2:])
        mse2 = torch.mean(torch.pow(x_hat-x_hat2,2))
        
        bits_est = net.get_estimate_bits(likelihoods)
        mse = torch.mean(torch.pow(x-x_hat,2))
        loss = bits_est + net.loss()
        
        bits_est.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1)
        optimizer.step()
        
        train_iter.set_description(
            f"Batch: {i:4}. "
            f"loss: {float(loss):.2f}. "
            f"bits_est: {float(bits_est):.2f}. "
            f"bits_act: {float(bits_act):.2f}. "
            f"MSE: {float(mse):.2f}. "
            f"MSE2: {float(mse2):.4f}. ")
    
if __name__ == '__main__':
    test()
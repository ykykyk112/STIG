import torch
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class LFLoss(nn.Module) :

    def __init__(self, opt) :
        super(LFLoss, self).__init__()
        self.dc_point = opt.size//2
        self.sigma = opt.sigma

    def forward(self, src_magnitude, tgt_magnitude) :
        
        src_plane = src_magnitude[:, :, self.dc_point-self.sigma:self.dc_point+self.sigma, self.dc_point-self.sigma:self.dc_point+self.sigma]
        tgt_plane = tgt_magnitude[:, :, self.dc_point-self.sigma:self.dc_point+self.sigma, self.dc_point-self.sigma:self.dc_point+self.sigma]

        loss = torch.sqrt(((src_plane-tgt_plane)**2).mean())
        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class PSNR(torch.nn.Module) :
    def __init__(self) :
        super(PSNR, self).__init__()

    def forward(self, img1, img2) :
        img1, img2 = img1 * 255., img2 * 255.
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255. / torch.sqrt(mse))

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class RadialAverage(nn.Module) :
    
    def __init__(self, image_size, device, only_psd = True) :
        super(RadialAverage, self).__init__()

        self.only_psd = only_psd

        rows, cols = image_size, image_size
        
        shift_rows = int(rows / 2)
        shift_cols = int(cols / 2)

        r = np.indices((rows,cols)) - np.array([[[shift_rows]],[[shift_cols]]])
        r = np.sqrt(r[0,:,:]**2+r[1,:,:]**2)
        r = r.astype(int)

        r_max = np.max(r)
        r = torch.from_numpy(r).expand(r_max+1,-1,-1)
        radius_to_slice = torch.arange(r_max+1).view(-1,1,1)
        
        self.mask = torch.where(
            r==radius_to_slice,
            torch.tensor(1,dtype=torch.float),
            torch.tensor(0,dtype=torch.float)
        )

        self.mask_n = torch.sum(self.mask,axis=(1,2))
        self.vector_length = r_max + 1

        self.mask = self.mask.to(device)
        self.mask_n = self.mask_n.to(device)


    def fft(self, x) :

        if not self.only_psd :
            fft = torch.fft.fft2(x)
            fft = torch.fft.fftshift(fft)
            mag = 20 * torch.log1p(torch.abs(fft))

            if len(mag.shape) == 4 and mag.shape[1] == 3:
            # convert to grayscale
                mag =  0.299 * mag[:,0,:,:] + \
                        0.587 * mag[:,1,:,:] + \
                        0.114 * mag[:,2,:,:]

        else :
            if len(x.shape) == 4 and x.shape[1] == 3:
            # convert to grayscale
                mag =  0.299 * x[:,0,:,:] + \
                        0.587 * x[:,1,:,:] + \
                        0.114 * x[:,2,:,:]
            else :
                mag = x
            mag = 20 * mag
        
        return mag

    def forward(self, x) :

        self.mag = self.fft(x)
        mag = self.mag.unsqueeze(1).expand(-1, self.vector_length, -1, -1)

        # apply mask and compute profile vector
        profile = (mag * self.mask)
        nbin = self.mask.sum((1,2))
        profile = profile.sum((2, 3)) / nbin
        # normalize profile into [0,1]
        profile = (profile - profile.min()) / (profile.max() - profile.min())
        
        return profile

class RectAverage(nn.Module) :
    
    def __init__(self, image_size, device, only_psd = True) :
        super(RectAverage, self).__init__()

        self.only_psd = only_psd

        w, h = image_size, image_size
        pyramid = np.zeros([w - 1, h - 1])
        x = pyramid.shape[0]
        y = pyramid.shape[1]

        for i in range(x // 2 + 1):
            v = x // 2 - i
            for j in range(i, x - i):
                for k in range(i, x - i):
                    pyramid[j, k] = v

        self.plane = np.ones([w, h]) * (w // 2)
        self.plane[1:, 1:] = pyramid
        
        mask = []

        for r in range(image_size//2+1) :
            px, py = np.where(self.plane == r)
            mask_plane = np.zeros((w, h))
            mask_plane[px, py] = 1
            mask.append(mask_plane)
            
        mask = np.stack(mask, axis = 0)
        self.mask = torch.from_numpy(mask).float().to(device)
        self.mask_n = torch.sum(self.mask,axis=(1,2))
        self.vector_length = len(self.mask)

    def fft(self, x) :

        if not self.only_psd :
            fft = torch.fft.fft2(x)
            fft = torch.fft.fftshift(fft)
            mag = 20 * torch.log1p(torch.abs(fft))

            if len(mag.shape) == 4 and mag.shape[1] == 3:
            # convert to grayscale
                mag =  0.299 * mag[:,0,:,:] + \
                        0.587 * mag[:,1,:,:] + \
                        0.114 * mag[:,2,:,:]

        else :
            if len(x.shape) == 4 and x.shape[1] == 3:
            # convert to grayscale
                mag =  0.299 * x[:,0,:,:] + \
                        0.587 * x[:,1,:,:] + \
                        0.114 * x[:,2,:,:]
            else :
                mag = x
            mag = 20 * mag
        
        return mag

    def forward(self, x) :

        self.mag = self.fft(x)
        print(self.mag.shape)
        mag = self.mag.unsqueeze(1).expand(-1, self.vector_length, -1, -1)

        # apply mask and compute profile vector
        profile = (mag * self.mask)
        nbin = self.mask.sum((1,2))
        profile = profile.sum((2, 3)) / nbin
        # normalize profile into [0,1]
        profile = (profile - profile.min()) / (profile.max() - profile.min())
        
        return profile


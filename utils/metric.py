from glob import glob

from model.loss import RadialAverage, RectAverage
from .processing import img2fft, normalize
from torchvision.transforms import ToPILImage, ToTensor
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

'''
FIDCalculator : Class for save samples from the trained STIG model.
                It saves real, generated and stig-refined image / magnitudes to defined path for saving.
                Saving results are devided into [clean, noise, denoised] & [clean_mag, noise_mag, denoised_mag]
                Each folder implies,
                clean(_mag) : real images (and magnitudes)
                noise(_mag) : generated images (and magnitudes)
                denoised(_mag) : stig-refined images (and magnitudes)
'''

class FIDCalculator :
    
    def __init__(self, opt, save_path, eval = False) :

        self.save_path = save_path

        self.baseline_config = False
        self.device = opt.device
        self.rectProfile = RectAverage(opt.size, self.device)
        self.radialProfile = RadialAverage(opt.size, self.device)
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()
        self.opt = opt

        self.clean_path = os.path.join(self.save_path, 'clean')
        self.noise_path = os.path.join(self.save_path, 'noise')
        self.denoised_path = os.path.join(self.save_path, 'denoised')

        self.clean_mag_path = os.path.join(self.save_path, 'clean_mag')
        self.noise_mag_path = os.path.join(self.save_path, 'noise_mag')
        self.denoised_mag_path = os.path.join(self.save_path, 'denoised_mag')
        
        self.psd_path = os.path.join(self.save_path, 'psd')
        self.averaged_magnitude_path = os.path.join(self.save_path, 'average')
        self.psd_feature_path = os.path.join(self.save_path, 'classification')

        if not eval :
            os.mkdir(self.clean_path)
            os.mkdir(self.noise_path)
            os.mkdir(self.denoised_path)

            os.mkdir(self.clean_mag_path)
            os.mkdir(self.noise_mag_path)
            os.mkdir(self.denoised_mag_path)

        self.real_psd, self.fake_psd = None, None

    def set_baseline_config(self, data, n) :
        
        clean = data['clean']
        noise = data['noise']

        clean_fft = img2fft(clean)
        noise_fft = img2fft(noise)

        clean_mag = torch.log1p(torch.abs(clean_fft))
        noise_mag = torch.log1p(torch.abs(noise_fft))

        clean_mag, _, _ = normalize(clean_mag)
        noise_mag, _, _ = normalize(noise_mag)

        clean_mag = self.zero_to_one(clean_mag).squeeze(0)
        noise_mag = self.zero_to_one(noise_mag).squeeze(0)

        clean_pil = self.to_pil(clean.squeeze(0))
        noise_pil = self.to_pil(noise.squeeze(0))
        
        clean_mag_pil = self.to_pil(clean_mag)
        noise_mag_pil = self.to_pil(noise_mag)

        clean_pil.save(os.path.join(self.clean_path, '{:06d}.png'.format(n)), 'png')
        noise_pil.save(os.path.join(self.noise_path, '{:06d}.png'.format(n)), 'png')

        clean_mag_pil.save(os.path.join(self.clean_mag_path, '{:06d}.png'.format(n)), 'png')
        noise_mag_pil.save(os.path.join(self.noise_mag_path, '{:06d}.png'.format(n)), 'png')

    def set_control_group(self, n) :

        img_pil = Image.open(self.control_groups[n])

        fft = img2fft(self.to_tensor(img_pil))
        mag = torch.log1p(torch.abs(fft))
        mag, _, _ = normalize(mag)
        mag = self.zero_to_one(mag).squeeze(0)

        mag_pil = self.to_pil(mag)

        img_pil.save(os.path.join(self.denoised_path, '{:06d}.png'.format(n)), 'png')
        mag_pil.save(os.path.join(self.denoised_mag_path, '{:06d}.png'.format(n)), 'png')

    def step(self, model, n) :

        denoised_img = model.denoised_image_normed.detach().squeeze(0)
        denoised_mag = model.denoised_mag.detach().squeeze(0)
        
        denoised_pil = self.to_pil(denoised_img)
        denoised_mag_pil = self.to_pil(denoised_mag)

        denoised_pil.save(os.path.join(self.denoised_path, '{:06d}.png'.format(n)), 'png')
        denoised_mag_pil.save(os.path.join(self.denoised_mag_path, '{:06d}.png'.format(n)), 'png')

    def compute_fid_score(self, baseline = False) :
        
        if not baseline :
            img_paths = [self.clean_path, self.denoised_path]
            mag_paths = [self.clean_mag_path, self.denoised_mag_path]
        else :
            img_paths = [self.clean_path, self.noise_path]
            mag_paths = [self.clean_mag_path, self.noise_mag_path]

        img_fid_score = calculate_fid_given_paths(img_paths, 64, self.device, 2048)
        mag_fid_score = calculate_fid_given_paths(mag_paths, 64, self.device, 2048)

        return img_fid_score, mag_fid_score

    def plot_psd_profile(self, epoch) :

        if (self.real_psd is None) and (self.fake_psd is None) :
            self.real_psd, self.real_rect, self.real_magnitude, real_psds, real_rects = self.get_psd_from_folder(self.clean_mag_path)
            self.fake_psd, self.fake_rect, self.fake_magnitude, fake_psds, fake_rects = self.get_psd_from_folder(self.noise_mag_path)
            torch.save(real_psds, os.path.join(self.psd_feature_path, 'real_psds.pt'))
            torch.save(fake_psds, os.path.join(self.psd_feature_path, 'fake_psds.pt'))
            torch.save(real_rects, os.path.join(self.psd_feature_path, 'real_rects.pt'))
            torch.save(fake_rects, os.path.join(self.psd_feature_path, 'fake_rects.pt'))
        self.fake_enhanced_psd, self.fake_enhanced_rect, fake_enhanced_magnitude, enhanced_psds, enhanced_rects = self.get_psd_from_folder(self.denoised_mag_path)

        torch.save(enhanced_psds, os.path.join(self.psd_feature_path, 'psd_{}.pt'.format(epoch+1)))
        torch.save(enhanced_rects, os.path.join(self.psd_feature_path, 'rects_{}.pt'.format(epoch+1)))

        self.baseline_psd_difference = abs((self.real_psd - self.fake_psd).sum())
        self.epoch_psd_difference = abs((self.real_psd - self.fake_enhanced_psd).sum())

        plt.plot(self.real_psd, label = 'real')
        plt.plot(self.fake_psd, label = 'fake')
        plt.plot(self.fake_enhanced_psd, label = 'enhanced')
        plt.legend()
        plt.savefig(os.path.join(self.psd_path, 'psd_{}.png'.format(epoch+1)))
        plt.close()

        plt.plot(self.real_rect, label = 'real')
        plt.plot(self.fake_rect, label = 'fake')
        plt.plot(self.fake_enhanced_rect, label = 'enhanced')
        plt.legend()
        plt.savefig(os.path.join(self.psd_path, 'rects_{}.png'.format(epoch+1)))
        plt.close()

        fig = plt.figure(figsize = (16, 6))

        rows = 2
        cols = 3

        ax = fig.add_subplot(rows, cols, 1)
        opts = {'vmin': 0., 'vmax': 1.}
        dif_opts = {'vmin': -0.2, 'vmax': 0.2}
        ax.imshow(self.real_magnitude, **opts)
        ax.axis('off')
        ax.set_title('Real Clean')
        ax = fig.add_subplot(rows, cols, 2)
        ax.imshow(self.fake_magnitude, **opts)
        ax.axis('off')
        ax.set_title('Real Noise')
        ax = fig.add_subplot(rows, cols, 3)
        ax.imshow(self.real_magnitude - self.fake_magnitude, **dif_opts)
        ax.axis('off')
        ax.set_title('Differences')
        ax = fig.add_subplot(rows, cols, 4)
        ax.imshow(self.real_magnitude, **opts)
        ax.axis('off')
        ax.set_title('Real Clean')
        ax = fig.add_subplot(rows, cols, 5)
        ax.imshow(fake_enhanced_magnitude, **opts)
        ax.axis('off')
        ax.set_title('Fake Clean')
        ax = fig.add_subplot(rows, cols, 6)
        ax.imshow(self.real_magnitude - fake_enhanced_magnitude, **dif_opts)
        ax.axis('off')
        ax.set_title('Result Differences')

        plt.savefig(os.path.join(self.averaged_magnitude_path, 'epoch_{}.png'.format(epoch+1)))
        plt.close()

    def get_psd_from_folder(self, path) :

        files = []
        for _, p in enumerate(tqdm(glob(os.path.join(path, '*.png')), desc="{:17s}".format('Loading Files'), mininterval=0.0001)) :
            files.append(self.to_tensor(Image.open(p)))
        files = torch.stack(files, dim = 0)
        average_magnitude = files.mean(1).mean(0)

        psds = []
        for _, magnitude in enumerate(tqdm(files, desc="{:17s}".format('Calculating Azimuthal'), mininterval=0.0001)) :
            psds.append(self.radialProfile(magnitude.unsqueeze(0).to(self.device)))
        psds = torch.stack(psds, dim = 0)
        psd = psds.mean(0).cpu().numpy()[0]

        rects = []
        for _, magnitude in enumerate(tqdm(files, desc="{:17s}".format('Calculating Rectangular'), mininterval=0.0001)) :
            rects.append(self.rectProfile(magnitude.unsqueeze(0).to(self.device)))
        rects = torch.stack(rects, dim = 0)
        rect = rects.mean(0).cpu().numpy()[0]
        
        return psd, rect, average_magnitude, psds.unsqueeze(1).cpu().numpy(), rects.unsqueeze(1).cpu().numpy()

    def get_image_mag_from_folder(self, path) :
        
        mag_save_path = os.path.join(self.opt.dst, '{}_mag'.format(self.opt.eval_name))
        os.makedirs(mag_save_path, exist_ok=True)

        files = []
        paths = glob(os.path.join(path, '*.png'))
        paths += glob(os.path.join(path, '*.jpg'))
        for _, p in enumerate(tqdm(paths, desc="{:17s}".format('Loading Files'), mininterval=0.0001)) :
            files.append(self.to_tensor(Image.open(p).convert('RGB')))
        files = torch.stack(files, dim = 0)
        average_images = files.mean(0)

        mags = []
        for n, images in enumerate(tqdm(files, desc="{:17s}".format('Calculating Rectangular'), mininterval=0.0001)) :
            mag = self.to_magnitude(images.to(self.device)).cpu()
            mag_pil = self.to_pil(mag.squeeze(0))
            mag_pil.save(os.path.join(mag_save_path, '{:06d}.png'.format(n)))
            mags.append(mag)
        mags = torch.stack(mags, dim = 0)
        average_magnitudes = mags.mean(0)

        return average_images, average_magnitudes
    
    def average_magnitude(self, path) :
        
        files = []
        paths = glob(os.path.join(path, '*.png'))
        paths += glob(os.path.join(path, '*.jpg'))
        for _, p in enumerate(tqdm(paths, desc="{:17s}".format('Loading Files'), mininterval=0.0001)) :
            files.append(self.to_tensor(Image.open(p).convert('RGB')))
        files = torch.stack(files, dim = 0)

        mags = []
        for n, images in enumerate(tqdm(files, desc="{:17s}".format('Calculating Rectangular'), mininterval=0.0001)) :
            mag = self.to_magnitude(images.to(self.device)).cpu()
            mags.append(mag)
        mags = torch.stack(mags, dim = 0)
        average_magnitudes = mags.mean(0)

        return average_magnitudes


    def zero_to_one(self, img) :

        return (img + 1.) * (0.5) 

    def to_magnitude(self, img) :

        fft = torch.fft.fft2(img)
        fft = torch.fft.fftshift(fft)
        mag = torch.log1p(torch.abs(fft))
        mag_scaled = (mag - mag.min()) / (mag.max() - mag.min())
        
        return mag_scaled
        
class InferenceModel :
    
    def __init__(self, opt, save_path) :

        self.save_path = save_path

        self.device = opt.device
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()
        self.opt = opt

        self.noise_path = os.path.join(self.save_path, 'noise')
        self.denoised_path = os.path.join(self.save_path, 'denoised')

        self.noise_mag_path = os.path.join(self.save_path, 'noise_mag')
        self.denoised_mag_path = os.path.join(self.save_path, 'denoised_mag')
        

        os.makedirs(self.noise_path, exist_ok = True)
        os.makedirs(self.denoised_path, exist_ok = True)

        os.makedirs(self.noise_mag_path, exist_ok = True)
        os.makedirs(self.denoised_mag_path, exist_ok = True)

    def step(self, model, input, n) :

        model.set_input(input, evaluation = True)

        input_img = model.input_image_normed.detach().squeeze(0)
        input_mag = model.input_mag.detach().squeeze(0).mean(0)

        denoised_img = model.denoised_image_normed.detach().squeeze(0)
        denoised_mag = model.denoised_mag.detach().squeeze(0).mean(0)
        
        # input_pil = self.to_pil(input_img)
        # input_mag_pil = self.to_pil(input_mag)

        # denoised_pil = self.to_pil(denoised_img)
        # denoised_mag_pil = self.to_pil(denoised_mag)

        # input_pil.save(os.path.join(self.noise_path, '{:06d}.png'.format(n)), 'png')
        # input_mag_pil.save(os.path.join(self.noise_mag_path, '{:06d}.png'.format(n)), 'png')

        # denoised_pil.save(os.path.join(self.denoised_path, '{:06d}.png'.format(n)), 'png')
        # denoised_mag_pil.save(os.path.join(self.denoised_mag_path, '{:06d}.png'.format(n)), 'png')

        input_img_np = np.transpose(input_img.cpu().numpy(), (1, 2, 0))
        input_mag_np = np.clip(input_mag.cpu().numpy(), a_min = 0.00, a_max = 1.0)

        denoised_img_np = np.transpose(denoised_img.cpu().numpy(), (1, 2, 0))
        denoised_mag_np = np.clip(denoised_mag.cpu().numpy(), a_min = 0.00, a_max = 1.0)
        print(input_img_np.shape, input_mag_np.shape, denoised_img_np.shape, denoised_mag_np.shape)

        plt.imsave(os.path.join(self.noise_mag_path, '{:06d}.png'.format(n)), input_mag_np, cmap = 'jet')
        plt.imsave(os.path.join(self.denoised_mag_path, '{:06d}.png'.format(n)), denoised_mag_np, cmap='jet')
        plt.imsave(os.path.join(self.noise_path, '{:06d}.png'.format(n)), input_img_np)
        plt.imsave(os.path.join(self.denoised_path, '{:06d}.png'.format(n)), denoised_img_np)

    def zero_to_one(self, img) :

        return (img + 1.) * (0.5) 

    def to_magnitude(self, img) :

        fft = torch.fft.fft2(img)
        fft = torch.fft.fftshift(fft)
        mag = torch.log1p(torch.abs(fft))
        mag_scaled = (mag - mag.min()) / (mag.max() - mag.min())
        
        return mag_scaled

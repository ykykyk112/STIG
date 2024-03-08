import os
from matplotlib import pyplot as plt
import torch.nn
import numpy as np
from model.loss import RadialAverage, RectAverage
from utils.processing import image_normalize

def inverse_norm(tensor) :
    return (tensor + 1.) * 0.5

class Visualizer :

    def __init__(self, opt) :

        self.size = opt.size
        self.batch_size = opt.batch_size
        self.rectProfile = RectAverage(self.size, torch.device(opt.device))
        self.radialProfile = RadialAverage(self.size, torch.device(opt.device))
        self.average_frequency = opt.average_frequency
        self.sample_frequency = opt.sample_frequency
        self.save_path = os.path.join('./results', opt.dst)
        
        self.iteration = 0
        self.current_epoch = 0

        self.real_noise_mean = torch.empty(self.average_frequency, 3, self.size, self.size).to(torch.device(opt.device))
        self.real_clean_mean = torch.empty(self.average_frequency, 3, self.size, self.size).to(torch.device(opt.device))
        self.fake_clean_recon_mean = torch.empty(self.average_frequency, 3, self.size, self.size).to(torch.device(opt.device))
        self.fake_clean_mag_mean = torch.empty(self.average_frequency, 3, self.size, self.size).to(torch.device(opt.device))
        #self.fake_clean_mean = torch.empty(self.average_frequency, 3, self.size, self.size)
        #self.idt_clean_mean = torch.empty(self.average_frequency, 3, self.size, self.size)

        self.colormap = 'viridis'

    def plot_sampled_profile(self, model, epoch, iter) :

        source = np.transpose(model.real_A.detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        target = np.transpose(model.real_B.detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        identity_recon = np.transpose(model.idt_B.detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        translated_recon = np.transpose(model.denoised_mag.detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        #identity = np.transpose(model.enhanced_mag_B.detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        #translated = np.transpose(model.enhanced_mag_A.detach().cpu().squeeze(0).numpy(), (1, 2, 0))

        source_img = np.transpose(image_normalize(model.image_A).detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        target_img = np.transpose(image_normalize(model.image_B).detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        identity_recon_img = np.transpose(image_normalize(model.identity_image_B).detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        translated_recon_img = np.transpose(image_normalize(model.denoised_image_normed).detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        #identity_img = np.transpose(image_normalize(model.enhanced_image_B).detach().cpu().squeeze(0).numpy(), (1, 2, 0))
        #translated_img = np.transpose(image_normalize(model.enhanced_image_A).detach().cpu().squeeze(0).numpy(), (1, 2, 0))

        fig = plt.figure(figsize = (16, 6))
        rows = 2
        cols = 4

        ax = fig.add_subplot(rows, cols, 1)
        ax.imshow(source_img)
        ax.axis('off')
        ax.set_title('Real Noise Image')
        ax = fig.add_subplot(rows, cols, 2)
        ax.imshow(inverse_norm(source.mean(2)), cmap = self.colormap)
        ax.axis('off')
        ax.set_title('Real Noise Mag')

        ax = fig.add_subplot(rows, cols, 3)
        ax.imshow(translated_recon_img)
        ax.axis('off')
        ax.set_title('Fake Clean Recon')
        ax = fig.add_subplot(rows, cols, 4)
        ax.imshow(inverse_norm(translated_recon.mean(2)), cmap = self.colormap)
        ax.axis('off')
        ax.set_title('Fake Clean Recon')

        # ax = fig.add_subplot(2, 6, 5)
        # ax.imshow(translated_img)
        # ax.axis('off')
        # ax.set_title('Fake Clean Image')
        # ax = fig.add_subplot(2, 6, 6)
        # ax.imshow(inverse_norm(translated.mean(2)), cmap = self.colormap)
        # ax.axis('off')
        # ax.set_title('Fake Clean Mag')

        ax = fig.add_subplot(rows, cols, 5)
        ax.imshow(target_img)
        ax.axis('off')
        ax.set_title('Real Clean Image')
        ax = fig.add_subplot(rows, cols, 6)
        ax.imshow(inverse_norm(target.mean(2)), cmap = self.colormap)
        ax.axis('off')
        ax.set_title('Real Clean Mag')

        ax = fig.add_subplot(rows, cols, 7)
        ax.imshow(identity_recon_img)
        ax.axis('off')
        ax.set_title('Identity Clean Recon')
        ax = fig.add_subplot(rows, cols, 8)
        ax.imshow(inverse_norm(identity_recon.mean(2)), cmap = self.colormap)
        ax.axis('off')
        ax.set_title('Identity Clean Recon')

        # ax = fig.add_subplot(2, 6, 11)
        # ax.imshow(identity_img)
        # ax.axis('off')
        # ax.set_title('Identity Clean Image')
        # ax = fig.add_subplot(2, 6, 12)
        # ax.imshow(inverse_norm(identity.mean(2)), cmap = self.colormap)
        # ax.axis('off')
        # ax.set_title('Identity Clean Mag')

        plt.savefig(os.path.join(self.save_path, 'sample/{}_{}.png'.format(epoch+1, iter)))
        plt.close()

    def plot_averaged_profile(self, epoch, iter) :

        self.real_noise_mean_np = self.real_noise_mean.detach().cpu().mean(1).mean(0).numpy()
        self.real_clean_mean_np = self.real_clean_mean.detach().cpu().mean(1).mean(0).numpy()
        self.fake_clean_recon_mean_np = self.fake_clean_recon_mean.detach().cpu().mean(1).mean(0).numpy()
        #self.fake_clean_mean_np = np.transpose(self.fake_clean_mean.detach().cpu().mean(1).numpy(), (1, 2, 0)).mean(2)
        self.fake_clean_mag_mean_np = self.fake_clean_mag_mean.detach().cpu().mean(1).mean(0).numpy()
        #self.idt_clean_mean_np = np.transpose(self.idt_clean_mean.detach().cpu().mean(1).numpy(), (1, 2, 0)).mean(2)
        
        fig = plt.figure(figsize = (16, 6))

        rows = 2
        cols = 3

        ax = fig.add_subplot(rows, cols, 1)
        opts = {'vmin': 0., 'vmax': 1.}
        dif_opts = {'vmin': -0.2, 'vmax': 0.2}
        ax.imshow(self.real_clean_mean_np, cmap = self.colormap, **opts)
        ax.axis('off')
        ax.set_title('Real Clean')
        ax = fig.add_subplot(rows, cols, 2)
        ax.imshow(self.real_noise_mean_np, cmap = self.colormap, **opts)
        ax.axis('off')
        ax.set_title('Real Noise')
        ax = fig.add_subplot(rows, cols, 3)
        ax.imshow(self.real_clean_mean_np - self.real_noise_mean_np, **dif_opts)
        ax.axis('off')
        ax.set_title('Differences')
        ax = fig.add_subplot(rows, cols, 4)
        ax.imshow(self.real_clean_mean_np, cmap = self.colormap, **opts)
        ax.axis('off')
        ax.set_title('Real Clean')
        ax = fig.add_subplot(rows, cols, 5)
        ax.imshow(self.fake_clean_recon_mean_np, cmap = self.colormap, **opts)
        ax.axis('off')
        ax.set_title('Fake Clean')
        ax = fig.add_subplot(rows, cols, 6)
        ax.imshow(self.real_clean_mean_np - self.fake_clean_recon_mean_np, **dif_opts)
        ax.axis('off')
        ax.set_title('Result Differences')

        plt.savefig(os.path.join(self.save_path, 'average/{}_{}.png'.format(epoch+1, iter+1)))
        plt.close()

    def plot_averaged_1d_psd(self, epoch, iter) :
        
        # real_noise_psd = self.radialProfile(self.real_noise_mean.detach().mean(0, keepdim = True)).cpu()
        # real_clean_psd = self.radialProfile(self.real_clean_mean.detach().mean(0, keepdim = True)).cpu()
        real_noise_rect = self.rectProfile(self.real_noise_mean.detach().mean(0, keepdim = True)).cpu()
        real_clean_rect = self.rectProfile(self.real_clean_mean.detach().mean(0, keepdim = True)).cpu()
        #fake_clean_psd = azimuthalAverage(self.fake_clean_mean_np)
        #idt_clean_psd = azimuthalAverage(self.idt_clean_mean_np)
        # fake_clean_recon_psd = self.radialProfile(self.fake_clean_recon_mean.detach().mean(0, keepdim = True)).cpu()
        # fake_clean_mag_psd = self.radialProfile(self.fake_clean_mag_mean.detach().mean(0, keepdim = True)).cpu()
        fake_clean_recon_rect = self.rectProfile(self.fake_clean_recon_mean.detach().mean(0, keepdim = True)).cpu()
        fake_clean_mag_rect = self.rectProfile(self.fake_clean_mag_mean.detach().mean(0, keepdim = True)).cpu()
        #idt_clean_recon_psd = azimuthalAverage(self.idt_clean_recon_mean_np)

        # plt.plot(real_noise_psd[0], label = 'Noise PSD')
        # plt.plot(real_clean_psd[0], label = 'Clean PSD')
        # plt.plot(fake_clean_recon_psd[0], label = 'Fake Clean - Translated')
        # plt.plot(fake_clean_mag_psd[0], label = 'Fake Clean PSD - Reconstructed')
        # #plt.plot(idt_clean_recon_psd, label = 'Idt Clean PSD - Recon')
        # #plt.plot(fake_clean_psd, label = 'Fake Clean PSD - Enhanced')
        # #plt.plot(idt_clean_psd, label = 'Idt Clean PSD - Enhanced')
        # plt.legend()
        # plt.savefig(os.path.join(self.save_path, 'psd/radial_{}_{}.png'.format(epoch+1, iter+1)))
        # plt.close()

        plt.plot(real_noise_rect[0], label = 'Noise PSD')
        plt.plot(real_clean_rect[0], label = 'Clean PSD')
        plt.plot(fake_clean_recon_rect[0], label = 'Fake Clean - Translated')
        plt.plot(fake_clean_mag_rect[0], label = 'Fake Clean PSD - Reconstructed')
        #plt.plot(idt_clean_recon_psd, label = 'Idt Clean PSD - Recon')
        #plt.plot(fake_clean_psd, label = 'Fake Clean PSD - Enhanced')
        #plt.plot(idt_clean_psd, label = 'Idt Clean PSD - Enhanced')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'psd/rect_{}_{}.png'.format(epoch+1, iter+1)))
        plt.close()


    def step(self, model, epoch, iter) :

        if self.current_epoch != epoch :
            self.iteration = 0
            self.current_epoch = epoch

        average_idx = self.iteration % self.average_frequency
        #start = average_idx * self.batch_size
        #end = (average_idx+1) * self.batch_size

        self.real_noise_mean[average_idx] = self.zero_to_one(model.real_A.detach())
        self.real_clean_mean[average_idx] = self.zero_to_one(model.real_B.detach())
        self.fake_clean_recon_mean[average_idx] = self.zero_to_one(model.fake_B.detach())
        self.fake_clean_mag_mean[average_idx] = model.denoised_mag.detach()
        #self.fake_clean_mean[average_idx] = model.enhanced_mag_A.detach()
        #self.idt_clean_mean[average_idx] = model.enhanced_mag_B.detach()

        if self.iteration != 0 and average_idx == self.average_frequency-1 :
            self.plot_averaged_profile(epoch, iter)
            self.plot_averaged_1d_psd(epoch, iter)

        if self.iteration % self.sample_frequency == 0 :
            self.plot_sampled_profile(model, epoch, iter)

        self.iteration += 1

    def zero_to_one(self, mag) :
        return (mag + 1.) * (0.5)

def azimuthalAverage(image, center=None):

    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

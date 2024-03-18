import os
import torch
from torch import nn
from torchsummary import summary
from .networks import NestedUNet, PatchSampleF, PatchDiscriminator
from .networks import get_norm_layer, init_net
from .patchnce import PatchNCELoss
from .loss import LFLoss, RectAverage, SSIM
from utils.processing import get_magnitude, img2fft, fft2polar, normalize, inverse_normalize, polar2img, image_normalize

'''
STIG model : The frequency domain method for enhancing the generated images.
Input : Real image & Generated image -> torch.Tensor
Output : Enhanced generated image -> torch.Tensor

Augument
- opt : option configurator
- device : torch.cude.device()
'''
class STIG(nn.Module) :
    
    def __init__(self, opt, device) :
        super(STIG, self).__init__()

        self.device = device
        self.lambda_PSD = None
        self.first_setting = False
        self.radial_profile = RectAverage(opt.size, self.device)

        self._load_argument_options(opt)
        self._build_modules()
        self._build_optimizers()

        self.dc_pos = self.opt.size // 2 + 1
        self.sigma = opt.sigma
        self.in_channels = self.opt.input_nc
        self.out_channels = self.opt.output_nc

    def set_input(self, input, evaluation = False) :

        if not evaluation :
            self.image_A = input['noise']
            self.image_B = input['clean']

            self.image_A = self.image_A.to(self.device)
            self.image_B = self.image_B.to(self.device)

        else :
            self.image_A = input
            self.image_A = self.image_A.to(self.device)

    def to_frequency_domain(self, image) :
        fft = img2fft(image)
        magnitude, phase = fft2polar(fft)
        return magnitude, phase

    def forward(self):

        # Transformation image domain to frequency domain
        # Normalization magnitude to value of [-1.0, 1.0]
        self.real_A, self.real_A_phase = self.to_frequency_domain(self.image_A)
        self.real_B, self.real_B_phase = self.to_frequency_domain(self.image_B)

        self.real_A, self.A_vmax, self.A_vmin = normalize(self.real_A)
        self.real_B, self.B_vmax, self.B_vmin = normalize(self.real_B)

        # Forward noisy magnitude to generator network to get the clean version
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt else self.real_A

        self.fake = self.generator(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        else :
            self.idt_B = torch.ones_like(self.fake_B)

        fake_B_inverse_normed = inverse_normalize(self.fake_B, self.A_vmax, self.A_vmin)
        idt_B_inverse_normed = inverse_normalize(self.idt_B, self.B_vmax, self.B_vmin)


        self.denoised_image_A = polar2img(fake_B_inverse_normed, self.real_A_phase)
        self.identity_image_B = polar2img(idt_B_inverse_normed, self.real_B_phase)

        self.denoised_image_normed = image_normalize(self.denoised_image_A)
        self.denoised_mag = get_magnitude(self.denoised_image_normed)

        self.real_psd = self.radial_profile(self.real_B)
        self.denoised_psd = self.radial_profile(self.denoised_mag)

    def evaluation(self) :

        self.real_A, self.real_A_phase = self.to_frequency_domain(self.image_A)
        self.real_A, self.A_vmax, self.A_vmin = normalize(self.real_A)
        
        self.input_image_normed = image_normalize(self.image_A)
        self.input_mag = get_magnitude(self.input_image_normed)

        with torch.no_grad():
            self.fake_B = self.generator(self.real_A)

            fake_B_inverse_normed = inverse_normalize(self.fake_B, self.A_vmax, self.A_vmin)
            self.denoised_image_A = polar2img(fake_B_inverse_normed, self.real_A_phase)
            self.denoised_image_normed = image_normalize(self.denoised_image_A)
            self.denoised_mag = get_magnitude(self.denoised_image_normed)

    def update_optimizer(self) :
        
        self.forward()

        # update optimizerD
        self.set_requires_grad(self.discriminator, True)
        self.optimizerD.zero_grad()
        self.optimizerS.zero_grad()
        self.loss_D_total = self.compute_D_loss()
        self.loss_D_total.backward()
        self.optimizerD.step()
        self.optimizerS.step()

        # update optimizerG & optimizerM
        self.set_requires_grad(self.discriminator, False)
        self.optimizerG.zero_grad()
        self.optimizerM.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizerG.step()
        self.optimizerM.step()


    def compute_G_loss(self) :

        fake = self.fake_B
        pred_fake = self.discriminator(fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))

        fake_psd = self.denoised_psd
        pred_s_fake = self.spectral_discriminator(fake_psd)
        self.loss_SG_GAN = self.criterionGAN(pred_s_fake, torch.ones_like(pred_s_fake))

        self.loss_identity = (1. - self.criterionSSIM(self.image_B, self.identity_image_B))
        self.loss_fake_identity = (1. - self.criterionSSIM(self.image_A, self.denoised_image_A))
        self.loss_LF = self.criterionLF(self.fake_B, self.real_A) + self.criterionLF(self.idt_B, self.real_B)

        self.loss_NCE = self.compute_NCE_loss(self.real_A, self.fake_B)

        if self.nce_idt :
            self.loss_idt_NCE = self.compute_NCE_loss(self.real_B, self.idt_B)
            self.loss_NCE_total = self.loss_NCE + self.loss_idt_NCE
        else :
            self.loss_idt_NCE = torch.tensor(0.)
            self.loss_NCE_total = self.loss_NCE

        self.loss_DC = 0.
        self.loss_PSD = 0.
        self.loss_SYM = 0.

        self.loss_G = self.loss_G_GAN * self.lambda_GAN + self.loss_SG_GAN * self.lambda_PSD + self.loss_NCE_total * self.lambda_NCE + 3.0 * self.loss_identity + 1.0 * self.loss_fake_identity + self.lambda_LF * self.loss_LF

        return self.loss_G


    def compute_D_loss(self) :

        pred_fake = self.discriminator(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))

        pred_real = self.discriminator(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))

        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        pred_s_fake = self.spectral_discriminator(self.denoised_psd.detach())
        self.loss_SD_fake = self.criterionGAN(pred_s_fake, torch.zeros_like(pred_s_fake))

        pred_s_real = self.spectral_discriminator(self.real_psd)
        self.loss_SD_real = self.criterionGAN(pred_s_real, torch.ones_like(pred_s_real))

        self.loss_SD = (self.loss_SD_real + self.loss_SD_fake) * 0.5

        return self.loss_D * 1.0 + self.loss_SD * 1.0


    def compute_NCE_loss(self, src, dst) :

        n_layers = len(self.nce_layers)

        feat_q = self.generator.encode_sample(dst)
        feat_k = self.generator.encode_sample(src)
        
        feat_k_pool, sample_ids = self.mlp(feat_k, self.n_patches, None)
        feat_q_pool, _ = self.mlp(feat_q, self.n_patches, sample_ids)

        total_nce_loss = 0.0

        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def _build_modules(self) :

        self.generator = NestedUNet(self.input_nc, self.output_nc, spatial = False)
        self.discriminator = PatchDiscriminator(self.input_nc).to(self.device)
        self.mlp = PatchSampleF(use_mlp=True, nc=self.n_patches, device = self.device).to(self.device)
        
        self.generator = init_net(self.generator)
        self.discriminator = init_net(self.discriminator)
        self.mlp = init_net(self.mlp)

        self.criterionGAN = nn.MSELoss()
        self.criterionNCE = [PatchNCELoss(self.opt)]
        self.criterionIDT = nn.L1Loss()
        self.criterionLF = LFLoss(self.opt)
        self.criterionSSIM = SSIM(window_size = 35)

    def _build_optimizers(self) :

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.beta1, self.beta2))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.beta1, self.beta2))

    def _load_argument_options(self, opt) :

        self.opt = opt
        self.norm = get_norm_layer(self.opt.norm)
        self.n_patches = self.opt.num_patch
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        self.nce_idt = self.opt.nce_idt
        self.lambda_GAN = self.opt.lambda_GAN
        self.lambda_NCE = self.opt.lambda_NCE
        self.lambda_PSD = self.opt.lambda_PSD
        self.lambda_LF = self.opt.lambda_LF
        self.lambda_identity = self.opt.lambda_identity
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

    def save_checkpoint(self, save_path, n, name = None) :

        checkpoint = {
            'Generator' : self.generator.state_dict(),
            'Discriminator' : self.discriminator.state_dict(),
            'MLP' : self.mlp.state_dict(),
            'PSD' : self.spectral_discriminator.state_dict()
        }

        if name is None :
            file_name = 'parameters_{}_epoch.pt'.format(n)
        else :
            file_name = name

        torch.save(checkpoint, os.path.join(save_path, file_name))

    def load_checkpoint(self, load_path) :
        
        checkpoint = torch.load(load_path)
        self.generator.load_state_dict(checkpoint['Generator'])
        print('Successfully loaded checkpoint...')

    def step_scheduler(self) :

        self.schedulerD.step()
        self.schedulerG.step()
        self.schedulerS.step()
        self.schedulerM.step()

    def data_dependent_init(self, data, n_train_samples) :

        self.set_input(data)
        self.forward()

        self.psd = self.radial_profile(self.real_A)
        self.spectral_discriminator = nn.Sequential(
            nn.Linear(self.psd.shape[1], 1),
            nn.Sigmoid()
        ).to(self.device)
        self.optimizerS = torch.optim.Adam(self.spectral_discriminator.parameters(), lr = self.opt.lr, betas = (self.beta1, self.beta2))

        self.compute_D_loss().backward()
        self.compute_G_loss().backward()

        self.schedulerG = self.get_lr_scheduler(self.optimizerG, n_train_samples * int(self.opt.epoch * 0.2), n_train_samples * int(self.opt.epoch * 0.8))
        self.schedulerD = self.get_lr_scheduler(self.optimizerD, n_train_samples * int(self.opt.epoch * 0.2), n_train_samples * int(self.opt.epoch * 0.8))
        self.schedulerS = self.get_lr_scheduler(self.optimizerS, n_train_samples * int(self.opt.epoch * 0.2), n_train_samples * int(self.opt.epoch * 0.8))
        if self.lambda_NCE > 0. :
            self.optimizerM = torch.optim.Adam(self.mlp.parameters(), lr = self.opt.lr, betas = (self.beta1, self.beta2))
            self.schedulerM = self.get_lr_scheduler(self.optimizerM, n_train_samples * int(self.opt.epoch * 0.2), n_train_samples * int(self.opt.epoch * 0.8))
        self.first_setting = True

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_lr_scheduler(self, optimizer, fixed_lr_step, decay_lr_step) :

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - fixed_lr_step) / float(decay_lr_step + 1)
            return lr_l

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler
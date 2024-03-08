import numpy as np
import torch
import torchvision
from utils.util import OptionConfigurator, fix_randomseed
import os
from PIL import Image
from utils.metric import FIDCalculator
from pytorch_fid.fid_score import calculate_fid_given_paths

if __name__ == '__main__' :

    # Preparing evaluation step
    # Fix the randomness for reproducibility
    # Set GPU environment and tensorboard logging
    fix_randomseed(42)

    opt = OptionConfigurator().parse_options()

    is_gpu = torch.cuda.is_available()
    device = torch.device(opt.device) if is_gpu else torch.device('cpu')

    root = os.path.join('./results', opt.eval_root, 'eval')

    # Calculate fid between real images and [fake iamges, stig-refined images]
    if opt.eval_mode == 'image_fid' :

        real_path = os.path.join(root, 'clean')
        fake_path = os.path.join(root, 'noise')
        stig_path = os.path.join(root, 'denoised')

        print('Calculate fid of baseline profile.')
        baseline_path = [real_path, fake_path]
        baseline_fid = calculate_fid_given_paths(baseline_path, 64, device, 2048)

        print('Calculate fid of stig profile.')
        stig_path = [real_path, stig_path]
        stig_fid = calculate_fid_given_paths(stig_path, 64, device, 2048)

        print('FID of the original generated images : {:.2f}'.format(baseline_fid))
        print('FID of the refined generated images : {:.2f}'.format(stig_fid))

    # Calculate fid between real spectrum and [fake spectrum, stig-refined spectrum]
    elif opt.eval_mode == 'magnitude_fid' :

        real_path = os.path.join(root, 'clean_mag')
        fake_path = os.path.join(root, 'noise_mag')
        stig_path = os.path.join(root, 'denoised_mag')

        print('Calculate fid of baseline profile.')
        baseline_path = [real_path, fake_path]
        baseline_fid = calculate_fid_given_paths(baseline_path, 64, device, 2048)

        print('Calculate fid of stig profile.')
        stig_path = [real_path, stig_path]
        stig_fid = calculate_fid_given_paths(stig_path, 64, device, 2048)

        print('FID of the original generated spectrum : {:.2f}'.format(baseline_fid))
        print('FID of the refined generated spectrum : {:.2f}'.format(stig_fid))

    # Calculate lfd between real spectrum and [fake spectrum, stig-refined spectrum]
    elif opt.eval_mode == 'lfd' :

        real_path = os.path.join(root, 'clean')
        fake_path = os.path.join(root, 'noise')
        stig_path = os.path.join(root, 'denoised')

        calculator = FIDCalculator(opt, opt.dst, eval = True)

        real_mag_mean = calculator.average_magnitude(real_path)
        fake_mag_mean = calculator.average_magnitude(fake_path)
        stig_mag_mean = calculator.average_magnitude(stig_path)

        to_pil = torchvision.transforms.ToPILImage()
        real_pil = to_pil(real_mag_mean)
        fake_pil = to_pil(fake_mag_mean)
        stig_pil = to_pil(stig_mag_mean)

        real_pil.save(os.path.join('./results', opt.eval_root, 'eval', 'real_mag_average.png'))
        fake_pil.save(os.path.join('./results', opt.eval_root, 'eval', 'fake_mag_average.png'))
        stig_pil.save(os.path.join('./results', opt.eval_root, 'eval', 'stig_mag_average.png'))

        real_mean = np.array(Image.open(os.path.join('./results', opt.eval_root, 'eval', 'real_mag_average.png')))
        fake_mean = np.array(Image.open(os.path.join('./results', opt.eval_root, 'eval', 'fake_mag_average.png')))
        stig_mean = np.array(Image.open(os.path.join('./results', opt.eval_root, 'eval', 'stig_mag_average.png')))

        baseline_lfd = np.log(np.square(real_mean-fake_mean).sum(axis=(0, 1, 2))+1)
        stig_lfd = np.log(np.square(real_mean-stig_mean).sum(axis=(0, 1, 2))+1)

        print('Log frequency distance of the original generated spectrum : {:.2f}'.format(baseline_lfd))
        print('Log frequency distance of the refined generated spectrum : {:.2f}'.format(stig_lfd))